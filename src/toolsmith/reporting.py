"""Comparisons of recorded evaluation runs on identical frozen tasks.

- Runs must share the dataset manifest, base model revision, decoding settings, and tasks.
- Missing episodes are errors, never a smaller denominator.
- Run differences use a paired, task-level bootstrap.
- Per-success costs are None when a run solves nothing.
"""

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

MATCHING_FIELDS = ("dataset_manifest_sha256", "model", "model_revision", "generation")


@dataclass(frozen=True)
class Run:
    name: str
    config: dict
    rows: dict[str, dict]


def load_run(directory: Path, name: str) -> Run:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    lines = (directory / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    rows: dict[str, dict] = {}
    for line in lines:
        row = json.loads(line)
        if row["task_id"] in rows:
            raise ValueError(f"{name}: Duplicate task ID {row['task_id']}")
        rows[row["task_id"]] = row
    if config["tasks_completed"] != config["tasks_requested"] or len(rows) != len(lines):
        raise ValueError(
            f"{name}: incomplete run ({config['tasks_completed']} of "
            f"{config['tasks_requested']} episodes)"
        )
    if len(rows) != config["tasks_requested"]:
        raise ValueError(f"{name}: incomplete run ({len(rows)} rows recorded)")
    return Run(name, config, rows)


def merge_runs(directories: list[Path], name: str) -> tuple[dict, list[dict]]:
    """Combine shard runs of one model into one run's config and rows.

    - Every shard must share the matching fields and the adapter.
    - Shard indices must cover 0..N-1 exactly once, and task sets must not overlap.
    - The merged config lists each shard's run name, shard, and wall time.
    """
    shards = [load_run(d, f"{name}[{i}]") for i, d in enumerate(directories)]
    first = shards[0].config
    for shard in shards[1:]:
        for field in (*MATCHING_FIELDS, "adapter"):
            if shard.config.get(field) != first.get(field):
                raise ValueError(f"{shard.name}: shard settings differ: {field}")
    specs = [tuple(s.config.get("shard") or (0, 1)) for s in shards]
    if sorted(specs) != [(i, specs[0][1]) for i in range(specs[0][1])]:
        raise ValueError(f"Shards {specs} do not cover every shard index exactly once")
    rows: dict[str, dict] = {}
    for shard in shards:
        if overlap := sorted(set(rows) & set(shard.rows)):
            raise ValueError(f"Shards overlap on {len(overlap)} tasks, e.g. {overlap[0]}")
        rows |= shard.rows
    skip = {"shard", "run_name", "wall_seconds", "tasks_requested", "tasks_completed"}
    config = {key: value for key, value in first.items() if key not in skip}
    config |= {
        "run_name": name,
        "shards": [
            {
                "run_name": s.config.get("run_name"),
                "shard": s.config.get("shard"),
                "wall_seconds": s.config.get("wall_seconds"),
            }
            for s in shards
        ],
        "tasks_requested": len(rows),
        "tasks_completed": len(rows),
        "stopped_by_time_guard": any(s.config.get("stopped_by_time_guard") for s in shards),
    }
    return config, [rows[task_id] for task_id in sorted(rows)]


def split_of(task_id: str) -> str:
    return task_id.split("-", 1)[0]


def paired_bootstrap(
    reference: list[bool], candidate: list[bool], *, seed: int = 17, resamples: int = 2000
) -> dict:
    """Candidate minus reference success rate with a 95% percentile interval."""
    if len(reference) != len(candidate) or not reference:
        raise ValueError("Bootstrap needs two equally long, non-empty paired outcome lists")
    deltas = [int(c) - int(r) for r, c in zip(reference, candidate, strict=True)]
    count = len(deltas)
    rng = random.Random(seed)
    samples = sorted(
        sum(deltas[rng.randrange(count)] for _ in range(count)) / count for _ in range(resamples)
    )
    return {
        "difference": sum(deltas) / count,
        "low": samples[int(0.025 * (resamples - 1))],
        "high": samples[int(round(0.975 * (resamples - 1)))],
        "resamples": resamples,
        "seed": seed,
    }


def _stats(rows: list[dict]) -> dict:
    count = len(rows)
    successes = sum(r["success"] for r in rows)
    tokens = sum(r["new_tokens"] for r in rows)
    seconds = sum(r["seconds"] for r in rows)
    return {
        "episodes": count,
        "successes": successes,
        "success_rate": successes / count,
        "infrastructure_errors": sum(r["reward"] is None for r in rows),
        "wrong_submissions": sum(
            r["terminal_reason"] == "submitted" and not r["success"] for r in rows
        ),
        "terminal_reasons": dict(sorted(Counter(r["terminal_reason"] for r in rows).items())),
        "mean_new_tokens": tokens / count,
        "mean_seconds": seconds / count,
        "mean_tool_calls": sum(r["tool_calls"] for r in rows) / count,
        "new_tokens_per_success": tokens / successes if successes else None,
        "seconds_per_success": seconds / successes if successes else None,
    }


def _check_comparable(runs: list[Run]) -> None:
    first = runs[0]
    for run in runs[1:]:
        for field in MATCHING_FIELDS:
            if run.config.get(field) != first.config.get(field):
                raise ValueError(f"{run.name} and {first.name} are not comparable: {field} differs")
        if set(run.rows) != set(first.rows):
            raise ValueError(f"{run.name} and {first.name}: task sets differ")


def compare(runs: list[Run], *, seed: int = 17, resamples: int = 2000) -> dict:
    """Per-run statistics and paired differences of each later run against the first."""
    if not runs:
        raise ValueError("No runs to compare")
    _check_comparable(runs)
    task_ids = sorted(runs[0].rows)
    by_split = defaultdict(list)
    for task_id in task_ids:
        by_split[split_of(task_id)].append(task_id)
    report = {
        "tasks": len(task_ids),
        "matching": {field: runs[0].config.get(field) for field in MATCHING_FIELDS},
        "runs": [],
        "pairs": [],
    }
    for run in runs:
        report["runs"].append(
            {
                "name": run.name,
                "adapter": run.config.get("adapter"),
                "overall": _stats([run.rows[t] for t in task_ids]),
                "splits": {
                    split: _stats([run.rows[t] for t in ids]) for split, ids in by_split.items()
                },
            }
        )
    reference = runs[0]
    for candidate in runs[1:]:
        report["pairs"].append(
            {
                "reference": reference.name,
                "candidate": candidate.name,
                "overall": _paired(reference, candidate, task_ids, seed, resamples),
                "splits": {
                    split: _paired(reference, candidate, ids, seed, resamples)
                    for split, ids in by_split.items()
                },
            }
        )
    return report


def _paired(reference: Run, candidate: Run, ids: list[str], seed: int, resamples: int) -> dict:
    before = [reference.rows[t]["success"] for t in ids]
    after = [candidate.rows[t]["success"] for t in ids]
    return {
        **paired_bootstrap(before, after, seed=seed, resamples=resamples),
        "gained": sum(a and not b for b, a in zip(before, after, strict=True)),
        "lost": sum(b and not a for b, a in zip(before, after, strict=True)),
    }


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_markdown(report: dict) -> str:
    runs = report["runs"]
    splits = list(runs[0]["splits"])
    lines = [
        "| Run | " + " | ".join(splits) + " | All | Wrong patches | Tokens/episode |",
        "| --- |" + " --- |" * (len(splits) + 3),
    ]
    for run in runs:
        cells = [f"{s['successes']}/{s['episodes']}" for s in (run["splits"][x] for x in splits)]
        overall = run["overall"]
        rate = _percent(overall["success_rate"])
        lines.append(
            f"| {run['name']} | "
            + " | ".join(cells)
            + f" | {overall['successes']}/{overall['episodes']} ({rate})"
            + f" | {overall['wrong_submissions']} | {overall['mean_new_tokens']:.0f} |"
        )
    if report["pairs"]:
        lines += [
            "",
            "| Comparison | Split | Difference (95% CI) | Gained | Lost |",
            "| --- | --- | --- | --- | --- |",
        ]
        for pair in report["pairs"]:
            label = f"{pair['candidate']} vs {pair['reference']}"
            for split, stats in [*pair["splits"].items(), ("all", pair["overall"])]:
                lines.append(
                    f"| {label} | {split} | {100 * stats['difference']:+.1f} pts "
                    f"({100 * stats['low']:+.1f} to {100 * stats['high']:+.1f}) | "
                    f"{stats['gained']} | {stats['lost']} |"
                )
    return "\n".join(lines) + "\n"


def write_report(output: Path, report: dict) -> str:
    """Write report.json and report.md to a new directory; never overwrite."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = render_markdown(report)
    (output / "report.md").write_text(markdown)
    return markdown
