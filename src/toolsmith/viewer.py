"""Static, read-only HTML viewer over recorded evaluation runs.

- Shows public task inputs (bug report, source, visible tests) and the router's decision.
- For each run shows outcome, summary score, last visible-test result, the tool timeline,
  and the submitted patch.
- Takes public ``BugTask`` inputs only; all recorded text is HTML-escaped.
- Policies without recorded runs are shown as "not run".
"""

import html
import json
from pathlib import Path

from toolsmith.reporting import Run, compare, split_of
from toolsmith.routing import RoutingPolicy, route_task
from toolsmith.tasks import BugTask

OBSERVATION_CHARS = 4000
STYLE = """
body { font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; }
body { padding: 0 1rem; }
pre { background: #f4f4f4; padding: .75rem; overflow-x: auto; white-space: pre-wrap; }
table { border-collapse: collapse; } td, th { border: 1px solid #ccc; padding: .25rem .5rem; }
.ok { color: #1a7f37; } .fail { color: #cf222e; } .muted { color: #666; }
section.run { border-top: 2px solid #ddd; margin-top: 2rem; }
"""


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
        f"<body>{body}</body></html>\n"
    )


def _pre(text: str) -> str:
    return f"<pre>{html.escape(text)}</pre>"


def _outcome(row: dict) -> str:
    if row["success"]:
        return '<span class="ok">passed</span>'
    return f'<span class="fail">{html.escape(row["terminal_reason"])}</span>'


def _observation(observation: dict) -> str:
    text = json.dumps(observation, indent=1, ensure_ascii=False)
    if len(text) > OBSERVATION_CHARS:
        omitted = len(text) - OBSERVATION_CHARS
        text = text[:OBSERVATION_CHARS] + f"\n… {omitted} more characters"
    return _pre(text)


def _run_section(run: Run, row: dict) -> str:
    score = row.get("score")
    visible = [e for e in row["events"] if e["tool"] == "run_visible_tests"]
    parts = [
        f'<section class="run"><h2>{html.escape(run.name)}: {_outcome(row)}</h2>',
        "<p>"
        f"Terminal reason: <code>{html.escape(row['terminal_reason'])}</code>; "
        f"tool calls: {row['tool_calls']}; new tokens: {row['new_tokens']}; "
        f"model seconds: {row['seconds']:.1f}</p>",
        "<p>Last visible tests: "
        + (
            f"<code>{html.escape(str(visible[-1]['observation'].get('status')))}</code>"
            if visible
            else '<span class="muted">not run</span>'
        )
        + "; hidden evaluator: "
        + (
            f"<code>{html.escape(score['status'])}</code>, reward {score['reward']}"
            if score
            else '<span class="muted">no patch scored</span>'
        )
        + "</p>",
        "<h3>Timeline</h3>",
    ]
    for turn in row["turns"]:
        flags = [name for name in ("truncated", "context_exhausted") if turn.get(name)]
        parts.append(
            f'<h4>Turn {turn["turn"]} <span class="muted">({turn["new_tokens"]} new tokens'
            + (f", {', '.join(flags)}" if flags else "")
            + ")</span></h4>"
        )
        parts.append(_pre(turn["text"]))
        for event in turn.get("dispatched", []):
            parts.append(f"<p>Tool result: <code>{html.escape(event['tool'])}</code></p>")
            parts.append(_observation(event["observation"]))
    parts.append("<h3>Submitted patch</h3>")
    patch = row.get("submitted_patch")
    parts.append(_pre(patch) if patch else '<p class="muted">none</p>')
    parts.append("</section>")
    return "".join(parts)


def _task_page(task: BugTask, runs: list[Run], policy: RoutingPolicy) -> str:
    decision = route_task(task, policy)
    files = "".join(
        f"<h3><code>{html.escape(f.path)}</code></h3>{_pre(f.content)}"
        for f in (*task.source_files, *task.visible_tests)
    )
    body = (
        '<p><a href="../index.html">← all tasks</a></p>'
        f"<h1><code>{html.escape(task.task_id)}</code></h1>"
        f"<h2>Bug report</h2>{_pre(task.issue)}"
        "<h2>Routing</h2>"
        f"<p>Upfront decision: <code>{html.escape(decision.route)}</code> "
        f"({html.escape(decision.reason)}). Large-model escalation: "
        '<span class="muted">not run (no escalation model configured)</span>.</p>'
        f"<h2>Public inputs</h2>{files}"
        + "".join(_run_section(run, run.rows[task.task_id]) for run in runs)
    )
    return _page(task.task_id, body)


def build_site(
    runs: list[Run], tasks: dict[str, BugTask], policy: RoutingPolicy, output: Path
) -> Path:
    """Write index.html and one page per task into a new directory; return the index."""
    report = compare(runs)  # also enforces identical, complete task sets
    task_ids = sorted(runs[0].rows)
    if missing := [t for t in task_ids if t not in tasks]:
        raise ValueError(f"Public inputs missing for {len(missing)} tasks, e.g. {missing[0]}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "tasks").mkdir()
    for task_id in task_ids:
        (output / "tasks" / f"{task_id}.html").write_text(
            _task_page(tasks[task_id], runs, policy), encoding="utf-8"
        )
    header = "".join(f"<th>{html.escape(run.name)}</th>" for run in runs)
    rows = "".join(
        f'<tr><td><a href="tasks/{html.escape(t)}.html"><code>{html.escape(t)}</code></a></td>'
        f"<td>{html.escape(split_of(t))}</td>"
        + "".join(f"<td>{_outcome(run.rows[t])}</td>" for run in runs)
        + "</tr>"
        for t in task_ids
    )
    adapters = "".join(
        f"<li><b>{html.escape(run.name)}</b>: "
        + (
            "adapter from <code>"
            + html.escape(
                run.config["adapter"].get("source") or f"kernel:{run.config['adapter']['kernel']}"
            )
            + "</code>, "
            f"sha256 <code>{html.escape(run.config['adapter']['sha256'][:16])}…</code>"
            if run.config.get("adapter")
            else "prompt-only (no adapter)"
        )
        + "</li>"
        for run in runs
    )
    body = (
        "<h1>Toolsmith recorded runs</h1>"
        f"<p>{report['tasks']} frozen tasks; dataset manifest "
        f"<code>{html.escape(str(report['matching']['dataset_manifest_sha256']))}</code>. "
        "Only public inputs and evaluator summaries are shown.</p>"
        f"<ul>{adapters}</ul>"
        f"<h2>Summary</h2>{_summary_tables(report)}"
        f"<h2>Tasks</h2><table><tr><th>Task</th><th>Split</th>{header}</tr>{rows}</table>"
    )
    index = output / "index.html"
    index.write_text(_page("Toolsmith runs", body), encoding="utf-8")
    return index


def _summary_tables(report: dict) -> str:
    splits = list(report["runs"][0]["splits"])
    head = "".join(f"<th>{html.escape(s)}</th>" for s in splits)
    rows = ""
    for run in report["runs"]:
        cells = "".join(
            f"<td>{run['splits'][s]['successes']}/{run['splits'][s]['episodes']}</td>"
            for s in splits
        )
        overall = run["overall"]
        rows += (
            f"<tr><td>{html.escape(run['name'])}</td>{cells}"
            f"<td><b>{overall['successes']}/{overall['episodes']}</b> "
            f"({100 * overall['success_rate']:.1f}%)</td>"
            f"<td>{overall['wrong_submissions']}</td><td>{overall['mean_new_tokens']:.0f}</td></tr>"
        )
    tables = (
        f"<table><tr><th>Run</th>{head}<th>All</th><th>Wrong patches</th>"
        f"<th>Tokens/episode</th></tr>{rows}</table>"
    )
    if report["pairs"]:
        pair_rows = ""
        for pair in report["pairs"]:
            label = html.escape(f"{pair['candidate']} vs {pair['reference']}")
            for split, s in [*pair["splits"].items(), ("all", pair["overall"])]:
                pair_rows += (
                    f"<tr><td>{label}</td><td>{html.escape(split)}</td>"
                    f"<td>{100 * s['difference']:+.1f} pts ({100 * s['low']:+.1f} to "
                    f"{100 * s['high']:+.1f})</td><td>{s['gained']}</td><td>{s['lost']}</td></tr>"
                )
        tables += (
            "<h3>Paired differences (task-level bootstrap, 95% CI)</h3>"
            "<table><tr><th>Comparison</th><th>Split</th><th>Difference</th>"
            f"<th>Gained</th><th>Lost</th></tr>{pair_rows}</table>"
        )
    return tables
