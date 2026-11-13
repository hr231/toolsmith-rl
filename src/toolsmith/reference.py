"""Execute dataset fixtures in Docker and build replay-checked reference trajectories.

- Verifies per task that the original source fails its visible test.
- Verifies that the reference patch passes every private check.
- Builds train/dev-only scripted demonstrations through the public Episode tools.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from toolsmith.dataset import SOURCE_PATH, SPLITS, DatasetRecord, load_records, validate_dataset
from toolsmith.episode import Episode
from toolsmith.evaluator import apply_patch, evaluate_patch
from toolsmith.sandbox import DEFAULT_IMAGE, Sandbox, SandboxError, SandboxLimits

DEMONSTRATION_SPLITS = ("train", "dev")
MAX_WORKERS = 8


def _workers(value: int) -> int:
    if type(value) is not int or not 1 <= value <= MAX_WORKERS:
        raise ValueError(f"workers must be an integer in [1, {MAX_WORKERS}]")
    return value


def _line(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"


def _private_counts(evaluation: dict) -> tuple[int, int]:
    cases = evaluation.get("cases", [])
    return sum(case["success"] for case in cases), len(cases)


def verify_record(record: DatasetRecord, *, image: str, limits: SandboxLimits) -> dict:
    """Run the original visible test and independently evaluate the reference patch."""
    result = {"task_id": record.task.task_id, "split": record.metadata["split"]}
    try:
        with Sandbox(record.task, image=image, limits=limits) as sandbox:
            original = sandbox.run_visible_tests()
    except SandboxError as error:
        return result | {"success": False, "status": "infrastructure_error", "error": str(error)}
    payload = original.payload or {}
    # A genuine defect fails an assertion; collection errors or crashes do not count.
    original_fails = (
        original.status == "failed"
        and payload.get("collected", 0) > 0
        and payload.get("failed", 0) > 0
        and payload.get("errors", 1) == 0
    )
    evaluation = evaluate_patch(
        record.task, record.reference_patch, record.checks, image=image, limits=limits
    )
    passed, total = _private_counts(evaluation)
    return result | {
        "success": original_fails and evaluation["success"],
        "original_status": original.status,
        "original_payload": original.payload,
        "original_fails_visibly": original_fails,
        "reference_status": evaluation["status"],
        "reference_visible_status": (evaluation["visible"] or {}).get("status"),
        "private_passed": passed,
        "private_total": total,
        "image_ids": sorted(
            {identifier for identifier in (original.image_id, evaluation["image_id"]) if identifier}
        ),
    }


def _new_directory(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)


def _image_identity(rows: list[dict]) -> str | None:
    identities = {identifier for row in rows for identifier in row.get("image_ids", [])}
    if len(identities) > 1:
        raise ValueError(f"Executions used multiple images: {sorted(identities)}")
    return next(iter(identities), None)


def verify_dataset(
    directory: Path,
    output: Path,
    *,
    image: str = DEFAULT_IMAGE,
    limits: SandboxLimits | None = None,
    workers: int = 4,
) -> dict:
    """Execute every task in every split; write a report bound to dataset hashes."""
    workers = _workers(workers)
    limits = limits or SandboxLimits()
    summary = validate_dataset(directory)
    records = [record for split in SPLITS for record in load_records(directory, split)]
    _new_directory(output)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda r: verify_record(r, image=image, limits=limits), records))
    failures = [row["task_id"] for row in rows if not row["success"]]
    report = {
        "success": not failures,
        "dataset_manifest_sha256": summary["manifest_sha256"],
        "dataset_files": summary["files"],
        "image_id": _image_identity(rows),
        "limits": asdict(limits),
        "counts": summary["counts"],
        "verified": len(rows) - len(failures),
        "total": len(rows),
        "failures": failures,
    }
    with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
        stream.writelines(_line(row) for row in rows)
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _reference_actions(record: DatasetRecord) -> list[tuple[str, dict]]:
    fixed = apply_patch(record.task, record.reference_patch)
    content = next(file.content for file in fixed.source_files if file.path == SOURCE_PATH)
    return [
        ("read_file", {"path": SOURCE_PATH}),
        ("run_visible_tests", {}),
        ("edit_file", {"path": SOURCE_PATH, "content": content}),
        ("run_visible_tests", {}),
        ("submit_patch", {}),
    ]


def _comparable(event) -> tuple:
    # Pytest timing text differs between runs; status and exit code must match.
    if event.tool == "run_visible_tests":
        observation = event.observation
        return (
            event.index,
            event.tool,
            event.arguments,
            observation.get("status"),
            observation.get("exit_code"),
        )
    return (event.index, event.tool, event.arguments_json, event.observation_json)


def build_demonstration(
    record: DatasetRecord, *, image: str, limits: SandboxLimits
) -> tuple[dict, dict]:
    """Return (public transcript, private verification) for one train/dev record."""
    task_id = record.task.task_id
    private = {"task_id": task_id, "split": record.metadata["split"], "success": False}
    if record.metadata["split"] not in DEMONSTRATION_SPLITS:
        raise ValueError("Demonstrations are restricted to train and dev tasks")
    actions = _reference_actions(record)
    try:
        with Episode(record.task, image=image, sandbox_limits=limits) as episode:
            for tool, arguments in actions:
                episode.dispatch(tool, arguments)
            events = episode.events
            transcript = {
                "schema_version": 1,
                "task_id": task_id,
                "split": record.metadata["split"],
                "initial_context": episode.initial_context,
                "events": [
                    {
                        "index": e.index,
                        "tool": e.tool,
                        "arguments": e.arguments,
                        "observation": e.observation,
                    }
                    for e in events
                ],
                "terminal_reason": episode.terminal_reason,
                "submitted_patch": episode.submitted_patch,
                "context_bytes": episode.context_bytes,
            }
            image_id = episode.image_id
        statuses = [e.observation.get("status") for e in events if e.tool == "run_visible_tests"]
        if statuses != ["failed", "passed"] or transcript["terminal_reason"] != "submitted":
            return transcript, private | {
                "reason": "unexpected_trajectory",
                "visible_statuses": statuses,
                "terminal_reason": transcript["terminal_reason"],
            }
        with Episode(record.task, image=image, sandbox_limits=limits) as replay:
            for tool, arguments in actions:
                replay.dispatch(tool, arguments)
            replay_matches = [_comparable(e) for e in replay.events] == [
                _comparable(e) for e in events
            ] and replay.submitted_patch == transcript["submitted_patch"]
    except SandboxError as error:
        return {}, private | {"reason": "infrastructure_error", "error": str(error)}
    evaluation = evaluate_patch(
        record.task, transcript["submitted_patch"], record.checks, image=image, limits=limits
    )
    passed, total = _private_counts(evaluation)
    return transcript, private | {
        "success": replay_matches and evaluation["success"],
        "replay_matches": replay_matches,
        "evaluation_status": evaluation["status"],
        "private_passed": passed,
        "private_total": total,
        "patch_sha256": evaluation["patch_sha256"],
        "image_ids": sorted({i for i in (image_id, evaluation["image_id"]) if i}),
    }


def build_demonstrations(
    directory: Path,
    output: Path,
    *,
    image: str = DEFAULT_IMAGE,
    limits: SandboxLimits | None = None,
    workers: int = 4,
) -> dict:
    """Write verified train/dev transcripts; unverified ones are reported, never kept."""
    workers = _workers(workers)
    limits = limits or SandboxLimits()
    summary = validate_dataset(directory)
    records = {split: load_records(directory, split) for split in DEMONSTRATION_SPLITS}
    _new_directory(output)
    (output / "private").mkdir()
    counts, failures, verification_rows = {}, [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for split, split_records in records.items():
            results = list(
                pool.map(
                    lambda r: build_demonstration(r, image=image, limits=limits), split_records
                )
            )
            kept = [public for public, private in results if private["success"]]
            failures.extend(private["task_id"] for _, private in results if not private["success"])
            verification_rows.extend(private for _, private in results)
            counts[split] = len(kept)
            with (output / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
                stream.writelines(_line(row) for row in kept)
    with (output / "private" / "verification.jsonl").open("x", encoding="utf-8") as stream:
        stream.writelines(_line(row) for row in verification_rows)
    files = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in [*(f"{s}.jsonl" for s in DEMONSTRATION_SPLITS), "private/verification.jsonl"]
    }
    manifest = {
        "success": not failures,
        "dataset_manifest_sha256": summary["manifest_sha256"],
        "image_id": _image_identity(verification_rows),
        "limits": asdict(limits),
        "counts": counts,
        "attempted": {split: len(rows) for split, rows in records.items()},
        "failures": failures,
        "files": files,
        "scope": "Scripted reference trajectories replayed through real tools; not model output.",
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
