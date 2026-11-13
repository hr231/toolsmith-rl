"""Apply source-only patches as data and score isolated runs on the host.

- Applies exact unified diffs to declared source files.
- Loads private checks and compares returned values on the host only.
- Writes evaluation artifacts to a new directory; never replaces an existing one.
"""

import hashlib
import json
import keyword
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from toolsmith.sandbox import DEFAULT_IMAGE, RunResult, Sandbox, SandboxError, SandboxLimits
from toolsmith.tasks import BugTask


class PatchError(ValueError):
    """A patch is not an exact, nonempty edit of declared source files."""


_HUNK = re.compile(r"@@ -([0-9]+)(?:,([0-9]+))? \+([0-9]+)(?:,([0-9]+))? @@(?: [^\n]*)?\n")
_INDEX = re.compile(r"index [0-9a-f]+\.\.[0-9a-f]+(?: 100644)?\n")
_NO_NEWLINE = "\\ No newline at end of file\n"
_MISSING = object()


def _manifest(task: BugTask) -> dict:
    return {
        "schema_version": 1,
        "task_id": task.task_id,
        "issue": task.issue,
        "source_files": {file.path: file.content for file in task.source_files},
        "visible_tests": {file.path: file.content for file in task.visible_tests},
    }


def _lines(text: str) -> list[str]:
    """Split only at LF; other Unicode line separators are ordinary source text."""
    parts = text.split("\n")
    return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _hunk(lines: list[str], position: int) -> tuple[int, int, list[str], list[str], int]:
    match = _HUNK.fullmatch(lines[position])
    if match is None:
        raise PatchError("Expected a standard unified hunk header")
    try:
        old_start, old_count, new_start, new_count = (
            int(value) if value is not None else 1 for value in match.groups()
        )
    except ValueError as error:
        raise PatchError("Hunk line counts are too large") from error
    if (old_count and not old_start) or (new_count and not new_start):
        raise PatchError("Nonempty hunk ranges must start at line 1 or later")
    old_offset = old_start - 1 if old_count else old_start
    new_offset = new_start - 1 if new_count else new_start
    old, new = [], []
    changed = False
    position += 1
    while len(old) < old_count or len(new) < new_count:
        if position >= len(lines) or lines[position][:1] not in {" ", "+", "-"}:
            raise PatchError("Hunk body does not match its declared line counts")
        line = lines[position]
        prefix, content = line[0], line[1:]
        position += 1
        if position < len(lines) and lines[position] == _NO_NEWLINE:
            content = content[:-1]
            position += 1
        if prefix in {" ", "-"}:
            old.append(content)
        if prefix in {" ", "+"}:
            new.append(content)
        changed |= prefix != " "
        if len(old) > old_count or len(new) > new_count:
            raise PatchError("Hunk body exceeds its declared line counts")
    if not changed:
        raise PatchError("Every hunk must contain a source edit")
    return old_offset, new_offset, old, new, position


def apply_patch(task: BugTask, patch: str) -> BugTask:
    """Apply an exact unified diff without invoking patch, Git, or project code.

    - Accepts unquoted ``diff --git`` paths and an optional regular-file index line.
    - Rejects modes, renames, binary patches, timestamps, creation/deletion headers,
      and undeclared paths.
    """
    if not isinstance(patch, str) or not patch or not patch.endswith("\n"):
        raise PatchError("A nonempty newline-terminated unified source patch is required")
    try:
        patch.encode("utf-8")
    except UnicodeEncodeError as error:
        raise PatchError("Patch must be valid UTF-8") from error
    data = _manifest(task)
    BugTask.from_dict(data)
    original = data["source_files"].copy()
    lines = _lines(patch)
    position = 0
    seen = set()
    while position < len(lines):
        metadata = None
        if lines[position].startswith("diff --git "):
            fields = lines[position].removesuffix("\n").split(" ")
            if len(fields) != 4 or fields[:2] != ["diff", "--git"]:
                raise PatchError("Unsupported git diff metadata")
            metadata = (fields[2], fields[3])
            position += 1
            if position < len(lines) and lines[position].startswith("index "):
                if _INDEX.fullmatch(lines[position]) is None:
                    raise PatchError("Unsupported git index metadata or file mode")
                position += 1
        if position + 1 >= len(lines):
            raise PatchError("Missing unified file headers")
        old_header, new_header = lines[position : position + 2]
        if not old_header.startswith("--- a/") or not new_header.startswith("+++ b/"):
            raise PatchError("Only canonical a/ and b/ source file headers are allowed")
        old_path, new_path = old_header[6:-1], new_header[6:-1]
        if old_path != new_path or old_path not in original:
            raise PatchError("Patch must edit the same existing declared source path")
        if metadata is not None and metadata != (f"a/{old_path}", f"b/{old_path}"):
            raise PatchError("Git metadata paths disagree with unified headers")
        if old_path in seen:
            raise PatchError("Duplicate file sections are forbidden")
        seen.add(old_path)
        position += 2
        source = _lines(original[old_path])
        output = []
        cursor = 0
        previous_offset = -1
        hunks = 0
        while position < len(lines) and lines[position].startswith("@@"):
            old_offset, new_offset, old, new, position = _hunk(lines, position)
            if old_offset < cursor or old_offset <= previous_offset or old_offset > len(source):
                raise PatchError("Hunks overlap, are out of order, or exceed the source")
            output.extend(source[cursor:old_offset])
            if new_offset != len(output):
                raise PatchError("New hunk position disagrees with preceding source edits")
            if source[old_offset : old_offset + len(old)] != old:
                raise PatchError("Hunk context does not exactly match the original source")
            output.extend(new)
            cursor = old_offset + len(old)
            previous_offset = old_offset
            hunks += 1
        if not hunks:
            raise PatchError("Every file section must contain a hunk")
        output.extend(source[cursor:])
        if any(not line.endswith("\n") for line in output[:-1]):
            raise PatchError("No-newline markers are only valid at the end of a file")
        content = "".join(output)
        if content == original[old_path]:
            raise PatchError("Unchanged source file sections are forbidden")
        data["source_files"][old_path] = content
    try:
        return BugTask.from_dict(data)
    except ValueError as error:
        raise PatchError(str(error)) from error


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
    return value


def _identifier(value: object, name: str) -> None:
    text = _text(value, name)
    if not text.isidentifier() or keyword.iskeyword(text):
        raise ValueError(f"{name} must be a Python identifier")


def _task_id(value: object) -> None:
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", _text(value, "task_id")) is None:
        raise ValueError("Invalid private-check task_id")


def _json_value(value: object) -> None:
    try:
        _validate_json_value(value)
    except RecursionError as error:
        raise ValueError("Private JSON is cyclic or nested too deeply") from error


def _validate_json_value(value: object) -> None:
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is str:
        _text(value, "JSON string")
        return
    if type(value) is list:
        for item in value:
            _validate_json_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            _text(key, "JSON object key")
            _validate_json_value(item)
        return
    raise ValueError("Private checks must contain only finite JSON values")


@dataclass(frozen=True)
class CheckSpec:
    id: str
    module: str
    function: str
    args: list
    kwargs: dict
    expected: object = field(default=_MISSING, repr=False)
    raises: str | None = None

    def __post_init__(self) -> None:
        if not _text(self.id, "case id").strip():
            raise ValueError("Private case id must be nonempty")
        for part in _text(self.module, "module").split("."):
            _identifier(part, "module")
        _identifier(self.function, "function")
        if type(self.args) is not list or type(self.kwargs) is not dict:
            raise ValueError("Case args must be a list and kwargs must be an object")
        _json_value(self.args)
        _json_value(self.kwargs)
        if (self.expected is _MISSING) == (self.raises is None):
            raise ValueError("Each private case requires exactly one of expected or raises")
        if self.raises is not None:
            _identifier(self.raises, "raises")
        else:
            _json_value(self.expected)


@dataclass(frozen=True)
class PrivateChecks:
    task_id: str
    cases: tuple[CheckSpec, ...]

    def __post_init__(self) -> None:
        _task_id(self.task_id)
        if type(self.cases) is not tuple or not self.cases:
            raise ValueError("Private cases must be a nonempty tuple")
        identifiers = set()
        for case in self.cases:
            if not isinstance(case, CheckSpec):
                raise ValueError("Private cases must contain CheckSpec records")
            case.__post_init__()
            if case.id in identifiers:
                raise ValueError("Duplicate private case id")
            identifiers.add(case.id)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")


def load_checks(path: Path) -> PrivateChecks:
    """Load exact private-check JSON; do not import any named module."""
    data = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_nonfinite,
    )
    if type(data) is not dict or set(data) != {"schema_version", "task_id", "cases"}:
        raise ValueError("Private checks require exactly schema_version, task_id, and cases")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("schema_version must be the integer 1")
    if type(data["cases"]) is not list:
        raise ValueError("Private cases must be a JSON array")
    cases = []
    fields = {"id", "module", "function", "args", "kwargs"}
    for case in data["cases"]:
        if type(case) is not dict or set(case) not in (fields | {"expected"}, fields | {"raises"}):
            raise ValueError("Invalid private case fields; require exactly one expectation")
        if "raises" in case and case["raises"] is None:
            raise ValueError("raises must be an exception class identifier")
        cases.append(CheckSpec(**case))
    return PrivateChecks(data["task_id"], tuple(cases))


def _equal_json(actual: object, expected: object) -> bool:
    if type(actual) in {int, float} and type(expected) in {int, float}:
        return actual == expected
    if type(actual) is not type(expected):
        return False
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _equal_json(left, right) for left, right in zip(actual, expected, strict=True)
        )
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(
            _equal_json(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


def _case_result(case: CheckSpec, run: RunResult) -> dict:
    payload = run.payload
    success = False
    if run.status == "passed" and isinstance(payload, dict):
        if case.raises is not None:
            success = payload.get("kind") == "exception" and payload.get("type") == case.raises
        elif payload.get("kind") == "return" and "value" in payload:
            success = _equal_json(payload["value"], case.expected)
    result = asdict(run)
    result.update(id=case.id, success=success)
    if run.status == "passed" and not success:
        result["status"] = "failed"
    return result


def _infrastructure(error: SandboxError, limits: SandboxLimits) -> RunResult:
    message = str(error).encode("utf-8")[: limits.output_bytes].decode("utf-8", errors="ignore")
    return RunResult("infrastructure_error", "", message, None, 0.0, "")


def evaluate_patch(
    task: BugTask,
    patch: str,
    checks: PrivateChecks,
    *,
    image: str = DEFAULT_IMAGE,
    limits: SandboxLimits | None = None,
) -> dict:
    """Evaluate a source patch in fresh containers, comparing private values on the host.

    - Reward 1: complete repair; 0: incorrect or resource-limited run.
    - Reward -1: invalid patch; None: infrastructure failure.
    - Raises ValueError for malformed evaluator configuration.
    """
    if not isinstance(checks, PrivateChecks):
        raise ValueError("checks must be PrivateChecks")
    checks.__post_init__()
    if checks.task_id != task.task_id:
        raise ValueError("Private checks and public task IDs must match")
    paths = {file.path for file in task.source_files}
    for case in checks.cases:
        base = "src/" + case.module.replace(".", "/")
        if base + ".py" not in paths and base + "/__init__.py" not in paths:
            raise ValueError(f"Private module {case.module!r} does not resolve to task source")
    if not isinstance(image, str) or not image or any(char.isspace() for char in image):
        raise ValueError("image must be a nonempty Docker image reference")
    if limits is None:
        limits = SandboxLimits()
    elif not isinstance(limits, SandboxLimits):
        raise ValueError("limits must be SandboxLimits")
    patch_hash = None
    if isinstance(patch, str):
        try:
            patch_hash = hashlib.sha256(patch.encode("utf-8")).hexdigest()
        except UnicodeEncodeError:
            pass
    result = {
        "task_id": task.task_id,
        "patch_sha256": patch_hash,
        "limits": asdict(limits),
        "image_id": None,
        "visible": None,
        "cases": [],
    }
    try:
        patched = apply_patch(task, patch)
    except PatchError as error:
        result.update(status="invalid_patch", success=False, reward=-1, reason=str(error))
        return result
    visible = None
    case_results = []
    try:
        with Sandbox(patched, image=image, limits=limits) as sandbox:
            try:
                visible = sandbox.run_visible_tests()
            except SandboxError as error:
                visible = _infrastructure(error, limits)
            for case in checks.cases:
                try:
                    run = sandbox.run_check(case.module, case.function, case.args, case.kwargs)
                except SandboxError as error:
                    run = _infrastructure(error, limits)
                case_results.append(_case_result(case, run))
    except SandboxError as error:
        failure = _infrastructure(error, limits)
        # Record a setup/teardown failure for every remaining case.
        visible = failure
        for case in checks.cases[len(case_results) :]:
            case_results.append(_case_result(case, failure))
    result["visible"] = asdict(visible)
    result["cases"] = case_results
    runs = [result["visible"], *case_results]
    result["image_id"] = next((run["image_id"] for run in runs if run["image_id"]), None)
    statuses = {run["status"] for run in runs}
    status = next(
        (
            name
            for name in ("infrastructure_error", "output_limit", "timeout", "failed")
            if name in statuses
        ),
        "passed",
    )
    result.update(
        status=status,
        success=status == "passed",
        reward=None if status == "infrastructure_error" else int(status == "passed"),
    )
    return result


def write_evaluation(output: Path, result: dict, task: BugTask, patch: str) -> None:
    """Create a new artifact directory without ever overwriting existing evidence."""
    task_data = _manifest(task)
    BugTask.from_dict(task_data)
    # Serialize all data before creating the destination directory.
    result_text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    task_text = json.dumps(task_data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    _text(patch, "patch")
    result_text.encode("utf-8")
    task_text.encode("utf-8")
    output.mkdir(parents=True, exist_ok=False)
    for name, content in (
        ("result.json", result_text),
        ("submitted.patch", patch),
        ("task.json", task_text),
    ):
        with (output / name).open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
