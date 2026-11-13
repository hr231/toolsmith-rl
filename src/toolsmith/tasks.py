"""Public bug-task data loaded as JSON without importing or executing project code."""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskFile:
    path: str
    content: str


@dataclass(frozen=True)
class BugTask:
    task_id: str
    issue: str
    source_files: tuple[TaskFile, ...]
    visible_tests: tuple[TaskFile, ...]

    @classmethod
    def from_dict(cls, data: object) -> "BugTask":
        """Validate the exact public schema; private evaluator fields are forbidden."""
        required = {"schema_version", "task_id", "issue", "source_files", "visible_tests"}
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError(
                f"Task must contain exactly these fields: {', '.join(sorted(required))}"
            )
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("schema_version must be the integer 1")
        task_id = _text(data["task_id"], "task_id")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", task_id):
            raise ValueError(
                "task_id must be 1-64 lowercase letters/digits/_/-, starting with a-z/0-9"
            )
        issue = _text(data["issue"], "issue")
        if not issue.strip():
            raise ValueError("issue must not be blank")
        return cls(
            task_id=task_id,
            issue=issue,
            source_files=_files(data["source_files"], "src"),
            visible_tests=_files(data["visible_tests"], "tests"),
        )

    @property
    def source_file_count(self) -> int:
        return len(self.source_files)

    @property
    def input_bytes(self) -> int:
        """Count public input bytes, not estimated tokens or private evaluation data."""
        size = len(self.issue.encode("utf-8"))
        for files in (self.source_files, self.visible_tests):
            for file in files:
                size += len(file.path.encode("utf-8")) + len(file.content.encode("utf-8"))
        return size


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field} must contain valid UTF-8 text") from error
    return value


def _files(value: object, root: str) -> tuple[TaskFile, ...]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{root} files must be a nonempty path-to-content object")
    files = []
    for path, content in value.items():
        path = _text(path, "file path")
        parts = path.split("/")
        if (
            len(parts) < 2
            or parts[0] != root
            or "\\" in path
            or "\x00" in path
            or any(not part or part.startswith(".") or part == "__pycache__" for part in parts)
            or not path.endswith(".py")
            or (root == "src" and parts[-1] == "conftest.py")
        ):
            raise ValueError(f"Invalid {root} file path: {path!r}")
        files.append(TaskFile(path, _text(content, f"content of {path!r}")))
    return tuple(sorted(files, key=lambda file: file.path))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_task(path: Path) -> BugTask:
    """Read a manifest as data, rejecting duplicate keys rather than losing fields."""
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    return BugTask.from_dict(data)
