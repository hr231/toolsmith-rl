import json

import pytest

from toolsmith.tasks import BugTask, load_task


def public_task():
    return {
        "schema_version": 1,
        "task_id": "pagination-01",
        "issue": "Preserve the final page.",
        "source_files": {"src/pages.py": "def pages():\n    return []\n"},
        "visible_tests": {"tests/test_pages.py": "def test_pages():\n    assert False\n"},
    }


def test_features_count_utf8_and_include_public_tests():
    payload = public_task()
    payload["issue"] = "Keep café entries."
    task = BugTask.from_dict(payload)
    expected_bytes = len(payload["issue"].encode("utf-8"))
    for group in ("source_files", "visible_tests"):
        for path, content in payload[group].items():
            expected_bytes += len(path.encode("utf-8")) + len(content.encode("utf-8"))
    assert task.source_file_count == 1
    assert task.input_bytes == expected_bytes
    payload["visible_tests"]["tests/test_pages.py"] += "\n# another public case\n"
    assert BugTask.from_dict(payload).input_bytes > task.input_bytes


def test_loaded_files_are_sorted_and_detached_from_input():
    payload = public_task()
    payload["source_files"] = {"src/z.py": "z = 1", "src/a.py": "a = 1"}
    task = BugTask.from_dict(payload)
    payload["source_files"]["src/a.py"] = "a = 999"
    assert tuple(file.path for file in task.source_files) == ("src/a.py", "src/z.py")
    assert task.source_files[0].content == "a = 1"
    with pytest.raises(AttributeError):
        task.source_files[0].content = "a = 2"


@pytest.mark.parametrize(
    "path",
    [
        "/src/a.py",
        "src/../outside.py",
        "src/./a.py",
        "src//a.py",
        "src\\a.py",
        "src/a\x00.py",
        "tests/test_a.py",
        "src/.hidden/a.py",
        "src/__pycache__/a.py",
        "src/conftest.py",
        "src/config.toml",
    ],
)
def test_rejects_noncanonical_or_non_source_paths(path):
    payload = public_task()
    payload["source_files"] = {path: "x = 1"}
    with pytest.raises(ValueError):
        BugTask.from_dict(payload)


def test_rejects_visible_test_outside_test_directory():
    payload = public_task()
    payload["visible_tests"] = {"src/test_pages.py": "assert False"}
    with pytest.raises(ValueError):
        BugTask.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("task_id", "../escape"),
        ("task_id", ""),
        ("issue", " \n"),
        ("issue", 123),
        ("issue", "invalid\ud800"),
        ("source_files", {}),
        ("visible_tests", {}),
        ("source_files", {"src/a.py": 42}),
        ("source_files", {"src/a.py": "\ud800"}),
    ],
)
def test_rejects_invalid_task_fields(field, value):
    payload = public_task()
    payload[field] = value
    with pytest.raises(ValueError):
        BugTask.from_dict(payload)


def test_private_expectations_cannot_enter_public_task():
    payload = public_task()
    payload["hidden_tests"] = {"private.py": "assert secret_answer == 3"}
    with pytest.raises(ValueError):
        BugTask.from_dict(payload)


def test_rejects_missing_required_fields():
    payload = public_task()
    del payload["visible_tests"]
    with pytest.raises(ValueError):
        BugTask.from_dict(payload)


def test_rejects_duplicate_file_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"source_files": {"src/a.py": "first", "src/a.py": "second"}}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_task(path)


def test_loading_never_executes_embedded_python(tmp_path):
    marker = tmp_path / "executed"
    payload = public_task()
    payload["source_files"]["src/pages.py"] = (
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    )
    manifest = tmp_path / "task.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    task = load_task(manifest)
    assert task.source_files[0].content == payload["source_files"]["src/pages.py"]
    assert not marker.exists()
