import difflib
import hashlib
import json
import os
from pathlib import Path

import pytest

from toolsmith.evaluator import (
    CheckSpec,
    PatchError,
    PrivateChecks,
    apply_patch,
    evaluate_patch,
    load_checks,
    write_evaluation,
)
from toolsmith.sandbox import DEFAULT_IMAGE, SandboxLimits
from toolsmith.tasks import BugTask, load_task

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/repair"


def task_with_source(source="value = 1\n", path="src/value.py"):
    return BugTask.from_dict(
        {
            "schema_version": 1,
            "task_id": "value-task",
            "issue": "Return the correct value.",
            "source_files": {path: source},
            "visible_tests": {"tests/test_value.py": "def test_visible():\n    assert True\n"},
        }
    )


def replacement_patch(before="value = 1\n", after="value = 2\n", path="src/value.py"):
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(
        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in lines
    )


def check_manifest(**case_fields):
    case = {
        "id": "value",
        "module": "value",
        "function": "get_value",
        "args": [],
        "kwargs": {},
        "expected": 2,
    }
    case.update(case_fields)
    return {"schema_version": 1, "task_id": "value-task", "cases": [case]}


def checks_from_manifest(tmp_path, manifest):
    path = tmp_path / "private.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return load_checks(path)


def test_patch_is_exact_immutable_and_preserves_public_task():
    task = task_with_source()
    patched = apply_patch(task, replacement_patch())
    assert patched.source_files[0].content == "value = 2\n"
    assert task.source_files[0].content == "value = 1\n"
    assert patched.visible_tests == task.visible_tests
    assert patched.issue == task.issue
    with pytest.raises(AttributeError):
        patched.source_files[0].content = "bad"


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("value = 1", "value = 2"),
        ("value = 1", "value = 2\n"),
        ("value = 1\n", "value = 2"),
        ("", "value = 2\n"),
        ("value = 1\n", ""),
    ],
)
def test_patch_preserves_final_newline_and_empty_existing_files(before, after):
    patched = apply_patch(task_with_source(before), replacement_patch(before, after))
    assert patched.source_files[0].content == after


def test_patch_handles_multiple_hunks_and_files_in_sorted_order():
    before = "".join(f"line{i}\n" for i in range(20))
    after = before.replace("line1\n", "first\n").replace("line18\n", "last\n")
    data = {
        "schema_version": 1,
        "task_id": "multiple",
        "issue": "Fix both files.",
        "source_files": {"src/z.py": before, "src/a.py": "a = 1\n"},
        "visible_tests": {"tests/test_a.py": "assert True\n"},
    }
    patch = replacement_patch(before, after, "src/z.py") + replacement_patch(
        "a = 1\n", "a = 2\n", "src/a.py"
    )
    patched = apply_patch(BugTask.from_dict(data), patch)
    assert [(file.path, file.content) for file in patched.source_files] == [
        ("src/a.py", "a = 2\n"),
        ("src/z.py", after),
    ]


@pytest.mark.parametrize(
    "patch",
    [
        "",
        " \n",
        "--- a/src/value.py\n+++ b/src/value.py\n",
        replacement_patch().replace("-value = 1", "-value = 99"),
        replacement_patch().replace("@@ -1 +1 @@", "@@ -2 +1 @@"),
        replacement_patch().replace("@@ -1 +1 @@", "@@ -1 +2 @@"),
        replacement_patch().replace("@@ -1 +1 @@", "@@ -1,2 +1 @@"),
        replacement_patch().replace("@@ -1 +1 @@", "@@ -0 +1 @@"),
        replacement_patch() + "trailing junk\n",
        replacement_patch() + replacement_patch(),
        replacement_patch() + "@@ -1 +1 @@\n-value = 1\n+value = 3\n",
        replacement_patch().replace("+value = 2", "+value = 1"),
        "--- a/src/value.py\n+++ b/src/value.py\n@@ -1 +1 @@\n value = 1\n",
        replacement_patch().replace("-value = 1\n", "-value = 1\n\\ No newline at end of file\n"),
        replacement_patch() + "\\ No newline at end of file\n\\ No newline at end of file\n",
        replacement_patch()[:-1],
    ],
)
def test_patch_rejects_context_counts_overlap_noops_and_junk(patch):
    with pytest.raises(PatchError):
        apply_patch(task_with_source(), patch)


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_value.py",
        "/src/value.py",
        "src/../src/value.py",
        "src//value.py",
        "src/./value.py",
        "src\\value.py",
        "src/new.py",
        "src/config.toml",
        "/dev/null",
    ],
)
def test_patch_rejects_non_source_paths_and_aliases(path):
    with pytest.raises(PatchError):
        apply_patch(task_with_source(), replacement_patch(path=path))


def test_patch_rejects_rename_and_creation_headers():
    for patch in (
        replacement_patch().replace("+++ b/src/value.py", "+++ b/src/other.py"),
        replacement_patch().replace("--- a/src/value.py", "--- /dev/null"),
        "GIT binary patch\nliteral 1\nx\n",
    ):
        with pytest.raises(PatchError):
            apply_patch(task_with_source(), patch)


def test_patch_accepts_consistent_git_metadata_and_rejects_inconsistent_metadata():
    metadata = "diff --git a/src/value.py b/src/value.py\nindex 1234567..abcdef0 100644\n"
    assert apply_patch(task_with_source(), metadata + replacement_patch()).source_files
    with pytest.raises(PatchError):
        apply_patch(
            task_with_source(), metadata.replace("b/src/value", "b/src/other") + replacement_patch()
        )


def test_applying_source_never_executes_it(tmp_path):
    marker = tmp_path / "host-executed"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    apply_patch(task_with_source(), replacement_patch(after=source))
    assert not marker.exists()


def test_loads_returns_null_and_exception_expectations(tmp_path):
    data = check_manifest(expected=None)
    data["cases"].append(
        {
            "id": "error",
            "module": "value",
            "function": "get_value",
            "args": [0],
            "kwargs": {},
            "raises": "ValueError",
        }
    )
    checks = checks_from_manifest(tmp_path, data)
    assert checks.task_id == "value-task"
    assert checks.cases[0].expected is None
    assert checks.cases[1].raises == "ValueError"
    assert isinstance(checks.cases, tuple)


@pytest.mark.parametrize(
    "fields",
    [
        {"id": ""},
        {"id": "  "},
        {"id": 1},
        {"module": "../value"},
        {"module": "value..other"},
        {"module": "value;exit()"},
        {"module": "class"},
        {"function": "x.y"},
        {"function": "return"},
        {"args": {}},
        {"kwargs": []},
        {"expected": float("nan")},
        {"expected": {"nested": float("inf")}},
        {"expected": "\ud800"},
        {"raises": "ValueError"},
        {"extra": 1},
    ],
)
def test_checks_reject_invalid_fields(tmp_path, fields):
    with pytest.raises(ValueError):
        checks_from_manifest(tmp_path, check_manifest(**fields))


def test_checks_reject_empty_duplicates_missing_expectations_and_schema(tmp_path):
    original = check_manifest()
    variants = [
        {**original, "schema_version": True},
        {**original, "schema_version": 2},
        {**original, "cases": []},
        {**original, "cases": original["cases"] * 2},
        {**original, "task_id": "../value"},
        {**original, "extra": 1},
    ]
    no_expected = check_manifest()
    del no_expected["cases"][0]["expected"]
    variants.append(no_expected)
    for data in variants:
        with pytest.raises(ValueError):
            checks_from_manifest(tmp_path, data)
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_checks(path)


def test_patch_rejects_oversized_hunk_counts():
    patch = replacement_patch().replace("@@ -1 +1 @@", f"@@ -1,{'9' * 5000} +1 @@")
    with pytest.raises(PatchError):
        apply_patch(task_with_source(), patch)


def test_checks_reject_cyclic_non_json_constructor_values():
    args = []
    args.append(args)
    with pytest.raises(ValueError):
        CheckSpec("value", "value", "get_value", args, {}, expected=2)


def test_checks_validate_direct_constructor_json():
    with pytest.raises(ValueError):
        CheckSpec("value", "value", "get_value", [], {1: "bad"}, expected=2)
    with pytest.raises(ValueError):
        CheckSpec("value", "value", "get_value", [], {}, raises="builtins.ValueError")
    with pytest.raises(ValueError):
        PrivateChecks("value-task", ())


def test_evaluation_validates_task_ids_and_modules_before_execution(tmp_path):
    checks = checks_from_manifest(tmp_path, check_manifest(module="missing"))
    with pytest.raises(ValueError, match="module"):
        evaluate_patch(task_with_source(), replacement_patch(), checks)
    checks = checks_from_manifest(tmp_path, {**check_manifest(), "task_id": "different"})
    with pytest.raises(ValueError, match="task"):
        evaluate_patch(task_with_source(), replacement_patch(), checks)


def test_evaluation_rejects_empty_patch_without_docker(tmp_path):
    checks = checks_from_manifest(tmp_path, check_manifest())
    result = evaluate_patch(task_with_source(), "", checks)
    assert result["status"] == "invalid_patch"
    assert result["success"] is False
    assert result["reward"] == -1
    assert result["reason"]


def test_artifacts_refuse_an_existing_file_or_symlink(tmp_path):
    task = task_with_source()
    existing = tmp_path / "existing"
    existing.write_text("original", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(existing)
    for output in (existing, link):
        with pytest.raises(FileExistsError):
            write_evaluation(output, {}, task, replacement_patch())
    assert existing.read_text() == "original"


def test_artifacts_are_new_roundtrippable_and_do_not_expose_private_checks(tmp_path):
    task = task_with_source()
    patch = replacement_patch()
    checks = checks_from_manifest(tmp_path, check_manifest())
    result = evaluate_patch(task, "", checks)
    output = tmp_path / "evaluation"
    write_evaluation(output, result, task, patch)
    assert json.loads((output / "result.json").read_text()) == result
    assert (output / "submitted.patch").read_text() == patch
    assert load_task(output / "task.json") == task
    assert set(json.loads((output / "task.json").read_text())) == {
        "schema_version",
        "task_id",
        "issue",
        "source_files",
        "visible_tests",
    }
    with pytest.raises(FileExistsError):
        write_evaluation(output, {"success": True}, task, patch)
    assert json.loads((output / "result.json").read_text()) == result


sandbox_only = pytest.mark.skipif(
    os.environ.get("TOOLSMITH_RUN_SANDBOX_TESTS") != "1",
    reason="Set TOOLSMITH_RUN_SANDBOX_TESTS=1 to run real Docker evaluations",
)


@pytest.mark.sandbox
@sandbox_only
@pytest.mark.parametrize(
    ("name", "status", "reward"),
    [
        ("correct", "passed", 1),
        ("broken", "failed", 0),
        ("hardcoded", "failed", 0),
        ("forbidden", "invalid_patch", -1),
    ],
)
def test_real_pagination_repair_outcomes(name, status, reward):
    task = load_task(ROOT / "examples/pagination/task.json")
    checks = load_checks(FIXTURES / "private.json")
    patch = (FIXTURES / f"{name}.patch").read_text(encoding="utf-8")
    result = evaluate_patch(
        task, patch, checks, image=os.environ.get("TOOLSMITH_SANDBOX_IMAGE", DEFAULT_IMAGE)
    )
    assert result["status"] == status, result
    assert result["success"] is (status == "passed")
    assert result["reward"] == reward
    assert result["patch_sha256"] == hashlib.sha256(patch.encode()).hexdigest()
    if name != "forbidden":
        assert len(result["cases"]) == len(checks.cases)
        assert result["image_id"].startswith("sha256:")
    if name == "hardcoded":
        assert result["visible"]["status"] == "passed"
        assert any(not case["success"] for case in result["cases"])


@pytest.mark.sandbox
@sandbox_only
@pytest.mark.parametrize(
    ("returned", "expected", "success"),
    [("False", 0, False), ("{'x': [True]}", {"x": [1]}, False), ("1.0", 1, True)],
)
def test_private_expected_comparison_uses_json_types(tmp_path, returned, expected, success):
    before = "def get_value():\n    return None\n"
    after = f"def get_value():\n    return {returned}\n"
    task = task_with_source(before)
    checks = checks_from_manifest(tmp_path, check_manifest(expected=expected))
    result = evaluate_patch(
        task,
        replacement_patch(before, after),
        checks,
        image=os.environ.get("TOOLSMITH_SANDBOX_IMAGE", DEFAULT_IMAGE),
    )
    assert result["success"] is success, result
    assert result["cases"][0]["success"] is success


@pytest.mark.sandbox
@sandbox_only
@pytest.mark.parametrize(
    ("body", "status", "limit"),
    [
        ("while True:\n        pass", "timeout", 65536),
        ("print('x' * 1000000)\n    return 2", "output_limit", 2048),
    ],
)
def test_evaluation_reports_resource_failures(tmp_path, body, status, limit):
    before = "def get_value():\n    return None\n"
    after = f"def get_value():\n    {body}\n"
    checks = checks_from_manifest(tmp_path, check_manifest())
    result = evaluate_patch(
        task_with_source(before),
        replacement_patch(before, after),
        checks,
        image=os.environ.get("TOOLSMITH_SANDBOX_IMAGE", DEFAULT_IMAGE),
        limits=SandboxLimits(timeout_seconds=3, output_bytes=limit),
    )
    assert result["status"] == status, result
    assert result["success"] is False
    assert result["reward"] == 0


@pytest.mark.sandbox
@sandbox_only
def test_expected_values_are_not_exposed_and_cases_get_fresh_containers(tmp_path):
    before = "def get_value():\n    return None\n"
    after = (
        "def get_value():\n"
        "    from pathlib import Path\n"
        "    import sys\n"
        "    marker = Path('/tmp/case-ran')\n"
        "    assert not marker.exists()\n"
        "    marker.touch()\n"
        "    assert not Path('/workspace/private.json').exists()\n"
        "    assert sys.stdin.read() == ''\n"
        "    return 2\n"
    )
    manifest = check_manifest()
    manifest["cases"].append({**manifest["cases"][0], "id": "second"})
    checks = checks_from_manifest(tmp_path, manifest)
    result = evaluate_patch(
        task_with_source(before),
        replacement_patch(before, after),
        checks,
        image=os.environ.get("TOOLSMITH_SANDBOX_IMAGE", DEFAULT_IMAGE),
    )
    assert result["success"] is True, result
    assert len(result["cases"]) == 2


@pytest.mark.sandbox
@sandbox_only
def test_missing_image_is_infrastructure_not_incorrect_patch(tmp_path):
    checks = checks_from_manifest(tmp_path, check_manifest())
    result = evaluate_patch(
        task_with_source(),
        replacement_patch(),
        checks,
        image="toolsmith-deliberately-missing-evaluator-image:never-pull",
    )
    assert result["status"] == "infrastructure_error", result
    assert result["success"] is False
    assert result["reward"] is None
    assert len(result["cases"]) == len(checks.cases)
