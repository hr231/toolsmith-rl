import os

import pytest

from toolsmith.sandbox import Sandbox
from toolsmith.tasks import BugTask

pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(
        os.environ.get("TOOLSMITH_RUN_SANDBOX_TESTS") != "1",
        reason="Set TOOLSMITH_RUN_SANDBOX_TESTS=1 after building the sandbox image",
    ),
]


def task_for(module, content):
    return BugTask.from_dict(
        {
            "schema_version": 1,
            "task_id": "runner-boundary",
            "issue": "Invoke only the declared source function, not a cached system module.",
            "source_files": {f"src/{module}.py": content},
            "visible_tests": {"tests/test_value.py": "def test_value():\n    assert 1 + 1 == 2\n"},
        }
    )


def test_cached_standard_module_cannot_substitute_for_task_source():
    task = task_for("os", "def getpid():\n    return 42\n")
    with Sandbox(task) as sandbox:
        result = sandbox.run_check("os", "getpid", [], {})
    assert result.status == "failed"
    assert result.payload is None


def test_import_failure_cannot_satisfy_an_expected_function_exception():
    task = task_for("broken_import", "raise ValueError('module is broken')\n")
    with Sandbox(task) as sandbox:
        result = sandbox.run_check("broken_import", "operation", [], {})
    assert result.status == "failed"
    assert result.payload is None
