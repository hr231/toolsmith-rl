import os
import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from toolsmith.sandbox import (
    DEFAULT_IMAGE,
    Sandbox,
    SandboxError,
    SandboxLimits,
    _capture,
    _protocol,
)
from toolsmith.tasks import BugTask, TaskFile


def make_task(source="def answer():\n    return 1\n", tests=None):
    return BugTask.from_dict(
        {
            "schema_version": 1,
            "task_id": "sandbox-test",
            "issue": "Return the correct answer.",
            "source_files": {"src/example.py": source},
            "visible_tests": {
                "tests/test_example.py": tests
                or "from example import answer\n\ndef test_answer():\n    assert answer() == 1\n"
            },
        }
    )


@pytest.mark.parametrize(
    "values",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": True},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": 10**1000},
        {"memory_mb": 0},
        {"memory_mb": 1.5},
        {"memory_mb": True},
        {"cpus": -1},
        {"cpus": 0.001},
        {"cpus": 10**1000},
        {"cpus": float("inf")},
        {"pids_limit": False},
        {"pids_limit": 0},
        {"output_bytes": 0},
        {"output_bytes": 0.5},
    ],
)
def test_limits_reject_invalid_values(values):
    with pytest.raises(ValueError):
        SandboxLimits(**values)


def test_limits_are_frozen():
    with pytest.raises(FrozenInstanceError):
        SandboxLimits().cpus = 2


def test_allowlisted_tools_need_no_docker_and_do_not_execute_source(tmp_path):
    marker = tmp_path / "executed"
    source = f"open({str(marker)!r}, 'w').write('bad')\n"
    with Sandbox(make_task(source)) as sandbox:
        assert sandbox.read_file("src/example.py") == source
        assert "test_answer" in sandbox.read_file("tests/test_example.py")
        sandbox.edit_file("src/example.py", source + "# edit\n")
        assert sandbox.submit_patch().endswith("+# edit\n")
    assert not marker.exists()


@pytest.mark.parametrize(
    "path",
    [
        "../src/example.py",
        "/src/example.py",
        "src/./example.py",
        "src//example.py",
        "src/../src/example.py",
        "src\\example.py",
        "src/new.py",
        "tests/new.py",
    ],
)
def test_tools_reject_nonallowlisted_paths(path):
    sandbox = Sandbox(make_task())
    with pytest.raises(ValueError):
        sandbox.read_file(path)
    with pytest.raises(ValueError):
        sandbox.edit_file(path, "pass\n")


def test_edits_cannot_change_tests_or_accept_non_utf8():
    sandbox = Sandbox(make_task())
    with pytest.raises(ValueError):
        sandbox.edit_file("tests/test_example.py", "pass\n")
    with pytest.raises(ValueError):
        sandbox.edit_file("src/example.py", "\ud800")
    with pytest.raises(ValueError):
        sandbox.edit_file("src/example.py", b"pass")


def test_instances_have_independent_memory_and_patch_uses_original():
    task = make_task()
    first, second = Sandbox(task), Sandbox(task)
    first.edit_file("src/example.py", "x = 2\n")
    first.edit_file("src/example.py", "x = 3\n")
    assert second.read_file("src/example.py") == task.source_files[0].content
    assert second.submit_patch() == ""
    assert first.submit_patch() == (
        "--- a/src/example.py\n+++ b/src/example.py\n@@ -1,2 +1 @@\n"
        "-def answer():\n-    return 1\n+x = 3\n"
    )
    first.edit_file("src/example.py", task.source_files[0].content)
    assert first.submit_patch() == ""


def test_patch_preserves_missing_final_newline():
    sandbox = Sandbox(make_task("x = 1"))
    sandbox.edit_file("src/example.py", "x = 2")
    assert sandbox.submit_patch() == (
        "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n"
        "-x = 1\n\\ No newline at end of file\n"
        "+x = 2\n\\ No newline at end of file\n"
    )


def test_revalidates_direct_dataclass_paths_and_parent_collisions():
    task = make_task()
    with pytest.raises(ValueError):
        Sandbox(
            BugTask(task.task_id, task.issue, (TaskFile("src/../bad.py", ""),), task.visible_tests)
        )
    with pytest.raises(ValueError):
        Sandbox(
            BugTask(
                task.task_id,
                task.issue,
                (TaskFile("src/a.py", ""), TaskFile("src/a.py/b.py", "")),
                task.visible_tests,
            )
        )
    with pytest.raises(ValueError):
        Sandbox(
            BugTask(
                task.task_id,
                task.issue,
                (TaskFile("src/a.py", ""), TaskFile("src/a.py", "")),
                task.visible_tests,
            )
        )


@pytest.mark.parametrize(
    "module,function,args,kwargs",
    [
        ("os", "system", [], {}),
        ("example;bad", "answer", [], {}),
        ("example", "answer()", [], {}),
        ("example", "answer", (), {}),
        ("example", "answer", [], {1: 2}),
        ("example", "answer", [float("nan")], {}),
        ("example", "answer", [{"nested": float("inf")}], {}),
        ("example", "answer", [object()], {}),
    ],
)
def test_check_rejects_invalid_requests_before_docker(module, function, args, kwargs):
    with pytest.raises(ValueError):
        Sandbox(make_task()).run_check(module, function, args, kwargs)


@pytest.fixture
def sandbox_image():
    if os.environ.get("TOOLSMITH_RUN_SANDBOX_TESTS") != "1":
        pytest.skip("Set TOOLSMITH_RUN_SANDBOX_TESTS=1 to run real Docker tests")
    return os.environ.get("TOOLSMITH_SANDBOX_IMAGE", DEFAULT_IMAGE)


def container_names():
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=toolsmith-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return set(result.stdout.splitlines())


@pytest.mark.sandbox
def test_real_visible_pass_wrong_edit_fail_and_isolation(sandbox_image):
    before = container_names()
    with Sandbox(make_task(), image=sandbox_image) as first:
        with Sandbox(make_task(), image=sandbox_image) as second:
            passed = first.run_visible_tests()
            assert passed.status == "passed"
            assert passed.payload["collected"] == 1
            assert passed.image_id.startswith("sha256:")
            first.edit_file("src/example.py", "def answer():\n    return 2\n")
            assert first.run_visible_tests().status == "failed"
            other = second.run_visible_tests()
            assert other.status == "passed"
            assert other.image_id == passed.image_id
    assert container_names() == before


@pytest.mark.sandbox
def test_real_restrictions_and_fresh_tmp(sandbox_image, monkeypatch):
    monkeypatch.setenv("TOOLSMITH_HOST_SECRET", "never-copy-this")
    source = """import os
import socket
from pathlib import Path

def probe():
    denied = []
    for name in ["/workspace/src/example.py", "/workspace/tests/test_example.py", "/root-file"]:
        try:
            Path(name).write_text("changed")
        except OSError:
            denied.append(name)
    sock = socket.socket()
    sock.settimeout(0.25)
    try:
        sock.connect(("1.1.1.1", 443))
        network = True
    except OSError:
        network = False
    finally:
        sock.close()
    marker = Path("/tmp/previous-run")
    fresh = not marker.exists()
    marker.write_text("state")
    readonly = all(os.statvfs(path).f_flag & os.ST_RDONLY
                   for path in ["/", "/workspace/src", "/workspace/tests"])
    return {"uid": os.getuid(), "denied": len(denied), "network": network, "fresh": fresh,
            "readonly": readonly, "host_secret": os.environ.get("TOOLSMITH_HOST_SECRET")}
"""
    with Sandbox(make_task(source), image=sandbox_image) as sandbox:
        for _ in range(2):
            result = sandbox.run_check("example", "probe", [], {})
            assert result.status == "passed"
            assert result.payload["value"] == {
                "uid": 65532,
                "denied": 3,
                "network": False,
                "fresh": True,
                "readonly": True,
                "host_secret": None,
            }


@pytest.mark.sandbox
@pytest.mark.parametrize(
    "source,status,limits",
    [
        (
            "def answer():\n    while True:\n        pass\n",
            "timeout",
            SandboxLimits(timeout_seconds=2),
        ),
        (
            "def answer():\n    while True:\n        print('x' * 8192, flush=True)\n",
            "output_limit",
            SandboxLimits(timeout_seconds=10, output_bytes=1024),
        ),
    ],
)
def test_real_limits_bound_output_and_remove_container(sandbox_image, source, status, limits):
    before = container_names()
    with Sandbox(make_task(source), image=sandbox_image, limits=limits) as sandbox:
        result = sandbox.run_check("example", "answer", [], {})
        assert result.status == status
        assert len(result.stdout.encode()) + len(result.stderr.encode()) <= limits.output_bytes
    assert container_names() == before


@pytest.mark.sandbox
def test_real_check_transports_arguments_and_exception(sandbox_image):
    source = "def answer(value, *, offset=0):\n    return value + offset\n"
    with Sandbox(make_task(source), image=sandbox_image) as sandbox:
        result = sandbox.run_check("example", "answer", [3], {"offset": 4})
        assert result.status == "passed"
        assert result.payload == {"kind": "return", "value": 7}
        raised = sandbox.run_check("example", "answer", [], {})
        assert raised.status == "passed"
        assert raised.payload == {"kind": "exception", "type": "TypeError"}


@pytest.mark.sandbox
@pytest.mark.parametrize(
    "wire",
    [
        "noise\\n",
        '{"kind":"return","value":1}{"kind":"return","value":2}',
        '{"kind":"return","kind":"return","value":1}',
        '{"kind":"return","value":NaN}',
        '{"kind":"return","value":1e999}',
    ],
)
def test_real_rejects_spoofed_or_malformed_protocol(sandbox_image, wire):
    source = f"import os\ndef answer():\n    os.write(1, {wire.encode()!r})\n    os._exit(0)\n"
    with Sandbox(make_task(source), image=sandbox_image) as sandbox:
        assert sandbox.run_check("example", "answer", [], {}).status == "failed"


@pytest.mark.sandbox
def test_real_missing_image_never_pulls_or_falls_back(sandbox_image):
    with Sandbox(make_task(), image="toolsmith-missing-image:no-such-local-tag") as sandbox:
        with pytest.raises(SandboxError):
            sandbox.run_visible_tests()


@pytest.mark.parametrize(
    "wire,exit_code,mode,status",
    [
        (b'{"kind":"return","value":1}', 0, "check", "passed"),
        (b'{"kind":"exception","type":"ValueError"}', 0, "check", "passed"),
        (b'{"kind":"return","value":1}', 1, "check", "failed"),
        (b'{"kind":"return","value":1,"extra":true}', 0, "check", "failed"),
        (b'{"kind":"return","value":1,"value":2}', 0, "check", "failed"),
        (b'{"kind":"return","value":NaN}', 0, "check", "failed"),
        (b'{"kind":"return","value":1e999}', 0, "check", "failed"),
        (b'{"kind":"return","value":"\\ud800"}', 0, "check", "failed"),
        (b'{"kind":"return","value":1}{}', 0, "check", "failed"),
        (b'{"kind":"exception","type":"bad.type"}', 0, "check", "failed"),
        (b"\xff", 0, "check", "failed"),
        (b"[]", 0, "check", "failed"),
        (b"", 0, "check", "failed"),
        (
            b'{"kind":"visible","collected":0,"passed":0,"failed":0,"errors":0,"skipped":0}',
            0,
            "visible",
            "failed",
        ),
        (
            b'{"kind":"visible","collected":1,"passed":1,"failed":0,"errors":0,"skipped":0}',
            0,
            "visible",
            "passed",
        ),
        (
            b'{"kind":"visible","collected":1,"passed":1,"failed":0,"errors":0,"skipped":0}',
            1,
            "visible",
            "failed",
        ),
        (
            b'{"kind":"visible","collected":true,"passed":1,"failed":0,"errors":0,"skipped":0}',
            0,
            "visible",
            "failed",
        ),
        (
            b'{"kind":"visible","collected":1,"passed":0,"failed":0,"errors":0,"skipped":1}',
            0,
            "visible",
            "failed",
        ),
    ],
)
def test_strict_protocol(wire, exit_code, mode, status):
    assert _protocol(wire, exit_code, mode)[0] == status


def test_streaming_capture_feeds_stdin_and_drains_both_output_pipes():
    data = b"x" * 200_000
    result = _capture(
        [
            sys.executable,
            "-c",
            "import os,sys; os.write(2,b'e'*100000); "
            "data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data)",
        ],
        stdin=data,
        timeout=5,
        output_limit=300_000,
    )
    assert result.exit_code == 0
    assert result.stopped is None
    assert result.stdout == data
    assert result.stderr == b"e" * 100_000


@pytest.mark.parametrize(
    "code,timeout,output_limit,stopped",
    [
        ("import time; time.sleep(60)", 0.1, 1024, "timeout"),
        ("import os\nwhile True: os.write(2, b'x'*8192)", 5, 1024, "output_limit"),
    ],
)
def test_streaming_capture_kills_real_process_at_limits(code, timeout, output_limit, stopped):
    result = _capture([sys.executable, "-c", code], timeout=timeout, output_limit=output_limit)
    assert result.stopped == stopped
    assert result.exit_code is not None
    assert len(result.stdout) + len(result.stderr) <= output_limit


def test_materialization_rejects_symlink_root_without_touching_target(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "link"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(SandboxError):
        Sandbox(make_task())._materialize(linked)
    assert tuple(target.iterdir()) == ()


def test_materialization_uses_current_source_and_original_tests(tmp_path):
    task = make_task()
    sandbox = Sandbox(task)
    sandbox.edit_file("src/example.py", "value = 99\n")
    sandbox._materialize(tmp_path.resolve())
    assert (tmp_path / "src/example.py").read_text() == "value = 99\n"
    assert (tmp_path / "tests/test_example.py").read_text() == task.visible_tests[0].content


def test_closed_sandbox_rejects_reuse():
    with Sandbox(make_task()) as sandbox:
        pass
    with pytest.raises(SandboxError):
        sandbox.run_visible_tests()
