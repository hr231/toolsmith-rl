"""Executable isolation checks of the local Docker sandbox."""

from dataclasses import asdict

from toolsmith.sandbox import DEFAULT_IMAGE, Sandbox, SandboxLimits
from toolsmith.tasks import BugTask

_PROBE_SOURCE = """def inspect_limits():
    import errno
    import os
    import socket
    from pathlib import Path

    result = {"nonroot": os.getuid() == 65532}
    for key, target in (
        ("root_readonly", "/etc/toolsmith-probe"),
        ("source_readonly", "/workspace/src/toolsmith-probe"),
        ("tests_readonly", "/workspace/tests/toolsmith-probe"),
    ):
        try:
            Path(target).write_text("probe")
        except OSError as error:
            result[key] = error.errno in (errno.EROFS, errno.EACCES, errno.EPERM)
        else:
            result[key] = False
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=0.5):
            result["external_network_blocked"] = False
    except OSError:
        result["external_network_blocked"] = True
    status = dict(
        line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    result["no_effective_capabilities"] = int(status["CapEff"].strip(), 16) == 0
    result["no_new_privileges"] = status["NoNewPrivs"].strip() == "1"
    temporary = Path("/tmp/toolsmith-probe")
    temporary.write_text("temporary")
    result["temporary_storage_available"] = temporary.read_text() == "temporary"
    return result
"""


def check_sandbox(*, image: str = DEFAULT_IMAGE, limits: SandboxLimits | None = None) -> dict:
    """Run a real container and verify observed identity and isolation controls."""
    resolved_limits = limits or SandboxLimits()
    task = BugTask.from_dict(
        {
            "schema_version": 1,
            "task_id": "sandbox-probe",
            "issue": "Inspect container isolation without accessing host resources.",
            "source_files": {"src/probe.py": _PROBE_SOURCE},
            "visible_tests": {
                "tests/test_probe.py": (
                    "from probe import inspect_limits\n\n"
                    "def test_nonroot():\n    assert inspect_limits()['nonroot'] is True\n"
                )
            },
        }
    )
    with Sandbox(task, image=image, limits=resolved_limits) as sandbox:
        result = sandbox.run_check("probe", "inspect_limits", [], {})
    expected = {
        "nonroot",
        "root_readonly",
        "source_readonly",
        "tests_readonly",
        "external_network_blocked",
        "no_effective_capabilities",
        "no_new_privileges",
        "temporary_storage_available",
    }
    payload = result.payload or {}
    observations = payload.get("value") if payload.get("kind") == "return" else None
    success = (
        result.status == "passed"
        and isinstance(observations, dict)
        and set(observations) == expected
        and all(value is True for value in observations.values())
    )
    return {
        "success": success,
        "status": "passed" if success else result.status if result.status != "passed" else "failed",
        "image_id": result.image_id,
        "limits": asdict(resolved_limits),
        "observations": observations,
        "execution": asdict(result),
        "scope": "Observed local controls; not proof of isolation against arbitrary hostile code.",
    }
