"""Container-only runner for visible tests and single-function checks.

- `visible`: runs pytest on the mounted tests and reports counts as JSON.
- `check`: calls one function from stdin JSON and returns its value as JSON.
- Receives no expected values; the host compares results after this process ends.
"""

import contextlib
import importlib
import json
import os
import sys
from pathlib import Path

import pytest


class VisibleSummary:
    def __init__(self):
        self.collected = 0
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.skipped = 0

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_collectreport(self, report):
        if report.failed:
            self.errors += 1
        elif report.skipped:
            self.skipped += 1

    def pytest_runtest_logreport(self, report):
        if report.skipped or hasattr(report, "wasxfail"):
            self.skipped += 1
        elif report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
        elif report.failed:
            self.errors += 1

    def payload(self):
        return {
            "kind": "visible",
            "collected": self.collected,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
        }


def visible():
    summary = VisibleSummary()
    code = pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            "--disable-warnings",
            "--confcutdir=/workspace/tests",
            "--import-mode=importlib",
            "/workspace/tests",
        ],
        plugins=[summary],
    )
    return summary.payload(), 0 if code == pytest.ExitCode.OK else 1


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def check():
    request_text = sys.stdin.read(1_048_577)
    if len(request_text.encode("utf-8")) > 1_048_576:
        raise ValueError("Check request exceeds 1 MiB")
    request = json.loads(request_text, parse_constant=reject_constant)
    if not isinstance(request, dict) or set(request) != {"module", "function", "args", "kwargs"}:
        raise ValueError("Invalid check request fields")
    if not isinstance(request["args"], list) or not isinstance(request["kwargs"], dict):
        raise ValueError("Invalid check arguments")
    module = importlib.import_module(request["module"])
    source = Path("/workspace/src").joinpath(*request["module"].split("."))
    origin = getattr(module, "__file__", None)
    if origin is None or Path(origin).resolve(strict=True) not in {
        source.with_suffix(".py"),
        source / "__init__.py",
    }:
        raise ValueError("Check module did not load from its declared source")
    function = getattr(module, request["function"])
    if not callable(function):
        raise ValueError("Check target must be callable")
    # Errors raised by the called function are reported as an "exception" result.
    try:
        result = function(*request["args"], **request["kwargs"])
    except BaseException as error:
        return {"kind": "exception", "type": type(error).__name__}, 0
    return {"kind": "return", "value": result}, 0


def main():
    # Capture output streams before loading untrusted source.
    output = sys.stdout
    diagnostics = sys.stderr
    encode = json.dumps
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.path.insert(0, "/workspace/src")
    try:
        with contextlib.redirect_stdout(diagnostics), contextlib.redirect_stderr(diagnostics):
            if sys.argv[1:] == ["visible"]:
                payload, code = visible()
            elif sys.argv[1:] == ["check"]:
                payload, code = check()
            else:
                raise ValueError("Runner mode must be visible or check")
        serialized = encode(payload, allow_nan=False, separators=(",", ":"))
    except BaseException as error:
        serialized = encode({"kind": "runner_error", "type": type(error).__name__})
        code = 1
    output.write(serialized + "\n")
    output.flush()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
