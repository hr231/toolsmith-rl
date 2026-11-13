"""Allowlisted in-memory editing and restricted, disposable Docker execution.

- Never imports project code on the host.
- Mounts fresh source and visible-test directories read-only for each execution.
- Leaves only the container's temporary storage writable.
- Treats Docker as a containment boundary, not protection against runtime exploits.
"""

import difflib
import json
import math
import os
import re
import selectors
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from toolsmith.tasks import BugTask

DEFAULT_IMAGE = "toolsmith-sandbox:py312-v1"
_CONTROL_TIMEOUT = 10.0
_CONTROL_OUTPUT = 4096


class SandboxError(RuntimeError):
    """Docker, image, daemon, or workspace setup is unavailable; no host fallback."""


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 10.0
    memory_mb: int = 256
    cpus: float = 1.0
    pids_limit: int = 64
    output_bytes: int = 65536

    def __post_init__(self) -> None:
        for name in ("timeout_seconds", "cpus"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not 0 < value <= sys.float_info.max:
                raise ValueError(f"{name} must be a positive finite number")
        for name in ("memory_mb", "pids_limit", "output_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.memory_mb < 6:
            raise ValueError("memory_mb must be at least Docker's 6 MiB minimum")
        if self.cpus < 0.01:
            raise ValueError("cpus must be at least 0.01")


@dataclass(frozen=True)
class RunResult:
    status: str
    stdout: str
    stderr: str
    exit_code: int | None
    elapsed_seconds: float
    image_id: str
    payload: dict | None = None


@dataclass(frozen=True)
class _ProcessResult:
    stdout: bytes
    stderr: bytes
    exit_code: int | None
    stopped: str | None


def _capture(
    command: list[str], *, stdin: bytes = b"", timeout: float, output_limit: int
) -> _ProcessResult:
    """Drain both pipes and feed stdin without unbounded capture or pipe deadlocks."""
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    stdout, stderr = bytearray(), bytearray()
    stopped = None
    position = 0
    try:
        with selectors.DefaultSelector() as selector:
            for stream, target in ((process.stdout, stdout), (process.stderr, stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, target)
            if stdin:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, None)
            else:
                process.stdin.close()
            while selector.get_map() or process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stopped = "timeout"
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    stream = key.fileobj
                    if key.data is None:
                        try:
                            position += os.write(
                                stream.fileno(), stdin[position : position + 65536]
                            )
                        except BrokenPipeError:
                            position = len(stdin)
                        except BlockingIOError:
                            continue
                        if position == len(stdin):
                            selector.unregister(stream)
                            stream.close()
                        continue
                    available = output_limit - len(stdout) - len(stderr)
                    try:
                        chunk = os.read(stream.fileno(), min(65536, available + 1))
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    key.data.extend(chunk[:available])
                    if len(chunk) > available:
                        stopped = "output_limit"
                        break
                if stopped:
                    break
            if stopped:
                process.kill()
            process.wait(timeout=max(0.1, deadline - time.monotonic()) if not stopped else 5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
    return _ProcessResult(bytes(stdout), bytes(stderr), process.returncode, stopped)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")


def _json_value(value: object) -> None:
    """Require real JSON types, including finite floats and string object keys."""
    if value is None or type(value) in (str, bool, int):
        if isinstance(value, str):
            value.encode("utf-8")
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for key, item in value.items():
            key.encode("utf-8")
            _json_value(item)
        return
    raise ValueError("Arguments and returned values must contain only finite JSON data")


def _protocol(stdout: bytes, exit_code: int | None, mode: str) -> tuple[str, dict | None]:
    try:
        payload = json.loads(
            stdout.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        _json_value(payload)
    except (ValueError, UnicodeError, RecursionError):
        return "failed", None
    if type(payload) is not dict:
        return "failed", None
    if mode == "visible":
        fields = {"collected", "passed", "failed", "errors", "skipped"}
        if (
            set(payload) != fields | {"kind"}
            or payload["kind"] != "visible"
            or any(type(payload[name]) is not int or payload[name] < 0 for name in fields)
        ):
            return "failed", None
        passed = (
            exit_code == 0
            and payload["collected"] > 0
            and payload["passed"] == payload["collected"]
            and all(payload[name] == 0 for name in ("failed", "errors", "skipped"))
        )
        return ("passed" if passed else "failed"), payload
    valid_return = set(payload) == {"kind", "value"} and payload["kind"] == "return"
    valid_exception = (
        set(payload) == {"kind", "type"}
        and payload["kind"] == "exception"
        and isinstance(payload["type"], str)
        and payload["type"].isidentifier()
    )
    if not (valid_return or valid_exception):
        return "failed", None
    return ("passed" if exit_code == 0 else "failed"), payload


def _bounded_text(raw: bytes, limit: int) -> str:
    # Invalid UTF-8 replacement can expand a byte; cap the encoded result, too.
    return (
        raw.decode("utf-8", errors="replace")
        .encode("utf-8")[:limit]
        .decode("utf-8", errors="ignore")
    )


def _lines(content: str) -> list[str]:
    # Unlike splitlines(), only LF is a patch line boundary (not Unicode separators).
    parts = content.split("\n")
    return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


class Sandbox:
    """A public task snapshot; Docker is resolved lazily before the first execution."""

    def __init__(
        self, task: BugTask, *, image: str = DEFAULT_IMAGE, limits: SandboxLimits | None = None
    ) -> None:
        if not isinstance(task, BugTask):
            raise ValueError("task must be a BugTask")
        all_files = task.source_files + task.visible_tests
        paths = {file.path for file in all_files}
        if len(all_files) != len(paths):
            raise ValueError("Task paths must be unique")
        self._task = BugTask.from_dict(
            {
                "schema_version": 1,
                "task_id": task.task_id,
                "issue": task.issue,
                "source_files": {file.path: file.content for file in task.source_files},
                "visible_tests": {file.path: file.content for file in task.visible_tests},
            }
        )
        for path in paths:
            if any(str(parent) in paths for parent in Path(path).parents):
                raise ValueError("Task file paths must not collide with parent directories")
        if (
            not isinstance(image, str)
            or not image
            or image.startswith("-")
            or any(character.isspace() or character == "\x00" for character in image)
        ):
            raise ValueError("image must be a nonempty Docker image reference")
        if limits is not None and not isinstance(limits, SandboxLimits):
            raise ValueError("limits must be SandboxLimits")
        self._image = image
        self._limits = limits if limits is not None else SandboxLimits()
        self._source = {file.path: file.content for file in self._task.source_files}
        self._tests = {file.path: file.content for file in self._task.visible_tests}
        self._image_id: str | None = None
        self._docker: str | None = None
        self._containers: set[str] = set()
        self._closed = False

    @property
    def task_id(self) -> str:
        return self._task.task_id

    def __enter__(self) -> "Sandbox":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            for name in tuple(self._containers):
                self._remove(name)
        finally:
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise SandboxError("Sandbox is closed")

    def read_file(self, path: str) -> str:
        self._ensure_open()
        if not isinstance(path, str) or path not in self._source and path not in self._tests:
            raise ValueError("Only exact declared source and visible-test paths may be read")
        return self._source[path] if path in self._source else self._tests[path]

    def edit_file(self, path: str, content: str) -> None:
        self._ensure_open()
        if not isinstance(path, str) or path not in self._source:
            raise ValueError("Only exact declared source paths may be edited")
        if not isinstance(content, str):
            raise ValueError("Source content must be UTF-8 text")
        try:
            content.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("Source content must be UTF-8 text") from error
        self._source[path] = content

    def submit_patch(self) -> str:
        self._ensure_open()
        patch = []
        for original in self._task.source_files:
            for line in difflib.unified_diff(
                _lines(original.content),
                _lines(self._source[original.path]),
                fromfile=f"a/{original.path}",
                tofile=f"b/{original.path}",
            ):
                patch.append(line)
                if not line.endswith("\n"):
                    patch.append("\n\\ No newline at end of file\n")
        return "".join(patch)

    def run_visible_tests(self) -> RunResult:
        return self._run("visible", b"")

    def run_check(self, module: str, function: str, args: list, kwargs: dict) -> RunResult:
        self._ensure_open()
        if not isinstance(module, str) or not all(
            part.isidentifier() for part in module.split(".")
        ):
            raise ValueError("Check module must be a dotted Python identifier")
        module_path = "src/" + module.replace(".", "/")
        if (
            module_path + ".py" not in self._source
            and module_path + "/__init__.py" not in self._source
        ):
            raise ValueError("Check module must correspond to a declared source file")
        if not isinstance(function, str) or not function.isidentifier():
            raise ValueError("Check function must be a Python identifier")
        if type(args) is not list or type(kwargs) is not dict:
            raise ValueError("Check args must be a list and kwargs must be an object")
        request = {"module": module, "function": function, "args": args, "kwargs": kwargs}
        try:
            _json_value(request)
            stdin = json.dumps(request, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ValueError("Check arguments must contain finite UTF-8 JSON data") from error
        return self._run("check", stdin)

    def _resolve_image(self) -> None:
        if self._image_id is not None:
            return
        self._docker = shutil.which("docker")
        if self._docker is None:
            raise SandboxError(
                "Docker is required; install/start Docker Desktop and build the image"
            )
        try:
            result = _capture(
                [self._docker, "image", "inspect", "--format", "{{.Id}}", self._image],
                timeout=_CONTROL_TIMEOUT,
                output_limit=_CONTROL_OUTPUT,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SandboxError(f"Cannot inspect local Docker image: {error}") from error
        image_id = result.stdout.decode("ascii", errors="replace").strip()
        if (
            result.stopped
            or result.exit_code != 0
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)
        ):
            detail = _bounded_text(result.stderr, _CONTROL_OUTPUT)
            raise SandboxError(
                f"Cannot resolve local image {self._image!r}; "
                f"start Docker and build it first. {detail}"
            )
        self._image_id = image_id

    def _materialize(self, root: Path) -> None:
        """Create only validated regular files under a fresh, private host directory."""
        if root.is_symlink() or not root.is_dir() or root.resolve() != root:
            raise SandboxError("Workspace root must be a real canonical directory")
        for name in ("src", "tests"):
            (root / name).mkdir(mode=0o755)
            (root / name).chmod(0o755)
        files = self._source | self._tests
        for relative in sorted(files):
            target = root / relative
            parent = root
            for part in Path(relative).parts[:-1]:
                parent = parent / part
                if not parent.exists():
                    parent.mkdir(mode=0o755)
                metadata = parent.lstat()
                if not stat.S_ISDIR(metadata.st_mode) or parent.is_symlink():
                    raise SandboxError("Workspace parent is not a real directory")
                parent.chmod(0o755)
            descriptor = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(files[relative].encode("utf-8"))
                os.fchmod(stream.fileno(), 0o644)
            if not stat.S_ISREG(target.lstat().st_mode):
                raise SandboxError("Workspace file is not a regular file")

    def _remove(self, name: str) -> None:
        try:
            result = _capture(
                [self._docker, "rm", "--force", name],
                timeout=_CONTROL_TIMEOUT,
                output_limit=_CONTROL_OUTPUT,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SandboxError(f"Cannot remove sandbox container {name}: {error}") from error
        if result.stopped or (result.exit_code != 0 and b"No such container" not in result.stderr):
            raise SandboxError(
                f"Cannot remove sandbox container {name}: "
                + _bounded_text(result.stderr, _CONTROL_OUTPUT)
            )
        self._containers.discard(name)

    def _run(self, mode: str, stdin: bytes) -> RunResult:
        self._ensure_open()
        self._resolve_image()
        started = time.monotonic()
        name = "toolsmith-" + uuid.uuid4().hex
        try:
            with tempfile.TemporaryDirectory(
                prefix="toolsmith-", dir=Path(tempfile.gettempdir()).resolve()
            ) as temporary:
                root = Path(temporary)
                self._materialize(root)
                # Docker --mount uses CSV; a comma in TMPDIR must not add mount options.
                if "," in temporary or "\n" in temporary or "\r" in temporary:
                    raise SandboxError("Docker workspace path cannot contain commas or newlines")
                # Container removal is done by explicit cleanup, not `docker run --rm`.
                command = [
                    self._docker,
                    "run",
                    "--pull=never",
                    "--name",
                    name,
                    "--interactive",
                    "--network=none",
                    "--ipc=none",
                    "--user=65532:65532",
                    "--read-only",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges",
                    f"--pids-limit={self._limits.pids_limit}",
                    f"--memory={self._limits.memory_mb}m",
                    f"--memory-swap={self._limits.memory_mb}m",
                    f"--cpus={self._limits.cpus}",
                    "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
                    "--log-driver=none",
                    "--workdir=/workspace",
                    "--env=PYTHONDONTWRITEBYTECODE=1",
                    "--env=PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                    "--mount",
                    f"type=bind,source={root / 'src'},target=/workspace/src,readonly",
                    "--mount",
                    f"type=bind,source={root / 'tests'},target=/workspace/tests,readonly",
                    "--entrypoint=python",
                    self._image_id,
                    "-I",
                    "/opt/toolsmith/runner.py",
                    mode,
                ]
                self._containers.add(name)
                cleanup_error = None
                try:
                    result = _capture(
                        command,
                        stdin=stdin,
                        timeout=self._limits.timeout_seconds,
                        output_limit=self._limits.output_bytes,
                    )
                finally:
                    try:
                        self._remove(name)
                    except SandboxError as error:
                        cleanup_error = str(error)
                if cleanup_error:
                    status, payload = "infrastructure_error", None
                elif result.stopped:
                    status, payload = result.stopped, None
                elif result.exit_code in (125, 126, 127):
                    status, payload = "infrastructure_error", None
                else:
                    status, payload = _protocol(result.stdout, result.exit_code, mode)
                stdout = _bounded_text(result.stdout, self._limits.output_bytes)
                raw_stderr = result.stderr
                if cleanup_error:
                    raw_stderr = (cleanup_error + "\n").encode("utf-8") + raw_stderr
                stderr = _bounded_text(raw_stderr, self._limits.output_bytes - len(stdout.encode()))
                return RunResult(
                    status=status,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=result.exit_code,
                    elapsed_seconds=time.monotonic() - started,
                    image_id=self._image_id,
                    payload=payload,
                )
        except (OSError, subprocess.SubprocessError) as error:
            raise SandboxError(f"Cannot set up or execute the Docker sandbox: {error}") from error
