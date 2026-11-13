"""Remote sandbox service and its standard-library client.

- Clients send a task ID plus the episode's current source files or a submitted patch.
- The service resolves the task in the validated dataset, runs it in the Docker sandbox,
  and returns only public observations.
- Private checks, reference patches, and metadata stay on the sandbox host.
- The server binds to 127.0.0.1; an authenticated tunnel is the only route in.
"""

import hmac
import json
import secrets
import stat
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from toolsmith.dataset import SPLITS, DatasetRecord, load_records, validate_dataset
from toolsmith.evaluator import evaluate_patch
from toolsmith.sandbox import DEFAULT_IMAGE, RunResult, Sandbox, SandboxError, SandboxLimits

PROTOCOL_VERSION = 1
MAX_BODY_BYTES = 512 * 1024
_SCORE_FIELDS = ("status", "success", "reward", "patch_sha256", "image_id")


class RemoteProtocolError(ValueError):
    """The service rejected a request as invalid; a configuration bug, not model failure."""


def ensure_token(path: Path) -> str:
    """Return the bearer token stored at ``path``, creating a private one if absent."""
    if path.exists():
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise ValueError(f"Token file {path} has unsafe permissions {oct(mode)}; use 0600")
        token = path.read_text(encoding="utf-8").strip()
        if len(token) < 32:
            raise ValueError(f"Token in {path} is too short")
        return token
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    token = secrets.token_urlsafe(32)
    with path.open("x", encoding="utf-8") as stream:
        path.chmod(0o600)
        stream.write(token + "\n")
    return token


def load_service_records(directory: Path) -> dict[str, DatasetRecord]:
    """Validate the dataset once, then index every split's records by task ID."""
    validate_dataset(directory)
    return {
        record.task.task_id: record for split in SPLITS for record in load_records(directory, split)
    }


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")


class _BadRequest(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class SandboxService:
    """Typed operations over a fixed dataset; no paths, images, or limits from callers."""

    def __init__(
        self,
        records: dict[str, DatasetRecord],
        *,
        token: str,
        image: str = DEFAULT_IMAGE,
        limits: SandboxLimits | None = None,
        workers: int = 4,
        queue_timeout: float = 60.0,
        score_log: Path | None = None,
    ) -> None:
        if not records:
            raise ValueError("Service needs at least one task")
        if not isinstance(token, str) or len(token) < 16:
            raise ValueError("Service token must be at least 16 characters")
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError("workers must be an integer in [1, 16]")
        self._records = records
        self._token = f"Bearer {token}".encode()
        self._image = image
        self._limits = limits or SandboxLimits()
        self._slots = threading.BoundedSemaphore(workers)
        self._queue_timeout = queue_timeout
        self.score_log = score_log
        self._log_lock = threading.Lock()

    def authorized(self, header: str | None) -> bool:
        return header is not None and hmac.compare_digest(header.encode(), self._token)

    def _record(self, payload: dict) -> DatasetRecord:
        task_id = payload["task_id"]
        if not isinstance(task_id, str) or task_id not in self._records:
            raise _BadRequest(400, "Unknown task_id")
        return self._records[task_id]

    def _acquire(self) -> bool:
        if self._queue_timeout <= 0:
            return self._slots.acquire(blocking=False)
        return self._slots.acquire(timeout=self._queue_timeout)

    def visible(self, payload: dict) -> dict:
        if set(payload) != {"task_id", "source_files"}:
            raise _BadRequest(400, "visible requires exactly task_id and source_files")
        record = self._record(payload)
        files = payload["source_files"]
        declared = {file.path for file in record.task.source_files}
        if type(files) is not dict or set(files) != declared:
            raise _BadRequest(400, "source_files must contain exactly the task's source paths")
        if not all(isinstance(content, str) for content in files.values()):
            raise _BadRequest(400, "source file contents must be strings")
        if not self._acquire():
            raise _BadRequest(503, "Sandbox service is busy")
        try:
            with Sandbox(record.task, image=self._image, limits=self._limits) as sandbox:
                for path, content in files.items():
                    sandbox.edit_file(path, content)
                result = sandbox.run_visible_tests()
        except SandboxError as error:
            return {
                "status": "infrastructure_error",
                "stdout": "",
                "stderr": str(error),
                "exit_code": None,
                "image_id": None,
            }
        finally:
            self._slots.release()
        return {
            "status": result.status,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "image_id": result.image_id,
        }

    def score(self, payload: dict) -> dict:
        if set(payload) != {"task_id", "patch"} or not isinstance(payload["patch"], str):
            raise _BadRequest(400, "score requires exactly task_id and a string patch")
        record = self._record(payload)
        if not self._acquire():
            raise _BadRequest(503, "Sandbox service is busy")
        try:
            result = evaluate_patch(
                record.task, payload["patch"], record.checks, image=self._image, limits=self._limits
            )
        finally:
            self._slots.release()
        if self.score_log is not None:
            row = {
                "time": datetime.now(UTC).isoformat(),
                "task_id": record.task.task_id,
                "result": result,
            }
            with self._log_lock, self.score_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        # Per-case values, expectations, and failure text stay in the local score log.
        return {field: result.get(field) for field in _SCORE_FIELDS}


class _Handler(BaseHTTPRequestHandler):
    server_version = "toolsmith-sandbox/1"
    timeout = 30

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        return  # request bodies may hold model code; the score log is the audit trail

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        if self.server.service.authorized(self.headers.get("Authorization")):
            return True
        self._send(401, {"error": "unauthorized"})
        return False

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if not self._authorized():
            return
        if self.path != "/v1/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {"ok": True, "protocol": PROTOCOL_VERSION})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if not self._authorized():
            return
        operation = {"/v1/visible": "visible", "/v1/score": "score"}.get(self.path)
        if operation is None:
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send(411, {"error": "Content-Length required"})
            return
        if not 0 < length <= MAX_BODY_BYTES:
            self._send(413, {"error": f"body must be 1..{MAX_BODY_BYTES} bytes"})
            return
        try:
            payload = json.loads(
                self.rfile.read(length).decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            if type(payload) is not dict:
                raise _BadRequest(400, "body must be a JSON object")
            self._send(200, getattr(self.server.service, operation)(payload))
        except _BadRequest as error:
            self._send(error.code, {"error": str(error)})
        except (ValueError, UnicodeDecodeError, RecursionError) as error:
            self._send(400, {"error": f"invalid JSON: {error}"})


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, service: SandboxService, port: int) -> None:
        super().__init__(("127.0.0.1", port), _Handler)
        self.service = service


def serve(service: SandboxService, *, port: int) -> ThreadingHTTPServer:
    """Start serving on 127.0.0.1 in a background thread and return the server."""
    httpd = _Server(service, port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


class RemoteClient:
    """HTTP client for the sandbox service; network and 5xx errors are infrastructure failures."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 120.0) -> None:
        if not base_url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ValueError("Remote sandbox URL must be HTTPS (or loopback for local testing)")
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(
            self._base + path,
            data=data,
            method=method,
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            if 400 <= error.code < 500:
                raise RemoteProtocolError(f"{error.code} from sandbox service: {detail}") from error
            raise SandboxError(f"{error.code} from sandbox service: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise SandboxError(f"Sandbox service unreachable: {error}") from error

    def health(self) -> dict:
        return self._call("GET", "/v1/health")

    def visible(self, task_id: str, source_files: dict[str, str]) -> RunResult:
        started = time.monotonic()
        try:
            body = self._call(
                "POST", "/v1/visible", {"task_id": task_id, "source_files": source_files}
            )
        except SandboxError as error:
            return RunResult(
                "infrastructure_error", "", str(error), None, time.monotonic() - started, ""
            )
        return RunResult(
            body["status"],
            body["stdout"],
            body["stderr"],
            body["exit_code"],
            time.monotonic() - started,
            body.get("image_id") or "",
        )

    def score(self, task_id: str, patch: str) -> dict:
        try:
            return self._call("POST", "/v1/score", {"task_id": task_id, "patch": patch})
        except SandboxError as error:
            return {
                "status": "infrastructure_error",
                "success": False,
                "reward": None,
                "patch_sha256": None,
                "image_id": None,
                "error": str(error),
            }


class RemoteSandbox(Sandbox):
    """Sandbox that keeps edits in memory and runs visible tests on the remote service."""

    def __init__(self, task, client: RemoteClient) -> None:
        super().__init__(task)
        self._client = client

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._closed = True  # no local containers exist

    def run_visible_tests(self) -> RunResult:
        self._ensure_open()
        return self._client.visible(self._task.task_id, dict(self._source))

    def run_check(self, module: str, function: str, args: list, kwargs: dict) -> RunResult:
        raise SandboxError("Private checks run only on the sandbox host through the score endpoint")


def run_service_forever(service: SandboxService, *, port: int) -> None:
    httpd = _Server(service, port)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
