"""Bounded public tool episodes; task source stays inert outside the Docker sandbox.

- Counts context as canonical UTF-8 JSON bytes: initial context plus recorded event
  envelopes (index, tool, arguments, observation).
- Omits rejected malformed arguments from the trace.
- Rejects oversized inputs before any tool runs.
- Replaces oversized outputs with a truncation event that ends the episode, keeping
  arguments only if they fit; raises EpisodeClosed if even that event cannot fit.
- Treats a submitted patch as a candidate, not a verification result.
"""

import json
from dataclasses import dataclass

from toolsmith.sandbox import DEFAULT_IMAGE, Sandbox, SandboxError, SandboxLimits
from toolsmith.tasks import BugTask


class EpisodeClosed(ValueError):
    """The episode cannot accept another action or return a bounded observation."""


@dataclass(frozen=True)
class EpisodeLimits:
    max_tool_calls: int = 8
    max_context_bytes: int = 16384

    def __post_init__(self) -> None:
        for name in ("max_tool_calls", "max_context_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ToolEvent:
    index: int
    tool: str
    arguments_json: str
    observation_json: str

    @property
    def arguments(self) -> dict:
        return json.loads(self.arguments_json)

    @property
    def observation(self) -> dict:
        return json.loads(self.observation_json)


def canonical_json(value: object) -> str:
    """The exact text of tool arguments/observations as shown to the model."""
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


class Episode:
    """One independently edited task with four tools and no private evaluator input."""

    _FIELDS = {
        "read_file": frozenset({"path"}),
        "edit_file": frozenset({"path", "content"}),
        "run_visible_tests": frozenset(),
        "submit_patch": frozenset(),
    }

    def __init__(
        self,
        task: BugTask,
        *,
        image: str = DEFAULT_IMAGE,
        sandbox_limits: SandboxLimits | None = None,
        limits: EpisodeLimits | None = None,
        sandbox: Sandbox | None = None,
    ) -> None:
        if limits is not None and not isinstance(limits, EpisodeLimits):
            raise ValueError("limits must be EpisodeLimits")
        self._limits = limits if limits is not None else EpisodeLimits()
        if sandbox is None:
            # Sandbox revalidates even directly constructed tasks and never resolves
            # Docker until execution. Each episode receives a fresh editable snapshot.
            self._sandbox = Sandbox(task, image=image, limits=sandbox_limits)
        else:
            # An injected sandbox (e.g. RemoteSandbox) carries its own execution settings,
            # so caller-supplied image or sandbox_limits are rejected.
            if not isinstance(sandbox, Sandbox) or sandbox.task_id != task.task_id:
                raise ValueError("sandbox must be a Sandbox for the same task")
            if image != DEFAULT_IMAGE or sandbox_limits is not None:
                raise ValueError("image and sandbox_limits belong to the injected sandbox")
            self._sandbox = sandbox
        self._task_id = task.task_id
        self._source_paths = frozenset(file.path for file in task.source_files)
        self._read_paths = self._source_paths | frozenset(file.path for file in task.visible_tests)
        self._initial_json = canonical_json(
            {
                "task_id": task.task_id,
                "issue": task.issue,
                "source_files": sorted(self._source_paths),
                "visible_tests": sorted(file.path for file in task.visible_tests),
                "tools": {
                    "read_file": "Read an exact declared source or visible-test path: {path}.",
                    "edit_file": "Replace one declared source file with UTF-8 text: "
                    "{path,content}.",
                    "run_visible_tests": "Run only visible tests in the sandbox: {}.",
                    "submit_patch": "Submit a nonempty source diff and end this episode: {}.",
                },
            }
        )
        self._context_bytes = len(self._initial_json.encode("utf-8"))
        if self._context_bytes > self._limits.max_context_bytes:
            raise ValueError("Initial context exceeds max_context_bytes")
        self._events: list[ToolEvent] = []
        self._calls = 0
        self._terminal_reason: str | None = None
        self._submitted_patch: str | None = None
        self._image_id: str | None = None
        self._closed = False
        self._entered = False

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def events(self) -> tuple[ToolEvent, ...]:
        return tuple(self._events)

    @property
    def terminal(self) -> bool:
        return self._terminal_reason is not None

    @property
    def terminal_reason(self) -> str | None:
        return self._terminal_reason

    @property
    def submitted_patch(self) -> str | None:
        return self._submitted_patch

    @property
    def image_id(self) -> str | None:
        """Actual execution image identity; not included in model-facing context."""
        return self._image_id

    @property
    def context_bytes(self) -> int:
        return self._context_bytes

    @property
    def initial_context(self) -> dict:
        return json.loads(self._initial_json)

    def __enter__(self) -> "Episode":
        self._ensure_open()
        if self._entered:
            raise EpisodeClosed("Episode context cannot be entered twice")
        self._sandbox.__enter__()
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self._sandbox.__exit__(exc_type, exc_value, traceback)
        except SandboxError:
            self._terminal_reason = "infrastructure_error"
            raise
        finally:
            self._closed = True
            if self._terminal_reason is None:
                self._terminal_reason = "context_closed"

    def _ensure_open(self) -> None:
        if self._closed or self.terminal:
            raise EpisodeClosed(f"Episode is closed: {self._terminal_reason}")

    def _validate_action(self, tool: str, arguments: dict) -> None:
        if type(tool) is not str or tool not in self._FIELDS:
            raise ValueError(
                "Only read_file, edit_file, run_visible_tests, submit_patch are allowed"
            )
        if type(arguments) is not dict or set(arguments) != self._FIELDS[tool]:
            raise ValueError("Tool arguments must contain exactly the declared fields")
        # The four schemas contain strings only, or no fields. Reject all other
        # JSON values (including NaN), bytes, tuples, and user-defined objects.
        for value in arguments.values():
            if type(value) is not str:
                raise ValueError("Tool argument values must be UTF-8 strings")
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("Tool argument values must be UTF-8 strings") from error
        if tool == "read_file" and arguments["path"] not in self._read_paths:
            raise ValueError("Only exact declared source and visible-test paths may be read")
        if tool == "edit_file" and arguments["path"] not in self._source_paths:
            raise ValueError("Only exact declared source paths may be edited")

    def _event_size(self, tool: str, arguments: dict, observation: dict) -> int:
        return len(
            canonical_json(
                {
                    "index": self._calls,
                    "tool": tool,
                    "arguments": arguments,
                    "observation": observation,
                }
            ).encode("utf-8")
        )

    def _fits(self, tool: str, arguments: dict, observation: dict) -> bool:
        return (
            self._context_bytes + self._event_size(tool, arguments, observation)
            <= self._limits.max_context_bytes
        )

    def _append(self, tool: str, arguments: dict, observation: dict) -> ToolEvent:
        event = ToolEvent(self._calls, tool, canonical_json(arguments), canonical_json(observation))
        self._context_bytes += self._event_size(tool, arguments, observation)
        self._events.append(event)
        return event

    def _context_exhausted(self, tool: str, arguments: dict) -> ToolEvent:
        self._terminal_reason = "context_budget_exhausted"
        observation = {"error": "context_budget_exhausted", "truncated": True}
        if self._fits(tool, arguments, observation):
            return self._append(tool, arguments, observation)
        observation["arguments_omitted"] = True
        if self._fits(tool, {}, observation):
            return self._append(tool, {}, observation)
        raise EpisodeClosed("Episode context budget cannot fit a truncation event")

    def _record(self, tool: str, arguments: dict, observation: dict) -> ToolEvent:
        if not self._fits(tool, arguments, observation):
            return self._context_exhausted(tool, arguments)
        event = self._append(tool, arguments, observation)
        if not self.terminal and self._calls >= self._limits.max_tool_calls:
            self._terminal_reason = "tool_budget_exhausted"
        return event

    def dispatch(self, tool: str, arguments: dict) -> ToolEvent:
        self._ensure_open()
        self._calls += 1
        try:
            self._validate_action(tool, arguments)
        except ValueError as error:
            # Do not serialize arbitrary invalid objects, nonfinite values, or
            # unbounded tool names into an otherwise bounded public trace.
            safe_tool = tool if type(tool) is str and tool in self._FIELDS else "<invalid>"
            self._terminal_reason = "invalid_action"
            return self._record(
                safe_tool,
                {},
                {"error": "invalid_action", "message": str(error), "arguments_omitted": True},
            )

        # Check the input envelope before execution. For edits also reserve the
        # entire deterministic acknowledgment, so exhaustion cannot mutate source.
        preflight = {"status": "edited"} if tool == "edit_file" else {}
        if not self._fits(tool, arguments, preflight):
            return self._context_exhausted(tool, arguments)
        try:
            if tool == "read_file":
                observation = {"content": self._sandbox.read_file(arguments["path"])}
            elif tool == "edit_file":
                self._sandbox.edit_file(arguments["path"], arguments["content"])
                observation = preflight
            elif tool == "run_visible_tests":
                result = self._sandbox.run_visible_tests()
                self._image_id = result.image_id
                observation = {
                    "status": result.status,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                }
                if result.status == "infrastructure_error":
                    self._terminal_reason = "infrastructure_error"
                elif result.status in {"timeout", "output_limit"}:
                    self._terminal_reason = "budget_exhausted"
            else:
                patch = self._sandbox.submit_patch()
                if not patch:
                    self._terminal_reason = "invalid_patch"
                    observation = {"error": "invalid_patch", "message": "Source diff is empty"}
                else:
                    observation = {"status": "submitted", "patch": patch}
                    if not self._fits(tool, arguments, observation):
                        return self._context_exhausted(tool, arguments)
                    self._submitted_patch = patch
                    self._terminal_reason = "submitted"
        except SandboxError as error:
            self._terminal_reason = "infrastructure_error"
            observation = {"error": "infrastructure_error", "message": str(error)}
        return self._record(tool, arguments, observation)
