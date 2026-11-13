"""Model-facing protocol for Dory: tool schemas, prompts, and strict XML tool-call parsing.

- Tool calls use ``<tool_call>``/``<function=NAME>``/``<parameter=P>`` XML blocks.
- Parsing is exact: malformed or extra output is a recorded failure, never repaired.
- Model execution imports torch lazily, so the module imports in the CPU-only install.
"""

import re

SYSTEM_PROMPT = (
    "You fix bugs in small Python projects. Use the tools to read the source, run the "
    "visible tests, edit the source file, and submit. You may call several tools in one "
    "reply; they run in order. Edit only files under src/; tests cannot be changed. "
    "When the visible tests pass, call submit_patch."
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a declared source or visible-test file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Exact file path."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace the entire contents of a declared source file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Exact source path under src/."},
                    "content": {"type": "string", "description": "Complete new file contents."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_visible_tests",
            "description": "Run the visible tests against the current source.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_patch",
            "description": "Submit the current source changes and end the task.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

_CALL = re.compile(
    r"<tool_call>\n<function=([A-Za-z_][A-Za-z0-9_]*)>\n(.*?)</function>\n</tool_call>", re.S
)
_PARAM = re.compile(r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\n(.*?)\n</parameter>\n", re.S)


class ToolCallError(ValueError):
    """Model output is not a sequence of well-formed tool calls."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def build_messages(initial_context: dict) -> list[dict]:
    """System prompt plus a user turn from the episode's public context (not its task ID)."""
    user = (
        f"Bug report:\n{initial_context['issue']}\n\n"
        f"Source files: {', '.join(initial_context['source_files'])}\n"
        f"Visible tests: {', '.join(initial_context['visible_tests'])}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def render_tool_call(name: str, arguments: dict[str, str]) -> str:
    """Render a call exactly as Dory's chat template does (used for checks and tests)."""
    body = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _arguments(body: str) -> dict[str, str]:
    arguments: dict[str, str] = {}
    position = 0
    for param in _PARAM.finditer(body):
        if param.start() != position or param.group(1) in arguments:
            raise ToolCallError("malformed_tool_call", "Unexpected text or duplicate parameter")
        arguments[param.group(1)] = param.group(2)
        position = param.end()
    if position != len(body):
        raise ToolCallError("malformed_tool_call", "Unparsed text inside the tool call")
    return arguments


def parse_tool_calls(text: str) -> tuple[list[tuple[str, dict[str, str]]], str]:
    """Return ([(tool name, string arguments), ...], preamble) or raise ToolCallError.

    Free text is allowed only before the first call; calls may be separated only by
    whitespace, and nothing but whitespace may follow the last call.
    """
    starts = text.count("<tool_call>")
    if starts == 0:
        raise ToolCallError("no_tool_call", "Reply contains no tool call")
    matches = list(_CALL.finditer(text))
    if len(matches) != starts:
        raise ToolCallError("malformed_tool_call", "A tool call does not follow the XML format")
    for previous, current in zip(matches, matches[1:], strict=False):
        if text[previous.end() : current.start()].strip():
            raise ToolCallError("text_between_tool_calls", "Text appears between tool calls")
    if text[matches[-1].end() :].strip():
        raise ToolCallError("text_after_tool_call", "Text follows the last tool call")
    calls = [(match.group(1), _arguments(match.group(2))) for match in matches]
    return calls, text[: matches[0].start()].strip()
