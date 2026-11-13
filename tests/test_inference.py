import pytest

from toolsmith.inference import (
    SYSTEM_PROMPT,
    TOOL_SCHEMAS,
    ToolCallError,
    build_messages,
    parse_tool_calls,
    render_tool_call,
)


def call(name, **params):
    return render_tool_call(name, params)


def test_parses_a_single_call_and_preserves_code_whitespace_exactly():
    code = "def solve(values):\n    return [v for v in values if v >= 3]\n"
    calls, preamble = parse_tool_calls(
        call("edit_file", path="src/solution.py", content=code) + "\n"
    )
    assert calls == [("edit_file", {"path": "src/solution.py", "content": code})]
    assert preamble == ""


def test_parses_consecutive_calls_in_order_as_dory_emits_them():
    text = (
        "\n"
        + call("read_file", path="src/solution.py")
        + "\n"
        + call("read_file", path="tests/test_solution.py")
        + "\n"
    )
    calls, _ = parse_tool_calls(text)
    assert calls == [
        ("read_file", {"path": "src/solution.py"}),
        ("read_file", {"path": "tests/test_solution.py"}),
    ]


def test_allows_reasoning_before_the_first_call_but_records_it():
    calls, preamble = parse_tool_calls("Let me read it first.\n" + call("run_visible_tests"))
    assert (calls, preamble) == ([("run_visible_tests", {})], "Let me read it first.")


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("The bug is on line 3.", "no_tool_call"),
        ("", "no_tool_call"),
        (
            call("read_file", path="a") + "\nthen\n" + call("submit_patch"),
            "text_between_tool_calls",
        ),
        (call("submit_patch") + "\nDone!", "text_after_tool_call"),
        (
            "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/a.py\n</function>",
            "malformed_tool_call",
        ),
        ("<tool_call>\nread_file(path='a')\n</tool_call>", "malformed_tool_call"),
        (call("submit_patch") + "<tool_call>\nbroken\n</tool_call>", "malformed_tool_call"),
        (call("read_file", path="a", **{"path ": "b"}), "malformed_tool_call"),
    ],
)
def test_rejects_without_repairing(text, reason):
    with pytest.raises(ToolCallError) as error:
        parse_tool_calls(text)
    assert error.value.reason == reason


def test_duplicate_parameters_are_rejected():
    text = (
        "<tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>\n"
        "<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>"
    )
    with pytest.raises(ToolCallError) as error:
        parse_tool_calls(text)
    assert error.value.reason == "malformed_tool_call"


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("read_file", {"path": "src/solution.py"}),
        ("edit_file", {"path": "src/solution.py", "content": "x = 1\n\n\ny = 2\n"}),
        ("edit_file", {"path": "src/solution.py", "content": "no trailing newline"}),
        ("run_visible_tests", {}),
        ("submit_patch", {}),
    ],
)
def test_rendered_calls_round_trip(name, arguments):
    assert parse_tool_calls(render_tool_call(name, arguments))[0] == [(name, arguments)]


def test_schemas_match_the_episode_tools_exactly():
    names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert names == {"read_file", "edit_file", "run_visible_tests", "submit_patch"}
    params = {
        schema["function"]["name"]: set(schema["function"]["parameters"]["properties"])
        for schema in TOOL_SCHEMAS
    }
    assert params == {
        "read_file": {"path"},
        "edit_file": {"path", "content"},
        "run_visible_tests": set(),
        "submit_patch": set(),
    }


def test_messages_carry_only_the_public_initial_context():
    context = {
        "task_id": "dev-0001-x",
        "issue": "Keep values >= 3.",
        "source_files": ["src/solution.py"],
        "visible_tests": ["tests/test_solution.py"],
        "tools": {"ignored": "episode descriptions are replaced by schemas"},
    }
    messages = build_messages(context)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert "Keep values >= 3." in messages[1]["content"]
    assert "src/solution.py" in messages[1]["content"]
    assert "dev-0001-x" not in messages[1]["content"]
    assert len(messages) == 2
