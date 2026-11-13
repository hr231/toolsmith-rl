import pytest

from toolsmith.sft import IGNORE_INDEX, demo_messages, supervised_labels


def transcript():
    return {
        "initial_context": {
            "task_id": "train-0000-x",
            "issue": "Fix it.",
            "source_files": ["src/solution.py"],
            "visible_tests": ["tests/test_solution.py"],
        },
        "events": [
            {
                "tool": "read_file",
                "arguments": {"path": "src/solution.py"},
                "observation": {"content": "x = 1\n"},
            },
            {
                "tool": "submit_patch",
                "arguments": {},
                "observation": {"status": "submitted", "patch": "p"},
            },
        ],
        "terminal_reason": "submitted",
    }


def test_conversation_ends_on_the_final_assistant_call_without_its_observation():
    messages = demo_messages(transcript())
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert messages[-1]["tool_calls"][0]["function"] == {"name": "submit_patch", "arguments": {}}
    assert messages[3]["content"] == '{"content":"x = 1\\n"}'  # same text inference sends


def test_unsubmitted_transcripts_are_rejected():
    bad = transcript() | {"terminal_reason": "tool_budget_exhausted"}
    with pytest.raises(ValueError):
        demo_messages(bad)


EOS = 2


def test_labels_keep_only_generated_assistant_tokens_through_eos():
    think = [7, 8]
    # Each masked span: think prefill, generated call, EOS, then a template newline (5).
    ids = [1, 3, 7, 8, 40, 41, EOS, 5, 9, 3, 7, 8, 50, EOS, 5]
    mask = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
    labels = supervised_labels(ids, mask, think, EOS)
    ignored = IGNORE_INDEX
    assert labels == [ignored] * 4 + [40, 41, EOS] + [ignored] * 5 + [50, EOS, ignored]


def test_a_span_without_prefill_eos_or_content_is_an_error():
    for ids, mask in (
        ([1, 40, EOS], [0, 1, 1]),  # no pre-filled think tokens
        ([1, 7, 8, 40], [0, 1, 1, 1]),  # never ends: truncated turn
        ([1, 7, 8, EOS], [0, 1, 1, 1]),  # nothing generated but EOS
    ):
        with pytest.raises(ValueError):
            supervised_labels(ids, mask, [7, 8], EOS)


def test_masks_must_align_with_ids():
    with pytest.raises(ValueError):
        supervised_labels([1, 2, 3], [0, 1], [7], EOS)
