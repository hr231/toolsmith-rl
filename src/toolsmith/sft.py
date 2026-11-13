"""Supervised fine-tuning data from verified demonstrations.

- Loss covers only generated tokens: assistant tool calls and the closing ``<|im_end|>``.
- System, user, and tool-response tokens are ignored, as is the ``<think></think>`` pair
  that inference pre-fills in the prompt.
- The assistant mask comes from TRL's training template; this module removes the
  pre-filled prefix and checks alignment.
"""

from toolsmith.episode import canonical_json
from toolsmith.inference import build_messages

IGNORE_INDEX = -100


def demo_messages(transcript: dict) -> list[dict]:
    """Rebuild the conversation of a submitted demonstration, ending on its last call."""
    if transcript.get("terminal_reason") != "submitted":
        raise ValueError("Only submitted demonstrations are training data")
    messages = build_messages(transcript["initial_context"])
    events = transcript["events"]
    for position, event in enumerate(events):
        call = {
            "type": "function",
            "function": {"name": event["tool"], "arguments": event["arguments"]},
        }
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        if position < len(events) - 1:
            messages.append({"role": "tool", "content": canonical_json(event["observation"])})
    return messages


def supervised_labels(
    ids: list[int], assistant_mask: list[int], think_ids: list[int], eos_id: int
) -> list[int]:
    """Labels for each assistant span: tokens after the pre-filled ``think_ids`` through
    the first ``eos_id``. Template text after EOS (a newline) is never generated."""
    if len(ids) != len(assistant_mask):
        raise ValueError("assistant mask must align with input ids")
    labels = [IGNORE_INDEX] * len(ids)
    position = 0
    while position < len(ids):
        if not assistant_mask[position]:
            position += 1
            continue
        end = position
        while end < len(ids) and assistant_mask[end]:
            end += 1
        span = ids[position:end]
        if span[: len(think_ids)] != think_ids:
            raise ValueError("assistant span does not start with the pre-filled think tokens")
        if eos_id not in span:
            raise ValueError("assistant span has no end-of-turn token (truncated)")
        start = position + len(think_ids)
        stop = position + span.index(eos_id) + 1
        if stop - start < 2:
            raise ValueError("assistant span has no generated tokens")
        labels[start:stop] = ids[start:stop]
        position = end
    return labels
