"""Multi-turn evaluation episodes: model replies -> strict parsing -> tools -> remote score.

- A policy produces one reply per turn from the structured conversation.
- Replies are parsed exactly (see ``toolsmith.inference``); malformed, truncated, or
  over-budget output ends the episode as a recorded failure, never repaired.
- Visible tests and the final score run through the remote sandbox service.
- Each tool call is re-rendered canonically by the chat template; raw model text is kept
  in the trace.
"""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from toolsmith.episode import Episode, EpisodeClosed, EpisodeLimits
from toolsmith.inference import TOOL_SCHEMAS, ToolCallError, build_messages, parse_tool_calls
from toolsmith.remote import RemoteClient, RemoteSandbox
from toolsmith.tasks import BugTask


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 256
    max_context_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    seed: int = 17


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int
    new_tokens: int
    truncated: bool
    seconds: float
    context_exhausted: bool = False


class Policy(Protocol):
    def reply(self, messages: list[dict], seed: int) -> Reply: ...


class ScriptedPolicy:
    """Deterministic replies for tests and scripted references (not a model)."""

    def __init__(self, replies: list[Reply]) -> None:
        self._replies = list(replies)

    def reply(self, messages: list[dict], seed: int) -> Reply:
        if not self._replies:
            return Reply("", 0, 0, False, 0.0)
        return self._replies.pop(0)


def episode_seed(seed: int, task_id: str, turn: int) -> int:
    digest = hashlib.sha256(f"{seed}:{task_id}:{turn}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def reward_for(terminal: str, score: dict | None) -> int | None:
    """Episode reward: None for infrastructure failures, 0 without a scored patch."""
    if terminal == "infrastructure_error" or (score and score["status"] == "infrastructure_error"):
        return None
    if score is None:
        return 0
    return score["reward"]


def run_episode(
    task: BugTask,
    policy: Policy,
    client: RemoteClient,
    config: GenerationConfig,
    *,
    limits: EpisodeLimits | None = None,
    sandbox=RemoteSandbox,
) -> dict:
    """Run one episode and return a JSON-serializable trace including the remote score."""
    episode = Episode(task, limits=limits, sandbox=sandbox(task, client))
    messages = build_messages(episode.initial_context)
    turns, terminal = [], None
    with episode:
        turn = 0
        while not episode.terminal:
            result = policy.reply(messages, episode_seed(config.seed, task.task_id, turn))
            record = {"turn": turn, **asdict(result)}
            turns.append(record)
            turn += 1
            if result.context_exhausted:
                terminal = "context_tokens_exhausted"
                break
            if result.truncated:
                terminal = "truncated"
                break
            try:
                calls, preamble = parse_tool_calls(result.text)
            except ToolCallError as error:
                terminal = error.reason
                break
            dispatched = []
            for name, arguments in calls:
                try:
                    dispatched.append(episode.dispatch(name, arguments))
                except EpisodeClosed:
                    break
                if episode.terminal:
                    break
            record["calls"] = len(calls)
            record["dispatched"] = [
                {"tool": e.tool, "observation": e.observation} for e in dispatched
            ]
            # All of a reply's tool results are appended together before the next
            # generation, which keeps earlier turns' rendering fixed.
            tool_calls = [
                {"type": "function", "function": {"name": e.tool, "arguments": e.arguments}}
                for e in dispatched
            ]
            messages = messages + [
                {"role": "assistant", "content": preamble, "tool_calls": tool_calls}
            ]
            messages += [{"role": "tool", "content": e.observation_json} for e in dispatched]
        terminal = terminal or episode.terminal_reason
        events = [
            {
                "index": e.index,
                "tool": e.tool,
                "arguments": e.arguments,
                "observation": e.observation,
            }
            for e in episode.events
        ]
        patch = episode.submitted_patch
        image_id = episode.image_id
    score = client.score(task.task_id, patch) if patch is not None else None
    reward = reward_for(terminal, score)
    return {
        "task_id": task.task_id,
        "terminal_reason": terminal,
        "turns": turns,
        "events": events,
        "submitted_patch": patch,
        "score": score,
        "reward": reward,
        "success": bool(score and score["success"]),
        "tool_calls": len(events),
        "new_tokens": sum(t["new_tokens"] for t in turns),
        "seconds": round(sum(t["seconds"] for t in turns), 3),
        "image_id": image_id or (score or {}).get("image_id"),
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate with every episode, including failures, in every denominator."""
    ids = [row["task_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs in evaluation rows")
    if not rows:
        raise ValueError("No evaluation rows")

    def stats(group: list[dict]) -> dict:
        count = len(group)
        return {
            "episodes": count,
            "success_rate": sum(r["success"] for r in group) / count,
            "infrastructure_error_rate": sum(r["reward"] is None for r in group) / count,
            "mean_tool_calls": sum(r["tool_calls"] for r in group) / count,
            "mean_new_tokens": sum(r["new_tokens"] for r in group) / count,
            "mean_seconds": sum(r["seconds"] for r in group) / count,
        }

    families = defaultdict(list)
    for row in rows:
        families[row["task_id"].split("-", 2)[2]].append(row)
    return {
        **stats(rows),
        "terminal_reasons": dict(Counter(r["terminal_reason"] for r in rows)),
        "by_family": {name: stats(group) for name, group in sorted(families.items())},
    }


def write_run(output: Path, config: dict, rows: list[dict]) -> dict:
    """Write config, trajectories, and summary to a new directory; never overwrite."""
    summary = summarize(rows)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    with (output / "trajectories.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


class HFPolicy:
    """Dory via Transformers on a CUDA device (imports torch lazily)."""

    def __init__(self, model, tokenizer, config: GenerationConfig, device: str = "cuda:0") -> None:
        self._model, self._tokenizer, self._config, self._device = model, tokenizer, config, device

    def reply(self, messages: list[dict], seed: int) -> Reply:
        import time

        import torch

        tokenizer = self._tokenizer
        text = tokenizer.apply_chat_template(
            messages,
            tools=TOOL_SCHEMAS,
            enable_thinking=False,
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        prompt_tokens = ids.shape[1]
        if prompt_tokens + self._config.max_new_tokens > self._config.max_context_tokens:
            return Reply("", prompt_tokens, 0, False, 0.0, context_exhausted=True)
        torch.manual_seed(seed)
        started = time.time()
        with torch.no_grad():
            output = self._model.generate(
                ids.to(self._device),
                max_new_tokens=self._config.max_new_tokens,
                do_sample=True,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        new = output[0, prompt_tokens:].tolist()
        ended = tokenizer.eos_token_id in new
        if ended:
            new = new[: new.index(tokenizer.eos_token_id)]
        return Reply(
            text=tokenizer.decode(new, skip_special_tokens=False),
            prompt_tokens=prompt_tokens,
            new_tokens=len(new) + int(ended),
            truncated=not ended,
            seconds=round(time.time() - started, 3),
        )
