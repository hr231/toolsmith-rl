"""Stateful GRPO rollouts for TRL's ``rollout_func``.

- Each prompt in a generation batch becomes an isolated episode with its own ``Episode``
  and remote sandbox; episodes advance in lockstep, one batched generation per turn.
- Context = sampled token IDs + template-rendered tool results.
- env_mask is 1 for sampled tokens and 0 for tool-result and template tokens.
- Reward is the remote score of the submitted patch (0 without one); infrastructure
  errors yield ``None``, which TRL excludes from the group baseline.
"""

import json
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Protocol

from toolsmith.episode import Episode, EpisodeClosed, EpisodeLimits
from toolsmith.evaluation import GenerationConfig, episode_seed, reward_for
from toolsmith.inference import TOOL_SCHEMAS, ToolCallError, build_messages, parse_tool_calls
from toolsmith.remote import RemoteClient, RemoteSandbox
from toolsmith.tasks import BugTask


@dataclass(frozen=True)
class Generated:
    ids: list[int]  # sampled tokens, ending with EOS when the turn ended
    text: str  # decoded text before EOS
    ended: bool


class Generator(Protocol):
    def __call__(self, contexts: dict[int, list[int]], seed: int) -> dict[int, Generated]: ...


class Renderer(Protocol):
    def prompt_ids(self, messages: list[dict]) -> list[int]: ...

    def tool_suffix_ids(self, tool_contents: list[str]) -> list[int]: ...


@dataclass
class Rollout:
    task_id: str
    prompt_ids: list[int]
    completion_ids: list[int] = field(default_factory=list)
    env_mask: list[int] = field(default_factory=list)
    terminal_reason: str | None = None
    reward: int | None = None
    score: dict | None = None
    patch: str | None = None
    turns: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def trace(self) -> dict:
        """JSON-serializable record without token IDs."""
        return {
            "task_id": self.task_id,
            "terminal_reason": self.terminal_reason,
            "reward": self.reward,
            "score": self.score,
            "turns": self.turns,
            "events": self.events,
            "submitted_patch": self.patch,
            "prompt_tokens": len(self.prompt_ids),
            "completion_tokens": len(self.completion_ids),
            "sampled_tokens": sum(self.env_mask),
        }


def run_group(
    tasks: list[BugTask],
    generate: Generator,
    render: Renderer,
    client: RemoteClient,
    config: GenerationConfig,
    *,
    seed_key: str,
    limits: EpisodeLimits | None = None,
    sandbox=RemoteSandbox,
) -> list[Rollout]:
    """Run one independent episode per task (repeats allowed) and score each."""
    with ExitStack() as stack:
        episodes = [
            stack.enter_context(Episode(task, limits=limits, sandbox=sandbox(task, client)))
            for task in tasks
        ]
        rollouts = []
        for task, episode in zip(tasks, episodes, strict=True):
            prompt = render.prompt_ids(build_messages(episode.initial_context))
            rollouts.append(Rollout(task.task_id, prompt))
        turn = 0
        while True:
            active = [
                i
                for i, (rollout, episode) in enumerate(zip(rollouts, episodes, strict=True))
                if rollout.terminal_reason is None and not episode.terminal
            ]
            ready = {}
            for i in active:
                context = rollouts[i].prompt_ids + rollouts[i].completion_ids
                if len(context) + config.max_new_tokens > config.max_context_tokens:
                    rollouts[i].terminal_reason = "context_tokens_exhausted"
                else:
                    ready[i] = context
            if not ready:
                break
            outputs = generate(ready, episode_seed(config.seed, seed_key, turn))
            for i in ready:
                _advance(rollouts[i], episodes[i], outputs[i], render, turn)
            turn += 1
        for rollout, episode in zip(rollouts, episodes, strict=True):
            rollout.terminal_reason = rollout.terminal_reason or episode.terminal_reason
            rollout.events = [
                {
                    "index": e.index,
                    "tool": e.tool,
                    "arguments": e.arguments,
                    "observation": e.observation,
                }
                for e in episode.events
            ]
            rollout.patch = episode.submitted_patch
    for rollout in rollouts:
        if rollout.patch is not None:
            rollout.score = client.score(rollout.task_id, rollout.patch)
        rollout.reward = reward_for(rollout.terminal_reason, rollout.score)
    return rollouts


def training_prompts(tasks: list[BugTask]) -> tuple[list[dict], dict[str, BugTask]]:
    """TRL dataset rows (public prompts only) and the prompt-to-task map for rollouts."""
    rows, by_prompt = [], {}
    for task in tasks:
        prompt = build_messages(Episode(task).initial_context)
        key = prompt_key(prompt)
        if key in by_prompt:
            raise ValueError(f"{task.task_id} has the same prompt as {by_prompt[key].task_id}")
        rows.append({"prompt": prompt})
        by_prompt[key] = task
    return rows, by_prompt


def _advance(rollout: Rollout, episode: Episode, output: Generated, render, turn: int) -> None:
    rollout.completion_ids += output.ids
    rollout.env_mask += [1] * len(output.ids)
    record = {"turn": turn, "text": output.text, "new_tokens": len(output.ids)}
    rollout.turns.append(record)
    if not output.ended:
        rollout.terminal_reason = "truncated"
        return
    try:
        calls, _ = parse_tool_calls(output.text)
    except ToolCallError as error:
        rollout.terminal_reason = error.reason
        return
    dispatched = []
    for name, arguments in calls:
        try:
            dispatched.append(episode.dispatch(name, arguments))
        except EpisodeClosed:
            break
        if episode.terminal:
            break
    record["calls"] = [e.tool for e in dispatched]
    if episode.terminal:
        return
    suffix = render.tool_suffix_ids([e.observation_json for e in dispatched])
    rollout.completion_ids += suffix
    rollout.env_mask += [0] * len(suffix)


class ChatRenderer:
    """Token rendering with a Hugging Face tokenizer, matching ``HFPolicy`` exactly."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def _ids(self, messages: list[dict], generation: bool) -> list[int]:
        text = self._tokenizer.apply_chat_template(
            messages,
            tools=TOOL_SCHEMAS,
            enable_thinking=False,
            add_generation_prompt=generation,
            tokenize=False,
        )
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        return self._ids(messages, generation=True)

    def tool_suffix_ids(self, tool_contents: list[str]) -> list[int]:
        """Tokens after an assistant turn's EOS: tool results, then the next generation prompt."""
        call = {"type": "function", "function": {"name": "read_file", "arguments": {"path": "x"}}}
        dummy = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
        ]
        prefix = self._ids(dummy, generation=False)
        eos = self._tokenizer.eos_token_id
        prefix = prefix[: len(prefix) - prefix[::-1].index(eos)]
        full = self._ids(dummy + [{"role": "tool", "content": c} for c in tool_contents], True)
        if full[: len(prefix)] != prefix:
            raise ValueError("Chat template does not extend an assistant turn at its EOS")
        return full[len(prefix) :]


def prompt_key(prompt: list[dict]) -> str:
    """Identify a task from its TRL prompt (the user turn) without exposing the task ID."""
    return json.dumps(prompt, sort_keys=True)


def make_rollout_func(
    tasks_by_prompt: dict[str, BugTask],
    generate: Generator,
    render: Renderer,
    client: RemoteClient,
    config: GenerationConfig,
    on_rollouts: Callable[[int, list[Rollout]], None],
):
    """Adapt ``run_group`` to TRL's ``rollout_func(prompts, trainer)`` contract."""

    def rollout_func(prompts: list, trainer) -> dict:
        tasks = [tasks_by_prompt[prompt_key(prompt)] for prompt in prompts]
        step = trainer.state.global_step
        rollouts = run_group(tasks, generate, render, client, config, seed_key=f"grpo-{step}")
        on_rollouts(step, rollouts)
        return {
            "prompt_ids": [r.prompt_ids for r in rollouts],
            "completion_ids": [r.completion_ids for r in rollouts],
            "logprobs": None,  # without vLLM, TRL recomputes log-probabilities itself
            "env_mask": [r.env_mask for r in rollouts],
            "toolsmith_reward": [r.reward for r in rollouts],
        }

    return rollout_func


def episode_reward(prompts, completions, toolsmith_reward, **kwargs) -> list[float | None]:
    """TRL reward function: the evaluator score computed during the rollout."""
    return [None if reward is None else float(reward) for reward in toolsmith_reward]
