import os

import pytest

from toolsmith.dataset import SOURCE_PATH, DatasetConfig, generate_dataset, load_records
from toolsmith.evaluation import GenerationConfig
from toolsmith.evaluator import apply_patch, evaluate_patch
from toolsmith.grpo import Generated, episode_reward, run_group
from toolsmith.inference import render_tool_call
from toolsmith.remote import RemoteClient, SandboxService, load_service_records, serve
from toolsmith.sandbox import DEFAULT_IMAGE, SandboxLimits

TOKEN = "grpo-test-token-0123456789abcdef"
SMALL = DatasetConfig(train=3, dev=3, test_iid=3, test_compositional=2, challenge=4)
EOS = 0


def ids(text):
    return [ord(c) for c in text]


class ToyRenderer:
    """Character-level stand-in for the chat template: prompt marker and tool wrapper."""

    def prompt_ids(self, messages):
        return ids("<prompt>")

    def tool_suffix_ids(self, contents):
        return ids("".join(f"<tool>{c}</tool>" for c in contents) + "<assistant>")


class Scripted:
    """Per-episode replies; a reply of None stands for a turn cut off without EOS."""

    def __init__(self, scripts):
        self.scripts = {i: list(s) for i, s in scripts.items()}
        self.calls = []

    def __call__(self, contexts, seed):
        self.calls.append(sorted(contexts))
        outputs = {}
        for i in contexts:
            text = self.scripts[i].pop(0)
            if text is None:
                outputs[i] = Generated(ids("<tool_call>"), "<tool_call>", ended=False)
            else:
                outputs[i] = Generated(ids(text) + [EOS], text, ended=True)
        return outputs


@pytest.fixture
def dataset(tmp_path):
    directory = tmp_path / "data"
    generate_dataset(SMALL, directory)
    return directory


def offline_client():
    return RemoteClient("http://127.0.0.1:9", TOKEN, timeout=2)


def test_sampled_tokens_train_and_tool_results_do_not(dataset):
    task = load_records(dataset, "train")[0].task
    read = render_tool_call("read_file", {"path": SOURCE_PATH})
    generate = Scripted({0: [read, "I am done."]})
    [rollout] = run_group(
        [task], generate, ToyRenderer(), offline_client(), GenerationConfig(), seed_key="t"
    )
    observation = rollout.events[0]["observation"]
    suffix = ToyRenderer().tool_suffix_ids([rollout_observation_json(rollout, 0)])
    sampled_1, sampled_2 = ids(read) + [EOS], ids("I am done.") + [EOS]
    assert rollout.completion_ids == sampled_1 + suffix + sampled_2
    assert rollout.env_mask == [1] * len(sampled_1) + [0] * len(suffix) + [1] * len(sampled_2)
    assert observation["content"] == task.source_files[0].content
    assert rollout.terminal_reason == "no_tool_call" and rollout.reward == 0


def rollout_observation_json(rollout, index):
    from toolsmith.episode import canonical_json

    return canonical_json(rollout.events[index]["observation"])


def test_rollouts_of_one_task_are_isolated_episodes(dataset):
    task = load_records(dataset, "train")[0].task
    edit = render_tool_call("edit_file", {"path": SOURCE_PATH, "content": "x = 1\n"})
    read = render_tool_call("read_file", {"path": SOURCE_PATH})
    generate = Scripted({0: [edit, read, "stop"], 1: [read, "stop"]})
    edited, untouched = run_group(
        [task, task], generate, ToyRenderer(), offline_client(), GenerationConfig(), seed_key="t"
    )
    assert edited.events[1]["observation"]["content"] == "x = 1\n"
    assert untouched.events[0]["observation"]["content"] == task.source_files[0].content
    # Lockstep: both episodes generate together until the shorter one ends.
    assert generate.calls == [[0, 1], [0, 1], [0]]


def test_truncation_and_context_budget_end_episodes_without_reward(dataset):
    task = load_records(dataset, "train")[0].task
    [cut] = run_group(
        [task],
        Scripted({0: [None]}),
        ToyRenderer(),
        offline_client(),
        GenerationConfig(),
        seed_key="t",
    )
    assert cut.terminal_reason == "truncated" and cut.reward == 0
    assert cut.env_mask == [1] * len(cut.completion_ids)

    def never(contexts, seed):
        raise AssertionError("generation must not start past the context budget")

    tight = GenerationConfig(max_new_tokens=256, max_context_tokens=260)
    [full] = run_group([task], never, ToyRenderer(), offline_client(), tight, seed_key="t")
    assert full.terminal_reason == "context_tokens_exhausted" and full.completion_ids == []


def test_infrastructure_failure_yields_no_reward_for_trl_to_exclude(dataset):
    task = load_records(dataset, "train")[0].task
    tests = render_tool_call("run_visible_tests", {})
    [rollout] = run_group(
        [task],
        Scripted({0: [tests]}),
        ToyRenderer(),
        offline_client(),
        GenerationConfig(),
        seed_key="t",
    )
    assert rollout.terminal_reason == "infrastructure_error"
    assert rollout.reward is None
    assert episode_reward([], [], toolsmith_reward=[rollout.reward, 1, 0]) == [None, 1.0, 0.0]


@pytest.mark.sandbox
@pytest.mark.skipif(os.environ.get("TOOLSMITH_RUN_SANDBOX_TESTS") != "1", reason="needs Docker")
def test_online_rollout_reward_matches_offline_evaluation(dataset, tmp_path):
    service = SandboxService(
        load_service_records(dataset),
        token=TOKEN,
        image=os.environ.get("TOOLSMITH_SANDBOX_IMAGE", DEFAULT_IMAGE),
        limits=SandboxLimits(),
        workers=2,
    )
    httpd = serve(service, port=0)
    record = load_records(dataset, "train")[1]
    fixed = apply_patch(record.task, record.reference_patch).source_files[0].content
    good = [
        render_tool_call("edit_file", {"path": SOURCE_PATH, "content": fixed}),
        render_tool_call("submit_patch", {}),
    ]
    wrong = record.task.source_files[0].content + "# reviewed\n"  # non-empty diff, bug kept
    bad = [
        render_tool_call("edit_file", {"path": SOURCE_PATH, "content": wrong}),
        render_tool_call("submit_patch", {}),
    ]
    try:
        client = RemoteClient(f"http://127.0.0.1:{httpd.server_address[1]}", TOKEN)
        rollouts = run_group(
            [record.task, record.task],
            Scripted({0: good, 1: bad}),
            ToyRenderer(),
            client,
            GenerationConfig(),
            seed_key="t",
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert [r.reward for r in rollouts] == [1, 0]
    for rollout in rollouts:
        offline = evaluate_patch(record.task, rollout.patch, record.checks, image=DEFAULT_IMAGE)
        assert offline["reward"] == rollout.reward
