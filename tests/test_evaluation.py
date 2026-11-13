import json
import os

import pytest

from toolsmith.dataset import SOURCE_PATH, DatasetConfig, generate_dataset, load_records
from toolsmith.evaluation import (
    GenerationConfig,
    Reply,
    ScriptedPolicy,
    run_episode,
    summarize,
    write_run,
)
from toolsmith.evaluator import apply_patch
from toolsmith.inference import render_tool_call
from toolsmith.remote import (
    RemoteClient,
    RemoteSandbox,
    SandboxService,
    load_service_records,
    serve,
)
from toolsmith.sandbox import DEFAULT_IMAGE, SandboxLimits

TOKEN = "evaluation-test-token-0123456789"
SMALL = DatasetConfig(train=3, dev=3, test_iid=3, test_compositional=2, challenge=4)


@pytest.fixture
def dataset(tmp_path):
    directory = tmp_path / "data"
    generate_dataset(SMALL, directory)
    return directory


def offline_client():
    return RemoteClient("http://127.0.0.1:9", TOKEN, timeout=2)


def remote(task, client):
    return RemoteSandbox(task, client)


def reply(text):
    return Reply(text=text, prompt_tokens=600, new_tokens=40, truncated=False, seconds=0.1)


def test_malformed_output_ends_the_episode_without_dispatch_or_score(dataset):
    record = load_records(dataset, "dev")[0]
    policy = ScriptedPolicy([reply("I think the bug is the comparison operator.")])
    trace = run_episode(record.task, policy, offline_client(), GenerationConfig(), sandbox=remote)
    assert trace["terminal_reason"] == "no_tool_call"
    assert trace["events"] == [] and trace["submitted_patch"] is None
    assert trace["score"] is None and trace["reward"] == 0


def test_truncated_and_context_exhausted_replies_are_terminal(dataset):
    record = load_records(dataset, "dev")[0]
    truncated = Reply(
        text="<tool_call>\n<function=edit",
        prompt_tokens=600,
        new_tokens=256,
        truncated=True,
        seconds=1.0,
    )
    trace = run_episode(
        record.task,
        ScriptedPolicy([truncated]),
        offline_client(),
        GenerationConfig(),
        sandbox=remote,
    )
    assert trace["terminal_reason"] == "truncated"
    exhausted = Reply(
        text="",
        prompt_tokens=4000,
        new_tokens=0,
        truncated=False,
        seconds=0.0,
        context_exhausted=True,
    )
    trace = run_episode(
        record.task,
        ScriptedPolicy([exhausted]),
        offline_client(),
        GenerationConfig(),
        sandbox=remote,
    )
    assert trace["terminal_reason"] == "context_tokens_exhausted"


def test_infrastructure_failure_has_no_reward(dataset):
    record = load_records(dataset, "dev")[0]
    policy = ScriptedPolicy([reply(render_tool_call("run_visible_tests", {}))])
    trace = run_episode(record.task, policy, offline_client(), GenerationConfig(), sandbox=remote)
    assert trace["terminal_reason"] == "infrastructure_error"
    assert trace["reward"] is None


def test_summary_keeps_failures_in_denominators_and_rejects_duplicates():
    rows = [
        {
            "task_id": "dev-0000-a",
            "terminal_reason": "submitted",
            "reward": 1,
            "success": True,
            "tool_calls": 5,
            "new_tokens": 200,
            "seconds": 3.0,
        },
        {
            "task_id": "dev-0001-b",
            "terminal_reason": "no_tool_call",
            "reward": 0,
            "success": False,
            "tool_calls": 0,
            "new_tokens": 30,
            "seconds": 1.0,
        },
        {
            "task_id": "dev-0002-a",
            "terminal_reason": "infrastructure_error",
            "reward": None,
            "success": False,
            "tool_calls": 1,
            "new_tokens": 20,
            "seconds": 1.0,
        },
    ]
    summary = summarize(rows)
    assert summary["episodes"] == 3
    assert summary["success_rate"] == pytest.approx(1 / 3)
    assert summary["infrastructure_error_rate"] == pytest.approx(1 / 3)
    assert summary["terminal_reasons"] == {
        "submitted": 1,
        "no_tool_call": 1,
        "infrastructure_error": 1,
    }
    assert summary["by_family"]["a"]["episodes"] == 2
    with pytest.raises(ValueError):
        summarize(rows + rows[:1])


def test_run_directory_is_new_and_complete(tmp_path):
    rows = [
        {
            "task_id": "dev-0000-a",
            "terminal_reason": "submitted",
            "reward": 1,
            "success": True,
            "tool_calls": 5,
            "new_tokens": 200,
            "seconds": 3.0,
        }
    ]
    output = tmp_path / "run"
    write_run(output, {"model": "m"}, rows)
    assert json.loads((output / "summary.json").read_text())["episodes"] == 1
    assert len((output / "trajectories.jsonl").read_text().splitlines()) == 1
    with pytest.raises(FileExistsError):
        write_run(output, {"model": "m"}, rows)


@pytest.mark.sandbox
@pytest.mark.skipif(os.environ.get("TOOLSMITH_RUN_SANDBOX_TESTS") != "1", reason="needs Docker")
def test_scripted_reference_policy_succeeds_through_the_remote_service(dataset, tmp_path):
    service = SandboxService(
        load_service_records(dataset),
        token=TOKEN,
        image=os.environ.get("TOOLSMITH_SANDBOX_IMAGE", DEFAULT_IMAGE),
        limits=SandboxLimits(),
        workers=2,
    )
    httpd = serve(service, port=0)
    try:
        client = RemoteClient(f"http://127.0.0.1:{httpd.server_address[1]}", TOKEN)
        record = load_records(dataset, "dev")[1]
        fixed = apply_patch(record.task, record.reference_patch).source_files[0].content
        texts = [
            render_tool_call("read_file", {"path": SOURCE_PATH}),
            render_tool_call("run_visible_tests", {}),
            render_tool_call("edit_file", {"path": SOURCE_PATH, "content": fixed}),
            render_tool_call("run_visible_tests", {}),
            render_tool_call("submit_patch", {}),
        ]
        trace = run_episode(
            record.task,
            ScriptedPolicy([reply(t) for t in texts]),
            client,
            GenerationConfig(),
            sandbox=remote,
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert trace["terminal_reason"] == "submitted"
    assert [e["tool"] for e in trace["events"]] == [
        "read_file",
        "run_visible_tests",
        "edit_file",
        "run_visible_tests",
        "submit_patch",
    ]
    assert trace["score"]["status"] == "passed" and trace["reward"] == 1 and trace["success"]
