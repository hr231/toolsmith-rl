import json
import os
from dataclasses import FrozenInstanceError

import pytest

from toolsmith.episode import Episode, EpisodeClosed, EpisodeLimits
from toolsmith.sandbox import DEFAULT_IMAGE, SandboxLimits
from toolsmith.tasks import BugTask

SOURCE = "def answer():\n    return 1\n"
FIXED = "def answer():\n    return 2\n"


def make_task(source=SOURCE, issue="Return 2."):
    return BugTask.from_dict(
        {
            "schema_version": 1,
            "task_id": "episode-test",
            "issue": issue,
            "source_files": {"src/example.py": source},
            "visible_tests": {
                "tests/test_example.py": (
                    "from example import answer\n\ndef test_answer():\n    assert answer() == 2\n"
                )
            },
        }
    )


def encoded_size(value):
    return len(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def event_size(event):
    return encoded_size(
        {
            "index": event.index,
            "tool": event.tool,
            "arguments": event.arguments,
            "observation": event.observation,
        }
    )


@pytest.mark.parametrize("field", ["max_tool_calls", "max_context_bytes"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, float("inf"), "8", None])
def test_limits_reject_nonpositive_or_noninteger_values(field, value):
    with pytest.raises(ValueError):
        EpisodeLimits(**{field: value})


def test_limits_defaults_and_immutability():
    limits = EpisodeLimits()
    assert (limits.max_tool_calls, limits.max_context_bytes) == (8, 16384)
    with pytest.raises(FrozenInstanceError):
        limits.max_tool_calls = 9


def test_initial_context_contains_only_public_issue_and_paths():
    with Episode(make_task()) as episode:
        context = episode.initial_context
        assert context["task_id"] == episode.task_id == "episode-test"
        assert context["issue"] == "Return 2."
        assert context["source_files"] == ["src/example.py"]
        assert context["visible_tests"] == ["tests/test_example.py"]
        assert episode.image_id is None
        assert SOURCE not in json.dumps(context)
        assert not ({"checks", "reference_patch", "expected"} & set(context))
        assert episode.context_bytes == encoded_size(context)
        context["source_files"].append("private.json")
        context["issue"] = "changed"
        assert episode.initial_context["source_files"] == ["src/example.py"]
        assert episode.initial_context["issue"] == "Return 2."
        assert episode.events == ()
        assert not episode.terminal
        assert episode.terminal_reason is None
        assert episode.submitted_patch is None


def test_public_tools_do_not_execute_source_on_host(tmp_path):
    marker = tmp_path / "host-execution"
    source = f"open({str(marker)!r}, 'w').write('executed')\n" + SOURCE
    with Episode(make_task(source)) as episode:
        read = episode.dispatch("read_file", {"path": "src/example.py"})
        assert read.observation["content"] == source
        visible = episode.dispatch("read_file", {"path": "tests/test_example.py"})
        assert "test_answer" in visible.observation["content"]
        episode.dispatch("edit_file", {"path": "src/example.py", "content": FIXED})
        submitted = episode.dispatch("submit_patch", {})
        assert episode.terminal_reason == "submitted"
        assert submitted.observation["patch"] == episode.submitted_patch
        assert "-    return 1\n+    return 2\n" in episode.submitted_patch
    assert not marker.exists()


def test_event_snapshots_are_immutable_ordered_and_replayable():
    with Episode(make_task()) as first:
        arguments = {"path": "src/example.py"}
        read = first.dispatch("read_file", arguments)
        snapshot = first.events
        arguments["path"] = "other"
        read.arguments["path"] = "other"
        read.observation["content"] = "other"
        with pytest.raises(FrozenInstanceError):
            read.tool = "submit_patch"
        assert read.arguments == {"path": "src/example.py"}
        assert read.observation["content"] == SOURCE
        first.dispatch("edit_file", {"path": "src/example.py", "content": FIXED})
        first.dispatch("submit_patch", {})
        assert snapshot == (read,)
        assert [event.index for event in first.events] == [1, 2, 3]
        assert first.context_bytes == encoded_size(first.initial_context) + sum(
            event_size(event) for event in first.events
        )
        with Episode(make_task()) as second:
            for event in first.events:
                replay = second.dispatch(event.tool, event.arguments)
                assert replay == event
            assert second.submitted_patch == first.submitted_patch


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("run_check", {}),
        ("private_tests", {}),
        ("shell", {"command": "true"}),
        (["read_file"], {}),
        ("read_file", None),
        ("read_file", []),
        ("read_file", {}),
        ("read_file", {"path": "src/example.py", "extra": 1}),
        ("read_file", {"path": ["src/example.py"]}),
        ("read_file", {"path": "src/../src/example.py"}),
        ("read_file", {"path": "src//example.py"}),
        ("read_file", {"path": "/src/example.py"}),
        ("read_file", {"path": "private/checks.json"}),
        ("read_file", {"path": "src/new.py"}),
        ("edit_file", {"path": "tests/test_example.py", "content": "pass\n"}),
        ("edit_file", {"path": "src/example.py", "content": b"pass\n"}),
        ("edit_file", {"path": "src/example.py", "content": "\ud800"}),
        ("edit_file", {"path": "src/example.py", "content": float("nan")}),
        ("edit_file", {"path": "src/example.py", "content": float("inf")}),
        ("run_visible_tests", {"checks": []}),
        ("submit_patch", {"patch": "fake"}),
    ],
)
def test_invalid_action_is_structured_terminal_and_does_not_edit(tool, arguments):
    with Episode(make_task()) as episode:
        event = episode.dispatch(tool, arguments)
        assert event.observation["error"] == "invalid_action"
        assert episode.terminal_reason == "invalid_action"
        assert episode.terminal
        assert episode.submitted_patch is None
        assert episode._sandbox.read_file("src/example.py") == SOURCE
        before = (episode.events, episode.context_bytes)
        with pytest.raises(EpisodeClosed):
            episode.dispatch("submit_patch", {})
        assert (episode.events, episode.context_bytes) == before


def test_constructor_does_not_accept_private_checks_or_reference_patch():
    for name in ("checks", "private_checks", "reference_patch"):
        with pytest.raises(TypeError):
            Episode(make_task(), **{name: {}})
    with pytest.raises(ValueError):
        Episode(make_task(), limits={"max_tool_calls": 8})


def test_last_allowed_call_can_submit_but_otherwise_exhausts_budget():
    with Episode(make_task(), limits=EpisodeLimits(max_tool_calls=2)) as episode:
        episode.dispatch("edit_file", {"path": "src/example.py", "content": FIXED})
        episode.dispatch("submit_patch", {})
        assert episode.terminal_reason == "submitted"
    with Episode(make_task(), limits=EpisodeLimits(max_tool_calls=1)) as episode:
        event = episode.dispatch("read_file", {"path": "src/example.py"})
        assert event.observation["content"] == SOURCE
        assert episode.terminal_reason == "tool_budget_exhausted"
        with pytest.raises(EpisodeClosed):
            episode.dispatch("submit_patch", {})
        assert episode.events == (event,)


def test_noop_submission_is_invalid_patch_even_at_call_boundary():
    with Episode(make_task(), limits=EpisodeLimits(max_tool_calls=1)) as episode:
        event = episode.dispatch("submit_patch", {})
        assert event.observation["error"] == "invalid_patch"
        assert episode.terminal_reason == "invalid_patch"
        assert episode.submitted_patch is None


def test_instances_have_independent_edits_and_events():
    task = make_task()
    with Episode(task) as first, Episode(task) as second:
        first.dispatch("edit_file", {"path": "src/example.py", "content": FIXED})
        assert second.events == ()
        assert (
            second.dispatch("read_file", {"path": "src/example.py"}).observation["content"]
            == SOURCE
        )
        first.dispatch("submit_patch", {})
        assert not second.terminal
        assert task.source_files[0].content == SOURCE


def test_context_exit_prevents_actions_and_reentry_without_mutating_history():
    with Episode(make_task()) as episode:
        episode.dispatch("read_file", {"path": "src/example.py"})
    assert episode.terminal_reason == "context_closed"
    before = (episode.events, episode.context_bytes)
    with pytest.raises(EpisodeClosed):
        episode.dispatch("submit_patch", {})
    with pytest.raises(EpisodeClosed):
        episode.__enter__()
    assert (episode.events, episode.context_bytes) == before


def test_initial_context_uses_utf8_bytes_and_rejects_oversized_issue():
    task = make_task(issue="Repair the café. " * 100)
    with Episode(task) as baseline:
        size = encoded_size(baseline.initial_context)
        assert size > len(
            json.dumps(baseline.initial_context, ensure_ascii=False, separators=(",", ":"))
        )
    with pytest.raises(ValueError, match="context"):
        Episode(task, limits=EpisodeLimits(max_context_bytes=size - 1))


def test_edit_context_budget_is_checked_before_mutation_including_acknowledgment():
    content = "# café\n" * 100
    with Episode(make_task()) as baseline:
        event = baseline.dispatch("edit_file", {"path": "src/example.py", "content": content})
        exact = baseline.context_bytes
        assert event.observation["status"] == "edited"
    with Episode(make_task(), limits=EpisodeLimits(max_context_bytes=exact - 1)) as episode:
        event = episode.dispatch("edit_file", {"path": "src/example.py", "content": content})
        assert episode.terminal_reason == "context_budget_exhausted"
        assert event.observation["truncated"] is True
        assert episode._sandbox.read_file("src/example.py") == SOURCE
        assert episode.context_bytes <= exact - 1
    with Episode(make_task(), limits=EpisodeLimits(max_context_bytes=exact)) as episode:
        episode.dispatch("edit_file", {"path": "src/example.py", "content": content})
        assert episode._sandbox.read_file("src/example.py") == content
        assert episode.context_bytes == exact


def test_oversized_read_does_not_retain_full_output_or_report_success():
    source = "# oversized source\n" * 1000
    with Episode(make_task(source), limits=EpisodeLimits(max_context_bytes=1024)) as episode:
        event = episode.dispatch("read_file", {"path": "src/example.py"})
        assert episode.terminal_reason == "context_budget_exhausted"
        assert event.observation["error"] == "context_budget_exhausted"
        assert event.observation["truncated"] is True
        assert "content" not in event.observation
        assert source not in repr(episode.events)
        assert episode.context_bytes <= 1024


def test_no_space_for_even_a_truncation_event_closes_without_retaining_it():
    task = make_task()
    with Episode(task) as baseline:
        size = baseline.context_bytes
    with Episode(task, limits=EpisodeLimits(max_context_bytes=size)) as episode:
        with pytest.raises(EpisodeClosed, match="context"):
            episode.dispatch("read_file", {"path": "src/example.py"})
        assert episode.terminal_reason == "context_budget_exhausted"
        assert episode.events == ()
        assert episode.context_bytes == size


def test_missing_docker_is_an_explicit_infrastructure_failure(monkeypatch):
    monkeypatch.setenv("PATH", "")
    with Episode(make_task()) as episode:
        event = episode.dispatch("run_visible_tests", {})
        assert event.observation["error"] == "infrastructure_error"
        assert episode.terminal_reason == "infrastructure_error"
        assert episode.submitted_patch is None


@pytest.fixture
def sandbox_image():
    if os.environ.get("TOOLSMITH_RUN_SANDBOX_TESTS") != "1":
        pytest.skip("Set TOOLSMITH_RUN_SANDBOX_TESTS=1 to run real Docker tests")
    return os.environ.get("TOOLSMITH_SANDBOX_IMAGE", DEFAULT_IMAGE)


@pytest.mark.sandbox
def test_real_five_step_reference_trace_fits_default_budget_and_replays(sandbox_image):
    with Episode(make_task(), image=sandbox_image) as episode:
        episode.dispatch("read_file", {"path": "src/example.py"})
        failed = episode.dispatch("run_visible_tests", {})
        assert failed.observation["status"] == "failed"
        assert not episode.terminal
        episode.dispatch("edit_file", {"path": "src/example.py", "content": FIXED})
        passed = episode.dispatch("run_visible_tests", {})
        assert passed.observation["status"] == "passed"
        assert set(passed.observation) == {"status", "stdout", "stderr", "exit_code"}
        episode.dispatch("submit_patch", {})
        assert episode.terminal_reason == "submitted"
        assert episode.context_bytes <= 16384
        with Episode(make_task(), image=sandbox_image) as replay:
            for event in episode.events:
                actual = replay.dispatch(event.tool, event.arguments)
                if event.tool == "run_visible_tests":
                    assert actual.observation["status"] == event.observation["status"]
                    assert actual.observation["exit_code"] == event.observation["exit_code"]
                else:
                    assert actual == event
            assert replay.submitted_patch == episode.submitted_patch


@pytest.mark.sandbox
def test_real_separate_instance_runs_original_after_other_instance_repairs(sandbox_image):
    with Episode(make_task(), image=sandbox_image) as first:
        first.dispatch("edit_file", {"path": "src/example.py", "content": FIXED})
        assert first.dispatch("run_visible_tests", {}).observation["status"] == "passed"
        with Episode(make_task(), image=sandbox_image) as second:
            assert second.dispatch("run_visible_tests", {}).observation["status"] == "failed"
            assert not second.terminal


@pytest.mark.sandbox
@pytest.mark.parametrize(
    "source,limits,status",
    [
        (
            "def answer():\n    while True:\n        pass\n",
            SandboxLimits(timeout_seconds=2),
            "timeout",
        ),
        (
            "def answer():\n    while True:\n        print('x' * 8192, flush=True)\n",
            SandboxLimits(output_bytes=1024),
            "output_limit",
        ),
    ],
)
def test_real_visible_resource_failures_are_terminal(sandbox_image, source, limits, status):
    with Episode(make_task(source), image=sandbox_image, sandbox_limits=limits) as episode:
        event = episode.dispatch("run_visible_tests", {})
        assert event.observation["status"] == status
        assert episode.terminal_reason == "budget_exhausted"
        assert episode.image_id.startswith("sha256:")


@pytest.mark.sandbox
def test_real_missing_image_is_not_a_failed_test_or_host_fallback(sandbox_image):
    with Episode(make_task(), image="toolsmith-missing-image:no-such-local-tag") as episode:
        event = episode.dispatch("run_visible_tests", {})
        assert event.observation["error"] == "infrastructure_error"
        assert episode.terminal_reason == "infrastructure_error"
