import pytest

from toolsmith.routing import AttemptOutcome, RoutingPolicy, load_policy, route_task
from toolsmith.tasks import BugTask


def task_with_files(count=1):
    return BugTask.from_dict(
        {
            "schema_version": 1,
            "task_id": "repair-task",
            "issue": "Fix the off-by-one error.",
            "source_files": {f"src/module_{i}.py": "value = 1\n" for i in range(count)},
            "visible_tests": {"tests/test_value.py": "assert value == 2\n"},
        }
    )


def test_small_eligibility_includes_both_exact_limits():
    task = task_with_files(2)
    policy = RoutingPolicy(small_max_files=2, small_max_input_bytes=task.input_bytes)
    assert route_task(task, policy).route == "small"


def test_one_excess_source_file_routes_directly_large():
    task = task_with_files(2)
    policy = RoutingPolicy(small_max_files=1, small_max_input_bytes=task.input_bytes)
    decision = route_task(task, policy, allow_large=True)
    assert decision.route == "large"
    assert decision.reason == "source_file_limit"


def test_one_excess_input_byte_routes_directly_large():
    task = task_with_files()
    policy = RoutingPolicy(small_max_files=2, small_max_input_bytes=task.input_bytes - 1)
    decision = route_task(task, policy, allow_large=True)
    assert decision.route == "large"
    assert decision.reason == "input_byte_limit"


def test_required_large_route_is_blocked_not_downgraded():
    task = task_with_files(2)
    decision = route_task(task, RoutingPolicy(small_max_files=1))
    assert decision.route == "stop"
    assert decision.reason == "source_file_limit"
    assert decision.blocked_by == "large_disabled"


@pytest.mark.parametrize(
    "outcome", [AttemptOutcome.VISIBLE_FAILED, AttemptOutcome.BUDGET_EXHAUSTED]
)
def test_small_failure_escalates_only_when_allowed(outcome):
    task = task_with_files()
    assert route_task(task, RoutingPolicy(), after_small=outcome, allow_large=True).route == "large"
    blocked = route_task(task, RoutingPolicy(), after_small=outcome)
    assert blocked.route == "stop"
    assert blocked.blocked_by == "large_disabled"


@pytest.mark.parametrize("stage", ["after_small", "after_large"])
def test_visible_pass_accepts_candidate_even_when_task_exceeds_initial_limits(stage):
    task = task_with_files(2)
    decision = route_task(
        task,
        RoutingPolicy(small_max_files=1, small_max_input_bytes=1),
        **{stage: AttemptOutcome.VISIBLE_PASSED},
    )
    assert decision.route == "accept_candidate"
    assert decision.blocked_by is None


@pytest.mark.parametrize("stage", ["after_small", "after_large"])
@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (AttemptOutcome.INFRASTRUCTURE_ERROR, "infrastructure_error"),
        (AttemptOutcome.INVALID_PATCH, "invalid_patch"),
    ],
)
def test_infrastructure_and_policy_failures_never_escalate(stage, outcome, reason):
    decision = route_task(task_with_files(), RoutingPolicy(), allow_large=True, **{stage: outcome})
    assert decision.route == "stop"
    assert decision.reason == reason


@pytest.mark.parametrize(
    "outcome", [AttemptOutcome.VISIBLE_FAILED, AttemptOutcome.BUDGET_EXHAUSTED]
)
def test_failed_large_attempt_is_terminal(outcome):
    decision = route_task(task_with_files(), RoutingPolicy(), after_large=outcome, allow_large=True)
    assert decision.route == "stop"
    assert decision.reason == f"large_{outcome.value}"


def test_conflicting_attempt_states_are_rejected():
    with pytest.raises(ValueError):
        route_task(
            task_with_files(),
            RoutingPolicy(),
            after_small=AttemptOutcome.VISIBLE_FAILED,
            after_large=AttemptOutcome.VISIBLE_PASSED,
        )


def test_unknown_outcome_does_not_trigger_a_paid_route():
    with pytest.raises(ValueError):
        route_task(task_with_files(), RoutingPolicy(), after_small="unknown", allow_large=True)


@pytest.mark.parametrize("field", ["small_max_files", "small_max_input_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_invalid_policy_limits_are_rejected(field, value):
    with pytest.raises(ValueError):
        RoutingPolicy(**{field: value})


def test_loaded_policy_changes_the_actual_route(tmp_path):
    path = tmp_path / "policy.toml"
    path.write_text("[router]\nsmall_max_files = 1\nsmall_max_input_bytes = 1000\n")
    decision = route_task(task_with_files(2), load_policy(path), allow_large=True)
    assert decision.route == "large"
    assert decision.reason == "source_file_limit"


@pytest.mark.parametrize(
    "config",
    [
        "[router]\nsmall_max_files = 1\nsmall_max_input_bytes = 1000\nallow_large = true\n",
        "[router]\nsmall_max_files = 1\n",
        "[router]\nsmall_max_files = true\nsmall_max_input_bytes = 1000\n",
        "[wrong_table]\nsmall_max_files = 1\nsmall_max_input_bytes = 1000\n",
    ],
)
def test_malformed_policy_cannot_silently_use_defaults(tmp_path, config):
    path = tmp_path / "policy.toml"
    path.write_text(config)
    with pytest.raises(ValueError):
        load_policy(path)
