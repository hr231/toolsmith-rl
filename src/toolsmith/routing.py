"""Transparent hybrid decisions based only on public inputs and visible outcomes."""

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from toolsmith.tasks import BugTask


class AttemptOutcome(StrEnum):
    VISIBLE_PASSED = "visible_passed"
    VISIBLE_FAILED = "visible_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    INVALID_PATCH = "invalid_patch"


@dataclass(frozen=True)
class RoutingPolicy:
    small_max_files: int = 3
    small_max_input_bytes: int = 12_000

    def __post_init__(self) -> None:
        for field in ("small_max_files", "small_max_input_bytes"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")


@dataclass(frozen=True)
class RouteDecision:
    route: Literal["small", "large", "accept_candidate", "stop"]
    reason: str
    blocked_by: str | None = None


def load_policy(path: Path) -> RoutingPolicy:
    """Read explicit thresholds; unknown or incomplete policy settings are errors."""
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if set(data) != {"router"} or not isinstance(data["router"], dict):
        raise ValueError("Policy must contain exactly one [router] table")
    fields = {"small_max_files", "small_max_input_bytes"}
    if set(data["router"]) != fields:
        raise ValueError("[router] requires exactly small_max_files and small_max_input_bytes")
    return RoutingPolicy(**data["router"])


def route_task(
    task: BugTask,
    policy: RoutingPolicy,
    *,
    after_small: AttemptOutcome | None = None,
    after_large: AttemptOutcome | None = None,
    allow_large: bool = False,
) -> RouteDecision:
    """Choose the next action from public task size and supplied attempt outcomes.

    - Accepts a candidate whose visible tests passed.
    - Stops on infrastructure errors, invalid patches, or after a large attempt.
    - Escalates to the large route only when allow_large is True.
    """
    if after_small is not None and after_large is not None:
        raise ValueError("Supply only one of after_small and after_large")
    outcome = after_small if after_small is not None else after_large
    if outcome is not None and not isinstance(outcome, AttemptOutcome):
        raise ValueError("Attempt outcome must be an AttemptOutcome value")
    if type(allow_large) is not bool:
        raise ValueError("allow_large must be an explicit boolean")

    if outcome == AttemptOutcome.VISIBLE_PASSED:
        return RouteDecision("accept_candidate", "visible_tests_passed")
    if outcome in (AttemptOutcome.INFRASTRUCTURE_ERROR, AttemptOutcome.INVALID_PATCH):
        return RouteDecision("stop", outcome.value)
    if after_large is not None:
        return RouteDecision("stop", f"large_{after_large.value}")
    if after_small is not None:
        reason = f"small_{after_small.value}"
    elif task.source_file_count > policy.small_max_files:
        reason = "source_file_limit"
    elif task.input_bytes > policy.small_max_input_bytes:
        reason = "input_byte_limit"
    else:
        return RouteDecision("small", "within_small_limits")

    if not allow_large:
        return RouteDecision("stop", reason, blocked_by="large_disabled")
    return RouteDecision("large", reason)
