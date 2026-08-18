"""Replay-safe autonomous orchestration cycles."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .calendar import ExchangeCalendar
from .schedule import (
    AutonomousSchedulePolicy,
    ScheduledInvocation,
    plan_exchange_session,
)


class InvocationStatus(str, Enum):
    EXECUTED = "EXECUTED"
    DUPLICATE = "DUPLICATE"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class InvocationResult:
    invocation: ScheduledInvocation
    status: InvocationStatus
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class OrchestrationCycleReport:
    window_started_at: datetime
    window_ended_at: datetime
    due_invocations: tuple[
        ScheduledInvocation,
        ...,
    ]
    results: tuple[InvocationResult, ...]

    @property
    def executed_count(self) -> int:
        return sum(
            item.status
            is InvocationStatus.EXECUTED
            for item in self.results
        )

    @property
    def duplicate_count(self) -> int:
        return sum(
            item.status
            is InvocationStatus.DUPLICATE
            for item in self.results
        )

    @property
    def failed_count(self) -> int:
        return sum(
            item.status
            is InvocationStatus.FAILED
            for item in self.results
        )


InvocationExecutor = Callable[
    [ScheduledInvocation],
    bool | None,
]


def _aware(
    name: str,
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{name} must be timezone-aware."
        )

    return value


def due_invocations(
    *,
    window_started_at: datetime,
    window_ended_at: datetime,
    calendar: ExchangeCalendar | None = None,
    policy: AutonomousSchedulePolicy | None = None,
) -> tuple[ScheduledInvocation, ...]:
    """Return invocations in the half-open recovery window."""

    started = _aware(
        "window_started_at",
        window_started_at,
    )
    ended = _aware(
        "window_ended_at",
        window_ended_at,
    )

    if ended < started:
        raise ValueError(
            "window_ended_at cannot be before "
            "window_started_at."
        )

    exchange_calendar = (
        calendar or ExchangeCalendar()
    )
    local_started = started.astimezone(
        exchange_calendar.timezone
    )
    local_ended = ended.astimezone(
        exchange_calendar.timezone
    )

    current_date = local_started.date()
    final_date = local_ended.date()
    planned: list[ScheduledInvocation] = []

    while current_date <= final_date:
        planned.extend(
            plan_exchange_session(
                current_date,
                calendar=exchange_calendar,
                policy=policy,
            )
        )
        current_date += timedelta(days=1)

    return tuple(
        item
        for item in sorted(
            planned,
            key=lambda candidate: (
                candidate.scheduled_for,
                candidate.job_kind.value,
            ),
        )
        if (
            started
            < item.scheduled_for
            <= ended
        )
    )


def run_orchestration_cycle(
    *,
    window_started_at: datetime,
    window_ended_at: datetime,
    executor: InvocationExecutor,
    calendar: ExchangeCalendar | None = None,
    policy: AutonomousSchedulePolicy | None = None,
    completed_keys: (
        Collection[str] | None
    ) = None,
) -> OrchestrationCycleReport:
    """Execute one deterministic, failure-isolated cycle.

    A restart can safely replay an earlier window. Keys already
    present in completed_keys are reported as duplicates, while
    downstream executors can also return False for their own
    idempotency decision.
    """

    if not callable(executor):
        raise ValueError(
            "executor must be callable."
        )

    due = due_invocations(
        window_started_at=window_started_at,
        window_ended_at=window_ended_at,
        calendar=calendar,
        policy=policy,
    )
    known = set(completed_keys or ())
    results: list[InvocationResult] = []

    for invocation in due:
        if invocation.idempotency_key in known:
            results.append(
                InvocationResult(
                    invocation=invocation,
                    status=(
                        InvocationStatus.DUPLICATE
                    ),
                )
            )
            continue

        try:
            executed = executor(invocation)

            if executed is False:
                status = (
                    InvocationStatus.DUPLICATE
                )
            else:
                status = (
                    InvocationStatus.EXECUTED
                )
                known.add(
                    invocation.idempotency_key
                )

            results.append(
                InvocationResult(
                    invocation=invocation,
                    status=status,
                )
            )
        except Exception as exc:
            results.append(
                InvocationResult(
                    invocation=invocation,
                    status=(
                        InvocationStatus.FAILED
                    ),
                    error_message=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )

    return OrchestrationCycleReport(
        window_started_at=_aware(
            "window_started_at",
            window_started_at,
        ),
        window_ended_at=_aware(
            "window_ended_at",
            window_ended_at,
        ),
        due_invocations=due,
        results=tuple(results),
    )
