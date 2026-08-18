"""Versioned deterministic autonomous-job schedule policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum

from src.strategy import StrategyHorizon

from .calendar import ExchangeCalendar


AUTONOMOUS_SCHEDULE_VERSION = (
    "p4.5-schedule-v1"
)


class AutonomousJobKind(str, Enum):
    """Jobs in the unattended paper operating cycle."""

    UNIVERSE_REFRESH = "UNIVERSE_REFRESH"
    SWING_SCAN = "SWING_SCAN"
    MEDIUM_TERM_SCAN = "MEDIUM_TERM_SCAN"
    POSITION_MONITOR = "POSITION_MONITOR"
    NOTIFICATION_DISPATCH = (
        "NOTIFICATION_DISPATCH"
    )
    POST_CLOSE_REPORT = (
        "POST_CLOSE_REPORT"
    )
    WEEKLY_REPORT = "WEEKLY_REPORT"


@dataclass(frozen=True, slots=True)
class AutonomousSchedulePolicy:
    """Validated cadence for one exchange."""

    policy_version: str = (
        AUTONOMOUS_SCHEDULE_VERSION
    )
    pre_session_minutes: int = 30
    swing_start_minutes_after_open: int = 30
    swing_end_minutes_before_close: int = 30
    swing_interval_minutes: int = 60
    position_monitor_interval_minutes: int = 15
    notification_dispatch_interval_minutes: int = 15
    medium_term_delay_minutes: int = 10
    post_close_report_delay_minutes: int = 30
    weekly_report_delay_minutes: int = 45

    def __post_init__(self) -> None:
        if (
            self.policy_version
            != AUTONOMOUS_SCHEDULE_VERSION
        ):
            raise ValueError(
                "Unsupported autonomous schedule "
                "policy version."
            )

        positive = (
            "pre_session_minutes",
            "swing_start_minutes_after_open",
            "swing_end_minutes_before_close",
            "swing_interval_minutes",
            "position_monitor_interval_minutes",
            "notification_dispatch_interval_minutes",
            "medium_term_delay_minutes",
            "post_close_report_delay_minutes",
            "weekly_report_delay_minutes",
        )

        for name in positive:
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive "
                    "integer."
                )

        if (
            self.post_close_report_delay_minutes
            <= self.medium_term_delay_minutes
        ):
            raise ValueError(
                "Post-close reporting must run "
                "after the medium-term scan."
            )

        if (
            self.weekly_report_delay_minutes
            <= self.post_close_report_delay_minutes
        ):
            raise ValueError(
                "Weekly reporting must run after "
                "the post-close report."
            )


@dataclass(frozen=True, slots=True)
class ScheduledInvocation:
    """One deterministic paper-only job invocation."""

    job_kind: AutonomousJobKind
    scheduled_for: datetime
    idempotency_key: str
    strategy_horizon: (
        StrategyHorizon | None
    ) = None

    def __post_init__(self) -> None:
        if (
            self.scheduled_for.tzinfo is None
            or self.scheduled_for.utcoffset()
            is None
        ):
            raise ValueError(
                "scheduled_for must be "
                "timezone-aware."
            )

        if not self.idempotency_key.strip():
            raise ValueError(
                "idempotency_key is required."
            )


def _timed_invocations(
    *,
    job_kind: AutonomousJobKind,
    start: datetime,
    end: datetime,
    interval_minutes: int,
    key_prefix: str,
    strategy_horizon: (
        StrategyHorizon | None
    ) = None,
) -> list[ScheduledInvocation]:
    results: list[ScheduledInvocation] = []
    current = start

    while current <= end:
        timestamp_key = current.strftime(
            "%Y%m%dT%H%M%z"
        )
        results.append(
            ScheduledInvocation(
                job_kind=job_kind,
                scheduled_for=current,
                idempotency_key=(
                    f"{key_prefix}:"
                    f"{job_kind.value}:"
                    f"{timestamp_key}"
                ),
                strategy_horizon=(
                    strategy_horizon
                ),
            )
        )
        current += timedelta(
            minutes=interval_minutes
        )

    return results


def plan_exchange_session(
    session_date: date,
    *,
    calendar: ExchangeCalendar | None = None,
    policy: AutonomousSchedulePolicy | None = None,
) -> tuple[ScheduledInvocation, ...]:
    """Build the complete deterministic plan for a session."""

    exchange_calendar = (
        calendar or ExchangeCalendar()
    )
    schedule = (
        policy or AutonomousSchedulePolicy()
    )
    session = exchange_calendar.session(
        session_date
    )

    if session is None:
        return ()

    key_prefix = (
        f"{schedule.policy_version}:"
        f"{session.exchange_code}:"
        f"{session.session_date.isoformat()}"
    )
    invocations: list[
        ScheduledInvocation
    ] = []

    def one(
        job_kind: AutonomousJobKind,
        at: datetime,
        *,
        horizon: (
            StrategyHorizon | None
        ) = None,
    ) -> None:
        invocations.append(
            ScheduledInvocation(
                job_kind=job_kind,
                scheduled_for=at,
                idempotency_key=(
                    f"{key_prefix}:"
                    f"{job_kind.value}:"
                    f"{at.strftime('%Y%m%dT%H%M%z')}"
                ),
                strategy_horizon=horizon,
            )
        )

    pre_session_at = (
        session.opens_at
        - timedelta(
            minutes=(
                schedule.pre_session_minutes
            )
        )
    )
    one(
        AutonomousJobKind.UNIVERSE_REFRESH,
        pre_session_at,
    )

    swing_start = (
        session.opens_at
        + timedelta(
            minutes=(
                schedule
                .swing_start_minutes_after_open
            )
        )
    )
    swing_end = (
        session.closes_at
        - timedelta(
            minutes=(
                schedule
                .swing_end_minutes_before_close
            )
        )
    )
    invocations.extend(
        _timed_invocations(
            job_kind=(
                AutonomousJobKind.SWING_SCAN
            ),
            start=swing_start,
            end=swing_end,
            interval_minutes=(
                schedule.swing_interval_minutes
            ),
            key_prefix=key_prefix,
            strategy_horizon=(
                StrategyHorizon.SWING
            ),
        )
    )

    invocations.extend(
        _timed_invocations(
            job_kind=(
                AutonomousJobKind
                .POSITION_MONITOR
            ),
            start=session.opens_at,
            end=session.closes_at,
            interval_minutes=(
                schedule
                .position_monitor_interval_minutes
            ),
            key_prefix=key_prefix,
        )
    )

    invocations.extend(
        _timed_invocations(
            job_kind=(
                AutonomousJobKind
                .NOTIFICATION_DISPATCH
            ),
            start=pre_session_at,
            end=(
                session.closes_at
                + timedelta(
                    minutes=(
                        schedule
                        .post_close_report_delay_minutes
                    )
                )
            ),
            interval_minutes=(
                schedule
                .notification_dispatch_interval_minutes
            ),
            key_prefix=key_prefix,
        )
    )

    one(
        AutonomousJobKind.MEDIUM_TERM_SCAN,
        session.closes_at
        + timedelta(
            minutes=(
                schedule
                .medium_term_delay_minutes
            )
        ),
        horizon=(
            StrategyHorizon.MEDIUM_TERM
        ),
    )
    one(
        AutonomousJobKind.POST_CLOSE_REPORT,
        session.closes_at
        + timedelta(
            minutes=(
                schedule
                .post_close_report_delay_minutes
            )
        ),
    )

    if exchange_calendar.is_last_session_of_week(
        session.session_date
    ):
        one(
            AutonomousJobKind.WEEKLY_REPORT,
            session.closes_at
            + timedelta(
                minutes=(
                    schedule
                    .weekly_report_delay_minutes
                )
            ),
        )

    return tuple(
        sorted(
            invocations,
            key=lambda item: (
                item.scheduled_for,
                item.job_kind.value,
            ),
        )
    )
