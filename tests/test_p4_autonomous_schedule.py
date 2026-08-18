from __future__ import annotations

from datetime import date, timezone

import pytest

from src.jobs import (
    AUTONOMOUS_SCHEDULE_VERSION,
    AutonomousJobKind,
    AutonomousSchedulePolicy,
    plan_exchange_session,
)
from src.strategy import StrategyHorizon


def test_closed_exchange_has_no_market_plan() -> None:
    assert (
        plan_exchange_session(
            date(2026, 8, 1)
        )
        == ()
    )


def test_session_plan_covers_autonomous_cycle() -> None:
    plan = plan_exchange_session(
        date(2026, 8, 3)
    )
    kinds = {
        item.job_kind
        for item in plan
    }

    assert {
        AutonomousJobKind.UNIVERSE_REFRESH,
        AutonomousJobKind.SWING_SCAN,
        AutonomousJobKind.MEDIUM_TERM_SCAN,
        AutonomousJobKind.POSITION_MONITOR,
        AutonomousJobKind.NOTIFICATION_DISPATCH,
        AutonomousJobKind.POST_CLOSE_REPORT,
    }.issubset(kinds)
    assert (
        AutonomousJobKind.WEEKLY_REPORT
        not in kinds
    )

    swing = [
        item
        for item in plan
        if item.job_kind
        is AutonomousJobKind.SWING_SCAN
    ]
    medium = [
        item
        for item in plan
        if item.job_kind
        is AutonomousJobKind.MEDIUM_TERM_SCAN
    ]

    assert len(swing) > 1
    assert all(
        item.strategy_horizon
        is StrategyHorizon.SWING
        for item in swing
    )
    assert len(medium) == 1
    assert (
        medium[0].strategy_horizon
        is StrategyHorizon.MEDIUM_TERM
    )


def test_plan_is_deterministic_and_keys_are_unique() -> None:
    first = plan_exchange_session(
        date(2026, 8, 3)
    )
    second = plan_exchange_session(
        date(2026, 8, 3)
    )

    assert first == second

    keys = [
        item.idempotency_key
        for item in first
    ]

    assert len(keys) == len(set(keys))
    assert all(
        key.startswith(
            AUTONOMOUS_SCHEDULE_VERSION
        )
        for key in keys
    )


def test_weekly_report_uses_final_session() -> None:
    friday = plan_exchange_session(
        date(2026, 8, 7)
    )
    thursday = plan_exchange_session(
        date(2026, 8, 6)
    )

    assert any(
        item.job_kind
        is AutonomousJobKind.WEEKLY_REPORT
        for item in friday
    )
    assert not any(
        item.job_kind
        is AutonomousJobKind.WEEKLY_REPORT
        for item in thursday
    )


def test_exchange_timezone_tracks_dst() -> None:
    summer = plan_exchange_session(
        date(2026, 8, 3)
    )
    winter = plan_exchange_session(
        date(2026, 12, 1)
    )

    summer_refresh = next(
        item
        for item in summer
        if item.job_kind
        is AutonomousJobKind.UNIVERSE_REFRESH
    )
    winter_refresh = next(
        item
        for item in winter
        if item.job_kind
        is AutonomousJobKind.UNIVERSE_REFRESH
    )

    assert (
        summer_refresh.scheduled_for
        .astimezone(timezone.utc)
        .hour
        == 13
    )
    assert (
        winter_refresh.scheduled_for
        .astimezone(timezone.utc)
        .hour
        == 14
    )


def test_schedule_policy_rejects_invalid_ordering() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Post-close reporting must run "
            "after"
        ),
    ):
        AutonomousSchedulePolicy(
            medium_term_delay_minutes=30,
            post_close_report_delay_minutes=10,
        )
