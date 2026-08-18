from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.jobs import (
    AutonomousInvocationDispatcher,
    AutonomousJobKind,
    ScheduledInvocation,
)
from src.paper import NotificationChannel
from src.scanner import StockUniverse
from src.strategy import StrategyHorizon


AT = datetime(
    2026,
    8,
    7,
    20,
    45,
    tzinfo=timezone.utc,
)


def invocation(
    kind: AutonomousJobKind,
    *,
    horizon: StrategyHorizon | None = None,
) -> ScheduledInvocation:
    return ScheduledInvocation(
        job_kind=kind,
        scheduled_for=AT,
        idempotency_key=(
            f"TEST:{kind.value}"
        ),
        strategy_horizon=horizon,
    )


class FakePaperRepository:
    def __init__(self) -> None:
        self.reconciled = True
        self.notifications = []

    def get_account(self, account_id):
        return SimpleNamespace(
            cash_balance=1800,
            reserved_cash=100,
            base_currency="EUR",
        )

    def reconcile_account(self, account_id):
        return SimpleNamespace(
            reconciled=self.reconciled,
        )

    def list_open_positions(self, account_id):
        return (object(),)

    def list_pending_orders(self, account_id):
        return (object(), object())

    def queue_notification(self, **kwargs):
        self.notifications.append(kwargs)


class FakeScanner:
    def __init__(self) -> None:
        self.calls = []

    def run_scan(self, **kwargs):
        self.calls.append(kwargs)


class FakeExecutionEngine:
    def __init__(self) -> None:
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)


class FakeNotificationService:
    def __init__(self) -> None:
        self.fanout_calls = []
        self.dispatch_calls = []

    def fan_out_internal(self, *args, **kwargs):
        self.fanout_calls.append(
            (args, kwargs)
        )
        return 0

    def dispatch_pending(self, *args, **kwargs):
        self.dispatch_calls.append(
            (args, kwargs)
        )


class FakeJobService:
    def __init__(self) -> None:
        self.calls = []
        self.duplicate = False

    def run_weekly_report(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            duplicate=self.duplicate,
        )


def make_dispatcher():
    paper = FakePaperRepository()
    scanner = FakeScanner()
    execution = FakeExecutionEngine()
    notifications = (
        FakeNotificationService()
    )
    jobs = FakeJobService()
    loads = []

    def load_universe():
        loads.append(True)
        return StockUniverse(
            name="autonomous",
            symbols=("AAPL",),
        )

    dispatcher = (
        AutonomousInvocationDispatcher(
            account_id="ACC-TEST",
            paper_repository=paper,
            scanner=scanner,
            execution_engine=execution,
            notification_service=(
                notifications
            ),
            job_service=jobs,
            universe_loader=load_universe,
            notification_channels=(
                NotificationChannel.EMAIL,
            ),
        )
    )

    return SimpleNamespace(
        dispatcher=dispatcher,
        paper=paper,
        scanner=scanner,
        execution=execution,
        notifications=notifications,
        jobs=jobs,
        loads=loads,
    )


def test_refresh_and_scans_use_horizon_scope() -> None:
    env = make_dispatcher()

    assert env.dispatcher(
        invocation(
            AutonomousJobKind
            .UNIVERSE_REFRESH
        )
    )
    assert env.dispatcher(
        invocation(
            AutonomousJobKind.SWING_SCAN,
            horizon=StrategyHorizon.SWING,
        )
    )
    assert env.dispatcher(
        invocation(
            AutonomousJobKind
            .MEDIUM_TERM_SCAN,
            horizon=(
                StrategyHorizon.MEDIUM_TERM
            ),
        )
    )

    assert len(env.loads) == 1
    assert [
        call["strategy_horizon"]
        for call in env.scanner.calls
    ] == [
        StrategyHorizon.SWING,
        StrategyHorizon.MEDIUM_TERM,
    ]
    assert [
        call["strategy_version"]
        for call in env.scanner.calls
    ] == [
        "p4.3-swing-v1",
        "p4.3-medium-term-v1",
    ]


def test_position_monitor_uses_invocation_key() -> None:
    env = make_dispatcher()
    item = invocation(
        AutonomousJobKind.POSITION_MONITOR
    )

    assert env.dispatcher(item)
    assert (
        env.execution.calls[0]["run_key"]
        == item.idempotency_key
    )
    assert (
        env.execution.calls[0]["run_at"]
        == AT
    )


def test_notification_dispatch_uses_channels() -> None:
    env = make_dispatcher()

    assert env.dispatcher(
        invocation(
            AutonomousJobKind
            .NOTIFICATION_DISPATCH
        )
    )
    assert len(
        env.notifications.fanout_calls
    ) == 1
    assert len(
        env.notifications.dispatch_calls
    ) == 1


def test_post_close_report_requires_reconciliation() -> None:
    env = make_dispatcher()
    item = invocation(
        AutonomousJobKind.POST_CLOSE_REPORT
    )

    env.paper.reconciled = False

    with pytest.raises(
        RuntimeError,
        match="reconciliation failed",
    ):
        env.dispatcher(item)

    assert env.paper.notifications == []

    env.paper.reconciled = True

    assert env.dispatcher(item)
    assert len(env.paper.notifications) == 1
    assert (
        env.paper.notifications[0][
            "reference_id"
        ]
        == item.idempotency_key
    )
    assert len(
        env.notifications.dispatch_calls
    ) == 1


def test_weekly_report_propagates_duplicate() -> None:
    env = make_dispatcher()
    item = invocation(
        AutonomousJobKind.WEEKLY_REPORT
    )

    assert env.dispatcher(item) is True

    env.jobs.duplicate = True

    assert env.dispatcher(item) is False
    assert len(env.jobs.calls) == 2


def test_scan_without_horizon_fails_closed() -> None:
    env = make_dispatcher()

    with pytest.raises(
        ValueError,
        match="approved strategy horizon",
    ):
        env.dispatcher(
            invocation(
                AutonomousJobKind.SWING_SCAN
            )
        )
