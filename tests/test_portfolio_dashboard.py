from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

from src.automation import (
    AutomationRepository,
    ExecutionRunStatus,
)
from src.jobs import (
    JobRepository,
    JobStatus,
    JobType,
)
from src.paper import (
    NotificationChannel,
    PaperExitReason,
    PaperRepository,
    PaperTradingService,
)
from src.portfolio_dashboard import (
    PortfolioDashboardRepository,
    PortfolioDashboardService,
)
from src.strategy import StrategyHorizon
from src.scanner import (
    ScannerRepository,
    StockUniverse,
)


T0 = datetime(
    2026,
    8,
    3,
    20,
    30,
    tzinfo=timezone.utc,
)


def make_environment(tmp_path):
    database_path = (
        tmp_path / "dashboard.db"
    )

    paper_repository = PaperRepository(
        database_path
    )

    paper_service = PaperTradingService(
        paper_repository,
        app_version="test-app",
        threshold_version=(
            "threshold-test"
        ),
    )

    account = paper_service.create_account(
        name="Dashboard Test",
        created_at=T0,
    )

    scanner_repository = ScannerRepository(
        database_path
    )

    automation_repository = (
        AutomationRepository(
            database_path
        )
    )

    job_repository = JobRepository(
        database_path
    )

    dashboard_repository = (
        PortfolioDashboardRepository(
            database_path
        )
    )

    dashboard_service = (
        PortfolioDashboardService(
            dashboard_repository
        )
    )

    return (
        paper_repository,
        paper_service,
        scanner_repository,
        automation_repository,
        job_repository,
        dashboard_repository,
        dashboard_service,
        account,
    )


def persist_signal(
    paper_service,
    account_id,
    *,
    signal_id,
    symbol,
    generated_at,
):
    return paper_service.persist_signal(
        account_id=account_id,
        signal_id=signal_id,
        symbol=symbol,
        generated_at=generated_at,
        expires_at=(
            generated_at
            + timedelta(days=7)
        ),
        strategy="trend_pullback",
        strategy_horizon=(
            StrategyHorizon.SWING
        ),
        strategy_version=(
            "p4.3-swing-v1"
        ),
        recommendation="BUY",
        market_regime="BULLISH",
        score=82,
        confidence=0.88,
        reward_to_risk=2.5,
        entry_low=99,
        entry_high=101,
        stop_price=95,
        targets=(110, 120),
        evidence=(
            "Trend is above the "
            "validated moving average.",
        ),
        conflicts=(),
    )


def create_closed_trade(
    paper_service,
    account_id,
):
    signal = persist_signal(
        paper_service,
        account_id,
        signal_id="SIGNAL-CLOSED",
        symbol="AAPL",
        generated_at=T0,
    )

    order, _ = (
        paper_service
        .create_automatic_buy(
            account_id=account_id,
            signal_id=signal.signal_id,
            quantity=10,
            idempotency_key=(
                "DASH-CLOSED"
            ),
            estimated_fees=1,
            created_at=T0,
        )
    )

    _, position = (
        paper_service
        .record_automatic_buy_fill(
            order_id=order.order_id,
            fill_price=100,
            fees=1,
            slippage=0.5,
            filled_at=(
                T0 + timedelta(days=1)
            ),
        )
    )

    trade = (
        paper_service
        .close_automatic_position(
            position_id=(
                position.position_id
            ),
            exit_price=110,
            exit_reason=(
                PaperExitReason.TARGET
            ),
            exit_fees=1,
            exit_slippage=0.5,
            closed_at=(
                T0 + timedelta(days=2)
            ),
        )
    )

    return signal, trade


def create_pending_order(
    paper_service,
    account_id,
):
    signal = persist_signal(
        paper_service,
        account_id,
        signal_id="SIGNAL-PENDING",
        symbol="MSFT",
        generated_at=(
            T0 + timedelta(minutes=1)
        ),
    )

    order, _ = (
        paper_service
        .create_automatic_buy(
            account_id=account_id,
            signal_id=signal.signal_id,
            quantity=5,
            idempotency_key=(
                "DASH-PENDING"
            ),
            estimated_fees=1,
            created_at=(
                T0 + timedelta(minutes=1)
            ),
        )
    )

    return signal, order


def create_execution_history(
    *,
    automation_repository,
    account_id,
):
    first, _ = (
        automation_repository
        .start_run(
            account_id=account_id,
            run_key="DASH-RUN-1",
            scan_id=None,
            configuration={},
            app_version="test",
            started_at=T0,
        )
    )

    first = (
        automation_repository
        .complete_run(
            first.run_id,
            status=(
                ExecutionRunStatus
                .COMPLETED
            ),
            completed_at=T0,
            created_orders=0,
            filled_orders=0,
            expired_orders=0,
            cancelled_orders=0,
            closed_positions=0,
            rejected_entries=0,
            error_count=0,
            entry_block_reasons=(),
        )
    )

    automation_repository.save_equity_snapshot(
        run_id=first.run_id,
        account_id=account_id,
        captured_at=T0,
        cash_balance=9000,
        reserved_cash=0,
        market_value=1000,
    )

    second_at = (
        T0 + timedelta(days=1)
    )

    second, _ = (
        automation_repository
        .start_run(
            account_id=account_id,
            run_key="DASH-RUN-2",
            scan_id=None,
            configuration={},
            app_version="test",
            started_at=second_at,
        )
    )

    second = (
        automation_repository
        .complete_run(
            second.run_id,
            status=(
                ExecutionRunStatus
                .COMPLETED
            ),
            completed_at=second_at,
            created_orders=0,
            filled_orders=0,
            expired_orders=0,
            cancelled_orders=0,
            closed_positions=0,
            rejected_entries=0,
            error_count=0,
            entry_block_reasons=(),
        )
    )

    automation_repository.save_equity_snapshot(
        run_id=second.run_id,
        account_id=account_id,
        captured_at=second_at,
        cash_balance=8500,
        reserved_cash=0,
        market_value=1000,
    )

    return first, second


def test_empty_snapshot_is_traceable(
    tmp_path,
) -> None:
    (
        _,
        _,
        _,
        _,
        _,
        _,
        service,
        account,
    ) = make_environment(tmp_path)

    snapshot = service.build_snapshot(
        account.account_id,
        generated_at=T0,
    )

    assert snapshot.account == account
    assert snapshot.open_positions == ()
    assert snapshot.pending_orders == ()
    assert snapshot.closed_trades == ()

    assert (
        snapshot.metadata["read_only"]
        is True
    )

    account_source = (
        snapshot.provenance_for(
            "account"
        )
    )

    assert (
        account.account_id
        in account_source.record_ids
    )


def test_trade_performance_and_evidence(
    tmp_path,
) -> None:
    (
        _,
        paper_service,
        _,
        _,
        _,
        _,
        service,
        account,
    ) = make_environment(tmp_path)

    signal, trade = create_closed_trade(
        paper_service,
        account.account_id,
    )

    snapshot = service.build_snapshot(
        account.account_id,
        generated_at=(
            T0 + timedelta(days=3)
        ),
    )

    assert (
        snapshot.performance.trade_count
        == 1
    )

    assert (
        snapshot.performance
        .winning_trades
        == 1
    )

    assert (
        snapshot.performance.net_pnl
        == trade.net_pnl
    )

    trace = next(
        item
        for item
        in snapshot.decision_traces
        if item.reference_id
        == trade.trade_id
    )

    assert (
        trace.signal_id
        == signal.signal_id
    )

    assert (
        trace.threshold_version
        == "threshold-test"
    )

    assert (
        trace.exit_reason
        == PaperExitReason.TARGET.value
    )

    assert trace.evidence


def test_pending_order_is_in_snapshot(
    tmp_path,
) -> None:
    (
        _,
        paper_service,
        _,
        _,
        _,
        _,
        service,
        account,
    ) = make_environment(tmp_path)

    signal, order = create_pending_order(
        paper_service,
        account.account_id,
    )

    snapshot = service.build_snapshot(
        account.account_id,
        generated_at=T0,
    )

    assert len(
        snapshot.pending_orders
    ) == 1

    assert (
        snapshot.pending_orders[0]
        .order_id
        == order.order_id
    )

    trace = next(
        item
        for item
        in snapshot.decision_traces
        if item.reference_type
        == "PENDING_ORDER"
    )

    assert (
        trace.signal_id
        == signal.signal_id
    )


def test_equity_curve_and_drawdown(
    tmp_path,
) -> None:
    (
        _,
        _,
        _,
        automation_repository,
        _,
        _,
        service,
        account,
    ) = make_environment(tmp_path)

    create_execution_history(
        automation_repository=(
            automation_repository
        ),
        account_id=account.account_id,
    )

    snapshot = service.build_snapshot(
        account.account_id,
        generated_at=(
            T0 + timedelta(days=2)
        ),
    )

    equity = snapshot.equity_performance

    assert equity.point_count == 2

    assert (
        equity.latest_equity
        == Decimal("9500.00000000")
    )

    assert (
        equity.peak_equity
        == Decimal("10000.00000000")
    )

    assert (
        equity.maximum_drawdown
        == Decimal("500.00000000")
    )

    assert (
        equity.maximum_drawdown_pct
        == 5.0
    )


def test_strategy_and_threshold_breakdowns(
    tmp_path,
) -> None:
    (
        _,
        paper_service,
        _,
        _,
        _,
        _,
        service,
        account,
    ) = make_environment(tmp_path)

    create_closed_trade(
        paper_service,
        account.account_id,
    )

    snapshot = service.build_snapshot(
        account.account_id,
        generated_at=T0,
    )

    strategy = next(
        row
        for row in snapshot.breakdowns
        if (
            row.dimension == "strategy"
            and row.key
            == "trend_pullback"
        )
    )

    horizon = next(
        row
        for row in snapshot.breakdowns
        if (
            row.dimension
            == "strategy_horizon"
            and row.key == "SWING"
        )
    )

    strategy_version = next(
        row
        for row in snapshot.breakdowns
        if (
            row.dimension
            == "strategy_version"
            and row.key
            == "p4.3-swing-v1"
        )
    )

    strategy_cohort = next(
        row
        for row in snapshot.breakdowns
        if (
            row.dimension
            == "strategy_cohort"
            and row.key
            == "SWING|p4.3-swing-v1"
        )
    )

    threshold = next(
        row
        for row in snapshot.breakdowns
        if (
            row.dimension
            == "threshold_version"
            and row.key
            == "threshold-test"
        )
    )

    assert strategy.trade_count == 1
    assert horizon.trade_count == 1
    assert strategy_version.trade_count == 1
    assert strategy_cohort.trade_count == 1
    assert strategy_cohort.total_costs > Decimal("0")
    assert threshold.trade_count == 1

    assert (
        threshold.provenance
        .record_count
        == 1
    )


def test_repository_lists_persisted_runs_and_scans(
    tmp_path,
) -> None:
    (
        _,
        _,
        scanner_repository,
        automation_repository,
        _,
        dashboard_repository,
        _,
        account,
    ) = make_environment(tmp_path)

    scan, _ = (
        scanner_repository.start_scan(
            account_id=(
                account.account_id
            ),
            universe=StockUniverse(
                name="dashboard",
                symbols=("AAPL",),
            ),
            configuration={},
            app_version="test",
            started_at=T0,
            scan_key="DASH-SCAN",
        )
    )

    scanner_repository.complete_scan(
        scan.scan_id,
        completed_at=T0,
    )

    first, second = (
        create_execution_history(
            automation_repository=(
                automation_repository
            ),
            account_id=(
                account.account_id
            ),
        )
    )

    scans = (
        dashboard_repository
        .list_scan_reports(
            account.account_id
        )
    )

    runs = (
        dashboard_repository
        .list_execution_runs(
            account.account_id
        )
    )

    equity = (
        dashboard_repository
        .list_equity_snapshots(
            account.account_id
        )
    )

    assert [
        item.scan.scan_id
        for item in scans
    ] == [scan.scan_id]

    assert [
        item.run_id
        for item in runs
    ] == [
        first.run_id,
        second.run_id,
    ]

    assert len(equity) == 2


def test_reliability_uses_persisted_records(
    tmp_path,
) -> None:
    (
        paper_repository,
        _,
        scanner_repository,
        automation_repository,
        job_repository,
        _,
        service,
        account,
    ) = make_environment(tmp_path)

    scan, _ = (
        scanner_repository.start_scan(
            account_id=(
                account.account_id
            ),
            universe=StockUniverse(
                name="dashboard",
                symbols=("AAPL",),
            ),
            configuration={},
            app_version="test",
            started_at=T0,
            scan_key="RELIABILITY-SCAN",
        )
    )

    scanner_repository.complete_scan(
        scan.scan_id,
        completed_at=T0,
    )

    create_execution_history(
        automation_repository=(
            automation_repository
        ),
        account_id=account.account_id,
    )

    job, _ = job_repository.start_job(
        account_id=account.account_id,
        job_key="DASH-JOB",
        job_type=JobType.MARKET_CYCLE,
        scheduled_for=T0,
        exchange_code="XNYS",
    )

    job_repository.complete_job(
        job.job_run_id,
        status=JobStatus.COMPLETED,
        completed_at=T0,
    )

    notification = (
        paper_repository
        .queue_notification(
            account_id=(
                account.account_id
            ),
            event_type=(
                "DASHBOARD_TEST"
            ),
            reference_type="JOB_RUN",
            reference_id=job.job_run_id,
            channel=(
                NotificationChannel.INTERNAL
            ),
            payload={
                "text": "Persisted.",
            },
            created_at=T0,
        )
    )

    paper_repository.mark_notification_sent(
        notification.notification_id,
        sent_at=T0,
    )

    paper_repository.record_system_event(
        account_id=account.account_id,
        event_type="DASHBOARD_TEST",
        severity="INFO",
        reference_type="JOB_RUN",
        reference_id=job.job_run_id,
        message="Persisted event.",
        created_at=T0,
    )

    snapshot = service.build_snapshot(
        account.account_id,
        generated_at=T0,
    )

    assert (
        snapshot.reliability
        .scans.successful
        == 1
    )

    assert (
        snapshot.reliability
        .execution_runs.successful
        == 2
    )

    assert (
        snapshot.reliability
        .scheduled_jobs.successful
        == 1
    )

    assert (
        snapshot.reliability
        .notifications.successful
        == 1
    )

    assert (
        snapshot.reliability
        .system_events.failed
        == 0
    )


def test_snapshot_is_deterministic(
    tmp_path,
) -> None:
    (
        _,
        paper_service,
        _,
        _,
        _,
        _,
        service,
        account,
    ) = make_environment(tmp_path)

    create_closed_trade(
        paper_service,
        account.account_id,
    )

    first = service.build_snapshot(
        account.account_id,
        generated_at=T0,
    )

    second = service.build_snapshot(
        account.account_id,
        generated_at=T0,
    )

    assert first == second
