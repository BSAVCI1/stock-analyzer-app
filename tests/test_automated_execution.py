from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
import sqlite3

import numpy as np
import pandas as pd

from src.analysis import Signal
from src.automation import (
    AutomatedExecutionConfig,
    AutomatedPaperExecutionEngine,
    AutomationRepository,
    ExecutionRunStatus,
)
from src.backtest import ExecutionCostModel, FillRule
from src.costs import (
    IBKRPricingPlan,
)
from src.data import MarketSnapshot
from src.paper import (
    PaperExitReason,
    PaperPortfolioConfig,
    PaperRepository,
    PaperTradingService,
)
from src.scanner import (
    ScannerAnalysisOutcome,
    ScannerRepository,
    ScanResult,
    ScanResultStatus,
    StockUniverse,
)


T0 = datetime(
    2026,
    8,
    1,
    20,
    0,
    tzinfo=timezone.utc,
)


def make_history(rows) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [row[1] for row in rows],
            "High": [row[2] for row in rows],
            "Low": [row[3] for row in rows],
            "Close": [row[4] for row in rows],
            "Volume": np.full(
                len(rows),
                1_000_000,
            ),
        },
        index=pd.DatetimeIndex(
            [row[0] for row in rows]
        ),
    )


def make_snapshot(
    symbol: str,
    history: pd.DataFrame,
) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        history=history,
        metadata={
            "symbol": symbol,
            "quoteType": "EQUITY",
            "currency": "USD",
            "exchange": "NMS",
        },
        fetched_at_utc=T0,
    )


def make_hold_outcome(
    snapshot: MarketSnapshot,
) -> ScannerAnalysisOutcome:
    generated_at = (
        snapshot.history.index[-1]
        .to_pydatetime()
    )

    return ScannerAnalysisOutcome(
        symbol=snapshot.symbol,
        generated_at=generated_at,
        strategy="trend_pullback",
        recommendation=Signal.HOLD,
        score=0,
        confidence=0.50,
        market_regime="NEUTRAL",
        regime_confidence=0.50,
        order=None,
        risk_vetoes=(),
        evidence=(
            {
                "code": "TEST_HOLD",
                "message":
                "Deterministic test outcome.",
            },
        ),
    )


def make_environment(
    tmp_path,
    *,
    history_by_symbol,
    costs=None,
):
    database_path = (
        tmp_path / "automation.db"
    )

    paper_repository = PaperRepository(
        database_path
    )

    paper_service = PaperTradingService(
        paper_repository,
        config=PaperPortfolioConfig(
            starting_balance=Decimal(
                "10000"
            ),
            maximum_allocation_fraction=(
                Decimal("0.20")
            ),
            risk_fraction_per_trade=(
                Decimal("0.01")
            ),
            maximum_open_risk_fraction=(
                Decimal("0.04")
            ),
        ),
        app_version="test",
        threshold_version="test",
    )

    account = paper_service.create_account(
        created_at=T0
    )

    scanner_repository = ScannerRepository(
        database_path
    )

    automation_repository = (
        AutomationRepository(
            database_path
        )
    )

    def snapshot_loader(symbol):
        return make_snapshot(
            symbol,
            history_by_symbol[symbol],
        )

    engine = AutomatedPaperExecutionEngine(
        paper_repository=paper_repository,
        paper_service=paper_service,
        scanner_repository=(
            scanner_repository
        ),
        automation_repository=(
            automation_repository
        ),
        config=AutomatedExecutionConfig(
            fill_rule=FillRule.LIMIT,
            costs=(
                costs
                or ExecutionCostModel()
            ),
        ),
        snapshot_loader=snapshot_loader,
        analysis_runner=make_hold_outcome,
        app_version="test",
    )

    return (
        paper_repository,
        paper_service,
        scanner_repository,
        automation_repository,
        engine,
        account,
    )


def create_candidate_scan(
    *,
    paper_service,
    scanner_repository,
    account_id,
    generated_at=T0,
    data_as_of=None,
    reward_to_risk=2.5,
    entry_low=99,
    entry_high=101,
    stop_price=95,
    targets=(110, 120),
    quote_currency=None,
):
    signal = paper_service.persist_signal(
        account_id=account_id,
        signal_id="SIG-AAPL-CANDIDATE",
        symbol="AAPL",
        quote_currency=quote_currency,
        generated_at=generated_at,
        expires_at=(
            generated_at
            + timedelta(days=5)
        ),
        strategy="trend_pullback",
        recommendation="BUY",
        market_regime="BULLISH",
        score=80,
        confidence=0.90,
        reward_to_risk=reward_to_risk,
        entry_low=entry_low,
        entry_high=entry_high,
        stop_price=stop_price,
        targets=targets,
        evidence=(
            "Approved deterministic signal.",
        ),
    )

    scan, _ = scanner_repository.start_scan(
        account_id=account_id,
        universe=StockUniverse(
            name="test",
            symbols=("AAPL",),
        ),
        configuration={},
        app_version="test",
        started_at=generated_at,
        scan_key=(
            "scan-"
            + generated_at.isoformat()
        ),
    )

    scanner_repository.save_result(
        ScanResult(
            result_id="RESULT-AAPL-CANDIDATE",
            scan_id=scan.scan_id,
            account_id=account_id,
            symbol="AAPL",
            status=(
                ScanResultStatus
                .ORDER_CANDIDATE
            ),
            processed_at=generated_at,
            data_as_of=(
                data_as_of
                or generated_at
            ),
            history_rows=260,
            latest_price=100,
            average_volume=1_000_000,
            average_dollar_volume=(
                100_000_000
            ),
            recommendation="BUY",
            strategy="trend_pullback",
            score=80,
            confidence=0.90,
            market_regime="BULLISH",
            reward_to_risk=reward_to_risk,
            release_eligible=True,
            rank_score=90,
            rank_position=1,
            signal_id=signal.signal_id,
            reasons=("Approved.",),
            evidence=(
                {
                    "code": "TEST",
                    "message": "Approved.",
                },
            ),
            metadata={},
        )
    )

    scanner_repository.complete_scan(
        scan.scan_id,
        completed_at=generated_at,
    )

    return scan


def open_position(
    *,
    paper_service,
    account_id,
    opened_at,
):
    signal_id = (
        "SIG-OPEN-"
        + str(int(opened_at.timestamp()))
    )

    signal = paper_service.persist_signal(
        account_id=account_id,
        signal_id=signal_id,
        symbol="AAPL",
        generated_at=(
            opened_at
            - timedelta(days=1)
        ),
        expires_at=(
            opened_at
            + timedelta(days=5)
        ),
        strategy="trend_pullback",
        recommendation="BUY",
        market_regime="BULLISH",
        score=80,
        confidence=0.90,
        reward_to_risk=2.5,
        entry_low=99,
        entry_high=101,
        stop_price=95,
        targets=(110, 120),
        evidence=("Open position.",),
    )

    order, _ = (
        paper_service.create_automatic_buy(
            account_id=account_id,
            signal_id=signal.signal_id,
            quantity=10,
            idempotency_key=(
                "OPEN-" + signal.signal_id
            ),
            estimated_fees=0,
            created_at=(
                opened_at
                - timedelta(hours=1)
            ),
        )
    )

    _, position = (
        paper_service
        .record_automatic_buy_fill(
            order_id=order.order_id,
            fill_price=100,
            fees=0,
            slippage=0,
            filled_at=opened_at,
        )
    )

    return position


def test_schema_version_three(
    tmp_path,
) -> None:
    history = make_history(
        [
            (
                T0,
                100,
                101,
                99,
                100,
            ),
        ]
    )

    (
        repository,
        _,
        _,
        _,
        _,
        _,
    ) = make_environment(
        tmp_path,
        history_by_symbol={
            "AAPL": history,
        },
    )

    import sqlite3

    connection = sqlite3.connect(
        repository.database_path
    )

    try:
        version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }
    finally:
        connection.close()

    assert version == 8

    assert {
        "paper_execution_runs",
        "paper_account_controls",
        "paper_exit_requests",
        "paper_equity_snapshots",
    }.issubset(tables)


def test_candidate_creates_idempotent_order(
    tmp_path,
) -> None:
    history = make_history(
        [
            (
                T0,
                100,
                101,
                99,
                100,
            ),
        ]
    )

    (
        paper_repository,
        paper_service,
        scanner_repository,
        _,
        engine,
        account,
    ) = make_environment(
        tmp_path,
        history_by_symbol={
            "AAPL": history,
        },
    )

    scan = create_candidate_scan(
        paper_service=paper_service,
        scanner_repository=(
            scanner_repository
        ),
        account_id=account.account_id,
    )

    first = engine.run(
        account_id=account.account_id,
        scan_id=scan.scan_id,
        run_key="candidate-run",
        run_at=T0,
    )

    second = engine.run(
        account_id=account.account_id,
        scan_id=scan.scan_id,
        run_key="candidate-run",
        run_at=T0,
    )

    assert first.run.created_orders == 1
    assert second.run.run_id == first.run.run_id

    orders = (
        paper_repository
        .list_pending_orders(
            account.account_id
        )
    )

    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"


def test_same_session_never_fills(
    tmp_path,
) -> None:
    history = make_history(
        [
            (
                T0,
                100,
                101,
                99,
                100,
            ),
        ]
    )

    (
        paper_repository,
        paper_service,
        scanner_repository,
        _,
        engine,
        account,
    ) = make_environment(
        tmp_path,
        history_by_symbol={
            "AAPL": history,
        },
    )

    scan = create_candidate_scan(
        paper_service=paper_service,
        scanner_repository=(
            scanner_repository
        ),
        account_id=account.account_id,
    )

    engine.run(
        account_id=account.account_id,
        scan_id=scan.scan_id,
        run_key="same-session-create",
        run_at=T0,
    )

    report = engine.run(
        account_id=account.account_id,
        run_key="same-session-execute",
        run_at=T0,
    )

    assert report.run.filled_orders == 0

    assert len(
        paper_repository.list_pending_orders(
            account.account_id
        )
    ) == 1


def test_next_session_fill_applies_costs(
    tmp_path,
) -> None:
    next_session = (
        T0 + timedelta(days=1)
    )

    history = make_history(
        [
            (
                T0,
                103,
                104,
                102,
                103,
            ),
            (
                next_session,
                100,
                101,
                99,
                100,
            ),
        ]
    )

    costs = ExecutionCostModel(
        fixed_fee="1",
        slippage_bps="50",
    )

    (
        paper_repository,
        paper_service,
        scanner_repository,
        _,
        engine,
        account,
    ) = make_environment(
        tmp_path,
        history_by_symbol={
            "AAPL": history,
        },
        costs=costs,
    )

    scan = create_candidate_scan(
        paper_service=paper_service,
        scanner_repository=(
            scanner_repository
        ),
        account_id=account.account_id,
    )

    engine.run(
        account_id=account.account_id,
        scan_id=scan.scan_id,
        run_key="fill-create",
        run_at=T0,
    )

    report = engine.run(
        account_id=account.account_id,
        run_key="fill-execute",
        run_at=next_session,
    )

    positions = (
        paper_repository
        .list_open_positions(
            account.account_id
        )
    )

    assert report.run.filled_orders == 1
    assert len(positions) == 1

    assert positions[0].entry_price == Decimal(
        "100.50000000"
    )

    assert report.reconciliation.reconciled


def test_stop_loss_closes_and_reconciles(
    tmp_path,
) -> None:
    opened_at = (
        T0 + timedelta(days=1)
    )

    stop_session = (
        T0 + timedelta(days=2)
    )

    history = make_history(
        [
            (
                opened_at,
                100,
                102,
                98,
                100,
            ),
            (
                stop_session,
                94,
                96,
                93,
                95,
            ),
        ]
    )

    (
        paper_repository,
        paper_service,
        _,
        _,
        engine,
        account,
    ) = make_environment(
        tmp_path,
        history_by_symbol={
            "AAPL": history,
        },
    )

    open_position(
        paper_service=paper_service,
        account_id=account.account_id,
        opened_at=opened_at,
    )

    report = engine.run(
        account_id=account.account_id,
        run_key="stop-loss-run",
        run_at=stop_session,
    )

    trades = (
        paper_repository
        .list_closed_trades(
            account.account_id
        )
    )

    assert report.run.closed_positions == 1
    assert len(trades) == 1

    assert (
        trades[0].exit_reason
        is PaperExitReason.STOP_LOSS
    )

    assert trades[0].exit_price == Decimal(
        "94.00000000"
    )

    assert report.reconciliation.reconciled


def test_target_closes_position(
    tmp_path,
) -> None:
    opened_at = (
        T0 + timedelta(days=1)
    )

    target_session = (
        T0 + timedelta(days=2)
    )

    history = make_history(
        [
            (
                opened_at,
                100,
                102,
                98,
                100,
            ),
            (
                target_session,
                108,
                111,
                107,
                110,
            ),
        ]
    )

    (
        paper_repository,
        paper_service,
        _,
        _,
        engine,
        account,
    ) = make_environment(
        tmp_path,
        history_by_symbol={
            "AAPL": history,
        },
    )

    open_position(
        paper_service=paper_service,
        account_id=account.account_id,
        opened_at=opened_at,
    )

    engine.run(
        account_id=account.account_id,
        run_key="target-run",
        run_at=target_session,
    )

    trade = (
        paper_repository
        .list_closed_trades(
            account.account_id
        )[0]
    )

    assert (
        trade.exit_reason
        is PaperExitReason.TARGET
    )

    assert trade.exit_price == Decimal(
        "110.00000000"
    )


def test_kill_switch_blocks_entries_but_allows_exit(
    tmp_path,
) -> None:
    opened_at = (
        T0 + timedelta(days=1)
    )

    stop_session = (
        T0 + timedelta(days=2)
    )

    history = make_history(
        [
            (
                opened_at,
                100,
                102,
                98,
                100,
            ),
            (
                stop_session,
                94,
                96,
                93,
                95,
            ),
        ]
    )

    (
        paper_repository,
        paper_service,
        scanner_repository,
        automation_repository,
        engine,
        account,
    ) = make_environment(
        tmp_path,
        history_by_symbol={
            "AAPL": history,
        },
    )

    open_position(
        paper_service=paper_service,
        account_id=account.account_id,
        opened_at=opened_at,
    )

    scan = create_candidate_scan(
        paper_service=paper_service,
        scanner_repository=(
            scanner_repository
        ),
        account_id=account.account_id,
        generated_at=opened_at,
        data_as_of=opened_at,
    )

    automation_repository.set_kill_switch(
        account.account_id,
        active=True,
        reason="Manual safety stop.",
        updated_at=stop_session,
    )

    report = engine.run(
        account_id=account.account_id,
        scan_id=scan.scan_id,
        run_key="kill-switch-run",
        run_at=stop_session,
    )

    assert report.entries_enabled is False

    assert "Manual safety stop." in (
        report.entry_block_reasons
    )

    assert report.run.created_orders == 0
    assert report.run.closed_positions == 1

    assert (
        paper_repository
        .list_open_positions(
            account.account_id
        )
        == ()
    )


def test_stale_candidate_is_rejected(
    tmp_path,
) -> None:
    stale_at = (
        T0 - timedelta(days=10)
    )

    history = make_history(
        [
            (
                stale_at,
                100,
                101,
                99,
                100,
            ),
        ]
    )

    (
        paper_repository,
        paper_service,
        scanner_repository,
        _,
        engine,
        account,
    ) = make_environment(
        tmp_path,
        history_by_symbol={
            "AAPL": history,
        },
    )

    scan = create_candidate_scan(
        paper_service=paper_service,
        scanner_repository=(
            scanner_repository
        ),
        account_id=account.account_id,
        data_as_of=stale_at,
    )

    report = engine.run(
        account_id=account.account_id,
        scan_id=scan.scan_id,
        run_key="stale-candidate-run",
        run_at=T0,
    )

    assert report.run.created_orders == 0
    assert report.run.rejected_entries == 1

    assert (
        paper_repository
        .list_pending_orders(
            account.account_id
        )
        == ()
    )


def test_empty_execution_run_reconciles(
    tmp_path,
) -> None:
    history = make_history(
        [
            (
                T0,
                100,
                101,
                99,
                100,
            ),
        ]
    )

    (
        _,
        _,
        _,
        _,
        engine,
        account,
    ) = make_environment(
        tmp_path,
        history_by_symbol={
            "AAPL": history,
        },
    )

    report = engine.run(
        account_id=account.account_id,
        run_key="reconciliation-run",
        run_at=T0,
    )

    assert (
        report.run.status
        is ExecutionRunStatus.COMPLETED
    )

    assert report.reconciliation.reconciled
    assert report.equity_snapshot is not None

def test_ibkr_cost_gate_is_disabled_by_default() -> None:
    config = AutomatedExecutionConfig()

    assert (
        config.ibkr_cost_gate_enabled
        is False
    )

    assert config.ibkr_pricing_plan is None
    assert config.ibkr_fx_mode is None


def test_ibkr_cost_gate_rejects_candidate_before_order(
    tmp_path,
) -> None:
    history = make_history(
        [
            (
                T0,
                10,
                10.10,
                9.90,
                10,
            ),
        ]
    )

    (
        paper_repository,
        paper_service,
        scanner_repository,
        _,
        engine,
        account,
    ) = make_environment(
        tmp_path,
        history_by_symbol={
            "AAPL": history,
        },
    )

    engine.config = AutomatedExecutionConfig(
        fill_rule=FillRule.LIMIT,
        costs=ExecutionCostModel(),
        ibkr_cost_gate_enabled=True,
        ibkr_pricing_plan=(
            IBKRPricingPlan.FIXED
        ),
    )

    scan = create_candidate_scan(
        paper_service=paper_service,
        scanner_repository=(
            scanner_repository
        ),
        account_id=account.account_id,
        reward_to_risk=2.02,
        entry_low=10,
        entry_high=10,
        stop_price=9.5,
        targets=(11.01,),
        quote_currency="USD",
        data_as_of=T0,
    )

    report = engine.run(
        account_id=account.account_id,
        scan_id=scan.scan_id,
        run_key="ibkr-cost-reject",
        run_at=T0,
    )

    assert report.run.created_orders == 0

    assert (
        report.run.rejected_entries
        == 1
    )

    assert (
        paper_repository
        .list_pending_orders(
            account.account_id
        )
        == ()
    )

    connection = sqlite3.connect(
        paper_repository.database_path
    )

    try:
        row = connection.execute(
            """
            SELECT metadata_json
            FROM paper_system_events
            WHERE account_id = ?
              AND event_type =
                  'AUTOMATIC_ENTRY_REJECTED'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (account.account_id,),
        ).fetchone()
    finally:
        connection.close()

    assert row is not None

    metadata = json.loads(
        row[0]
    )

    assert (
        metadata["error_type"]
        == "IBKRCostGateRejected"
    )

    assert (
        "UNECONOMIC_AFTER_COSTS"
        in metadata["reason"]
    )

    assert (
        "net_rr="
        in metadata["reason"]
    )


def test_ibkr_tiered_gate_fails_closed_without_route_cost(
    tmp_path,
) -> None:
    history = make_history(
        [
            (
                T0,
                10,
                10.10,
                9.90,
                10,
            ),
        ]
    )

    (
        paper_repository,
        paper_service,
        scanner_repository,
        _,
        engine,
        account,
    ) = make_environment(
        tmp_path,
        history_by_symbol={
            "AAPL": history,
        },
    )

    engine.config = AutomatedExecutionConfig(
        fill_rule=FillRule.LIMIT,
        costs=ExecutionCostModel(),
        ibkr_cost_gate_enabled=True,
        ibkr_pricing_plan=(
            IBKRPricingPlan.TIERED
        ),
    )

    scan = create_candidate_scan(
        paper_service=paper_service,
        scanner_repository=(
            scanner_repository
        ),
        account_id=account.account_id,
        reward_to_risk=3.0,
        entry_low=10,
        entry_high=10,
        stop_price=9,
        targets=(13,),
        quote_currency="USD",
        data_as_of=T0,
    )

    report = engine.run(
        account_id=account.account_id,
        scan_id=scan.scan_id,
        run_key="ibkr-tiered-incomplete",
        run_at=T0,
    )

    assert report.run.created_orders == 0

    assert (
        report.run.rejected_entries
        == 1
    )

    connection = sqlite3.connect(
        paper_repository.database_path
    )

    try:
        row = connection.execute(
            """
            SELECT metadata_json
            FROM paper_system_events
            WHERE account_id = ?
              AND event_type =
                  'AUTOMATIC_ENTRY_REJECTED'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (account.account_id,),
        ).fetchone()
    finally:
        connection.close()

    assert row is not None

    metadata = json.loads(
        row[0]
    )

    assert (
        metadata["error_type"]
        == "IBKRCostGateRejected"
    )

    assert (
        "INCOMPLETE_COST_ESTIMATE"
        in metadata["reason"]
    )
