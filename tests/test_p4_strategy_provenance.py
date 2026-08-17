from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
import sqlite3

from src.paper import (
    PaperExitReason,
    PaperRepository,
    PaperTradingService,
)
from src.paper import migrations
from src.paper.migrations import (
    initialize_database,
)
from src.scanner import (
    ScanResult,
    ScanResultStatus,
    ScannerRepository,
    StockUniverse,
)
from src.strategy import StrategyHorizon


T0 = datetime(
    2026,
    8,
    17,
    12,
    0,
    tzinfo=timezone.utc,
)


EXPECTED_TABLES = {
    "paper_scan_results",
    "paper_signals",
    "paper_orders",
    "paper_positions",
    "paper_closed_trades",
}


def test_strategy_horizon_domain_has_only_approved_values(
) -> None:
    assert {
        item.value
        for item in StrategyHorizon
    } == {
        "SWING",
        "MEDIUM_TERM",
    }


def test_genuine_v8_database_upgrades_to_v10(
    tmp_path,
) -> None:
    database = (
        tmp_path
        / "legacy-v8.db"
    )

    connection = sqlite3.connect(
        database
    )

    try:
        for version in range(
            1,
            9,
        ):
            connection.executescript(
                getattr(
                    migrations,
                    f"_SCHEMA_V{version}",
                )
            )

        assert (
            connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]
            == 8
        )
    finally:
        connection.close()

    initialize_database(
        database
    )

    connection = sqlite3.connect(
        database
    )

    try:
        assert (
            connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]
            == 10
        )

        for table in EXPECTED_TABLES:
            columns = {
                row[1]
                for row in connection.execute(
                    f"PRAGMA table_info({table})"
                )
            }

            assert (
                "strategy_horizon"
                in columns
            )

            assert (
                "strategy_version"
                in columns
            )

            if table == "paper_positions":
                assert (
                    "maximum_holding_sessions"
                    in columns
                )
    finally:
        connection.close()


def test_paper_lifecycle_preserves_strategy_provenance(
    tmp_path,
) -> None:
    database = (
        tmp_path
        / "lifecycle.db"
    )

    repository = PaperRepository(
        database
    )

    service = PaperTradingService(
        repository,
        app_version="p4.3-test",
        threshold_version="p4.3-test",
    )

    account = service.create_account(
        created_at=T0
    )

    signal = service.persist_signal(
        account_id=account.account_id,
        signal_id="SIG-P43-SWING",
        symbol="AAPL",
        quote_currency="USD",
        generated_at=T0,
        expires_at=(
            T0
            + timedelta(days=7)
        ),
        strategy="trend_pullback",
        strategy_horizon=(
            StrategyHorizon.SWING
        ),
        strategy_version=(
            "trend-pullback-swing-v1"
        ),
        recommendation="BUY",
        market_regime="BULLISH",
        score=85,
        confidence=0.90,
        reward_to_risk=3.0,
        entry_low=99,
        entry_high=100,
        stop_price=95,
        targets=(110,),
        evidence=(
            "P4.3 provenance fixture.",
        ),
    )

    assert (
        signal.strategy_horizon
        is StrategyHorizon.SWING
    )
    assert (
        signal.strategy_version
        == "trend-pullback-swing-v1"
    )

    order, created = (
        service.create_automatic_buy(
            account_id=account.account_id,
            signal_id=signal.signal_id,
            quantity=1,
            idempotency_key=(
                "P43-PROVENANCE"
            ),
            estimated_fees=1,
            created_at=(
                T0
                + timedelta(hours=1)
            ),
        )
    )

    assert created is True
    assert (
        order.strategy_horizon
        is StrategyHorizon.SWING
    )
    assert (
        order.strategy_version
        == "trend-pullback-swing-v1"
    )

    _, position = (
        service.record_automatic_buy_fill(
            order_id=order.order_id,
            fill_price=100,
            fees=1,
            slippage=0,
            filled_at=(
                T0
                + timedelta(days=1)
            ),
        )
    )

    assert (
        position.strategy_horizon
        is StrategyHorizon.SWING
    )
    assert (
        position.strategy_version
        == "trend-pullback-swing-v1"
    )
    assert (
        position.maximum_holding_sessions
        == 20
    )
    assert (
        position.expires_at
        == order.expires_at
    )

    trade = (
        service.close_automatic_position(
            position_id=position.position_id,
            exit_price=110,
            exit_reason=(
                PaperExitReason.TARGET
            ),
            exit_fees=1,
            exit_slippage=0,
            closed_at=(
                T0
                + timedelta(days=2)
            ),
        )
    )

    assert (
        trade.strategy_horizon
        is StrategyHorizon.SWING
    )
    assert (
        trade.strategy_version
        == "trend-pullback-swing-v1"
    )

    assert (
        repository.reconcile_account(
            account.account_id
        ).reconciled
        is True
    )


def test_scanner_result_persists_strategy_provenance(
    tmp_path,
) -> None:
    database = (
        tmp_path
        / "scanner.db"
    )

    repository = PaperRepository(
        database
    )

    service = PaperTradingService(
        repository
    )

    account = service.create_account(
        created_at=T0
    )

    scanner = ScannerRepository(
        database
    )

    scan, created = scanner.start_scan(
        account_id=account.account_id,
        universe=StockUniverse(
            name="p4.3-test",
            symbols=("AAPL",),
        ),
        configuration={},
        app_version="p4.3-test",
        started_at=T0,
        scan_key="p4.3-provenance",
    )

    assert created is True

    result = ScanResult(
        result_id="RESULT-P43",
        scan_id=scan.scan_id,
        account_id=account.account_id,
        symbol="AAPL",
        status=(
            ScanResultStatus.ORDER_CANDIDATE
        ),
        processed_at=T0,
        data_as_of=T0,
        history_rows=260,
        latest_price=100,
        average_volume=1_000_000,
        average_dollar_volume=(
            100_000_000
        ),
        recommendation="BUY",
        strategy="trend_pullback",
        strategy_horizon=(
            StrategyHorizon.MEDIUM_TERM
        ),
        strategy_version=(
            "trend-pullback-medium-v1"
        ),
        score=85,
        confidence=0.90,
        market_regime="BULLISH",
        reward_to_risk=3.0,
        release_eligible=True,
        rank_score=90,
        rank_position=1,
        signal_id=None,
        reasons=("Approved.",),
        evidence=(),
        metadata={},
    )

    scanner.save_result(
        result
    )

    connection = sqlite3.connect(
        database
    )

    connection.row_factory = sqlite3.Row

    try:
        row = connection.execute(
            """
            SELECT *
            FROM paper_scan_results
            WHERE result_id = ?
            """,
            ("RESULT-P43",),
        ).fetchone()
    finally:
        connection.close()

    assert row is not None

    mapped = scanner._result_from_row(
        row
    )

    assert (
        mapped.strategy_horizon
        is StrategyHorizon.MEDIUM_TERM
    )

    assert (
        mapped.strategy_version
        == "trend-pullback-medium-v1"
    )
