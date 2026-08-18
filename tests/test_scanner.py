from __future__ import annotations

from dataclasses import dataclass
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    PaperOrder,
    Signal,
)
from src.data import MarketSnapshot
from src.paper import (
    PaperPortfolioConfig,
    PaperRepository,
    PaperTradingService,
)
from src.scanner import (
    AutomaticMarketScanner,
    ScannerAnalysisOutcome,
    ScannerRepository,
    ScannerThresholds,
    ScanResultStatus,
    ScanStatus,
    StockUniverse,
    WatchlistState,
    load_stock_universe,
    run_deterministic_scanner_analysis,
)
from src.strategy import StrategyHorizon


T0 = datetime(
    2026,
    8,
    1,
    20,
    0,
    tzinfo=timezone.utc,
)


@dataclass(frozen=True)
class FakeReleaseReport:
    alert_scheduling_eligible: bool
    reasons: tuple[str, ...]


def make_snapshot(
    symbol: str,
    *,
    rows: int = 260,
    volume: float = 1_000_000,
    end: str = "2026-07-31",
) -> MarketSnapshot:
    index = pd.date_range(
        end=end,
        periods=rows,
        freq="B",
        tz="UTC",
    )

    close = np.linspace(
        90.0,
        110.0,
        rows,
    )

    history = pd.DataFrame(
        {
            "Open": close - 0.25,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(
                rows,
                volume,
            ),
        },
        index=index,
    )

    return MarketSnapshot(
        symbol=symbol,
        history=history,
        metadata={
            "symbol": symbol,
            "shortName":
            f"{symbol} Corporation",
            "quoteType": "EQUITY",
            "currency": "USD",
            "exchange": "NMS",
        },
        fetched_at_utc=T0,
    )


def make_outcome(
    snapshot: MarketSnapshot,
    *,
    signal: Signal = Signal.BUY,
    strategy: str = "trend_pullback",
    score: float = 80.0,
    confidence: float = 0.85,
    reward_to_risk: float = 2.5,
    include_order: bool = True,
) -> ScannerAnalysisOutcome:
    generated_at = (
        pd.Timestamp(
            snapshot.history.index[-1]
        )
        .tz_convert("UTC")
        .to_pydatetime()
    )

    order = None

    if include_order:
        order = PaperOrder(
            symbol=snapshot.symbol,
            signal=signal,
            created_at=generated_at,
            expires_at=(
                generated_at
                + timedelta(days=5)
            ),
            entry_low=100.0,
            entry_high=101.0,
            stop_price=95.0,
            targets=(
                113.0,
                120.0,
            ),
            risk_per_share=6.0,
            reward_to_risk=(
                reward_to_risk
            ),
            paper_only=True,
        )

    return ScannerAnalysisOutcome(
        symbol=snapshot.symbol,
        generated_at=generated_at,
        strategy=strategy,
        recommendation=signal,
        score=score,
        confidence=confidence,
        market_regime="BULLISH",
        regime_confidence=0.90,
        order=order,
        risk_vetoes=(),
        evidence=(
            {
                "code": "TEST_SIGNAL",
                "reason":
                "Deterministic test evidence.",
            },
        ),
    )


def make_services(
    tmp_path,
):
    database_path = (
        tmp_path / "scanner.db"
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
        ),
        app_version="test-version",
        threshold_version=(
            "test-thresholds"
        ),
    )

    account = paper_service.create_account(
        created_at=T0
    )

    scanner_repository = ScannerRepository(
        database_path
    )

    thresholds = ScannerThresholds(
        minimum_history_rows=200,
        maximum_staleness_days=7,
        minimum_price=5,
        minimum_average_volume=1_000,
        minimum_average_dollar_volume=(
            10_000
        ),
        liquidity_lookback_sessions=20,
    )

    return (
        database_path,
        paper_repository,
        paper_service,
        scanner_repository,
        account,
        thresholds,
    )


def test_universe_normalises_and_deduplicates() -> None:
    universe = StockUniverse(
        name="test",
        symbols=(
            " aapl ",
            "MSFT",
            "AAPL",
        ),
    )

    assert universe.symbols == (
        "AAPL",
        "MSFT",
    )


def test_configured_universe_loads() -> None:
    universe = load_stock_universe()

    assert universe.name == (
        "personal_us_large_cap_v1"
    )

    assert "AAPL" in universe.symbols
    assert "MSFT" in universe.symbols
    assert len(universe.symbols) >= 20
    assert (
        universe.policy_version
        == "p4.4-universe-v1"
    )
    assert universe.included_symbols == ()
    assert universe.excluded_symbols == ()


def test_versioned_universe_applies_curated_lists(
    tmp_path,
) -> None:
    path = tmp_path / "universe.json"

    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "policy_version":
                "p4.4-universe-v1",
                "name": "fixture",
                "description": "fixture",
                "base_symbols": [
                    "AAPL",
                    "MSFT",
                ],
                "include_symbols": [
                    "SPY",
                    "QQQ",
                ],
                "exclude_symbols": [
                    "MSFT",
                ],
            }
        ),
        encoding="utf-8",
    )

    universe = load_stock_universe(
        path
    )

    assert universe.symbols == (
        "AAPL",
        "SPY",
        "QQQ",
    )
    assert universe.included_symbols == (
        "SPY",
        "QQQ",
    )
    assert universe.excluded_symbols == (
        "MSFT",
    )


def test_versioned_universe_rejects_list_conflict(
    tmp_path,
) -> None:
    path = tmp_path / "universe.json"

    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "policy_version":
                "p4.4-universe-v1",
                "name": "fixture",
                "description": "fixture",
                "base_symbols": [
                    "AAPL",
                ],
                "include_symbols": [
                    "SPY",
                ],
                "exclude_symbols": [
                    "SPY",
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must be disjoint",
    ):
        load_stock_universe(path)


def test_schema_migrates_to_latest_version(
    tmp_path,
) -> None:
    (
        database_path,
        _,
        _,
        _,
        _,
        _,
    ) = make_services(tmp_path)

    connection = sqlite3.connect(
        database_path
    )

    try:
        version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'paper_scan_results'
            """
        ).fetchone()
    finally:
        connection.close()

    assert version == 10
    assert table is not None


def test_insufficient_history_is_rejected(
    tmp_path,
) -> None:
    (
        _,
        _,
        paper_service,
        scanner_repository,
        account,
        thresholds,
    ) = make_services(tmp_path)

    scanner = AutomaticMarketScanner(
        scanner_repository=(
            scanner_repository
        ),
        paper_service=paper_service,
        release_gate_lookup=(
            lambda strategy:
            FakeReleaseReport(
                True,
                ("Approved.",),
            )
        ),
        thresholds=thresholds,
        snapshot_loader=(
            lambda symbol:
            make_snapshot(
                symbol,
                rows=100,
            )
        ),
        analysis_runner=lambda snapshot: (
            make_outcome(snapshot)
        ),
    )

    report = scanner.run_scan(
        account_id=account.account_id,
        universe=StockUniverse(
            name="short-history",
            symbols=("AAPL",),
        ),
        started_at=T0,
    )

    result = report.results[0]

    assert (
        result.status
        is ScanResultStatus.DATA_REJECTED
    )

    assert result.signal_id is None
    assert (
        result.watchlist_state
        is WatchlistState.REJECT
    )
    assert "at least 200" in result.reasons[0]



def test_stale_history_has_explicit_stale_state(
    tmp_path,
) -> None:
    (
        _,
        _,
        paper_service,
        scanner_repository,
        account,
        thresholds,
    ) = make_services(tmp_path)

    scanner = AutomaticMarketScanner(
        scanner_repository=(
            scanner_repository
        ),
        paper_service=paper_service,
        release_gate_lookup=(
            lambda strategy:
            FakeReleaseReport(
                True,
                ("Approved.",),
            )
        ),
        thresholds=thresholds,
        snapshot_loader=(
            lambda symbol:
            make_snapshot(
                symbol,
                end="2026-06-01",
            )
        ),
        analysis_runner=make_outcome,
    )

    report = scanner.run_scan(
        account_id=account.account_id,
        universe=StockUniverse(
            name="stale",
            symbols=("AAPL",),
        ),
        started_at=T0,
    )

    result = report.results[0]

    assert (
        result.status
        is ScanResultStatus.DATA_REJECTED
    )
    assert (
        result.watchlist_state
        is WatchlistState.STALE
    )
    assert (
        result.metadata[
            "watchlist_state"
        ]
        == "STALE"
    )


def test_watch_result_is_persisted_without_signal(
    tmp_path,
) -> None:
    (
        _,
        _,
        paper_service,
        scanner_repository,
        account,
        thresholds,
    ) = make_services(tmp_path)

    scanner = AutomaticMarketScanner(
        scanner_repository=(
            scanner_repository
        ),
        paper_service=paper_service,
        release_gate_lookup=(
            lambda strategy:
            FakeReleaseReport(
                True,
                ("Approved.",),
            )
        ),
        thresholds=thresholds,
        snapshot_loader=make_snapshot,
        analysis_runner=lambda snapshot: (
            make_outcome(
                snapshot,
                signal=Signal.WATCH,
                include_order=False,
            )
        ),
    )

    report = scanner.run_scan(
        account_id=account.account_id,
        universe=StockUniverse(
            name="watch",
            symbols=("AAPL",),
        ),
        started_at=T0,
    )

    result = report.results[0]

    assert (
        result.status
        is ScanResultStatus.WATCH
    )

    assert (
        result.watchlist_state
        is WatchlistState.WATCH
    )
    assert result.signal_id is None


def test_missing_release_report_is_ineligible(
    tmp_path,
) -> None:
    (
        _,
        _,
        paper_service,
        scanner_repository,
        account,
        thresholds,
    ) = make_services(tmp_path)

    scanner = AutomaticMarketScanner(
        scanner_repository=(
            scanner_repository
        ),
        paper_service=paper_service,
        release_gate_lookup=(
            lambda strategy: None
        ),
        thresholds=thresholds,
        snapshot_loader=make_snapshot,
        analysis_runner=make_outcome,
    )

    report = scanner.run_scan(
        account_id=account.account_id,
        universe=StockUniverse(
            name="release-check",
            symbols=("AAPL",),
        ),
        started_at=T0,
    )

    result = report.results[0]

    assert (
        result.status
        is ScanResultStatus
        .RELEASE_INELIGIBLE
    )

    assert result.release_eligible is False
    assert (
        result.watchlist_state
        is WatchlistState.PREPARE
    )
    assert result.signal_id is None


def test_rejected_release_report_is_ineligible(
    tmp_path,
) -> None:
    (
        _,
        _,
        paper_service,
        scanner_repository,
        account,
        thresholds,
    ) = make_services(tmp_path)

    scanner = AutomaticMarketScanner(
        scanner_repository=(
            scanner_repository
        ),
        paper_service=paper_service,
        release_gate_lookup=(
            lambda strategy:
            FakeReleaseReport(
                False,
                (
                    "Strategy did not pass "
                    "the release gate.",
                ),
            )
        ),
        thresholds=thresholds,
        snapshot_loader=make_snapshot,
        analysis_runner=make_outcome,
    )

    report = scanner.run_scan(
        account_id=account.account_id,
        universe=StockUniverse(
            name="rejected-release",
            symbols=("AAPL",),
        ),
        started_at=T0,
    )

    assert (
        report.results[0].status
        is ScanResultStatus
        .RELEASE_INELIGIBLE
    )


def test_candidates_are_ranked_and_signals_persisted(
    tmp_path,
) -> None:
    (
        _,
        paper_repository,
        paper_service,
        scanner_repository,
        account,
        thresholds,
    ) = make_services(tmp_path)

    def analysis_runner(
        snapshot,
    ):
        return make_outcome(
            snapshot,
            score=(
                92
                if snapshot.symbol == "MSFT"
                else 78
            ),
            confidence=(
                0.95
                if snapshot.symbol == "MSFT"
                else 0.80
            ),
        )

    scanner = AutomaticMarketScanner(
        scanner_repository=(
            scanner_repository
        ),
        paper_service=paper_service,
        release_gate_lookup=(
            lambda strategy:
            FakeReleaseReport(
                True,
                ("P2 release approved.",),
            )
        ),
        thresholds=thresholds,
        snapshot_loader=make_snapshot,
        analysis_runner=analysis_runner,
    )

    report = scanner.run_scan(
        account_id=account.account_id,
        universe=StockUniverse(
            name="candidates",
            symbols=(
                "AAPL",
                "MSFT",
            ),
        ),
        started_at=T0,
        strategy_horizon=(
            StrategyHorizon.SWING
        ),
        strategy_version=(
            "p4.3-swing-v1"
        ),
    )

    assert (
        report.scan.configuration[
            "strategy_horizon"
        ]
        == "SWING"
    )
    assert (
        report.scan.configuration[
            "strategy_version"
        ]
        == "p4.3-swing-v1"
    )
    assert (
        report.scan.configuration[
            "universe_policy"
        ]["policy_version"]
        == "legacy-universe-v1"
    )

    assert [
        result.symbol
        for result in report.candidates
    ] == [
        "MSFT",
        "AAPL",
    ]

    assert [
        result.rank_position
        for result in report.candidates
    ] == [1, 2]

    assert all(
        result.signal_id is not None
        for result in report.candidates
    )
    assert all(
        result.watchlist_state
        is WatchlistState.ACTIONABLE
        for result in report.candidates
    )
    assert all(
        set(result.score_components)
        == {
            "analysis_score",
            "confidence",
            "reward_to_risk",
            "regime_confidence",
        }
        for result in report.candidates
    )
    assert all(
        sum(
            result.score_components.values()
        )
        == pytest.approx(
            result.rank_score
        )
        for result in report.candidates
    )

    for result in report.candidates:
        signal = paper_repository.get_signal(
            result.signal_id
        )

        assert signal.scan_id == (
            report.scan.scan_id
        )

        assert signal.recommendation == "BUY"
        assert signal.quote_currency == "USD"
        assert (
            signal.strategy_horizon
            is StrategyHorizon.SWING
        )
        assert (
            signal.strategy_version
            == "p4.3-swing-v1"
        )


def test_horizon_watchlist_ranks_restart_independently(
    tmp_path,
) -> None:
    (
        _,
        _,
        paper_service,
        scanner_repository,
        account,
        thresholds,
    ) = make_services(tmp_path)

    def analysis_runner(snapshot):
        return make_outcome(
            snapshot,
            score=(
                90
                if snapshot.symbol == "MSFT"
                else 75
            ),
        )

    scanner = AutomaticMarketScanner(
        scanner_repository=scanner_repository,
        paper_service=paper_service,
        release_gate_lookup=(
            lambda strategy:
            FakeReleaseReport(
                True,
                ("P2 release approved.",),
            )
        ),
        thresholds=thresholds,
        snapshot_loader=make_snapshot,
        analysis_runner=analysis_runner,
    )
    universe = StockUniverse(
        name="horizon-watchlists",
        symbols=("AAPL", "MSFT"),
    )

    reports = [
        scanner.run_scan(
            account_id=account.account_id,
            universe=universe,
            started_at=(
                T0
                + timedelta(days=index)
            ),
            scan_key=(
                f"horizon-{horizon.value}"
            ),
            strategy_horizon=horizon,
            strategy_version=version,
        )
        for index, (horizon, version)
        in enumerate(
            (
                (
                    StrategyHorizon.SWING,
                    "p4.3-swing-v1",
                ),
                (
                    StrategyHorizon.MEDIUM_TERM,
                    "p4.3-medium-term-v1",
                ),
            )
        )
    ]

    for report, expected_horizon in zip(
        reports,
        (
            StrategyHorizon.SWING,
            StrategyHorizon.MEDIUM_TERM,
        ),
        strict=True,
    ):
        assert [
            result.rank_position
            for result in report.candidates
        ] == [1, 2]
        assert all(
            result.strategy_horizon
            is expected_horizon
            for result in report.results
        )

    assert (
        reports[0].scan.configuration[
            "strategy_horizon"
        ]
        == "SWING"
    )
    assert (
        reports[1].scan.configuration[
            "strategy_horizon"
        ]
        == "MEDIUM_TERM"
    )


def test_horizon_and_version_must_be_supplied_together(
    tmp_path,
) -> None:
    (
        _,
        _,
        paper_service,
        scanner_repository,
        account,
        thresholds,
    ) = make_services(tmp_path)

    scanner = AutomaticMarketScanner(
        scanner_repository=scanner_repository,
        paper_service=paper_service,
        release_gate_lookup=(
            lambda strategy: None
        ),
        thresholds=thresholds,
        snapshot_loader=make_snapshot,
        analysis_runner=make_outcome,
    )

    with pytest.raises(
        ValueError,
        match=(
            "strategy_horizon and "
            "strategy_version"
        ),
    ):
        scanner.run_scan(
            account_id=account.account_id,
            universe=StockUniverse(
                name="invalid-horizon-scope",
                symbols=("AAPL",),
            ),
            started_at=T0,
            strategy_horizon=(
                StrategyHorizon.SWING
            ),
        )


def test_scan_key_prevents_duplicate_scan(
    tmp_path,
) -> None:
    (
        _,
        _,
        paper_service,
        scanner_repository,
        account,
        thresholds,
    ) = make_services(tmp_path)

    calls = {"count": 0}

    def loader(symbol):
        calls["count"] += 1
        return make_snapshot(symbol)

    scanner = AutomaticMarketScanner(
        scanner_repository=(
            scanner_repository
        ),
        paper_service=paper_service,
        release_gate_lookup=(
            lambda strategy:
            FakeReleaseReport(
                True,
                ("Approved.",),
            )
        ),
        thresholds=thresholds,
        snapshot_loader=loader,
        analysis_runner=make_outcome,
    )

    universe = StockUniverse(
        name="idempotent",
        symbols=("AAPL",),
    )

    first = scanner.run_scan(
        account_id=account.account_id,
        universe=universe,
        started_at=T0,
        scan_key="daily-2026-08-01",
    )

    second = scanner.run_scan(
        account_id=account.account_id,
        universe=universe,
        started_at=T0,
        scan_key="daily-2026-08-01",
    )

    assert first.scan.scan_id == (
        second.scan.scan_id
    )

    assert calls["count"] == 1


def test_symbol_failure_is_persisted_as_scan_error(
    tmp_path,
) -> None:
    (
        _,
        _,
        paper_service,
        scanner_repository,
        account,
        thresholds,
    ) = make_services(tmp_path)

    def loader(symbol):
        if symbol == "MSFT":
            raise RuntimeError(
                "Provider unavailable."
            )

        return make_snapshot(symbol)

    scanner = AutomaticMarketScanner(
        scanner_repository=(
            scanner_repository
        ),
        paper_service=paper_service,
        release_gate_lookup=(
            lambda strategy:
            FakeReleaseReport(
                True,
                ("Approved.",),
            )
        ),
        thresholds=thresholds,
        snapshot_loader=loader,
        analysis_runner=make_outcome,
    )

    report = scanner.run_scan(
        account_id=account.account_id,
        universe=StockUniverse(
            name="partial",
            symbols=(
                "AAPL",
                "MSFT",
            ),
        ),
        started_at=T0,
    )

    assert (
        report.scan.status
        is ScanStatus.COMPLETED_WITH_ERRORS
    )

    errors = [
        result
        for result in report.results
        if result.status
        is ScanResultStatus.SCAN_ERROR
    ]

    assert len(errors) == 1
    assert errors[0].symbol == "MSFT"


def test_candidate_without_valid_currency_is_not_persisted(
    tmp_path,
) -> None:
    (
        _,
        paper_repository,
        paper_service,
        scanner_repository,
        account,
        thresholds,
    ) = make_services(tmp_path)

    def loader(symbol):
        snapshot = make_snapshot(symbol)

        snapshot.metadata.pop(
            "currency",
            None,
        )

        return snapshot

    scanner = AutomaticMarketScanner(
        scanner_repository=(
            scanner_repository
        ),
        paper_service=paper_service,
        release_gate_lookup=(
            lambda strategy:
            FakeReleaseReport(
                True,
                ("Approved.",),
            )
        ),
        thresholds=thresholds,
        snapshot_loader=loader,
        analysis_runner=make_outcome,
    )

    report = scanner.run_scan(
        account_id=account.account_id,
        universe=StockUniverse(
            name="missing-currency",
            symbols=("AAPL",),
        ),
        started_at=T0,
    )

    assert len(report.candidates) == 0

    result = report.results[0]

    assert (
        result.status
        is ScanResultStatus.SCAN_ERROR
    )

    assert result.signal_id is None
    assert result.rank_score is None
    assert result.score_components == {}

    assert (
        "valid three-letter quote currency"
        in result.reasons[0]
    )

    expected_signal_id = (
        f"SIG-{report.scan.scan_id}-AAPL"
    )

    try:
        paper_repository.get_signal(
            expected_signal_id
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid-currency candidate signal "
            "was unexpectedly persisted."
        )


def test_watch_without_currency_remains_watch(
    tmp_path,
) -> None:
    (
        _,
        _,
        paper_service,
        scanner_repository,
        account,
        thresholds,
    ) = make_services(tmp_path)

    def loader(symbol):
        snapshot = make_snapshot(symbol)

        snapshot.metadata.pop(
            "currency",
            None,
        )

        return snapshot

    scanner = AutomaticMarketScanner(
        scanner_repository=(
            scanner_repository
        ),
        paper_service=paper_service,
        release_gate_lookup=(
            lambda strategy:
            FakeReleaseReport(
                True,
                ("Approved.",),
            )
        ),
        thresholds=thresholds,
        snapshot_loader=loader,
        analysis_runner=(
            lambda snapshot:
            make_outcome(
                snapshot,
                signal=Signal.WATCH,
                include_order=False,
            )
        ),
    )

    report = scanner.run_scan(
        account_id=account.account_id,
        universe=StockUniverse(
            name="watch-missing-currency",
            symbols=("AAPL",),
        ),
        started_at=T0,
    )

    result = report.results[0]

    assert (
        result.status
        is ScanResultStatus.WATCH
    )

    assert result.signal_id is None

    assert len(
        [
            item
            for item in report.results
            if item.status
            is ScanResultStatus.SCAN_ERROR
        ]
    ) == 0


def test_real_analysis_adapter_uses_existing_engine() -> None:
    outcome = (
        run_deterministic_scanner_analysis(
            make_snapshot("AAPL")
        )
    )

    assert outcome.symbol == "AAPL"
    assert isinstance(
        outcome.recommendation,
        Signal,
    )
    assert outcome.strategy
    assert 0 <= outcome.confidence <= 1
    assert outcome.evidence
