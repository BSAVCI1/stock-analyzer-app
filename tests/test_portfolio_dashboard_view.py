from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal
from enum import Enum
from types import SimpleNamespace

from src.portfolio_dashboard import (
    closed_trade_rows,
    decision_trace_rows,
    equity_rows,
    format_money,
    metric_cards,
    open_position_rows,
    pending_order_rows,
    provenance_rows,
    reliability_rows,
    scan_result_rows,
    scan_rows,
)


T0 = datetime(
    2026,
    8,
    3,
    20,
    30,
    tzinfo=timezone.utc,
)


class Value(str, Enum):
    OPEN = "OPEN"
    BUY = "BUY"
    COMPLETED = "COMPLETED"
    TARGET = "TARGET"


def test_format_money_is_deterministic() -> None:
    assert format_money(
        Decimal("1234.5678"),
        "usd",
    ) == "1,234.57 USD"


def test_metric_cards_use_snapshot_values() -> None:
    snapshot = SimpleNamespace(
        account=SimpleNamespace(
            account_id="ACC-1",
            base_currency="USD",
            cash_balance=Decimal(
                "9000"
            ),
            reserved_cash=Decimal(
                "1000"
            ),
        ),
        open_positions=(1, 2),
        pending_orders=(1,),
        performance=SimpleNamespace(
            net_pnl=Decimal("250")
        ),
        reconciliation=SimpleNamespace(
            reconciled=True
        ),
    )

    cards = metric_cards(snapshot)

    assert cards[0]["value"] == (
        "9,000.00 USD"
    )

    assert cards[1]["value"] == (
        "8,000.00 USD"
    )

    assert cards[2]["value"] == "2"
    assert cards[3]["value"] == "1"
    assert cards[5]["value"] == "Yes"


def test_position_and_order_rows_keep_ids() -> None:
    position = SimpleNamespace(
        position_id="POS-1",
        order_id="ORD-1",
        fill_id="FILL-1",
        symbol="AAPL",
        side=Value.BUY,
        quantity=Decimal("10"),
        entry_price=Decimal("100"),
        stop_price=Decimal("95"),
        targets=(
            Decimal("110"),
            Decimal("120"),
        ),
        opened_at=T0,
        expires_at=(
            T0 + timedelta(days=7)
        ),
        status=Value.OPEN,
    )

    order = SimpleNamespace(
        order_id="ORD-2",
        signal_id="SIG-2",
        idempotency_key="KEY-2",
        symbol="MSFT",
        side=Value.BUY,
        quantity=Decimal("5"),
        entry_low=Decimal("99"),
        entry_high=Decimal("101"),
        stop_price=Decimal("95"),
        targets=(Decimal("110"),),
        reserved_cash=Decimal("505"),
        status=Value.OPEN,
        created_at=T0,
        expires_at=(
            T0 + timedelta(days=7)
        ),
    )

    snapshot = SimpleNamespace(
        open_positions=(position,),
        pending_orders=(order,),
    )

    positions = open_position_rows(
        snapshot
    )

    orders = pending_order_rows(
        snapshot
    )

    assert (
        positions[0]["position_id"]
        == "POS-1"
    )

    assert (
        positions[0]["order_id"]
        == "ORD-1"
    )

    assert (
        orders[0]["signal_id"]
        == "SIG-2"
    )


def test_trade_and_trace_rows_preserve_evidence() -> None:
    trade = SimpleNamespace(
        trade_id="TRADE-1",
        position_id="POS-1",
        signal_id="SIG-1",
        symbol="AAPL",
        strategy="trend_pullback",
        market_regime="BULLISH",
        entry_time=T0,
        entry_price=Decimal("100"),
        exit_time=(
            T0 + timedelta(days=2)
        ),
        exit_price=Decimal("110"),
        exit_reason=Value.TARGET,
        quantity=Decimal("10"),
        gross_pnl=Decimal("100"),
        fees=Decimal("2"),
        slippage=Decimal("1"),
        net_pnl=Decimal("97"),
        return_pct=9.7,
        holding_seconds=172800,
    )

    trace = SimpleNamespace(
        reference_type="CLOSED_TRADE",
        reference_id="TRADE-1",
        signal_id="SIG-1",
        symbol="AAPL",
        strategy="trend_pullback",
        recommendation="BUY",
        market_regime="BULLISH",
        score=82,
        confidence=0.88,
        reward_to_risk=2.5,
        threshold_version=(
            "threshold-test"
        ),
        app_version="test",
        exit_reason="TARGET",
        evidence=(
            "Evidence one.",
            "Evidence two.",
        ),
        conflicts=(),
        provenance=SimpleNamespace(
            source_tables=(
                "paper_closed_trades",
                "paper_signals",
            ),
            record_ids=(
                "TRADE-1",
                "SIG-1",
            ),
        ),
    )

    snapshot = SimpleNamespace(
        closed_trades=(trade,),
        decision_traces=(trace,),
    )

    trades = closed_trade_rows(
        snapshot
    )

    traces = decision_trace_rows(
        snapshot
    )

    assert (
        trades[0]["trade_id"]
        == "TRADE-1"
    )

    assert (
        traces[0]["evidence"]
        == "Evidence one. | Evidence two."
    )

    assert "SIG-1" in (
        traces[0][
            "source_record_ids"
        ]
    )


def test_equity_rows_are_chronological() -> None:
    later = SimpleNamespace(
        snapshot_id="EQ-2",
        run_id="RUN-2",
        captured_at=(
            T0 + timedelta(days=1)
        ),
        cash_balance=Decimal("8500"),
        reserved_cash=Decimal("0"),
        market_value=Decimal("1000"),
        equity=Decimal("9500"),
    )

    earlier = SimpleNamespace(
        snapshot_id="EQ-1",
        run_id="RUN-1",
        captured_at=T0,
        cash_balance=Decimal("9000"),
        reserved_cash=Decimal("0"),
        market_value=Decimal("1000"),
        equity=Decimal("10000"),
    )

    rows = equity_rows(
        SimpleNamespace(
            equity_snapshots=(
                later,
                earlier,
            )
        )
    )

    assert [
        row["snapshot_id"]
        for row in rows
    ] == [
        "EQ-1",
        "EQ-2",
    ]

    assert rows[1]["equity"] == 9500.0


def test_scan_rows_keep_scan_and_result_ids() -> None:
    result = SimpleNamespace(
        result_id="RESULT-1",
        scan_id="SCAN-1",
        symbol="AAPL",
        status=Value.COMPLETED,
        recommendation="BUY",
        strategy="trend_pullback",
        score=80,
        confidence=0.9,
        market_regime="BULLISH",
        reward_to_risk=2.5,
        release_eligible=True,
        rank_score=90,
        rank_position=1,
        signal_id="SIG-1",
        reasons=("Approved.",),
    )

    scan = SimpleNamespace(
        scan_id="SCAN-1",
        scan_key="KEY-1",
        universe="test",
        status=Value.COMPLETED,
        started_at=T0,
        completed_at=T0,
        requested_count=1,
        processed_count=1,
        rejected_count=0,
        signal_count=1,
        order_count=1,
        app_version="test",
        error_message=None,
    )

    snapshot = SimpleNamespace(
        scan_reports=(
            SimpleNamespace(
                scan=scan,
                results=(result,),
            ),
        )
    )

    scans = scan_rows(snapshot)
    results = scan_result_rows(
        snapshot
    )

    assert scans[0]["scan_id"] == "SCAN-1"

    assert (
        results[0]["result_id"]
        == "RESULT-1"
    )

    assert (
        results[0]["signal_id"]
        == "SIG-1"
    )


def test_reliability_rows_include_source_count() -> None:
    def metric(name):
        return SimpleNamespace(
            name=name,
            total=2,
            successful=1,
            failed=1,
            pending_or_other=0,
            success_rate_pct=50.0,
            provenance=SimpleNamespace(
                source_tables=(
                    f"paper_{name}",
                ),
                record_count=2,
            ),
        )

    snapshot = SimpleNamespace(
        reliability=SimpleNamespace(
            scans=metric("scans"),
            execution_runs=metric(
                "execution_runs"
            ),
            scheduled_jobs=metric(
                "scheduled_jobs"
            ),
            notifications=metric(
                "notifications"
            ),
            system_events=metric(
                "system_events"
            ),
        )
    )

    rows = reliability_rows(
        snapshot
    )

    assert len(rows) == 5

    assert (
        rows[0][
            "source_record_count"
        ]
        == 2
    )


def test_provenance_rows_expose_sources() -> None:
    provenance = SimpleNamespace(
        source_tables=(
            "paper_accounts",
            "paper_ledger_entries",
        ),
        record_count=3,
        record_ids=(
            "ACC-1",
            "LEDGER-1",
            "LEDGER-2",
        ),
        filters=(
            "account_id=ACC-1",
        ),
        calculation=(
            "Stored cash minus ledger cash."
        ),
    )

    snapshot = SimpleNamespace(
        section_provenance=(
            SimpleNamespace(
                section="reconciliation",
                provenance=provenance,
            ),
        )
    )

    rows = provenance_rows(
        snapshot
    )

    assert (
        rows[0]["section"]
        == "reconciliation"
    )

    assert (
        rows[0]["record_count"]
        == 3
    )

    assert "paper_accounts" in (
        rows[0]["source_tables"]
    )
