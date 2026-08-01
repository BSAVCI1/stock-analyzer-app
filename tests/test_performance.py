from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal
from math import sqrt
from statistics import stdev

import pandas as pd
import pytest

from src.backtest import (
    ClosedTradeRecord,
    ExecutionCostModel,
    ExitReason,
    PositionSide,
    SettledTrade,
    calculate_max_drawdown,
    calculate_performance,
    calculate_sharpe_ratio,
    settle_trade,
)


T0 = datetime(
    2026,
    1,
    1,
    tzinfo=timezone.utc,
)


def make_trade(
    trade_id: str,
    *,
    opened_day: int,
    closed_day: int,
    entry_price: float,
    exit_price: float,
    quantity: float = 10.0,
) -> ClosedTradeRecord:
    return ClosedTradeRecord(
        trade_id=trade_id,
        position_id=f"POS-{trade_id}",
        order_id=f"ORD-{trade_id}",
        fill_id=f"FILL-{trade_id}",
        signal_id=f"SIG-{trade_id}",
        symbol="TEST",
        side=PositionSide.LONG,
        opened_at=(
            T0
            + timedelta(days=opened_day)
        ),
        closed_at=(
            T0
            + timedelta(days=closed_day)
        ),
        expires_at=(
            T0
            + timedelta(days=30)
        ),
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        stop_price=50.0,
        targets=(250.0, 300.0),
        exit_reason=ExitReason.MANUAL,
    )


def make_history(
    prices: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": prices},
        index=pd.date_range(
            T0,
            periods=len(prices),
            freq="D",
            tz="UTC",
        ),
    )


def make_settled_trade(
    trade: ClosedTradeRecord,
    opening_balance: Decimal,
) -> SettledTrade:
    settlement = settle_trade(
        trade,
        opening_balance,
        costs=ExecutionCostModel(),
    )

    return SettledTrade(
        trade=trade,
        settlement=settlement,
    )


def test_complete_metrics_match_hand_calculation() -> None:
    first = make_settled_trade(
        make_trade(
            "001",
            opened_day=1,
            closed_day=3,
            entry_price=100.0,
            exit_price=110.0,
        ),
        Decimal("10000"),
    )

    second = make_settled_trade(
        make_trade(
            "002",
            opened_day=5,
            closed_day=7,
            entry_price=100.0,
            exit_price=95.0,
        ),
        first.settlement.ending_balance,
    )

    report = calculate_performance(
        (first, second),
        starting_balance=Decimal("10000"),
        period_start=T0,
        period_end=T0 + timedelta(days=10),
        market_history=make_history(
            [
                100, 101, 102, 103, 104, 105,
                106, 107, 108, 109, 110,
            ]
        ),
    )

    assert report.ending_balance == Decimal(
        "10050.00000000"
    )

    assert report.total_net_pnl == Decimal(
        "50.00000000"
    )

    assert report.total_return == pytest.approx(
        0.005
    )

    assert report.annualised_return == pytest.approx(
        (1.005 ** (365 / 10)) - 1
    )

    assert report.trade_count == 2
    assert report.winning_trades == 1
    assert report.losing_trades == 1
    assert report.breakeven_trades == 0
    assert report.win_rate == pytest.approx(0.5)

    assert report.gross_profit == Decimal(
        "100.00000000"
    )

    assert report.gross_loss == Decimal(
        "50.00000000"
    )

    assert report.profit_factor == pytest.approx(
        2.0
    )

    assert report.max_drawdown == pytest.approx(
        50 / 10100
    )

    assert report.exposure == pytest.approx(
        0.4
    )

    assert (
        report.average_holding_period_days
        == pytest.approx(2.0)
    )

    assert (
        report.benchmark.total_return
        == pytest.approx(0.10)
    )

    assert (
        report.benchmark.strategy_excess_return
        == pytest.approx(-0.095)
    )


def test_max_drawdown_matches_peak_to_trough_fixture() -> None:
    assert calculate_max_drawdown(
        [100, 120, 90, 108]
    ) == pytest.approx(0.25)


def test_sharpe_matches_hand_calculated_fixture() -> None:
    returns = [0.01, 0.02, -0.01]

    expected = (
        sqrt(252)
        * (
            sum(returns)
            / len(returns)
        )
        / stdev(returns)
    )

    assert calculate_sharpe_ratio(
        returns
    ) == pytest.approx(expected)


def test_overlapping_trades_do_not_exceed_full_exposure() -> None:
    first = make_settled_trade(
        make_trade(
            "001",
            opened_day=1,
            closed_day=5,
            entry_price=100,
            exit_price=101,
        ),
        Decimal("10000"),
    )

    second = make_settled_trade(
        make_trade(
            "002",
            opened_day=3,
            closed_day=7,
            entry_price=100,
            exit_price=101,
        ),
        first.settlement.ending_balance,
    )

    report = calculate_performance(
        (first, second),
        starting_balance=Decimal("10000"),
        period_start=T0,
        period_end=T0 + timedelta(days=10),
        market_history=make_history(
            [100] * 11
        ),
    )

    # Union of day 1–5 and day 3–7 is day 1–7.
    assert report.exposure == pytest.approx(
        0.6
    )


def test_no_trades_produces_flat_strategy_metrics() -> None:
    report = calculate_performance(
        (),
        starting_balance=Decimal("10000"),
        period_start=T0,
        period_end=T0 + timedelta(days=10),
        market_history=make_history(
            [100] * 11
        ),
    )

    assert report.trade_count == 0
    assert report.total_return == 0
    assert report.annualised_return == 0
    assert report.max_drawdown == 0
    assert report.win_rate == 0
    assert report.profit_factor is None
    assert report.exposure == 0
    assert report.average_holding_period_days == 0
    assert report.sharpe_ratio == 0


def test_settlement_sequence_must_reconcile() -> None:
    first = make_settled_trade(
        make_trade(
            "001",
            opened_day=1,
            closed_day=3,
            entry_price=100,
            exit_price=110,
        ),
        Decimal("10000"),
    )

    second = make_settled_trade(
        make_trade(
            "002",
            opened_day=4,
            closed_day=6,
            entry_price=100,
            exit_price=105,
        ),
        Decimal("9999"),
    )

    with pytest.raises(ValueError):
        calculate_performance(
            (first, second),
            starting_balance=Decimal("10000"),
            period_start=T0,
            period_end=T0 + timedelta(days=10),
            market_history=make_history(
                [100] * 11
            ),
        )


def test_duplicate_trade_ids_are_rejected() -> None:
    first = make_settled_trade(
        make_trade(
            "001",
            opened_day=1,
            closed_day=3,
            entry_price=100,
            exit_price=101,
        ),
        Decimal("10000"),
    )

    second = make_settled_trade(
        make_trade(
            "001",
            opened_day=4,
            closed_day=6,
            entry_price=100,
            exit_price=101,
        ),
        first.settlement.ending_balance,
    )

    with pytest.raises(ValueError):
        calculate_performance(
            (first, second),
            starting_balance=Decimal("10000"),
            period_start=T0,
            period_end=T0 + timedelta(days=10),
            market_history=make_history(
                [100] * 11
            ),
        )


def test_trade_outside_test_window_is_rejected() -> None:
    trade = make_settled_trade(
        make_trade(
            "001",
            opened_day=1,
            closed_day=12,
            entry_price=100,
            exit_price=101,
        ),
        Decimal("10000"),
    )

    with pytest.raises(ValueError):
        calculate_performance(
            (trade,),
            starting_balance=Decimal("10000"),
            period_start=T0,
            period_end=T0 + timedelta(days=10),
            market_history=make_history(
                [100] * 11
            ),
        )


def test_invalid_benchmark_prices_are_rejected() -> None:
    with pytest.raises(ValueError):
        calculate_performance(
            (),
            starting_balance=Decimal("10000"),
            period_start=T0,
            period_end=T0 + timedelta(days=2),
            market_history=make_history(
                [100, 0, 101]
            ),
        )
