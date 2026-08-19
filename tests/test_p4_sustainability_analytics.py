"""P4.10 cost-adjusted and version-safe analytics tests."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.portfolio_dashboard import (
    calculate_breakdowns,
    calculate_performance,
    performance_breakdown_rows,
)
from src.strategy import StrategyHorizon


def _trade(
    trade_id: str,
    *,
    horizon: StrategyHorizon,
    version: str,
    gross_pnl: str,
    fees: str,
    slippage: str,
    net_pnl: str,
    return_pct: float,
):
    return SimpleNamespace(
        trade_id=trade_id,
        signal_id=f"SIG-{trade_id}",
        strategy="trend_pullback",
        strategy_horizon=horizon,
        strategy_version=version,
        symbol="AAPL",
        market_regime="BULLISH",
        gross_pnl=Decimal(gross_pnl),
        fees=Decimal(fees),
        slippage=Decimal(slippage),
        net_pnl=Decimal(net_pnl),
        return_pct=return_pct,
    )


def _trades():
    return (
        _trade(
            "T1",
            horizon=StrategyHorizon.SWING,
            version="swing-v1",
            gross_pnl="20",
            fees="3",
            slippage="2",
            net_pnl="15",
            return_pct=1.5,
        ),
        _trade(
            "T2",
            horizon=StrategyHorizon.SWING,
            version="swing-v2",
            gross_pnl="-5",
            fees="3",
            slippage="2",
            net_pnl="-10",
            return_pct=-1.0,
        ),
        _trade(
            "T3",
            horizon=StrategyHorizon.MEDIUM_TERM,
            version="medium-v1",
            gross_pnl="30",
            fees="3",
            slippage="2",
            net_pnl="25",
            return_pct=2.5,
        ),
    )


def test_headline_performance_exposes_cost_drag() -> None:
    performance = calculate_performance(_trades())

    assert performance.gross_pnl == Decimal("45")
    assert performance.total_fees == Decimal("9")
    assert performance.total_slippage == Decimal("6")
    assert performance.total_costs == Decimal("15")
    assert performance.net_pnl == Decimal("30")
    assert performance.expectancy == Decimal("10")
    assert performance.profit_factor == 4.0
    assert performance.cost_drag_pct == pytest.approx(
        33.3333333333
    )


def test_horizon_and_version_cohorts_never_mix() -> None:
    trades = _trades()
    signals = {
        trade.signal_id: SimpleNamespace(
            threshold_version="threshold-v1"
        )
        for trade in trades
    }
    breakdowns = calculate_breakdowns(
        trades,
        signals,
    )
    cohorts = tuple(
        row
        for row in breakdowns
        if row.dimension == "strategy_cohort"
    )

    assert {row.key for row in cohorts} == {
        "MEDIUM_TERM|medium-v1",
        "SWING|swing-v1",
        "SWING|swing-v2",
    }
    assert all(row.trade_count == 1 for row in cohorts)

    swing_v1 = next(
        row
        for row in cohorts
        if row.key == "SWING|swing-v1"
    )
    assert swing_v1.gross_pnl == Decimal("20")
    assert swing_v1.total_costs == Decimal("5")
    assert swing_v1.net_pnl == Decimal("15")
    assert swing_v1.expectancy == Decimal("15")
    assert swing_v1.profit_factor == float("inf")

    rows = performance_breakdown_rows(
        SimpleNamespace(breakdowns=cohorts)
    )
    assert rows[0]["gross_pnl"]
    assert rows[0]["total_costs"]
    assert rows[0]["net_pnl"]
