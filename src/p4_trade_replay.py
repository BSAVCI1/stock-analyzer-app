"""Chronological, cost-aware replay for the approved trend-pullback strategy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from src.analysis import Signal, evaluate_trend_pullback
from src.analysis.regime import classify_history
from src.analysis.trend_pullback import TrendPullbackThresholds
from src.costs import (
    IBKRPricingPlan,
    IBKRTradeSide,
    calculate_us_stock_reference_fees,
)
from src.data import MarketSnapshot
from src.scanner import build_scanner_analysis_snapshot


_MAX_HOLDING = {"swing": 20, "medium_term": 65}


@dataclass(frozen=True, slots=True)
class ReplaySignal:
    actionable: bool
    atr: float


SignalEvaluator = Callable[
    [pd.DataFrame, str, Mapping[str, object]], ReplaySignal
]
FeeEstimator = Callable[[float, float, IBKRTradeSide], float]


def _default_signal(
    history: pd.DataFrame,
    symbol: str,
    parameters: Mapping[str, object],
) -> ReplaySignal:
    snapshot = MarketSnapshot(
        symbol=symbol,
        history=history,
        metadata={"symbol": symbol, "quoteType": "EQUITY", "currency": "USD"},
        fetched_at_utc=history.index[-1].to_pydatetime(),
        warnings=(),
    )
    analysis, clean = build_scanner_analysis_snapshot(snapshot)
    result = evaluate_trend_pullback(
        analysis,
        classify_history(clean),
        TrendPullbackThresholds(**dict(parameters)),
    )
    return ReplaySignal(
        actionable=result.signal is Signal.BUY and not result.vetoed,
        atr=analysis.indicators.atr,
    )


def _default_fee(quantity: float, value: float, side: IBKRTradeSide) -> float:
    estimate = calculate_us_stock_reference_fees(
        quantity=quantity,
        trade_value_usd=value,
        pricing_plan=IBKRPricingPlan.FIXED,
        side=side,
        fractional=True,
    )
    if not estimate.complete:
        raise ValueError("IBKR US stock cost estimate is incomplete.")
    return float(estimate.total_known_cost)


def _frame(rows: object, symbol: str) -> pd.DataFrame:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{symbol} rows must be a non-empty array.")
    frame = pd.DataFrame(rows).rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    required = ["at", "Open", "High", "Low", "Close", "Volume"]
    if any(column not in frame for column in required):
        raise ValueError(f"{symbol} rows are missing required fields.")
    frame.index = pd.to_datetime(frame.pop("at"), utc=True, errors="coerce")
    if frame.index.isna().any() or frame.index.duplicated().any():
        raise ValueError(f"{symbol} rows contain invalid timestamps.")
    frame = frame.sort_index()
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.isna().any().any():
        raise ValueError(f"{symbol} rows contain invalid numeric values.")
    return frame


def _fx_series(dataset: Mapping[str, object]) -> pd.Series:
    fx = dataset.get("fx")
    if not isinstance(fx, Mapping) or not isinstance(fx.get("rates"), list):
        raise ValueError("dataset requires historical FX rates.")
    rates = pd.DataFrame(fx["rates"])
    if rates.empty or "at" not in rates or "rate" not in rates:
        raise ValueError("dataset FX rates are empty or malformed.")
    index = pd.to_datetime(rates["at"], utc=True, errors="coerce")
    values = pd.to_numeric(rates["rate"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=index).sort_index()
    if series.index.isna().any() or series.index.duplicated().any():
        raise ValueError("dataset FX timestamps are invalid or duplicated.")
    if series.isna().any() or (series <= 0).any():
        raise ValueError("dataset FX rates must be positive numbers.")
    return series


def _rate_at(rates: pd.Series, at: pd.Timestamp) -> float:
    eligible = rates.loc[rates.index <= at]
    if eligible.empty:
        raise ValueError(f"No historical FX rate exists at or before {at.isoformat()}.")
    return float(eligible.iloc[-1])


def replay_trend_pullback(
    dataset: Mapping[str, object],
    *,
    parameters: Mapping[str, object],
    test_start: datetime,
    test_end: datetime,
    target_order_value_usd: float = 100.0,
    signal_evaluator: SignalEvaluator = _default_signal,
    fee_estimator: FeeEstimator = _default_fee,
) -> dict[str, Any]:
    """Replay one out-of-sample window without looking beyond its end."""

    if dataset.get("schema_version") != 2:
        raise ValueError("trade replay requires validation dataset schema_version 2.")
    horizon = str(dataset.get("horizon", "")).strip().lower()
    if horizon not in _MAX_HOLDING:
        raise ValueError("dataset horizon must be swing or medium_term.")
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("test_start and test_end must be ordered aware datetimes.")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    target_value = float(target_order_value_usd)
    if target_value <= 0:
        raise ValueError("target_order_value_usd must be positive.")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("parameters must be a non-empty object.")
    instruments = dataset.get("instruments")
    if not isinstance(instruments, list) or not instruments:
        raise ValueError("dataset instruments must be a non-empty array.")
    rates = _fx_series(dataset)
    trades: list[dict[str, object]] = []

    for item in instruments:
        if not isinstance(item, Mapping):
            raise ValueError("dataset instrument must be an object.")
        symbol = str(item.get("symbol", "")).strip().upper()
        frame = _frame(item.get("rows"), symbol)
        index = frame.index
        position = max(199, int(index.searchsorted(start)))
        while position < len(frame) - 1 and index[position] <= end:
            history = frame.iloc[:position + 1].copy()
            decision = signal_evaluator(history, symbol, parameters)
            if not isinstance(decision, ReplaySignal):
                raise ValueError("signal_evaluator must return ReplaySignal.")
            if not decision.actionable:
                position += 1
                continue
            if decision.atr <= 0:
                raise ValueError("actionable signal ATR must be positive.")
            entry_position = position + 1
            if index[entry_position] > end:
                break
            entry_at = index[entry_position]
            entry = float(frame["Open"].iloc[entry_position])
            quantity = target_value / entry
            stop = entry - 1.5 * decision.atr
            target = entry + 3.0 * decision.atr
            final_position = min(
                entry_position + _MAX_HOLDING[horizon],
                int(index.searchsorted(end, side="right")) - 1,
                len(frame) - 1,
            )
            exit_position = final_position
            exit_price = float(frame["Close"].iloc[final_position])
            reason = "TIME_EXIT"
            for candidate_position in range(entry_position, final_position + 1):
                low = float(frame["Low"].iloc[candidate_position])
                high = float(frame["High"].iloc[candidate_position])
                if low <= stop:
                    exit_position, exit_price, reason = candidate_position, stop, "STOP"
                    break
                if high >= target:
                    exit_position, exit_price, reason = candidate_position, target, "TARGET"
                    break
            exit_at = index[exit_position]
            entry_fee = fee_estimator(
                quantity, quantity * entry, IBKRTradeSide.BUY
            )
            exit_fee = fee_estimator(
                quantity, quantity * exit_price, IBKRTradeSide.SELL
            )
            entry_fx, exit_fx = _rate_at(rates, entry_at), _rate_at(rates, exit_at)
            gross_eur = quantity * exit_price * exit_fx - quantity * entry * entry_fx
            costs_eur = entry_fee * entry_fx + exit_fee * exit_fx
            trades.append({
                "symbol": symbol,
                "signal_at": index[position].isoformat(),
                "entry_at": entry_at.isoformat(),
                "exit_at": exit_at.isoformat(),
                "entry_price_usd": entry,
                "exit_price_usd": exit_price,
                "quantity": quantity,
                "entry_fx_usd_eur": entry_fx,
                "exit_fx_usd_eur": exit_fx,
                "gross_pnl_eur": gross_eur,
                "execution_costs_eur": costs_eur,
                "net_pnl_eur": gross_eur - costs_eur,
                "exit_reason": reason,
            })
            position = exit_position + 1

    return {
        "horizon": horizon,
        "strategy": "trend_pullback",
        "test_start": start.isoformat(),
        "test_end": end.isoformat(),
        "selected_parameters": dict(parameters),
        "trade_count": len(trades),
        "gross_pnl": sum(float(item["gross_pnl_eur"]) for item in trades),
        "execution_costs": sum(
            float(item["execution_costs_eur"]) for item in trades
        ),
        "net_pnl": sum(float(item["net_pnl_eur"]) for item in trades),
        "currency": "EUR",
        "trades": trades,
    }


__all__ = ["ReplaySignal", "replay_trend_pullback"]
