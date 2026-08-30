"""Independent walk-forward selection over immutable P4 horizon datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
import json
from typing import Any

import pandas as pd

from src.p4_trade_replay import replay_trend_pullback


_WINDOWS = {
    "swing": {"train": 126, "test": 63},
    "medium_term": {"train": 26, "test": 13},
}
ReplayRunner = Callable[..., dict[str, Any]]


def _aware(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware datetime.")
    return value


def _candidates(value: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("parameter_candidates must be an array.")
    result = tuple(dict(sorted(candidate.items())) for candidate in value)
    if not result or any(not candidate for candidate in result):
        raise ValueError("parameter_candidates must contain non-empty objects.")
    keys = tuple(result[0])
    if any(tuple(candidate) != keys for candidate in result):
        raise ValueError("all parameter candidates must use identical keys.")
    canonical = tuple(
        json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        for candidate in result
    )
    if len(set(canonical)) != len(canonical):
        raise ValueError("parameter_candidates contains duplicates.")
    return result


def _timeline(dataset: Mapping[str, object]) -> pd.DatetimeIndex:
    instruments = dataset.get("instruments")
    if not isinstance(instruments, list) or not instruments:
        raise ValueError("dataset instruments must be a non-empty array.")
    first = instruments[0]
    if not isinstance(first, Mapping) or not isinstance(first.get("rows"), list):
        raise ValueError("dataset first instrument rows are malformed.")
    index = pd.to_datetime(
        [row.get("at") for row in first["rows"] if isinstance(row, Mapping)],
        utc=True,
        errors="coerce",
    )
    if index.isna().any() or index.duplicated().any():
        raise ValueError("dataset reference timeline is invalid.")
    return index.sort_values()


def _ranking(result: Mapping[str, object], candidate: Mapping[str, object]) -> tuple:
    net = float(result.get("net_pnl", 0.0))
    trades = int(result.get("trade_count", 0))
    return (
        net,
        trades,
        json.dumps(candidate, sort_keys=True, separators=(",", ":")),
    )


def _diagnostics(result: Mapping[str, object]) -> dict[str, object]:
    trades = result.get("trades")
    expected = int(result.get("trade_count", 0))
    if not isinstance(trades, list) or len(trades) != expected:
        raise ValueError("replay trade details must match trade_count.")
    reasons: Counter[str] = Counter()
    symbols: Counter[str] = Counter()
    winners = losers = breakeven = gross_winners = 0
    gross_profit = gross_loss = total_costs = 0.0
    for position, trade in enumerate(trades, start=1):
        if not isinstance(trade, Mapping):
            raise ValueError(f"trade {position} must be an object.")
        reason = str(trade.get("exit_reason", "")).strip()
        symbol = str(trade.get("symbol", "")).strip().upper()
        if not reason or not symbol:
            raise ValueError(f"trade {position} requires symbol and exit_reason.")
        gross = float(trade.get("gross_pnl_eur"))
        costs = float(trade.get("execution_costs_eur"))
        net = float(trade.get("net_pnl_eur"))
        reasons[reason] += 1
        symbols[symbol] += 1
        total_costs += costs
        if gross > 0:
            gross_winners += 1
            gross_profit += gross
        elif gross < 0:
            gross_loss += gross
        if net > 0:
            winners += 1
        elif net < 0:
            losers += 1
        else:
            breakeven += 1
    raw_signal_rejections = result.get("signal_rejection_counts", {})
    if not isinstance(raw_signal_rejections, Mapping):
        raise ValueError("signal_rejection_counts must be an object.")
    return {
        "winning_trades": winners,
        "losing_trades": losers,
        "breakeven_trades": breakeven,
        "gross_profitable_trades": gross_winners,
        "gross_gains_erased_by_costs": max(0, gross_winners - winners),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "execution_costs": total_costs,
        "economic_rejection_count": int(
            result.get("economic_rejection_count", 0)
        ),
        "risk_geometry_rejection_count": int(
            result.get("risk_geometry_rejection_count", 0)
        ),
        "signal_evaluation_count": int(result.get("signal_evaluation_count", 0)),
        "actionable_signal_count": int(result.get("actionable_signal_count", 0)),
        "signal_rejection_counts": dict(sorted(
            (str(key), int(value))
            for key, value in raw_signal_rejections.items()
        )),
        "exit_reason_counts": dict(sorted(reasons.items())),
        "symbol_counts": dict(sorted(symbols.items())),
    }


def _aggregate_diagnostics(folds: Sequence[Mapping[str, object]]) -> dict[str, object]:
    totals: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    symbols: Counter[str] = Counter()
    signal_rejections: Counter[str] = Counter()
    numeric_keys = (
        "winning_trades", "losing_trades", "breakeven_trades",
        "gross_profitable_trades", "gross_gains_erased_by_costs",
        "gross_profit", "gross_loss", "execution_costs",
        "economic_rejection_count", "risk_geometry_rejection_count",
        "signal_evaluation_count", "actionable_signal_count",
    )
    for fold in folds:
        diagnostics = fold["diagnostics"]
        for key in numeric_keys:
            totals[key] += diagnostics[key]
        reasons.update(diagnostics["exit_reason_counts"])
        symbols.update(diagnostics["symbol_counts"])
        signal_rejections.update(diagnostics["signal_rejection_counts"])
    return {
        **dict(totals),
        "exit_reason_counts": dict(sorted(reasons.items())),
        "symbol_counts": dict(sorted(symbols.items())),
        "signal_rejection_counts": dict(sorted(signal_rejections.items())),
    }


def run_walk_forward_study(
    dataset: Mapping[str, object],
    *,
    parameter_candidates: Sequence[Mapping[str, object]],
    generated_at: datetime,
    cost_model_id: str,
    minimum_trade_count: int = 20,
    minimum_parameter_stability: float = 0.5,
    minimum_net_expectancy: float = 0.0,
    replay_runner: ReplayRunner = replay_trend_pullback,
) -> dict[str, Any]:
    """Select on training data and replay once on each untouched test fold."""

    if dataset.get("schema_version") != 2:
        raise ValueError("walk-forward study requires dataset schema_version 2.")
    horizon = str(dataset.get("horizon", "")).strip().lower()
    if horizon not in _WINDOWS:
        raise ValueError("dataset horizon must be swing or medium_term.")
    strategy_version = str(dataset.get("strategy_version", "")).strip()
    dataset_id = str(dataset.get("dataset_id", "")).strip()
    if not dataset_id.startswith("sha256:"):
        raise ValueError("dataset_id must be a SHA-256 identifier.")
    if not cost_model_id.strip():
        raise ValueError("cost_model_id is required.")
    if minimum_trade_count < 1:
        raise ValueError("minimum_trade_count must be positive.")
    if not 0 <= minimum_parameter_stability <= 1:
        raise ValueError("minimum_parameter_stability must be between 0 and 1.")
    candidates = _candidates(parameter_candidates)
    timeline = _timeline(dataset)
    train_size = _WINDOWS[horizon]["train"]
    test_size = _WINDOWS[horizon]["test"]
    warmup_end = 199
    folds = []
    train_end_position = warmup_end + train_size

    while train_end_position + test_size < len(timeline):
        train_start = timeline[warmup_end]
        train_end = timeline[train_end_position]
        test_start = timeline[train_end_position + 1]
        test_end = timeline[train_end_position + test_size]
        training_results = []
        for candidate in candidates:
            result = replay_runner(
                dataset,
                parameters=candidate,
                test_start=train_start.to_pydatetime(),
                test_end=train_end.to_pydatetime(),
            )
            training_results.append((candidate, result))
        selected, selected_training = max(
            training_results,
            key=lambda item: _ranking(item[1], item[0]),
        )
        test_result = replay_runner(
            dataset,
            parameters=selected,
            test_start=test_start.to_pydatetime(),
            test_end=test_end.to_pydatetime(),
        )
        diagnostics = _diagnostics(test_result)
        folds.append({
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "selected_parameters": selected,
            "training_trade_count": int(selected_training["trade_count"]),
            "training_net_pnl": float(selected_training["net_pnl"]),
            "trade_count": int(test_result["trade_count"]),
            "gross_pnl": float(test_result["gross_pnl"]),
            "execution_costs": float(test_result["execution_costs"]),
            "net_pnl": float(test_result["net_pnl"]),
            "economic_rejection_count": int(
                test_result.get("economic_rejection_count", 0)
            ),
            "risk_geometry_rejection_count": int(
                test_result.get("risk_geometry_rejection_count", 0)
            ),
            "trade_ids": [
                f"{trade['symbol']}:{trade['entry_at']}:{trade['exit_at']}"
                for trade in test_result.get("trades", [])
            ],
            "diagnostics": diagnostics,
        })
        train_end_position += test_size

    if len(folds) < 2:
        raise ValueError("dataset does not contain enough history for two folds.")
    generated = _aware(generated_at, "generated_at")
    return {
        "schema_version": 1,
        "horizon": horizon,
        "strategy_version": strategy_version,
        "generated_at": generated.isoformat(),
        "dataset_id": dataset_id,
        "cost_model_id": cost_model_id.strip(),
        "costs_included": True,
        "requirements": {
            "minimum_trade_count": minimum_trade_count,
            "minimum_parameter_stability": minimum_parameter_stability,
            "minimum_net_expectancy": minimum_net_expectancy,
        },
        "folds": folds,
        "diagnostics": _aggregate_diagnostics(folds),
    }


__all__ = ["run_walk_forward_study"]
