"""Derive horizon validation reports from costed walk-forward observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from math import isclose, isfinite
from typing import Any


_VERSIONS = {
    "swing": "p4.3-swing-v1",
    "medium_term": "p4.3-medium-term-v1",
}


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required.")
    return value.strip()


def _time(value: object, label: str) -> datetime:
    text = _text(value, label)
    try:
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be a timezone-aware ISO timestamp.") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{label} must be a timezone-aware ISO timestamp.")
    return result


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}.")
    return value


def _parameters(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty object.")
    result: dict[str, object] = {}
    for key, item in value.items():
        name = _text(key, f"{label} parameter name")
        if not isinstance(item, (str, int, float, bool, type(None))):
            raise ValueError(f"{label}.{name} has an unsupported value.")
        result[name] = item
    return dict(sorted(result.items()))


def _stability(selections: Sequence[Mapping[str, object]]) -> float:
    names = tuple(selections[0])
    if any(tuple(item) != names for item in selections):
        raise ValueError("selected parameter keys must be identical across folds.")
    if len(selections) == 1:
        return 1.0
    scores = []
    for name in names:
        changes = sum(
            current[name] != previous[name]
            for previous, current in zip(selections, selections[1:])
        )
        scores.append(1.0 - changes / (len(selections) - 1))
    return sum(scores) / len(scores)


def build_validation_report(
    observation: Mapping[str, object],
    *,
    approved_cost_model_id: str,
) -> dict[str, Any]:
    """Return a fail-closed report compatible with P4 horizon evidence."""

    if observation.get("schema_version") != 1:
        raise ValueError("observation schema_version must be 1.")
    horizon = _text(observation.get("horizon"), "horizon").lower()
    if horizon not in _VERSIONS:
        raise ValueError("horizon must be swing or medium_term.")
    if observation.get("strategy_version") != _VERSIONS[horizon]:
        raise ValueError(f"strategy_version must be {_VERSIONS[horizon]!r}.")
    generated_at = _time(observation.get("generated_at"), "generated_at")
    dataset_id = _text(observation.get("dataset_id"), "dataset_id")
    if not dataset_id.startswith("sha256:"):
        raise ValueError("dataset_id must be a SHA-256 identifier.")
    cost_model_id = _text(observation.get("cost_model_id"), "cost_model_id")
    approved_cost = _text(approved_cost_model_id, "approved_cost_model_id")
    if cost_model_id != approved_cost:
        raise ValueError("cost_model_id does not match approved product policy.")
    if observation.get("costs_included") is not True:
        raise ValueError("costs_included must be true.")

    requirements = observation.get("requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("requirements must be an object.")
    minimum_trades = _integer(
        requirements.get("minimum_trade_count"),
        "minimum_trade_count", minimum=1,
    )
    minimum_stability = _number(
        requirements.get("minimum_parameter_stability"),
        "minimum_parameter_stability",
    )
    minimum_expectancy = _number(
        requirements.get("minimum_net_expectancy"),
        "minimum_net_expectancy",
    )
    if not 0 <= minimum_stability <= 1:
        raise ValueError("minimum_parameter_stability must be between 0 and 1.")

    folds = observation.get("folds")
    if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)):
        raise ValueError("folds must be an array.")
    if len(folds) < 2:
        raise ValueError("at least two walk-forward folds are required.")

    previous_test_end: datetime | None = None
    selections: list[Mapping[str, object]] = []
    total_trades = 0
    total_gross = 0.0
    total_costs = 0.0
    total_net = 0.0
    normalised_folds = []
    for position, raw in enumerate(folds, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"folds[{position}] must be an object.")
        train_start = _time(raw.get("train_start"), f"folds[{position}].train_start")
        train_end = _time(raw.get("train_end"), f"folds[{position}].train_end")
        test_start = _time(raw.get("test_start"), f"folds[{position}].test_start")
        test_end = _time(raw.get("test_end"), f"folds[{position}].test_end")
        if not train_start < train_end < test_start <= test_end:
            raise ValueError(f"folds[{position}] has invalid chronology.")
        if previous_test_end is not None and test_start <= previous_test_end:
            raise ValueError("out-of-sample fold windows must not overlap.")
        previous_test_end = test_end
        trade_count = _integer(raw.get("trade_count"), f"folds[{position}].trade_count")
        gross = _number(raw.get("gross_pnl"), f"folds[{position}].gross_pnl")
        costs = _number(raw.get("execution_costs"), f"folds[{position}].execution_costs")
        net = _number(raw.get("net_pnl"), f"folds[{position}].net_pnl")
        if costs < 0 or not isclose(net, gross - costs, abs_tol=1e-8):
            raise ValueError(f"folds[{position}] net_pnl must equal gross_pnl minus costs.")
        selected = _parameters(
            raw.get("selected_parameters"), f"folds[{position}].selected_parameters"
        )
        selections.append(selected)
        total_trades += trade_count
        total_gross += gross
        total_costs += costs
        total_net += net
        normalised_folds.append({
            "fold": position,
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "trade_count": trade_count,
            "gross_pnl": gross,
            "execution_costs": costs,
            "net_pnl": net,
            "selected_parameters": selected,
        })

    stability = _stability(selections)
    net_expectancy = total_net / total_trades if total_trades else 0.0
    out_of_sample_passed = (
        total_trades >= minimum_trades and net_expectancy >= minimum_expectancy
    )
    return {
        "schema_version": 1,
        "horizon": horizon,
        "strategy_version": _VERSIONS[horizon],
        "generated_at": generated_at.isoformat(),
        "dataset_id": dataset_id,
        "cost_model_id": cost_model_id,
        "validation": {
            "out_of_sample_passed": out_of_sample_passed,
            "walk_forward_passed": True,
            "costs_included": True,
            "observed_trade_count": total_trades,
            "minimum_trade_count": minimum_trades,
            "parameter_stability": stability,
            "minimum_parameter_stability": minimum_stability,
        },
        "observations": {
            "fold_count": len(normalised_folds),
            "gross_pnl": total_gross,
            "execution_costs": total_costs,
            "net_pnl": total_net,
            "net_expectancy": net_expectancy,
            "minimum_net_expectancy": minimum_expectancy,
            "folds": normalised_folds,
        },
    }


__all__ = ["build_validation_report"]
