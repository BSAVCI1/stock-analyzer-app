"""Strategy-horizon provenance shared across project layers."""

from __future__ import annotations

from enum import Enum


class StrategyHorizon(str, Enum):
    """Approved non-intraday strategy horizons."""

    SWING = "SWING"
    MEDIUM_TERM = "MEDIUM_TERM"


def coerce_strategy_horizon(
    value: object | None,
) -> StrategyHorizon | None:
    """Normalize and validate optional horizon provenance."""

    if value is None:
        return None

    if isinstance(
        value,
        StrategyHorizon,
    ):
        return value

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            "strategy_horizon must be "
            "SWING, MEDIUM_TERM or None."
        )

    normalized = (
        value.strip()
        .upper()
        .replace("-", "_")
    )

    try:
        return StrategyHorizon(
            normalized
        )
    except ValueError as exc:
        raise ValueError(
            "strategy_horizon must be "
            "SWING or MEDIUM_TERM."
        ) from exc


def strategy_horizon_value(
    value: object | None,
) -> str | None:
    """Return canonical storage representation."""

    horizon = coerce_strategy_horizon(
        value
    )

    return (
        horizon.value
        if horizon is not None
        else None
    )


def normalise_strategy_version(
    value: object | None,
) -> str | None:
    """Normalize optional strategy-version provenance."""

    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            "strategy_version must be "
            "a string or None."
        )

    result = value.strip()

    if not result:
        raise ValueError(
            "strategy_version cannot be blank."
        )

    return result
