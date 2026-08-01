"""Deterministic scanner data-quality and liquidity filters."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite

import pandas as pd

from src.data import MarketSnapshot

from .models import (
    DataQualityMetrics,
    ScannerThresholds,
)


def _utc_datetime(
    value: object,
) -> datetime:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "UTC"
        )
    else:
        timestamp = timestamp.tz_convert(
            "UTC"
        )

    return timestamp.to_pydatetime()


def evaluate_market_snapshot(
    snapshot: MarketSnapshot,
    *,
    thresholds: ScannerThresholds,
    scan_started_at: datetime,
) -> tuple[
    DataQualityMetrics,
    tuple[str, ...],
]:
    """Calculate traceable quality metrics and rejection reasons."""

    if (
        scan_started_at.tzinfo is None
        or scan_started_at.utcoffset() is None
    ):
        raise ValueError(
            "scan_started_at must be timezone-aware."
        )

    history = snapshot.history

    if not isinstance(
        history,
        pd.DataFrame,
    ) or history.empty:
        raise ValueError(
            "Market snapshot history is empty."
        )

    required = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    missing = sorted(
        required.difference(history.columns)
    )

    if missing:
        raise ValueError(
            "Market history is missing columns: "
            + ", ".join(missing)
            + "."
        )

    clean = history.loc[
        :,
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ],
    ].copy()

    for column in clean.columns:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="coerce",
        )

    clean = clean.dropna(
        subset=["Close", "Volume"]
    )

    if clean.empty:
        raise ValueError(
            "Market history has no usable rows."
        )

    data_as_of = _utc_datetime(
        clean.index[-1]
    )

    scan_utc = scan_started_at.astimezone(
        timezone.utc
    )

    staleness_days = (
        scan_utc.date()
        - data_as_of.date()
    ).days

    lookback = clean.tail(
        thresholds.liquidity_lookback_sessions
    )

    latest_price = float(
        clean["Close"].iloc[-1]
    )

    average_volume = float(
        lookback["Volume"].mean()
    )

    average_dollar_volume = float(
        (
            lookback["Close"]
            * lookback["Volume"]
        ).mean()
    )

    metadata = snapshot.metadata or {}

    quote_type = str(
        metadata.get("quoteType")
        or metadata.get("instrumentType")
        or "UNKNOWN"
    ).strip().upper()

    currency = str(
        metadata.get("currency")
        or "UNKNOWN"
    ).strip().upper()

    exchange = str(
        metadata.get("exchange")
        or metadata.get("fullExchangeName")
        or "UNKNOWN"
    ).strip().upper()

    reasons: list[str] = []

    if len(clean) < thresholds.minimum_history_rows:
        reasons.append(
            f"History contains {len(clean)} rows; "
            f"at least "
            f"{thresholds.minimum_history_rows} "
            "are required."
        )

    if staleness_days < 0:
        reasons.append(
            "Market history is dated after "
            "the scan timestamp."
        )
    elif (
        staleness_days
        > thresholds.maximum_staleness_days
    ):
        reasons.append(
            f"Market history is {staleness_days} "
            "calendar days old; maximum permitted "
            f"is "
            f"{thresholds.maximum_staleness_days}."
        )

    if (
        not isfinite(latest_price)
        or latest_price
        < thresholds.minimum_price
    ):
        reasons.append(
            f"Latest price is below the minimum "
            f"{thresholds.minimum_price:.2f}."
        )

    if (
        not isfinite(average_volume)
        or average_volume
        < thresholds.minimum_average_volume
    ):
        reasons.append(
            "Average volume is below the configured "
            "liquidity minimum."
        )

    if (
        not isfinite(average_dollar_volume)
        or average_dollar_volume
        < thresholds
        .minimum_average_dollar_volume
    ):
        reasons.append(
            "Average dollar volume is below the "
            "configured liquidity minimum."
        )

    if (
        quote_type
        not in thresholds.allowed_quote_types
    ):
        reasons.append(
            f"Quote type {quote_type} is not enabled "
            "for automatic scanning."
        )

    metrics = DataQualityMetrics(
        symbol=snapshot.symbol,
        data_as_of=data_as_of,
        history_rows=len(clean),
        latest_price=latest_price,
        average_volume=average_volume,
        average_dollar_volume=(
            average_dollar_volume
        ),
        quote_type=quote_type,
        currency=currency,
        exchange=exchange,
        staleness_days=staleness_days,
        provider_warnings=tuple(
            snapshot.warnings
        ),
    )

    return metrics, tuple(reasons)
