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


def _fractional_eligibility(
    metadata: dict[str, object],
) -> bool | None:
    for key in (
        "fractionalTradingEligible",
        "fractionalable",
        "fractional_eligible",
    ):
        if key not in metadata:
            continue

        value = metadata[key]

        if isinstance(value, bool):
            return value

        if (
            isinstance(value, int)
            and value in {0, 1}
        ):
            return bool(value)

        if isinstance(value, str):
            normalized = (
                value.strip().lower()
            )

            if normalized in {
                "true",
                "yes",
                "eligible",
                "1",
            }:
                return True

            if normalized in {
                "false",
                "no",
                "ineligible",
                "0",
            }:
                return False

        return None

    return None


def _next_event_at(
    metadata: dict[str, object],
) -> tuple[datetime | None, bool]:
    for key in (
        "nextEventAt",
        "earningsTimestamp",
        "earningsTimestampStart",
        "earningsDate",
    ):
        if key not in metadata:
            continue

        value = metadata[key]

        try:
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                timestamp = pd.Timestamp(
                    value,
                    unit="s",
                    tz="UTC",
                )
            else:
                timestamp = pd.Timestamp(
                    value
                )

                if timestamp.tzinfo is None:
                    timestamp = (
                        timestamp.tz_localize(
                            "UTC"
                        )
                    )
                else:
                    timestamp = (
                        timestamp.tz_convert(
                            "UTC"
                        )
                    )

            return (
                timestamp.to_pydatetime(),
                False,
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return None, True

    return None, False


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

    metadata = dict(
        snapshot.metadata or {}
    )

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

    fractional_eligible = (
        _fractional_eligibility(
            metadata
        )
    )
    next_event_at, event_date_invalid = (
        _next_event_at(metadata)
    )

    reasons: list[str] = []
    reason_codes: list[str] = []

    if len(clean) < thresholds.minimum_history_rows:
        reason_codes.append(
            "INSUFFICIENT_HISTORY"
        )
        reasons.append(
            f"History contains {len(clean)} rows; "
            f"at least "
            f"{thresholds.minimum_history_rows} "
            "are required."
        )

    if staleness_days < 0:
        reason_codes.append(
            "FUTURE_DATED_DATA"
        )
        reasons.append(
            "Market history is dated after "
            "the scan timestamp."
        )
    elif (
        staleness_days
        > thresholds.maximum_staleness_days
    ):
        reason_codes.append(
            "STALE_DATA"
        )
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
        reason_codes.append(
            "PRICE_BELOW_MINIMUM"
        )
        reasons.append(
            f"Latest price is below the minimum "
            f"{thresholds.minimum_price:.2f}."
        )

    if (
        not isfinite(average_volume)
        or average_volume
        < thresholds.minimum_average_volume
    ):
        reason_codes.append(
            "VOLUME_BELOW_MINIMUM"
        )
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
        reason_codes.append(
            "DOLLAR_VOLUME_BELOW_MINIMUM"
        )
        reasons.append(
            "Average dollar volume is below the "
            "configured liquidity minimum."
        )

    if (
        quote_type
        not in thresholds.allowed_quote_types
    ):
        reason_codes.append(
            "QUOTE_TYPE_NOT_ALLOWED"
        )
        reasons.append(
            f"Quote type {quote_type} is not enabled "
            "for automatic scanning."
        )

    if (
        latest_price
        > thresholds.maximum_order_value
        and fractional_eligible is not True
    ):
        reason_codes.append(
            "FRACTIONAL_ELIGIBILITY_REQUIRED"
        )
        reasons.append(
            "Latest price exceeds the maximum "
            "order value and verified fractional "
            "eligibility is required."
        )

    if event_date_invalid:
        reason_codes.append(
            "EVENT_DATE_INVALID"
        )
        reasons.append(
            "Known event metadata contains an "
            "invalid date."
        )
    elif next_event_at is not None:
        event_days = (
            next_event_at.date()
            - scan_utc.date()
        ).days

        if (
            0 <= event_days
            <= thresholds.event_blackout_days
        ):
            reason_codes.append(
                "EVENT_RISK_BLACKOUT"
            )
            reasons.append(
                "A known market event occurs "
                f"within {event_days} days; "
                "new candidates are blocked "
                "during the configured blackout."
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
        fractional_eligible=(
            fractional_eligible
        ),
        next_event_at=next_event_at,
        filter_reason_codes=tuple(
            reason_codes
        ),
    )

    return metrics, tuple(reasons)
