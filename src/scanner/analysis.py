"""Adapter from validated market data to the Trading Expert."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Mapping

import numpy as np
import pandas as pd

from src.analysis import (
    AnalysisSnapshot,
    IndicatorSnapshot,
    build_trading_expert_report,
)
from src.data import MarketSnapshot

from .models import ScannerAnalysisOutcome


def _json_safe(
    value: object,
) -> object:
    if is_dataclass(value):
        return _json_safe(asdict(value))

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        (str, int, float, bool, type(None)),
    ):
        return value

    return str(value)


def _aware_timestamp(
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


def build_scanner_analysis_snapshot(
    snapshot: MarketSnapshot,
) -> tuple[
    AnalysisSnapshot,
    pd.DataFrame,
]:
    """Build the canonical model using approved default indicators."""

    history = snapshot.history.loc[
        :,
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ],
    ].copy()

    history = history.sort_index()

    for column in history.columns:
        history[column] = pd.to_numeric(
            history[column],
            errors="coerce",
        )

    history = history.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )

    if len(history) < 200:
        raise ValueError(
            "Scanner analysis requires at least "
            "200 completed sessions."
        )

    history["MA20"] = (
        history["Close"]
        .rolling(20)
        .mean()
    )

    history["MA50"] = (
        history["Close"]
        .rolling(50)
        .mean()
    )

    history["MA200"] = (
        history["Close"]
        .rolling(200)
        .mean()
    )

    delta = history["Close"].diff()

    gain = (
        delta.clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta.clip(upper=0)
        .rolling(14)
        .mean()
    )

    relative_strength = gain / loss

    history["RSI"] = (
        100
        - (
            100
            / (1 + relative_strength)
        )
    )

    history["EMAf"] = (
        history["Close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    history["EMAs"] = (
        history["Close"]
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    history["MACD"] = (
        history["EMAf"]
        - history["EMAs"]
    )

    history["MACDs"] = (
        history["MACD"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    history["MACD_h"] = (
        history["MACD"]
        - history["MACDs"]
    )

    history["BBm"] = (
        history["Close"]
        .rolling(20)
        .mean()
    )

    history["BBstd"] = (
        history["Close"]
        .rolling(20)
        .std()
    )

    history["BBu"] = (
        history["BBm"]
        + 2 * history["BBstd"]
    )

    history["BBl"] = (
        history["BBm"]
        - 2 * history["BBstd"]
    )

    history["BBpctB"] = (
        (
            history["Close"]
            - history["BBl"]
        )
        / (
            history["BBu"]
            - history["BBl"]
        )
    )

    true_range = pd.concat(
        [
            history["High"]
            - history["Low"],
            (
                history["High"]
                - history["Close"].shift()
            ).abs(),
            (
                history["Low"]
                - history["Close"].shift()
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    history["ATR"] = (
        true_range
        .rolling(14)
        .mean()
    )

    history["OBV"] = (
        np.sign(
            history["Close"].diff()
        )
        * history["Volume"]
    ).fillna(0).cumsum()

    required_indicators = [
        "MA20",
        "MA50",
        "MA200",
        "RSI",
        "MACD",
        "MACDs",
        "MACD_h",
        "BBpctB",
        "ATR",
        "OBV",
    ]

    usable = history.dropna(
        subset=required_indicators
    )

    if usable.empty:
        raise ValueError(
            "Required scanner indicators could "
            "not be calculated."
        )

    latest = usable.iloc[-1]
    latest_index = usable.index[-1]

    cutoff = (
        pd.Timestamp(latest_index)
        - pd.Timedelta(days=90)
    )

    recent = history.loc[
        history.index >= cutoff
    ]

    if recent.empty:
        recent = history.tail(63)

    support = float(
        np.percentile(
            recent["Low"],
            10,
        )
    )

    resistance = float(
        np.percentile(
            recent["High"],
            90,
        )
    )

    metadata = snapshot.metadata or {}

    analysis = AnalysisSnapshot(
        symbol=snapshot.symbol,
        display_name=str(
            metadata.get("shortName")
            or metadata.get("longName")
            or snapshot.symbol
        ),
        fetched_at_utc=(
            snapshot.fetched_at_utc
        ),
        history_rows=len(history),
        indicators=IndicatorSnapshot(
            as_of=_aware_timestamp(
                latest_index
            ),
            close=float(latest["Close"]),
            volume=float(latest["Volume"]),
            ma20=float(latest["MA20"]),
            ma50=float(latest["MA50"]),
            ma200=float(latest["MA200"]),
            rsi=float(latest["RSI"]),
            macd=float(latest["MACD"]),
            macd_signal=float(
                latest["MACDs"]
            ),
            macd_histogram=float(
                latest["MACD_h"]
            ),
            bollinger_percent_b=float(
                latest["BBpctB"]
            ),
            atr=float(latest["ATR"]),
            obv=float(latest["OBV"]),
            support=support,
            resistance=resistance,
        ),
        quote_type=str(
            metadata.get("quoteType")
            or "UNKNOWN"
        ),
        currency=str(
            metadata.get("currency")
            or "UNKNOWN"
        ),
        exchange=str(
            metadata.get("exchange")
            or metadata.get(
                "fullExchangeName"
            )
            or "UNKNOWN"
        ),
        warnings=tuple(snapshot.warnings),
    )

    return analysis, history


def run_deterministic_scanner_analysis(
    snapshot: MarketSnapshot,
) -> ScannerAnalysisOutcome:
    """Run the existing Trading Expert without recalculation elsewhere."""

    analysis, history = (
        build_scanner_analysis_snapshot(
            snapshot
        )
    )

    report = build_trading_expert_report(
        analysis,
        history,
        snapshot.metadata,
    )

    recommendation = (
        report.risk_decision.recommendation
    )

    order = report.risk_decision.order

    raw_regime = getattr(
        report.regime,
        "regime",
        "UNKNOWN",
    )

    market_regime = (
        raw_regime.value
        if isinstance(raw_regime, Enum)
        else str(raw_regime)
    )

    regime_confidence = float(
        getattr(
            report.regime,
            "confidence",
            0.0,
        )
    )

    evidence = tuple(
        _json_safe(item)
        for item in recommendation.evidence
    )

    generated_at = (
        order.created_at
        if order is not None
        else analysis.indicators.as_of
    )

    return ScannerAnalysisOutcome(
        symbol=analysis.symbol,
        generated_at=generated_at,
        strategy=recommendation.strategy,
        recommendation=(
            recommendation.signal
        ),
        score=recommendation.score,
        confidence=(
            recommendation.confidence
        ),
        market_regime=market_regime,
        regime_confidence=(
            regime_confidence
        ),
        order=order,
        risk_vetoes=tuple(
            report.risk_decision.risk_vetoes
        ),
        evidence=tuple(
            value
            if isinstance(value, Mapping)
            else {"value": value}
            for value in evidence
        ),
        warnings=tuple(
            analysis.warnings
        ),
    )
