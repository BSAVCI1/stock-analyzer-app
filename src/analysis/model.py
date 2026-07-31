"""Canonical immutable models used by the trading-expert signal engine.

This module contains no Streamlit or provider-specific code. It defines the
validated objects that later regime classifiers, strategies, scoring and risk
management components will consume and produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Iterable


class Signal(str, Enum):
    """Final or strategy-level recommendation."""

    BUY = "BUY"
    WATCH = "WATCH"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    SELL = "SELL"


class EvidenceDirection(str, Enum):
    """Directional meaning of one evidence item."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc

    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number.")

    return result


def _positive_number(name: str, value: object) -> float:
    result = _finite_number(name, value)

    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return result


def _non_negative_number(name: str, value: object) -> float:
    result = _finite_number(name, value)

    if result < 0:
        raise ValueError(f"{name} cannot be negative.")

    return result


def _aware_datetime(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")

    return value


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    """Immutable technical-indicator values for one completed session."""

    as_of: datetime
    close: float
    volume: float

    ma20: float
    ma50: float
    ma200: float

    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float

    bollinger_percent_b: float
    atr: float
    obv: float

    support: float
    resistance: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "as_of",
            _aware_datetime("as_of", self.as_of),
        )

        for name in (
            "close",
            "ma20",
            "ma50",
            "ma200",
            "support",
            "resistance",
        ):
            object.__setattr__(
                self,
                name,
                _positive_number(name, getattr(self, name)),
            )

        for name in ("volume", "atr"):
            object.__setattr__(
                self,
                name,
                _non_negative_number(name, getattr(self, name)),
            )

        for name in (
            "macd",
            "macd_signal",
            "macd_histogram",
            "bollinger_percent_b",
            "obv",
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(name, getattr(self, name)),
            )

        rsi = _finite_number("rsi", self.rsi)

        if not 0 <= rsi <= 100:
            raise ValueError("rsi must be between 0 and 100.")

        object.__setattr__(self, "rsi", rsi)

        if self.support > self.resistance:
            raise ValueError(
                "support cannot be greater than resistance."
            )

        expected_histogram = self.macd - self.macd_signal
        tolerance = max(
            1e-9,
            abs(expected_histogram) * 1e-7,
        )

        if abs(self.macd_histogram - expected_histogram) > tolerance:
            raise ValueError(
                "macd_histogram must equal macd minus macd_signal."
            )


@dataclass(frozen=True, slots=True)
class AnalysisSnapshot:
    """Canonical instrument and indicator state used by every strategy."""

    symbol: str
    display_name: str
    fetched_at_utc: datetime
    history_rows: int
    indicators: IndicatorSnapshot

    quote_type: str = "UNKNOWN"
    currency: str = "UNKNOWN"
    exchange: str = "UNKNOWN"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        symbol = _required_text("symbol", self.symbol).upper()
        display_name = _required_text(
            "display_name",
            self.display_name,
        )

        fetched_at = _aware_datetime(
            "fetched_at_utc",
            self.fetched_at_utc,
        ).astimezone(timezone.utc)

        if isinstance(self.history_rows, bool):
            raise ValueError("history_rows must be an integer.")

        try:
            history_rows = int(self.history_rows)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "history_rows must be an integer."
            ) from exc

        if history_rows < 200:
            raise ValueError(
                "history_rows must contain at least 200 sessions."
            )

        if not isinstance(self.indicators, IndicatorSnapshot):
            raise ValueError(
                "indicators must be an IndicatorSnapshot."
            )

        indicator_time_utc = self.indicators.as_of.astimezone(
            timezone.utc
        )

        if indicator_time_utc > fetched_at:
            raise ValueError(
                "indicator as_of cannot be later than fetched_at_utc."
            )

        warnings = tuple(
            warning.strip()
            for warning in self.warnings
            if isinstance(warning, str) and warning.strip()
        )

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "fetched_at_utc", fetched_at)
        object.__setattr__(self, "history_rows", history_rows)
        object.__setattr__(
            self,
            "quote_type",
            str(self.quote_type or "UNKNOWN").strip().upper(),
        )
        object.__setattr__(
            self,
            "currency",
            str(self.currency or "UNKNOWN").strip().upper(),
        )
        object.__setattr__(
            self,
            "exchange",
            str(self.exchange or "UNKNOWN").strip().upper(),
        )
        object.__setattr__(self, "warnings", warnings)

    @property
    def latest_close(self) -> float:
        return self.indicators.close

    @property
    def as_of(self) -> datetime:
        return self.indicators.as_of


@dataclass(frozen=True, slots=True)
class Evidence:
    """One traceable fact supporting or opposing a strategy."""

    code: str
    message: str
    direction: EvidenceDirection
    strength: float
    observed_value: float | str | None = None

    def __post_init__(self) -> None:
        code = _required_text("code", self.code).upper()
        message = _required_text("message", self.message)

        if not isinstance(self.direction, EvidenceDirection):
            raise ValueError(
                "direction must be an EvidenceDirection."
            )

        strength = _finite_number("strength", self.strength)

        if not 0 <= strength <= 1:
            raise ValueError(
                "strength must be between 0 and 1."
            )

        observed_value = self.observed_value

        if isinstance(observed_value, bool):
            raise ValueError(
                "observed_value cannot be a boolean."
            )

        if isinstance(observed_value, (int, float)):
            observed_value = _finite_number(
                "observed_value",
                observed_value,
            )
        elif isinstance(observed_value, str):
            observed_value = observed_value.strip()

            if not observed_value:
                observed_value = None
        elif observed_value is not None:
            raise ValueError(
                "observed_value must be a number, string or None."
            )

        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "strength", strength)
        object.__setattr__(
            self,
            "observed_value",
            observed_value,
        )


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """Validated output produced by one deterministic strategy."""

    strategy: str
    signal: Signal
    score: float
    confidence: float
    evidence: tuple[Evidence, ...]

    vetoed: bool = False
    veto_reason: str | None = None

    def __post_init__(self) -> None:
        strategy = _required_text("strategy", self.strategy)

        if not isinstance(self.signal, Signal):
            raise ValueError("signal must be a Signal enum value.")

        score = _finite_number("score", self.score)

        if not -100 <= score <= 100:
            raise ValueError(
                "score must be between -100 and 100."
            )

        confidence = _finite_number(
            "confidence",
            self.confidence,
        )

        if not 0 <= confidence <= 1:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        evidence = tuple(self.evidence)

        if not evidence:
            raise ValueError(
                "strategy results require at least one evidence item."
            )

        if not all(
            isinstance(item, Evidence)
            for item in evidence
        ):
            raise ValueError(
                "evidence must contain only Evidence objects."
            )

        if self.signal is Signal.BUY and score <= 0:
            raise ValueError(
                "BUY requires a positive score."
            )

        if self.signal in {Signal.REDUCE, Signal.SELL} and score >= 0:
            raise ValueError(
                f"{self.signal.value} requires a negative score."
            )

        veto_reason = (
            self.veto_reason.strip()
            if isinstance(self.veto_reason, str)
            else None
        )

        if self.vetoed:
            if not veto_reason:
                raise ValueError(
                    "A vetoed result requires a veto_reason."
                )

            if self.signal in {Signal.BUY, Signal.SELL}:
                raise ValueError(
                    "A vetoed result cannot issue BUY or SELL."
                )
        elif veto_reason:
            raise ValueError(
                "veto_reason is only allowed when vetoed is true."
            )

        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "veto_reason", veto_reason)
