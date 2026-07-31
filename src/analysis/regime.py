"""Deterministic market-regime classification.

This module contains no Streamlit or provider-specific code. It converts
validated OHLC history into explicit regime inputs and classifies the market
as bullish, bearish, sideways or high volatility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite

import pandas as pd


class MarketRegime(str, Enum):
    """Mutually exclusive market regimes used by strategy modules."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


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


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")

    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc

    if result < 1:
        raise ValueError(f"{name} must be a positive integer.")

    return result


@dataclass(frozen=True, slots=True)
class RegimeInputs:
    """Normalised numerical inputs consumed by the regime classifier.

    Percentage values use percentage points rather than decimal fractions.
    For example, ``2.5`` means 2.5%.
    """

    short_trend_slope_pct: float
    long_trend_slope_pct: float
    price_vs_ma200_pct: float
    short_vs_long_ma_pct: float
    atr_pct: float

    def __post_init__(self) -> None:
        for name in (
            "short_trend_slope_pct",
            "long_trend_slope_pct",
            "price_vs_ma200_pct",
            "short_vs_long_ma_pct",
            "atr_pct",
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(name, getattr(self, name)),
            )

        if self.atr_pct < 0:
            raise ValueError("atr_pct cannot be negative.")


@dataclass(frozen=True, slots=True)
class RegimeThresholds:
    """Thresholds governing deterministic regime classification."""

    short_slope_pct: float = 0.08
    long_slope_pct: float = 0.03
    price_location_pct: float = 2.0
    moving_average_spread_pct: float = 0.5
    high_volatility_atr_pct: float = 4.0

    def __post_init__(self) -> None:
        for name in (
            "short_slope_pct",
            "long_slope_pct",
            "price_location_pct",
            "moving_average_spread_pct",
            "high_volatility_atr_pct",
        ):
            value = _finite_number(name, getattr(self, name))

            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")

            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class RegimeClassification:
    """Immutable result returned by the regime classifier."""

    regime: MarketRegime
    confidence: float
    bullish_votes: int
    bearish_votes: int
    inputs: RegimeInputs
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.regime, MarketRegime):
            raise ValueError("regime must be a MarketRegime.")

        confidence = _finite_number("confidence", self.confidence)

        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1.")

        for name in ("bullish_votes", "bearish_votes"):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")

            if not 0 <= value <= 4:
                raise ValueError(f"{name} must be between 0 and 4.")

        if not isinstance(self.inputs, RegimeInputs):
            raise ValueError("inputs must be RegimeInputs.")

        reasons = tuple(
            reason.strip()
            for reason in self.reasons
            if isinstance(reason, str) and reason.strip()
        )

        if not reasons:
            raise ValueError(
                "A regime classification requires at least one reason."
            )

        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "reasons", reasons)


def build_regime_inputs(
    history: pd.DataFrame,
    *,
    short_window: int = 20,
    long_window: int = 50,
    trend_lookback: int = 10,
    atr_window: int = 14,
) -> RegimeInputs:
    """Derive normalised regime inputs from validated OHLC history.

    Trend slopes are expressed as average percentage movement per session over
    ``trend_lookback`` sessions. Price location compares the latest close with
    the 200-session moving average. Volatility is ATR divided by latest close.
    """

    if not isinstance(history, pd.DataFrame) or history.empty:
        raise ValueError("history must be a non-empty pandas DataFrame.")

    short_window = _positive_integer("short_window", short_window)
    long_window = _positive_integer("long_window", long_window)
    trend_lookback = _positive_integer(
        "trend_lookback",
        trend_lookback,
    )
    atr_window = _positive_integer("atr_window", atr_window)

    if short_window >= long_window:
        raise ValueError(
            "short_window must be smaller than long_window."
        )

    required_columns = ("High", "Low", "Close")
    missing = [
        column
        for column in required_columns
        if column not in history.columns
    ]

    if missing:
        raise ValueError(
            "history is missing required columns: "
            + ", ".join(missing)
        )

    clean = history.loc[:, list(required_columns)].copy()

    for column in required_columns:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="coerce",
        )

    clean = clean.dropna(subset=list(required_columns))

    minimum_rows = max(
        200,
        long_window + trend_lookback,
        atr_window + 1,
    )

    if len(clean) < minimum_rows:
        raise ValueError(
            f"history requires at least {minimum_rows} usable rows."
        )

    if (clean["Close"] <= 0).any():
        raise ValueError("closing prices must be greater than zero.")

    if (clean["High"] < clean["Low"]).any():
        raise ValueError("history contains High values below Low values.")

    close = clean["Close"]

    short_ma = close.rolling(short_window).mean()
    long_ma = close.rolling(long_window).mean()
    ma200 = close.rolling(200).mean()

    latest_close = float(close.iloc[-1])
    latest_short_ma = float(short_ma.iloc[-1])
    latest_long_ma = float(long_ma.iloc[-1])
    latest_ma200 = float(ma200.iloc[-1])

    previous_short_ma = float(
        short_ma.iloc[-(trend_lookback + 1)]
    )
    previous_long_ma = float(
        long_ma.iloc[-(trend_lookback + 1)]
    )

    for name, value in (
        ("latest_short_ma", latest_short_ma),
        ("latest_long_ma", latest_long_ma),
        ("latest_ma200", latest_ma200),
        ("previous_short_ma", previous_short_ma),
        ("previous_long_ma", previous_long_ma),
    ):
        if not isfinite(value) or value <= 0:
            raise ValueError(
                f"{name} could not be calculated from history."
            )

    short_slope = (
        (
            latest_short_ma / previous_short_ma
            - 1
        )
        * 100
        / trend_lookback
    )

    long_slope = (
        (
            latest_long_ma / previous_long_ma
            - 1
        )
        * 100
        / trend_lookback
    )

    price_vs_ma200 = (
        latest_close / latest_ma200
        - 1
    ) * 100

    short_vs_long_ma = (
        latest_short_ma / latest_long_ma
        - 1
    ) * 100

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            clean["High"] - clean["Low"],
            (clean["High"] - previous_close).abs(),
            (clean["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    latest_atr = float(
        true_range.rolling(atr_window).mean().iloc[-1]
    )

    if not isfinite(latest_atr) or latest_atr < 0:
        raise ValueError("ATR could not be calculated from history.")

    atr_pct = latest_atr / latest_close * 100

    return RegimeInputs(
        short_trend_slope_pct=short_slope,
        long_trend_slope_pct=long_slope,
        price_vs_ma200_pct=price_vs_ma200,
        short_vs_long_ma_pct=short_vs_long_ma,
        atr_pct=atr_pct,
    )


def classify_market_regime(
    inputs: RegimeInputs,
    thresholds: RegimeThresholds | None = None,
) -> RegimeClassification:
    """Classify one set of normalised regime inputs deterministically."""

    if not isinstance(inputs, RegimeInputs):
        raise ValueError("inputs must be RegimeInputs.")

    thresholds = thresholds or RegimeThresholds()

    if not isinstance(thresholds, RegimeThresholds):
        raise ValueError("thresholds must be RegimeThresholds.")

    bullish_checks = (
        inputs.short_trend_slope_pct
        >= thresholds.short_slope_pct,
        inputs.long_trend_slope_pct
        >= thresholds.long_slope_pct,
        inputs.price_vs_ma200_pct
        >= thresholds.price_location_pct,
        inputs.short_vs_long_ma_pct
        >= thresholds.moving_average_spread_pct,
    )

    bearish_checks = (
        inputs.short_trend_slope_pct
        <= -thresholds.short_slope_pct,
        inputs.long_trend_slope_pct
        <= -thresholds.long_slope_pct,
        inputs.price_vs_ma200_pct
        <= -thresholds.price_location_pct,
        inputs.short_vs_long_ma_pct
        <= -thresholds.moving_average_spread_pct,
    )

    bullish_votes = sum(bullish_checks)
    bearish_votes = sum(bearish_checks)

    if inputs.atr_pct >= thresholds.high_volatility_atr_pct:
        excess_ratio = (
            inputs.atr_pct
            - thresholds.high_volatility_atr_pct
        ) / thresholds.high_volatility_atr_pct

        confidence = min(
            1.0,
            0.70 + 0.30 * excess_ratio,
        )

        return RegimeClassification(
            regime=MarketRegime.HIGH_VOLATILITY,
            confidence=round(confidence, 4),
            bullish_votes=bullish_votes,
            bearish_votes=bearish_votes,
            inputs=inputs,
            reasons=(
                (
                    f"ATR is {inputs.atr_pct:.2f}% of price, "
                    f"above the "
                    f"{thresholds.high_volatility_atr_pct:.2f}% "
                    "high-volatility threshold."
                ),
            ),
        )

    if bullish_votes >= 3 and bullish_votes > bearish_votes:
        confidence = min(
            1.0,
            max(
                0.5,
                (
                    bullish_votes
                    - 0.5 * bearish_votes
                ) / 4,
            ),
        )

        return RegimeClassification(
            regime=MarketRegime.BULLISH,
            confidence=round(confidence, 4),
            bullish_votes=bullish_votes,
            bearish_votes=bearish_votes,
            inputs=inputs,
            reasons=(
                (
                    f"{bullish_votes} of 4 directional inputs "
                    "support a bullish regime."
                ),
                (
                    f"Price is {inputs.price_vs_ma200_pct:.2f}% "
                    "relative to MA200."
                ),
                (
                    f"Short and long trend slopes are "
                    f"{inputs.short_trend_slope_pct:.3f}% and "
                    f"{inputs.long_trend_slope_pct:.3f}% "
                    "per session."
                ),
            ),
        )

    if bearish_votes >= 3 and bearish_votes > bullish_votes:
        confidence = min(
            1.0,
            max(
                0.5,
                (
                    bearish_votes
                    - 0.5 * bullish_votes
                ) / 4,
            ),
        )

        return RegimeClassification(
            regime=MarketRegime.BEARISH,
            confidence=round(confidence, 4),
            bullish_votes=bullish_votes,
            bearish_votes=bearish_votes,
            inputs=inputs,
            reasons=(
                (
                    f"{bearish_votes} of 4 directional inputs "
                    "support a bearish regime."
                ),
                (
                    f"Price is {inputs.price_vs_ma200_pct:.2f}% "
                    "relative to MA200."
                ),
                (
                    f"Short and long trend slopes are "
                    f"{inputs.short_trend_slope_pct:.3f}% and "
                    f"{inputs.long_trend_slope_pct:.3f}% "
                    "per session."
                ),
            ),
        )

    confidence = max(
        0.5,
        1 - abs(bullish_votes - bearish_votes) / 4,
    )

    return RegimeClassification(
        regime=MarketRegime.SIDEWAYS,
        confidence=round(confidence, 4),
        bullish_votes=bullish_votes,
        bearish_votes=bearish_votes,
        inputs=inputs,
        reasons=(
            (
                "Directional evidence is mixed or insufficient: "
                f"{bullish_votes} bullish and "
                f"{bearish_votes} bearish votes."
            ),
            (
                f"ATR is {inputs.atr_pct:.2f}% of price and does "
                "not trigger the high-volatility regime."
            ),
        ),
    )


def classify_history(
    history: pd.DataFrame,
    *,
    thresholds: RegimeThresholds | None = None,
    short_window: int = 20,
    long_window: int = 50,
    trend_lookback: int = 10,
    atr_window: int = 14,
) -> RegimeClassification:
    """Build regime inputs from history and classify them."""

    inputs = build_regime_inputs(
        history,
        short_window=short_window,
        long_window=long_window,
        trend_lookback=trend_lookback,
        atr_window=atr_window,
    )

    return classify_market_regime(
        inputs,
        thresholds=thresholds,
    )
