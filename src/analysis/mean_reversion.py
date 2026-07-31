"""Deterministic support-based mean-reversion strategy.

The strategy looks for a completed-session reversal from a defined support
zone. It is enabled only in suitable market regimes and explicitly vetoes
strong bearish trends, high-volatility conditions and material support breaks.

This module contains no Streamlit or provider-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .model import (
    AnalysisSnapshot,
    Evidence,
    EvidenceDirection,
    Signal,
    StrategyResult,
)
from .regime import MarketRegime, RegimeClassification


STRATEGY_NAME = "mean_reversion"


def _finite_number(name: str, value: object) -> float:
    """Return a validated finite floating-point number."""

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
    """Return a validated strictly positive number."""

    result = _finite_number(name, value)

    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return result


@dataclass(frozen=True, slots=True)
class MeanReversionContext:
    """Previous-session context required for reversal confirmation."""

    previous_close: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "previous_close",
            _positive_number(
                "previous_close",
                self.previous_close,
            ),
        )


@dataclass(frozen=True, slots=True)
class MeanReversionThresholds:
    """Configuration governing deterministic mean-reversion evaluation.

    Percentage fields use percentage points. For example, ``2.0`` means 2%.
    """

    max_support_zone_above_pct: float = 3.0
    prior_touch_tolerance_pct: float = 0.5
    max_prior_support_breach_pct: float = 1.0
    max_current_support_break_pct: float = 1.5

    min_reversal_pct: float = 0.5

    rsi_min: float = 30.0
    rsi_max: float = 50.0

    bollinger_min: float = -0.25
    bollinger_max: float = 0.35

    min_macd_histogram: float = 0.0

    watch_score: float = 55.0
    buy_score: float = 75.0

    def __post_init__(self) -> None:
        for name in (
            "max_support_zone_above_pct",
            "prior_touch_tolerance_pct",
            "max_prior_support_breach_pct",
            "max_current_support_break_pct",
            "min_reversal_pct",
            "rsi_min",
            "rsi_max",
            "bollinger_min",
            "bollinger_max",
            "min_macd_histogram",
            "watch_score",
            "buy_score",
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(name, getattr(self, name)),
            )

        if self.max_support_zone_above_pct <= 0:
            raise ValueError(
                "max_support_zone_above_pct must be greater than zero."
            )

        if self.prior_touch_tolerance_pct < 0:
            raise ValueError(
                "prior_touch_tolerance_pct cannot be negative."
            )

        if self.max_prior_support_breach_pct < 0:
            raise ValueError(
                "max_prior_support_breach_pct cannot be negative."
            )

        if self.max_current_support_break_pct <= 0:
            raise ValueError(
                "max_current_support_break_pct must be greater than zero."
            )

        if self.min_reversal_pct <= 0:
            raise ValueError(
                "min_reversal_pct must be greater than zero."
            )

        if not 0 <= self.rsi_min < self.rsi_max <= 100:
            raise ValueError(
                "RSI thresholds must satisfy "
                "0 <= rsi_min < rsi_max <= 100."
            )

        if self.bollinger_min >= self.bollinger_max:
            raise ValueError(
                "bollinger_min must be less than bollinger_max."
            )

        if not 0 <= self.watch_score < self.buy_score <= 100:
            raise ValueError(
                "Scores must satisfy "
                "0 <= watch_score < buy_score <= 100."
            )


def _criterion_evidence(
    *,
    code: str,
    passed: bool,
    success_message: str,
    failure_message: str,
    strength: float,
    observed_value: float | str,
    failure_direction: EvidenceDirection = EvidenceDirection.NEUTRAL,
) -> Evidence:
    """Create consistent evidence for one strategy criterion."""

    return Evidence(
        code=code,
        message=success_message if passed else failure_message,
        direction=(
            EvidenceDirection.BULLISH
            if passed
            else failure_direction
        ),
        strength=strength,
        observed_value=observed_value,
    )


def evaluate_mean_reversion(
    analysis: AnalysisSnapshot,
    regime: RegimeClassification,
    context: MeanReversionContext,
    thresholds: MeanReversionThresholds | None = None,
) -> StrategyResult:
    """Evaluate a deterministic support-zone mean-reversion setup.

    A confirmed mean-reversion entry requires:

    - a sideways market regime
    - a completed-session close inside the support zone
    - evidence that the previous session tested support
    - a positive completed-session reversal
    - RSI in a constructive recovery range
    - price positioned near the lower Bollinger area
    - non-negative MACD histogram

    Strong bearish trends, high-volatility regimes and material support breaks
    veto the strategy.
    """

    if not isinstance(analysis, AnalysisSnapshot):
        raise ValueError("analysis must be an AnalysisSnapshot.")

    if not isinstance(regime, RegimeClassification):
        raise ValueError("regime must be a RegimeClassification.")

    if not isinstance(context, MeanReversionContext):
        raise ValueError(
            "context must be a MeanReversionContext."
        )

    thresholds = thresholds or MeanReversionThresholds()

    if not isinstance(thresholds, MeanReversionThresholds):
        raise ValueError(
            "thresholds must be MeanReversionThresholds."
        )

    indicators = analysis.indicators

    close = indicators.close
    previous_close = context.previous_close
    support = indicators.support

    support_distance_pct = (
        close / support - 1
    ) * 100

    previous_support_distance_pct = (
        previous_close / support - 1
    ) * 100

    reversal_pct = (
        close / previous_close - 1
    ) * 100

    suitable_regime = (
        regime.regime is MarketRegime.SIDEWAYS
    )

    support_zone = (
        0
        <= support_distance_pct
        <= thresholds.max_support_zone_above_pct
    )

    prior_support_test = (
        -thresholds.max_prior_support_breach_pct
        <= previous_support_distance_pct
        <= thresholds.prior_touch_tolerance_pct
    )

    reversal_confirmation = (
        close > support
        and reversal_pct >= thresholds.min_reversal_pct
    )

    rsi_confirmation = (
        thresholds.rsi_min
        <= indicators.rsi
        <= thresholds.rsi_max
    )

    bollinger_confirmation = (
        thresholds.bollinger_min
        <= indicators.bollinger_percent_b
        <= thresholds.bollinger_max
    )

    momentum_confirmation = (
        indicators.macd_histogram
        >= thresholds.min_macd_histogram
    )

    strong_bearish_inputs = (
        regime.bearish_votes >= 3
        and regime.inputs.short_trend_slope_pct < 0
        and regime.inputs.long_trend_slope_pct < 0
        and regime.inputs.price_vs_ma200_pct < 0
    )

    strong_bearish_trend = (
        regime.regime is MarketRegime.BEARISH
        or strong_bearish_inputs
    )

    material_support_break = (
        support_distance_pct
        < -thresholds.max_current_support_break_pct
    )

    evidence: list[Evidence] = []

    evidence.append(
        Evidence(
            code="REGIME_SUITABILITY",
            message=(
                "The sideways market regime supports "
                "mean-reversion strategies."
                if suitable_regime
                else (
                    "The current market regime is not suitable "
                    "for this mean-reversion strategy."
                )
            ),
            direction=(
                EvidenceDirection.BULLISH
                if suitable_regime
                else (
                    EvidenceDirection.BEARISH
                    if strong_bearish_trend
                    else EvidenceDirection.NEUTRAL
                )
            ),
            strength=0.20,
            observed_value=regime.regime.value,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="SUPPORT_ZONE",
            passed=support_zone,
            success_message=(
                "The completed-session close is inside "
                "the configured support zone."
            ),
            failure_message=(
                "The completed-session close is outside "
                "the configured support zone."
            ),
            strength=0.25,
            observed_value=support_distance_pct,
            failure_direction=EvidenceDirection.BEARISH,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="PRIOR_SUPPORT_TEST",
            passed=prior_support_test,
            success_message=(
                "The previous session tested the support area."
            ),
            failure_message=(
                "The previous session did not provide a valid "
                "support-area test."
            ),
            strength=0.15,
            observed_value=previous_support_distance_pct,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="REVERSAL_CONFIRMATION",
            passed=reversal_confirmation,
            success_message=(
                "Price confirmed a completed-session reversal "
                "away from support."
            ),
            failure_message=(
                "Price has not confirmed a completed-session "
                "reversal away from support."
            ),
            strength=0.20,
            observed_value=reversal_pct,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="RSI_CONFIRMATION",
            passed=rsi_confirmation,
            success_message=(
                "RSI is inside the constructive "
                "mean-reversion recovery range."
            ),
            failure_message=(
                "RSI is outside the constructive "
                "mean-reversion recovery range."
            ),
            strength=0.10,
            observed_value=indicators.rsi,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="BOLLINGER_CONFIRMATION",
            passed=bollinger_confirmation,
            success_message=(
                "Price remains near the lower Bollinger area."
            ),
            failure_message=(
                "Price is not positioned in the configured "
                "lower Bollinger area."
            ),
            strength=0.05,
            observed_value=indicators.bollinger_percent_b,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="MOMENTUM_CONFIRMATION",
            passed=momentum_confirmation,
            success_message=(
                "MACD histogram confirms stabilising "
                "or improving momentum."
            ),
            failure_message=(
                "MACD histogram has not confirmed "
                "stabilising momentum."
            ),
            strength=0.05,
            observed_value=indicators.macd_histogram,
        )
    )

    invalidation_reasons: list[str] = []

    if strong_bearish_trend:
        invalidation_reasons.append(
            "A strong bearish trend invalidates "
            "the mean-reversion strategy."
        )

    if regime.regime is MarketRegime.HIGH_VOLATILITY:
        invalidation_reasons.append(
            "A high-volatility regime invalidates "
            "the mean-reversion strategy."
        )

    if material_support_break:
        invalidation_reasons.append(
            "Price has materially broken below local support."
        )

    if invalidation_reasons:
        veto_reason = " ".join(invalidation_reasons)

        evidence.append(
            Evidence(
                code="MEAN_REVERSION_INVALIDATED",
                message=veto_reason,
                direction=EvidenceDirection.BEARISH,
                strength=1.0,
                observed_value=support_distance_pct,
            )
        )

        score = max(
            -100.0,
            -55.0 - 15.0 * (len(invalidation_reasons) - 1),
        )

        confidence = round(
            min(
                1.0,
                max(0.70, regime.confidence),
            ),
            4,
        )

        return StrategyResult(
            strategy=STRATEGY_NAME,
            signal=Signal.HOLD,
            score=score,
            confidence=confidence,
            evidence=tuple(evidence),
            vetoed=True,
            veto_reason=veto_reason,
        )

    score = 0.0

    if suitable_regime:
        score += 20.0

    if support_zone:
        score += 25.0

    if prior_support_test:
        score += 15.0

    if reversal_confirmation:
        score += 20.0

    if rsi_confirmation:
        score += 10.0

    if bollinger_confirmation:
        score += 5.0

    if momentum_confirmation:
        score += 5.0

    setup_present = (
        suitable_regime
        and support_zone
        and prior_support_test
        and reversal_confirmation
    )

    setup_confirmed = (
        setup_present
        and rsi_confirmation
        and bollinger_confirmation
        and momentum_confirmation
    )

    if (
        setup_confirmed
        and score >= thresholds.buy_score
    ):
        signal = Signal.BUY
        state_code = "MEAN_REVERSION_CONFIRMED"
        state_message = (
            "Support-zone mean-reversion setup is confirmed."
        )
        state_direction = EvidenceDirection.BULLISH
        state_strength = 1.0

    elif (
        setup_present
        and score >= thresholds.watch_score
    ):
        signal = Signal.WATCH
        state_code = "MEAN_REVERSION_AWAITING_CONFIRMATION"
        state_message = (
            "Support reversal is present, but one or more "
            "confirmation conditions remain incomplete."
        )
        state_direction = EvidenceDirection.NEUTRAL
        state_strength = 0.8

    else:
        signal = Signal.HOLD
        state_code = "MEAN_REVERSION_ABSENT"
        state_message = (
            "A confirmed support-zone mean-reversion "
            "setup is not present."
        )
        state_direction = EvidenceDirection.NEUTRAL
        state_strength = 0.6

    evidence.append(
        Evidence(
            code=state_code,
            message=state_message,
            direction=state_direction,
            strength=state_strength,
            observed_value=score,
        )
    )

    if signal in {Signal.BUY, Signal.WATCH}:
        confidence = min(
            1.0,
            max(
                0.50,
                score
                / 100
                * max(0.50, regime.confidence),
            ),
        )
    else:
        confidence = min(
            1.0,
            max(
                0.35,
                1 - score / 200,
            ),
        )

    return StrategyResult(
        strategy=STRATEGY_NAME,
        signal=signal,
        score=round(score, 4),
        confidence=round(confidence, 4),
        evidence=tuple(evidence),
    )
