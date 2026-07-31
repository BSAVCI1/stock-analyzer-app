"""Deterministic range-breakout strategy.

The strategy evaluates whether price has closed above a previously established
range resistance with sufficient volume confirmation. It also filters stale,
overextended and failed breakouts.

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


STRATEGY_NAME = "breakout"


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


@dataclass(frozen=True, slots=True)
class BreakoutContext:
    """Historical range and volume context required by the strategy.

    ``range_resistance`` must be calculated from sessions before the current
    completed session. This prevents the current breakout bar from redefining
    its own resistance level.
    """

    range_resistance: float
    average_volume: float
    previous_close: float
    range_sessions: int = 20

    def __post_init__(self) -> None:
        for name in (
            "range_resistance",
            "average_volume",
            "previous_close",
        ):
            object.__setattr__(
                self,
                name,
                _positive_number(name, getattr(self, name)),
            )

        if (
            isinstance(self.range_sessions, bool)
            or not isinstance(self.range_sessions, int)
        ):
            raise ValueError("range_sessions must be an integer.")

        if self.range_sessions < 5:
            raise ValueError(
                "range_sessions must contain at least 5 sessions."
            )


@dataclass(frozen=True, slots=True)
class BreakoutThresholds:
    """Configuration governing deterministic breakout evaluation.

    Percentage fields use percentage points. For example, ``0.5`` means 0.5%.
    """

    min_close_breakout_pct: float = 0.5
    max_close_extension_pct: float = 6.0
    prior_close_tolerance_pct: float = 0.25

    min_volume_ratio: float = 1.5
    min_macd_histogram: float = 0.0

    rsi_min: float = 50.0
    rsi_max: float = 78.0

    watch_score: float = 55.0
    buy_score: float = 75.0

    def __post_init__(self) -> None:
        for name in (
            "min_close_breakout_pct",
            "max_close_extension_pct",
            "prior_close_tolerance_pct",
            "min_volume_ratio",
            "min_macd_histogram",
            "rsi_min",
            "rsi_max",
            "watch_score",
            "buy_score",
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(name, getattr(self, name)),
            )

        if self.min_close_breakout_pct <= 0:
            raise ValueError(
                "min_close_breakout_pct must be greater than zero."
            )

        if (
            self.max_close_extension_pct
            <= self.min_close_breakout_pct
        ):
            raise ValueError(
                "max_close_extension_pct must be greater than "
                "min_close_breakout_pct."
            )

        if self.prior_close_tolerance_pct < 0:
            raise ValueError(
                "prior_close_tolerance_pct cannot be negative."
            )

        if self.min_volume_ratio <= 1:
            raise ValueError(
                "min_volume_ratio must be greater than 1."
            )

        if not 0 <= self.rsi_min < self.rsi_max <= 100:
            raise ValueError(
                "RSI thresholds must satisfy "
                "0 <= rsi_min < rsi_max <= 100."
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
    """Create consistent evidence for one breakout criterion."""

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


def evaluate_breakout(
    analysis: AnalysisSnapshot,
    regime: RegimeClassification,
    context: BreakoutContext,
    thresholds: BreakoutThresholds | None = None,
) -> StrategyResult:
    """Evaluate a deterministic range-breakout setup.

    A confirmed breakout requires all of the following:

    - suitable bullish or sideways regime
    - completed-session close above prior range resistance
    - previous close not already materially above resistance
    - current volume at or above the required average-volume multiple
    - acceptable breakout extension
    - constructive RSI
    - non-negative MACD histogram

    Bearish regimes, high-volatility regimes, failed breakouts and materially
    overextended closes veto the setup.
    """

    if not isinstance(analysis, AnalysisSnapshot):
        raise ValueError("analysis must be an AnalysisSnapshot.")

    if not isinstance(regime, RegimeClassification):
        raise ValueError("regime must be a RegimeClassification.")

    if not isinstance(context, BreakoutContext):
        raise ValueError("context must be a BreakoutContext.")

    thresholds = thresholds or BreakoutThresholds()

    if not isinstance(thresholds, BreakoutThresholds):
        raise ValueError(
            "thresholds must be BreakoutThresholds."
        )

    indicators = analysis.indicators

    close = indicators.close
    volume = indicators.volume
    resistance = context.range_resistance

    breakout_pct = (close / resistance - 1) * 100
    previous_close_pct = (
        context.previous_close / resistance - 1
    ) * 100
    volume_ratio = volume / context.average_volume

    suitable_regime = regime.regime in {
        MarketRegime.BULLISH,
        MarketRegime.SIDEWAYS,
    }

    close_confirmation = (
        breakout_pct >= thresholds.min_close_breakout_pct
    )

    fresh_breakout = (
        previous_close_pct
        <= thresholds.prior_close_tolerance_pct
    )

    volume_confirmation = (
        volume_ratio >= thresholds.min_volume_ratio
    )

    extension_filter = (
        breakout_pct <= thresholds.max_close_extension_pct
    )

    rsi_confirmation = (
        thresholds.rsi_min
        <= indicators.rsi
        <= thresholds.rsi_max
    )

    momentum_confirmation = (
        indicators.macd_histogram
        >= thresholds.min_macd_histogram
    )

    prior_confirmed_breakout = (
        previous_close_pct
        >= thresholds.min_close_breakout_pct
    )

    failed_breakout = (
        prior_confirmed_breakout
        and close <= resistance
    )

    evidence: list[Evidence] = []

    evidence.append(
        Evidence(
            code="REGIME_SUPPORT",
            message=(
                "The market regime supports breakout strategies."
                if suitable_regime
                else (
                    "The market regime does not support a new "
                    "long breakout entry."
                )
            ),
            direction=(
                EvidenceDirection.BULLISH
                if suitable_regime
                else EvidenceDirection.BEARISH
            ),
            strength=0.15,
            observed_value=regime.regime.value,
        )
    )

    evidence.append(
        Evidence(
            code="RANGE_CONTEXT",
            message=(
                f"Breakout resistance was derived from "
                f"{context.range_sessions} prior sessions."
            ),
            direction=EvidenceDirection.NEUTRAL,
            strength=0.05,
            observed_value=resistance,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="CLOSE_CONFIRMATION",
            passed=close_confirmation,
            success_message=(
                "The completed-session close is above "
                "range resistance by the required margin."
            ),
            failure_message=(
                "The completed-session close has not confirmed "
                "a range breakout."
            ),
            strength=0.30,
            observed_value=breakout_pct,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="FRESH_BREAKOUT",
            passed=fresh_breakout,
            success_message=(
                "The previous close was not already materially "
                "above resistance."
            ),
            failure_message=(
                "The move is not a fresh breakout because the "
                "previous close was already above resistance."
            ),
            strength=0.10,
            observed_value=previous_close_pct,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="VOLUME_CONFIRMATION",
            passed=volume_confirmation,
            success_message=(
                "Volume confirms the breakout."
            ),
            failure_message=(
                "Volume does not yet confirm the breakout."
            ),
            strength=0.25,
            observed_value=volume_ratio,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="EXTENSION_FILTER",
            passed=extension_filter,
            success_message=(
                "The breakout close is not materially overextended."
            ),
            failure_message=(
                "The breakout close is materially overextended "
                "above resistance."
            ),
            strength=0.05,
            observed_value=breakout_pct,
            failure_direction=EvidenceDirection.BEARISH,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="RSI_CONFIRMATION",
            passed=rsi_confirmation,
            success_message=(
                "RSI remains within the constructive breakout range."
            ),
            failure_message=(
                "RSI is outside the constructive breakout range."
            ),
            strength=0.05,
            observed_value=indicators.rsi,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="MOMENTUM_CONFIRMATION",
            passed=momentum_confirmation,
            success_message=(
                "MACD histogram confirms constructive momentum."
            ),
            failure_message=(
                "MACD histogram does not confirm constructive momentum."
            ),
            strength=0.10,
            observed_value=indicators.macd_histogram,
        )
    )

    invalidation_reasons: list[str] = []

    if regime.regime is MarketRegime.BEARISH:
        invalidation_reasons.append(
            "Bearish regime invalidates new long breakout entries."
        )

    if regime.regime is MarketRegime.HIGH_VOLATILITY:
        invalidation_reasons.append(
            "High-volatility regime invalidates this breakout setup."
        )

    if failed_breakout:
        invalidation_reasons.append(
            "Price closed back below resistance after a prior breakout."
        )

    if close_confirmation and not extension_filter:
        invalidation_reasons.append(
            "The breakout close is excessively extended above resistance."
        )

    if invalidation_reasons:
        veto_reason = " ".join(invalidation_reasons)

        evidence.append(
            Evidence(
                code="BREAKOUT_INVALIDATED",
                message=veto_reason,
                direction=EvidenceDirection.BEARISH,
                strength=1.0,
                observed_value=breakout_pct,
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
        score += 15.0

    if close_confirmation:
        score += 30.0

    if fresh_breakout:
        score += 10.0

    if volume_confirmation:
        score += 25.0

    if extension_filter:
        score += 5.0

    if rsi_confirmation:
        score += 5.0

    if momentum_confirmation:
        score += 10.0

    breakout_detected = (
        suitable_regime
        and close_confirmation
        and fresh_breakout
        and extension_filter
    )

    breakout_confirmed = (
        breakout_detected
        and volume_confirmation
        and rsi_confirmation
        and momentum_confirmation
    )

    if (
        breakout_confirmed
        and score >= thresholds.buy_score
    ):
        signal = Signal.BUY
        state_code = "BREAKOUT_CONFIRMED"
        state_message = (
            "Range breakout is confirmed by both close and volume."
        )
        state_direction = EvidenceDirection.BULLISH
        state_strength = 1.0

    elif (
        breakout_detected
        and score >= thresholds.watch_score
    ):
        signal = Signal.WATCH
        state_code = "BREAKOUT_AWAITING_CONFIRMATION"
        state_message = (
            "Price has broken resistance, but one or more "
            "confirmation conditions remain incomplete."
        )
        state_direction = EvidenceDirection.NEUTRAL
        state_strength = 0.8

    else:
        signal = Signal.HOLD
        state_code = "BREAKOUT_ABSENT"
        state_message = (
            "A fresh confirmed range breakout is not present."
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
