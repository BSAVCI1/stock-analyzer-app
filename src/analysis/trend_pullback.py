"""Deterministic trend-pullback strategy.

The strategy identifies controlled pullbacks inside an established bullish
trend. It produces explicit evidence for setup detection, confirmation and
invalidation without depending on Streamlit or market-data providers.
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


STRATEGY_NAME = "trend_pullback"


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


@dataclass(frozen=True, slots=True)
class TrendPullbackThresholds:
    """Configuration governing deterministic trend-pullback evaluation.

    Percentage fields use percentage points. For example, ``1.5`` means 1.5%.
    """

    max_close_above_ma20_pct: float = 1.5
    max_close_below_ma20_pct: float = 4.0

    rsi_min: float = 40.0
    rsi_max: float = 65.0

    min_macd_histogram: float = 0.0
    min_support_buffer_pct: float = 0.5

    watch_score: float = 55.0
    buy_score: float = 75.0

    def __post_init__(self) -> None:
        for name in (
            "max_close_above_ma20_pct",
            "max_close_below_ma20_pct",
            "rsi_min",
            "rsi_max",
            "min_macd_histogram",
            "min_support_buffer_pct",
            "watch_score",
            "buy_score",
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(name, getattr(self, name)),
            )

        if self.max_close_above_ma20_pct < 0:
            raise ValueError(
                "max_close_above_ma20_pct cannot be negative."
            )

        if self.max_close_below_ma20_pct <= 0:
            raise ValueError(
                "max_close_below_ma20_pct must be greater than zero."
            )

        if not 0 <= self.rsi_min < self.rsi_max <= 100:
            raise ValueError(
                "RSI thresholds must satisfy "
                "0 <= rsi_min < rsi_max <= 100."
            )

        if self.min_support_buffer_pct < 0:
            raise ValueError(
                "min_support_buffer_pct cannot be negative."
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


def evaluate_trend_pullback(
    analysis: AnalysisSnapshot,
    regime: RegimeClassification,
    thresholds: TrendPullbackThresholds | None = None,
) -> StrategyResult:
    """Evaluate a deterministic bullish trend-pullback setup.

    Setup requirements
    ------------------
    - bullish market regime
    - MA20 above MA50 above MA200
    - price above MA50 and MA200
    - price has pulled back near MA20
    - RSI has reset into a constructive range

    Confirmation requirements
    -------------------------
    - non-negative MACD histogram
    - price remains safely above local support

    Invalidation rules
    ------------------
    - bearish or high-volatility regime
    - closing price at or below support
    - closing price below MA50
    """

    if not isinstance(analysis, AnalysisSnapshot):
        raise ValueError("analysis must be an AnalysisSnapshot.")

    if not isinstance(regime, RegimeClassification):
        raise ValueError("regime must be a RegimeClassification.")

    thresholds = thresholds or TrendPullbackThresholds()

    if not isinstance(thresholds, TrendPullbackThresholds):
        raise ValueError(
            "thresholds must be TrendPullbackThresholds."
        )

    indicators = analysis.indicators

    close = indicators.close
    ma20 = indicators.ma20
    ma50 = indicators.ma50
    ma200 = indicators.ma200
    support = indicators.support

    distance_to_ma20_pct = (close / ma20 - 1) * 100
    support_buffer_pct = (close / support - 1) * 100

    bullish_regime = regime.regime is MarketRegime.BULLISH

    trend_alignment = ma20 > ma50 > ma200

    price_structure = (
        close >= ma50
        and close >= ma200
    )

    pullback_zone = (
        -thresholds.max_close_below_ma20_pct
        <= distance_to_ma20_pct
        <= thresholds.max_close_above_ma20_pct
    )

    rsi_reset = (
        thresholds.rsi_min
        <= indicators.rsi
        <= thresholds.rsi_max
    )

    momentum_confirmation = (
        indicators.macd_histogram
        >= thresholds.min_macd_histogram
    )

    support_confirmation = (
        support_buffer_pct
        >= thresholds.min_support_buffer_pct
    )

    evidence: list[Evidence] = []

    evidence.append(
        Evidence(
            code="BULLISH_REGIME",
            message=(
                "The market regime supports bullish "
                "trend-pullback strategies."
                if bullish_regime
                else (
                    "The market regime does not currently support "
                    "a bullish trend-pullback entry."
                )
            ),
            direction=(
                EvidenceDirection.BULLISH
                if bullish_regime
                else (
                    EvidenceDirection.NEUTRAL
                    if regime.regime is MarketRegime.SIDEWAYS
                    else EvidenceDirection.BEARISH
                )
            ),
            strength=0.20,
            observed_value=regime.regime.value,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="TREND_ALIGNMENT",
            passed=trend_alignment,
            success_message=(
                "Moving averages are positively aligned: "
                "MA20 > MA50 > MA200."
            ),
            failure_message=(
                "Moving averages are not positively aligned."
            ),
            strength=0.20,
            observed_value=(
                f"{ma20:.4f}/{ma50:.4f}/{ma200:.4f}"
            ),
            failure_direction=EvidenceDirection.BEARISH,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="PRICE_STRUCTURE",
            passed=price_structure,
            success_message=(
                "Price remains above MA50 and MA200."
            ),
            failure_message=(
                "Price has fallen below an important trend average."
            ),
            strength=0.15,
            observed_value=close,
            failure_direction=EvidenceDirection.BEARISH,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="PULLBACK_ZONE",
            passed=pullback_zone,
            success_message=(
                "Price has pulled back into the configured "
                "MA20 entry zone."
            ),
            failure_message=(
                "Price is outside the configured MA20 "
                "pullback zone."
            ),
            strength=0.20,
            observed_value=distance_to_ma20_pct,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="RSI_RESET",
            passed=rsi_reset,
            success_message=(
                "RSI has reset into the constructive "
                "pullback range."
            ),
            failure_message=(
                "RSI is outside the constructive "
                "pullback range."
            ),
            strength=0.10,
            observed_value=indicators.rsi,
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
                "MACD histogram has not yet confirmed "
                "improving momentum."
            ),
            strength=0.10,
            observed_value=indicators.macd_histogram,
        )
    )

    evidence.append(
        _criterion_evidence(
            code="SUPPORT_BUFFER",
            passed=support_confirmation,
            success_message=(
                "Price maintains the required buffer "
                "above local support."
            ),
            failure_message=(
                "Price is too close to local support "
                "for confirmation."
            ),
            strength=0.05,
            observed_value=support_buffer_pct,
            failure_direction=EvidenceDirection.BEARISH,
        )
    )

    invalidation_reasons: list[str] = []

    if regime.regime is MarketRegime.BEARISH:
        invalidation_reasons.append(
            "Bearish regime invalidates bullish pullback entries."
        )

    if regime.regime is MarketRegime.HIGH_VOLATILITY:
        invalidation_reasons.append(
            "High-volatility regime invalidates this setup."
        )

    if close <= support:
        invalidation_reasons.append(
            "Closing price is at or below local support."
        )

    if close < ma50:
        invalidation_reasons.append(
            "Closing price is below MA50."
        )

    if invalidation_reasons:
        veto_reason = " ".join(invalidation_reasons)

        evidence.append(
            Evidence(
                code="SETUP_INVALIDATED",
                message=veto_reason,
                direction=EvidenceDirection.BEARISH,
                strength=1.0,
                observed_value=close,
            )
        )

        score = max(
            -100.0,
            -50.0 - 15.0 * (len(invalidation_reasons) - 1),
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

    if bullish_regime:
        score += 20.0

    if trend_alignment:
        score += 20.0

    if price_structure:
        score += 15.0

    if pullback_zone:
        score += 20.0

    if rsi_reset:
        score += 10.0

    if momentum_confirmation:
        score += 10.0

    if support_confirmation:
        score += 5.0

    setup_ready = (
        bullish_regime
        and trend_alignment
        and price_structure
        and pullback_zone
        and rsi_reset
    )

    setup_confirmed = (
        setup_ready
        and momentum_confirmation
        and support_confirmation
    )

    if setup_confirmed and score >= thresholds.buy_score:
        signal = Signal.BUY
        state_code = "SETUP_CONFIRMED"
        state_message = (
            "Trend-pullback setup is confirmed."
        )
        state_direction = EvidenceDirection.BULLISH
        state_strength = 1.0

    elif setup_ready and score >= thresholds.watch_score:
        signal = Signal.WATCH
        state_code = "SETUP_FORMING"
        state_message = (
            "Trend-pullback setup is present but "
            "awaiting full confirmation."
        )
        state_direction = EvidenceDirection.NEUTRAL
        state_strength = 0.8

    else:
        signal = Signal.HOLD
        state_code = "SETUP_ABSENT"
        state_message = (
            "Trend-pullback requirements are not fully present."
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
