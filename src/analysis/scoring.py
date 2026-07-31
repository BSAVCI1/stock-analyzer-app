"""Deterministic weighted scoring and strategy-conflict resolution.

The module combines six normalized component scores into one recommendation:

- trend
- setup
- momentum
- volume
- volatility
- fundamental quality

Each component must be between -100 and 100. Positive volatility values mean
that volatility conditions are supportive; negative values mean that risk or
instability is unfavorable.

Documented conflict priority:

1. A sufficiently strong strategy veto blocks positive recommendations.
2. Direct positive-versus-negative strategy conflicts require a clear
   conviction margin.
3. Strategy direction cannot reverse the weighted component direction.
   A contradiction is reduced to HOLD rather than silently flipped.
4. If no conflict applies, the weighted component score determines the final
   recommendation using fixed thresholds.
5. Strategy input order never changes the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Iterable

from .model import (
    Evidence,
    EvidenceDirection,
    Signal,
    StrategyResult,
)


FINAL_STRATEGY_NAME = "final_recommendation"

_POSITIVE_SIGNALS = frozenset(
    {
        Signal.BUY,
        Signal.WATCH,
    }
)

_NEGATIVE_SIGNALS = frozenset(
    {
        Signal.REDUCE,
        Signal.SELL,
    }
)


def _finite_number(name: str, value: object) -> float:
    """Return a validated finite number."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc

    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number.")

    return result


def _bounded_score(name: str, value: object) -> float:
    """Validate a normalized score between -100 and 100."""

    result = _finite_number(name, value)

    if not -100 <= result <= 100:
        raise ValueError(
            f"{name} must be between -100 and 100."
        )

    return result


@dataclass(frozen=True, slots=True)
class ScoreComponents:
    """Normalized scoring inputs for the final recommendation."""

    trend: float
    setup: float
    momentum: float
    volume: float
    volatility: float
    fundamental: float

    def __post_init__(self) -> None:
        for name in (
            "trend",
            "setup",
            "momentum",
            "volume",
            "volatility",
            "fundamental",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_score(name, getattr(self, name)),
            )

    def as_dict(self) -> dict[str, float]:
        """Return components in stable deterministic order."""

        return {
            "trend": self.trend,
            "setup": self.setup,
            "momentum": self.momentum,
            "volume": self.volume,
            "volatility": self.volatility,
            "fundamental": self.fundamental,
        }


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    """Weights applied to the six normalized component scores."""

    trend: float = 0.25
    setup: float = 0.25
    momentum: float = 0.15
    volume: float = 0.10
    volatility: float = 0.10
    fundamental: float = 0.15

    def __post_init__(self) -> None:
        values: list[float] = []

        for name in (
            "trend",
            "setup",
            "momentum",
            "volume",
            "volatility",
            "fundamental",
        ):
            value = _finite_number(name, getattr(self, name))

            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} weight must be between 0 and 1."
                )

            object.__setattr__(self, name, value)
            values.append(value)

        if not isclose(
            sum(values),
            1.0,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Score weights must sum to exactly 1.0."
            )

    def as_dict(self) -> dict[str, float]:
        """Return weights in stable deterministic order."""

        return {
            "trend": self.trend,
            "setup": self.setup,
            "momentum": self.momentum,
            "volume": self.volume,
            "volatility": self.volatility,
            "fundamental": self.fundamental,
        }


@dataclass(frozen=True, slots=True)
class RecommendationThresholds:
    """Thresholds and conflict rules for the final recommendation."""

    buy_score: float = 70.0
    watch_score: float = 35.0
    reduce_score: float = -35.0
    sell_score: float = -70.0

    conflict_margin: float = 15.0
    hard_veto_score: float = -60.0

    def __post_init__(self) -> None:
        for name in (
            "buy_score",
            "watch_score",
            "reduce_score",
            "sell_score",
            "conflict_margin",
            "hard_veto_score",
        ):
            object.__setattr__(
                self,
                name,
                _finite_number(name, getattr(self, name)),
            )

        if not (
            -100
            <= self.sell_score
            < self.reduce_score
            < 0
            < self.watch_score
            < self.buy_score
            <= 100
        ):
            raise ValueError(
                "Thresholds must satisfy "
                "-100 <= sell < reduce < 0 < watch < buy <= 100."
            )

        if not 0 < self.conflict_margin <= 100:
            raise ValueError(
                "conflict_margin must be between 0 and 100."
            )

        if not -100 <= self.hard_veto_score < 0:
            raise ValueError(
                "hard_veto_score must be between -100 and 0."
            )


def weighted_component_score(
    components: ScoreComponents,
    weights: ScoreWeights | None = None,
) -> float:
    """Return the deterministic weighted component score."""

    if not isinstance(components, ScoreComponents):
        raise ValueError(
            "components must be ScoreComponents."
        )

    weights = weights or ScoreWeights()

    if not isinstance(weights, ScoreWeights):
        raise ValueError(
            "weights must be ScoreWeights."
        )

    component_values = components.as_dict()
    weight_values = weights.as_dict()

    score = sum(
        component_values[name] * weight_values[name]
        for name in component_values
    )

    return round(score, 4)


def _signal_from_score(
    score: float,
    thresholds: RecommendationThresholds,
) -> Signal:
    """Map one weighted score to the recommendation scale."""

    if score >= thresholds.buy_score:
        return Signal.BUY

    if score >= thresholds.watch_score:
        return Signal.WATCH

    if score <= thresholds.sell_score:
        return Signal.SELL

    if score <= thresholds.reduce_score:
        return Signal.REDUCE

    return Signal.HOLD


def _direction_for_score(
    score: float,
) -> EvidenceDirection:
    if score > 0:
        return EvidenceDirection.BULLISH

    if score < 0:
        return EvidenceDirection.BEARISH

    return EvidenceDirection.NEUTRAL


def _direction_for_signal(
    signal: Signal,
) -> EvidenceDirection:
    if signal in _POSITIVE_SIGNALS:
        return EvidenceDirection.BULLISH

    if signal in _NEGATIVE_SIGNALS:
        return EvidenceDirection.BEARISH

    return EvidenceDirection.NEUTRAL


def _strategy_conviction(
    result: StrategyResult,
) -> float:
    """Confidence-adjusted strategy conviction."""

    return abs(result.score) * result.confidence


def _strategy_summary(
    results: tuple[StrategyResult, ...],
) -> str:
    if not results:
        return "No strategy-level recommendations were supplied."

    return "; ".join(
        (
            f"{result.strategy}={result.signal.value}"
            f"/{result.score:.2f}"
            f"/{result.confidence:.2f}"
            f"{'/VETO' if result.vetoed else ''}"
        )
        for result in results
    )


def resolve_recommendation(
    components: ScoreComponents,
    strategy_results: Iterable[StrategyResult] = (),
    *,
    weights: ScoreWeights | None = None,
    thresholds: RecommendationThresholds | None = None,
) -> StrategyResult:
    """Produce one deterministic recommendation.

    Strategy results are used as confirmation, conflict and veto inputs. The
    weighted component score remains the canonical final score.

    A strategy conflict may suppress a recommendation to HOLD, but it cannot
    reverse a positive component score into SELL or a negative component score
    into BUY.
    """

    if not isinstance(components, ScoreComponents):
        raise ValueError(
            "components must be ScoreComponents."
        )

    weights = weights or ScoreWeights()
    thresholds = thresholds or RecommendationThresholds()

    if not isinstance(weights, ScoreWeights):
        raise ValueError(
            "weights must be ScoreWeights."
        )

    if not isinstance(
        thresholds,
        RecommendationThresholds,
    ):
        raise ValueError(
            "thresholds must be RecommendationThresholds."
        )

    raw_results = tuple(strategy_results)

    if not all(
        isinstance(result, StrategyResult)
        for result in raw_results
    ):
        raise ValueError(
            "strategy_results must contain only StrategyResult objects."
        )

    strategy_names = [
        result.strategy
        for result in raw_results
    ]

    if len(strategy_names) != len(set(strategy_names)):
        raise ValueError(
            "strategy_results cannot contain duplicate strategy names."
        )

    # Stable ordering guarantees deterministic evidence and equality.
    results = tuple(
        sorted(
            raw_results,
            key=lambda result: (
                result.strategy,
                result.signal.value,
                result.score,
                result.confidence,
            ),
        )
    )

    score = weighted_component_score(
        components,
        weights,
    )

    base_signal = _signal_from_score(
        score,
        thresholds,
    )

    positive_results = tuple(
        result
        for result in results
        if result.signal in _POSITIVE_SIGNALS
        and not result.vetoed
    )

    negative_results = tuple(
        result
        for result in results
        if result.signal in _NEGATIVE_SIGNALS
        and not result.vetoed
    )

    positive_conviction = sum(
        _strategy_conviction(result)
        for result in positive_results
    )

    negative_conviction = sum(
        _strategy_conviction(result)
        for result in negative_results
    )

    hard_vetoes = tuple(
        result
        for result in results
        if result.vetoed
        and result.score <= thresholds.hard_veto_score
    )

    final_signal = base_signal
    final_vetoed = False
    veto_reason: str | None = None
    resolution_reason = (
        "No strategy conflict changed the weighted recommendation."
    )

    # Priority 1: a sufficiently strong veto blocks a positive recommendation.
    if (
        hard_vetoes
        and base_signal in _POSITIVE_SIGNALS
    ):
        final_signal = Signal.HOLD
        final_vetoed = True

        veto_names = ", ".join(
            result.strategy
            for result in hard_vetoes
        )

        veto_reason = (
            "Positive recommendation blocked by strategy veto: "
            f"{veto_names}."
        )

        resolution_reason = veto_reason

    # Priority 2: resolve direct positive-versus-negative conflicts.
    elif positive_results and negative_results:
        conviction_difference = (
            positive_conviction
            - negative_conviction
        )

        if (
            abs(conviction_difference)
            < thresholds.conflict_margin
        ):
            final_signal = Signal.HOLD
            resolution_reason = (
                "Positive and negative strategy conviction "
                "did not exceed the required conflict margin."
            )

        elif conviction_difference > 0:
            if base_signal in _POSITIVE_SIGNALS:
                resolution_reason = (
                    "Positive strategy conviction exceeded negative "
                    "conviction and agreed with the weighted score."
                )
            else:
                final_signal = Signal.HOLD
                resolution_reason = (
                    "Positive strategies dominated, but the weighted "
                    "component score did not support a positive decision."
                )

        else:
            if base_signal in _NEGATIVE_SIGNALS:
                resolution_reason = (
                    "Negative strategy conviction exceeded positive "
                    "conviction and agreed with the weighted score."
                )
            else:
                final_signal = Signal.HOLD
                resolution_reason = (
                    "Negative strategies dominated, but the weighted "
                    "component score did not support a negative decision."
                )

    # Priority 3: one-sided strategy contradiction suppresses the decision.
    elif (
        negative_results
        and base_signal in _POSITIVE_SIGNALS
    ):
        final_signal = Signal.HOLD
        resolution_reason = (
            "Negative strategy evidence contradicted the positive "
            "weighted recommendation."
        )

    elif (
        positive_results
        and base_signal in _NEGATIVE_SIGNALS
    ):
        final_signal = Signal.HOLD
        resolution_reason = (
            "Positive strategy evidence contradicted the negative "
            "weighted recommendation."
        )

    component_values = components.as_dict()
    weight_values = weights.as_dict()

    evidence: list[Evidence] = []

    for name in component_values:
        raw_value = component_values[name]
        weight = weight_values[name]
        contribution = raw_value * weight

        evidence.append(
            Evidence(
                code=f"{name}_score",
                message=(
                    f"{name.replace('_', ' ').title()} score "
                    f"{raw_value:.2f} with weight {weight:.2f} "
                    f"contributed {contribution:.2f} points."
                ),
                direction=_direction_for_score(raw_value),
                strength=weight,
                observed_value=round(contribution, 4),
            )
        )

    evidence.append(
        Evidence(
            code="weighted_component_score",
            message=(
                "The six weighted components produced a "
                f"composite score of {score:.2f}."
            ),
            direction=_direction_for_score(score),
            strength=1.0,
            observed_value=score,
        )
    )

    total_directional_conviction = (
        positive_conviction
        + negative_conviction
    )

    if total_directional_conviction == 0:
        strategy_agreement = 0.5
        consensus_direction = EvidenceDirection.NEUTRAL
    else:
        strategy_agreement = (
            max(
                positive_conviction,
                negative_conviction,
            )
            / total_directional_conviction
        )

        if positive_conviction > negative_conviction:
            consensus_direction = EvidenceDirection.BULLISH
        elif negative_conviction > positive_conviction:
            consensus_direction = EvidenceDirection.BEARISH
        else:
            consensus_direction = EvidenceDirection.NEUTRAL

    evidence.append(
        Evidence(
            code="strategy_consensus",
            message=_strategy_summary(results),
            direction=consensus_direction,
            strength=round(strategy_agreement, 4),
            observed_value=(
                f"positive={positive_conviction:.2f};"
                f"negative={negative_conviction:.2f}"
            ),
        )
    )

    evidence.append(
        Evidence(
            code="conflict_resolution",
            message=resolution_reason,
            direction=_direction_for_signal(final_signal),
            strength=1.0,
            observed_value=final_signal.value,
        )
    )

    component_confidence = abs(score) / 100

    confidence = (
        0.40
        + 0.40 * component_confidence
        + 0.20 * strategy_agreement
    )

    # A suppressed recommendation carries lower actionable confidence.
    if (
        final_signal is Signal.HOLD
        and base_signal is not Signal.HOLD
    ):
        confidence *= 0.80

    if final_vetoed:
        confidence = max(
            confidence,
            max(
                result.confidence
                for result in hard_vetoes
            ),
        )

    confidence = round(
        min(1.0, max(0.0, confidence)),
        4,
    )

    return StrategyResult(
        strategy=FINAL_STRATEGY_NAME,
        signal=final_signal,
        score=score,
        confidence=confidence,
        evidence=tuple(evidence),
        vetoed=final_vetoed,
        veto_reason=veto_reason,
    )
