from __future__ import annotations

import pytest

from src.analysis import (
    Evidence,
    EvidenceDirection,
    RecommendationThresholds,
    ScoreComponents,
    ScoreWeights,
    Signal,
    StrategyResult,
    resolve_recommendation,
    weighted_component_score,
)


def make_components(
    *,
    trend: float = 0,
    setup: float = 0,
    momentum: float = 0,
    volume: float = 0,
    volatility: float = 0,
    fundamental: float = 0,
) -> ScoreComponents:
    return ScoreComponents(
        trend=trend,
        setup=setup,
        momentum=momentum,
        volume=volume,
        volatility=volatility,
        fundamental=fundamental,
    )


def make_strategy_result(
    strategy: str,
    signal: Signal,
    score: float,
    *,
    confidence: float = 0.9,
    vetoed: bool = False,
) -> StrategyResult:
    if score > 0:
        direction = EvidenceDirection.BULLISH
    elif score < 0:
        direction = EvidenceDirection.BEARISH
    else:
        direction = EvidenceDirection.NEUTRAL

    return StrategyResult(
        strategy=strategy,
        signal=signal,
        score=score,
        confidence=confidence,
        evidence=(
            Evidence(
                code=f"{strategy}_test",
                message="Deterministic strategy fixture.",
                direction=direction,
                strength=1.0,
                observed_value=score,
            ),
        ),
        vetoed=vetoed,
        veto_reason=(
            "Deterministic veto fixture."
            if vetoed
            else None
        ),
    )


def test_weighted_component_score_uses_documented_weights() -> None:
    components = make_components(
        trend=100,
        setup=50,
        momentum=0,
        volume=-50,
        volatility=25,
        fundamental=75,
    )

    assert weighted_component_score(components) == 46.25


def test_strong_positive_components_return_buy() -> None:
    result = resolve_recommendation(
        make_components(
            trend=100,
            setup=100,
            momentum=100,
            volume=100,
            volatility=100,
            fundamental=100,
        )
    )

    assert result.signal is Signal.BUY
    assert result.score == 100.0
    assert result.vetoed is False


def test_moderate_positive_components_return_watch() -> None:
    result = resolve_recommendation(
        make_components(
            trend=50,
            setup=50,
            momentum=50,
            volume=50,
            volatility=50,
            fundamental=50,
        )
    )

    assert result.signal is Signal.WATCH
    assert result.score == 50.0


def test_neutral_components_return_hold() -> None:
    result = resolve_recommendation(
        make_components()
    )

    assert result.signal is Signal.HOLD
    assert result.score == 0.0


def test_moderate_negative_components_return_reduce() -> None:
    result = resolve_recommendation(
        make_components(
            trend=-50,
            setup=-50,
            momentum=-50,
            volume=-50,
            volatility=-50,
            fundamental=-50,
        )
    )

    assert result.signal is Signal.REDUCE
    assert result.score == -50.0


def test_strong_negative_components_return_sell() -> None:
    result = resolve_recommendation(
        make_components(
            trend=-100,
            setup=-100,
            momentum=-100,
            volume=-100,
            volatility=-100,
            fundamental=-100,
        )
    )

    assert result.signal is Signal.SELL
    assert result.score == -100.0


def test_hard_veto_blocks_positive_recommendation() -> None:
    result = resolve_recommendation(
        make_components(
            trend=90,
            setup=90,
            momentum=90,
            volume=90,
            volatility=90,
            fundamental=90,
        ),
        (
            make_strategy_result(
                "risk_filter",
                Signal.HOLD,
                -80,
                confidence=0.95,
                vetoed=True,
            ),
        ),
    )

    assert result.signal is Signal.HOLD
    assert result.vetoed is True
    assert "risk_filter" in result.veto_reason


def test_balanced_positive_and_negative_conflict_returns_hold() -> None:
    result = resolve_recommendation(
        make_components(
            trend=90,
            setup=90,
            momentum=90,
            volume=90,
            volatility=90,
            fundamental=90,
        ),
        (
            make_strategy_result(
                "positive_strategy",
                Signal.BUY,
                90,
                confidence=0.9,
            ),
            make_strategy_result(
                "negative_strategy",
                Signal.SELL,
                -85,
                confidence=0.9,
            ),
        ),
    )

    assert result.signal is Signal.HOLD
    assert result.vetoed is False


def test_dominant_positive_conflict_allows_positive_score() -> None:
    result = resolve_recommendation(
        make_components(
            trend=90,
            setup=90,
            momentum=90,
            volume=90,
            volatility=90,
            fundamental=90,
        ),
        (
            make_strategy_result(
                "breakout",
                Signal.BUY,
                100,
                confidence=1.0,
            ),
            make_strategy_result(
                "risk_warning",
                Signal.REDUCE,
                -40,
                confidence=0.5,
            ),
        ),
    )

    assert result.signal is Signal.BUY
    assert result.vetoed is False


def test_dominant_negative_conflict_allows_negative_score() -> None:
    result = resolve_recommendation(
        make_components(
            trend=-90,
            setup=-90,
            momentum=-90,
            volume=-90,
            volatility=-90,
            fundamental=-90,
        ),
        (
            make_strategy_result(
                "positive_strategy",
                Signal.WATCH,
                40,
                confidence=0.5,
            ),
            make_strategy_result(
                "negative_strategy",
                Signal.SELL,
                -100,
                confidence=1.0,
            ),
        ),
    )

    assert result.signal is Signal.SELL
    assert result.vetoed is False


def test_strategy_cannot_reverse_component_direction() -> None:
    result = resolve_recommendation(
        make_components(
            trend=90,
            setup=90,
            momentum=90,
            volume=90,
            volatility=90,
            fundamental=90,
        ),
        (
            make_strategy_result(
                "negative_strategy",
                Signal.SELL,
                -100,
                confidence=1.0,
            ),
        ),
    )

    assert result.signal is Signal.HOLD
    assert result.score == 90.0


def test_strategy_order_does_not_change_result() -> None:
    components = make_components(
        trend=80,
        setup=80,
        momentum=80,
        volume=80,
        volatility=80,
        fundamental=80,
    )

    first_strategy = make_strategy_result(
        "alpha",
        Signal.BUY,
        80,
        confidence=0.8,
    )

    second_strategy = make_strategy_result(
        "beta",
        Signal.WATCH,
        60,
        confidence=0.7,
    )

    first = resolve_recommendation(
        components,
        (
            first_strategy,
            second_strategy,
        ),
    )

    second = resolve_recommendation(
        components,
        (
            second_strategy,
            first_strategy,
        ),
    )

    assert first == second


def test_identical_inputs_always_return_identical_decision() -> None:
    components = make_components(
        trend=75,
        setup=80,
        momentum=65,
        volume=55,
        volatility=45,
        fundamental=70,
    )

    strategies = (
        make_strategy_result(
            "breakout",
            Signal.BUY,
            85,
            confidence=0.8,
        ),
        make_strategy_result(
            "trend_pullback",
            Signal.WATCH,
            65,
            confidence=0.75,
        ),
    )

    first = resolve_recommendation(
        components,
        strategies,
    )

    second = resolve_recommendation(
        components,
        strategies,
    )

    assert first == second


def test_duplicate_strategy_names_are_rejected() -> None:
    duplicate_one = make_strategy_result(
        "duplicate",
        Signal.BUY,
        80,
    )

    duplicate_two = make_strategy_result(
        "duplicate",
        Signal.WATCH,
        60,
    )

    with pytest.raises(ValueError):
        resolve_recommendation(
            make_components(),
            (
                duplicate_one,
                duplicate_two,
            ),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "trend",
        "setup",
        "momentum",
        "volume",
        "volatility",
        "fundamental",
    ],
)
def test_components_reject_scores_outside_range(
    field_name: str,
) -> None:
    values = {
        "trend": 0,
        "setup": 0,
        "momentum": 0,
        "volume": 0,
        "volatility": 0,
        "fundamental": 0,
    }

    values[field_name] = 101

    with pytest.raises(ValueError):
        ScoreComponents(**values)


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        ScoreWeights(
            trend=0.20,
            setup=0.20,
            momentum=0.20,
            volume=0.20,
            volatility=0.20,
            fundamental=0.20,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "sell_score": -30,
            "reduce_score": -40,
        },
        {
            "watch_score": 80,
            "buy_score": 70,
        },
        {
            "conflict_margin": 0,
        },
        {
            "hard_veto_score": 10,
        },
    ],
)
def test_thresholds_reject_invalid_configuration(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        RecommendationThresholds(**kwargs)
