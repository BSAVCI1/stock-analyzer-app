from datetime import datetime, timedelta, timezone

import pytest

from src.analysis import (
    AnalysisSnapshot,
    Evidence,
    EvidenceDirection,
    IndicatorSnapshot,
    PaperOrder,
    RiskDecision,
    RiskThresholds,
    Signal,
    StrategyResult,
    apply_risk_management,
)


AS_OF = datetime(
    2026,
    7,
    31,
    20,
    0,
    tzinfo=timezone.utc,
)


def make_analysis(
    *,
    close: float = 100.0,
    atr: float = 2.0,
    support: float = 96.5,
    resistance: float = 108.0,
) -> AnalysisSnapshot:
    return AnalysisSnapshot(
        symbol="TEST",
        display_name="Test Instrument",
        fetched_at_utc=AS_OF + timedelta(minutes=10),
        history_rows=260,
        indicators=IndicatorSnapshot(
            as_of=AS_OF,
            close=close,
            volume=1_500_000,
            ma20=98.0,
            ma50=96.0,
            ma200=92.0,
            rsi=60.0,
            macd=1.5,
            macd_signal=1.0,
            macd_histogram=0.5,
            bollinger_percent_b=0.75,
            atr=atr,
            obv=25_000_000,
            support=support,
            resistance=resistance,
        ),
        quote_type="EQUITY",
        currency="USD",
        exchange="NMS",
    )


def make_recommendation(
    signal: Signal = Signal.BUY,
    *,
    score: float | None = None,
    confidence: float = 0.9,
) -> StrategyResult:
    if score is None:
        if signal in {Signal.BUY, Signal.WATCH}:
            score = 80.0
        elif signal in {Signal.SELL, Signal.REDUCE}:
            score = -80.0
        else:
            score = 0.0

    if score > 0:
        direction = EvidenceDirection.BULLISH
    elif score < 0:
        direction = EvidenceDirection.BEARISH
    else:
        direction = EvidenceDirection.NEUTRAL

    return StrategyResult(
        strategy="final_recommendation",
        signal=signal,
        score=score,
        confidence=confidence,
        evidence=(
            Evidence(
                code="FINAL_SCORE",
                message="Deterministic recommendation fixture.",
                direction=direction,
                strength=1.0,
                observed_value=score,
            ),
        ),
    )


def evidence_codes(
    result: StrategyResult,
) -> set[str]:
    return {
        item.code
        for item in result.evidence
    }


def test_valid_buy_generates_paper_order() -> None:
    decision = apply_risk_management(
        make_analysis(),
        make_recommendation(Signal.BUY),
    )

    assert decision.recommendation.signal is Signal.BUY
    assert decision.recommendation.vetoed is False
    assert decision.risk_vetoes == ()
    assert decision.order is not None

    order = decision.order

    assert order.paper_only is True
    assert order.signal is Signal.BUY
    assert order.stop_price < order.entry_low
    assert all(
        target > order.entry_high
        for target in order.targets
    )
    assert order.reward_to_risk >= 2.0
    assert order.invalidation_price == order.stop_price
    assert "RISK_GATE_PASSED" in evidence_codes(
        decision.recommendation
    )


def test_valid_sell_generates_paper_order() -> None:
    analysis = make_analysis(
        support=92.0,
        resistance=103.5,
    )

    decision = apply_risk_management(
        analysis,
        make_recommendation(Signal.SELL),
    )

    assert decision.recommendation.signal is Signal.SELL
    assert decision.recommendation.vetoed is False
    assert decision.order is not None

    order = decision.order

    assert order.signal is Signal.SELL
    assert order.stop_price > order.entry_high
    assert all(
        target < order.entry_low
        for target in order.targets
    )
    assert order.reward_to_risk >= 2.0
    assert order.invalidation_price == order.stop_price


def test_buy_below_minimum_reward_to_risk_is_vetoed() -> None:
    analysis = make_analysis(
        support=88.0,
        resistance=105.0,
    )

    decision = apply_risk_management(
        analysis,
        make_recommendation(Signal.BUY),
    )

    assert decision.recommendation.signal is Signal.HOLD
    assert decision.recommendation.vetoed is True
    assert decision.order is None
    assert decision.risk_vetoes
    assert "Reward-to-risk" in (
        decision.recommendation.veto_reason
    )
    assert "RISK_VETO" in evidence_codes(
        decision.recommendation
    )


def test_excessive_stop_distance_is_vetoed() -> None:
    analysis = make_analysis(
        support=70.0,
        resistance=110.0,
    )

    decision = apply_risk_management(
        analysis,
        make_recommendation(Signal.BUY),
    )

    assert decision.recommendation.signal is Signal.HOLD
    assert decision.recommendation.vetoed is True
    assert decision.order is None
    assert "maximum" in (
        decision.recommendation.veto_reason
    )


def test_non_actionable_signal_has_no_order() -> None:
    decision = apply_risk_management(
        make_analysis(),
        make_recommendation(
            Signal.WATCH,
            score=50.0,
        ),
    )

    assert decision.recommendation.signal is Signal.WATCH
    assert decision.recommendation.vetoed is False
    assert decision.order is None
    assert decision.risk_vetoes == ()
    assert (
        "PAPER_ORDER_NOT_APPLICABLE"
        in evidence_codes(decision.recommendation)
    )


def test_order_has_defined_expiry() -> None:
    thresholds = RiskThresholds(
        expiry_days=7,
    )

    decision = apply_risk_management(
        make_analysis(),
        make_recommendation(Signal.BUY),
        thresholds,
    )

    assert decision.order is not None
    assert decision.order.created_at == AS_OF
    assert decision.order.expires_at == (
        AS_OF + timedelta(days=7)
    )


def test_buy_and_sell_always_have_invalidation_point() -> None:
    buy = apply_risk_management(
        make_analysis(),
        make_recommendation(Signal.BUY),
    )

    sell = apply_risk_management(
        make_analysis(
            support=92.0,
            resistance=103.5,
        ),
        make_recommendation(Signal.SELL),
    )

    assert buy.order is not None
    assert sell.order is not None
    assert buy.order.invalidation_price > 0
    assert sell.order.invalidation_price > 0


def test_identical_inputs_produce_identical_order() -> None:
    analysis = make_analysis()
    recommendation = make_recommendation(
        Signal.BUY
    )

    first = apply_risk_management(
        analysis,
        recommendation,
    )

    second = apply_risk_management(
        analysis,
        recommendation,
    )

    assert first == second


def test_actionable_risk_decision_requires_order() -> None:
    with pytest.raises(ValueError):
        RiskDecision(
            recommendation=make_recommendation(
                Signal.BUY
            ),
            order=None,
        )


def test_paper_order_rejects_buy_stop_above_entry() -> None:
    with pytest.raises(ValueError):
        PaperOrder(
            symbol="TEST",
            signal=Signal.BUY,
            created_at=AS_OF,
            expires_at=AS_OF + timedelta(days=5),
            entry_low=99.0,
            entry_high=101.0,
            stop_price=100.0,
            targets=(108.0, 112.0),
            risk_per_share=1.0,
            reward_to_risk=2.0,
        )


def test_paper_order_rejects_live_execution() -> None:
    with pytest.raises(ValueError):
        PaperOrder(
            symbol="TEST",
            signal=Signal.BUY,
            created_at=AS_OF,
            expires_at=AS_OF + timedelta(days=5),
            entry_low=99.0,
            entry_high=101.0,
            stop_price=96.0,
            targets=(108.0, 112.0),
            risk_per_share=4.0,
            reward_to_risk=2.0,
            paper_only=False,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"entry_zone_atr_fraction": 0},
        {"atr_stop_multiple": 0},
        {"structure_buffer_atr": 0},
        {"target_atr_multiple": 0},
        {"min_reward_to_risk": 0},
        {"min_stop_distance_pct": 20},
        {"expiry_days": 0},
    ],
)
def test_thresholds_reject_invalid_configuration(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        RiskThresholds(**kwargs)
