from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest

from src.analysis import (
    AnalysisSnapshot,
    Evidence,
    EvidenceDirection,
    IndicatorSnapshot,
    Signal,
    StrategyResult,
)


AS_OF = datetime(2026, 7, 30, 20, 0, tzinfo=timezone.utc)
FETCHED_AT = AS_OF + timedelta(hours=1)


@pytest.fixture
def indicator_snapshot() -> IndicatorSnapshot:
    return IndicatorSnapshot(
        as_of=AS_OF,
        close=150.0,
        volume=1_200_000,
        ma20=148.0,
        ma50=145.0,
        ma200=130.0,
        rsi=58.0,
        macd=2.5,
        macd_signal=2.0,
        macd_histogram=0.5,
        bollinger_percent_b=0.72,
        atr=3.4,
        obv=25_000_000,
        support=140.0,
        resistance=160.0,
    )


@pytest.fixture
def bullish_evidence() -> Evidence:
    return Evidence(
        code="trend_above_ma200",
        message="Price is above its 200-session moving average.",
        direction=EvidenceDirection.BULLISH,
        strength=0.8,
        observed_value=150.0,
    )


def test_signal_enum_contains_required_values() -> None:
    assert [signal.value for signal in Signal] == [
        "BUY",
        "WATCH",
        "HOLD",
        "REDUCE",
        "SELL",
    ]


def test_indicator_snapshot_is_immutable(
    indicator_snapshot: IndicatorSnapshot,
) -> None:
    with pytest.raises(FrozenInstanceError):
        indicator_snapshot.close = 151.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("close", 0),
        ("volume", -1),
        ("rsi", 101),
        ("atr", -0.01),
        ("support", 170),
    ],
)
def test_indicator_snapshot_rejects_invalid_values(
    indicator_snapshot: IndicatorSnapshot,
    field_name: str,
    invalid_value: float,
) -> None:
    with pytest.raises(ValueError):
        replace(
            indicator_snapshot,
            **{field_name: invalid_value},
        )


def test_indicator_snapshot_rejects_inconsistent_macd_histogram(
    indicator_snapshot: IndicatorSnapshot,
) -> None:
    with pytest.raises(
        ValueError,
        match="macd_histogram",
    ):
        replace(
            indicator_snapshot,
            macd_histogram=9.0,
        )


def test_analysis_snapshot_normalises_context(
    indicator_snapshot: IndicatorSnapshot,
) -> None:
    snapshot = AnalysisSnapshot(
        symbol="aapl",
        display_name="Apple Inc.",
        fetched_at_utc=FETCHED_AT,
        history_rows=505,
        indicators=indicator_snapshot,
        quote_type="equity",
        currency="usd",
        exchange="nms",
        warnings=[" Metadata fallback used. ", ""],
    )

    assert snapshot.symbol == "AAPL"
    assert snapshot.currency == "USD"
    assert snapshot.exchange == "NMS"
    assert snapshot.quote_type == "EQUITY"
    assert snapshot.warnings == ("Metadata fallback used.",)
    assert snapshot.latest_close == 150.0
    assert snapshot.as_of == AS_OF


def test_analysis_snapshot_requires_200_sessions(
    indicator_snapshot: IndicatorSnapshot,
) -> None:
    with pytest.raises(
        ValueError,
        match="at least 200",
    ):
        AnalysisSnapshot(
            symbol="AAPL",
            display_name="Apple Inc.",
            fetched_at_utc=FETCHED_AT,
            history_rows=199,
            indicators=indicator_snapshot,
        )


def test_analysis_snapshot_rejects_future_indicator(
    indicator_snapshot: IndicatorSnapshot,
) -> None:
    with pytest.raises(
        ValueError,
        match="cannot be later",
    ):
        AnalysisSnapshot(
            symbol="AAPL",
            display_name="Apple Inc.",
            fetched_at_utc=AS_OF - timedelta(minutes=1),
            history_rows=505,
            indicators=indicator_snapshot,
        )


def test_evidence_is_normalised_and_immutable() -> None:
    evidence = Evidence(
        code=" rsi_neutral ",
        message=" RSI is within the neutral range. ",
        direction=EvidenceDirection.NEUTRAL,
        strength=0.4,
        observed_value=55,
    )

    assert evidence.code == "RSI_NEUTRAL"
    assert evidence.message == "RSI is within the neutral range."
    assert evidence.observed_value == 55.0

    with pytest.raises(FrozenInstanceError):
        evidence.strength = 0.9  # type: ignore[misc]


@pytest.mark.parametrize("strength", [-0.1, 1.1])
def test_evidence_rejects_invalid_strength(
    strength: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        Evidence(
            code="TEST",
            message="Test evidence.",
            direction=EvidenceDirection.NEUTRAL,
            strength=strength,
        )


def test_valid_strategy_result(
    bullish_evidence: Evidence,
) -> None:
    result = StrategyResult(
        strategy="trend_pullback",
        signal=Signal.BUY,
        score=72,
        confidence=0.81,
        evidence=(bullish_evidence,),
    )

    assert result.signal is Signal.BUY
    assert result.score == 72.0
    assert result.confidence == 0.81


@pytest.mark.parametrize(
    ("signal", "score"),
    [
        (Signal.BUY, -10),
        (Signal.SELL, 10),
        (Signal.REDUCE, 5),
    ],
)
def test_strategy_result_rejects_directional_inconsistency(
    bullish_evidence: Evidence,
    signal: Signal,
    score: float,
) -> None:
    with pytest.raises(ValueError):
        StrategyResult(
            strategy="test_strategy",
            signal=signal,
            score=score,
            confidence=0.5,
            evidence=(bullish_evidence,),
        )


def test_strategy_result_requires_evidence() -> None:
    with pytest.raises(
        ValueError,
        match="at least one evidence",
    ):
        StrategyResult(
            strategy="test_strategy",
            signal=Signal.HOLD,
            score=0,
            confidence=0.2,
            evidence=(),
        )


def test_vetoed_result_requires_reason_and_safe_signal(
    bullish_evidence: Evidence,
) -> None:
    valid = StrategyResult(
        strategy="trend_pullback",
        signal=Signal.WATCH,
        score=40,
        confidence=0.7,
        evidence=(bullish_evidence,),
        vetoed=True,
        veto_reason="Reward-to-risk threshold not met.",
    )

    assert valid.vetoed is True

    with pytest.raises(
        ValueError,
        match="cannot issue BUY or SELL",
    ):
        replace(valid, signal=Signal.BUY)

    with pytest.raises(
        ValueError,
        match="requires a veto_reason",
    ):
        replace(valid, veto_reason=None)
