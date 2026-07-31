from datetime import datetime, timedelta, timezone

import pytest

from src.analysis import (
    AnalysisSnapshot,
    BreakoutContext,
    BreakoutThresholds,
    IndicatorSnapshot,
    MarketRegime,
    RegimeClassification,
    RegimeInputs,
    Signal,
    evaluate_breakout,
)


AS_OF = datetime(2026, 7, 31, 20, 0, tzinfo=timezone.utc)


def make_analysis(
    *,
    close: float = 104.0,
    volume: float = 2_000_000,
    resistance: float = 100.0,
    rsi: float = 62.0,
    macd: float = 1.5,
    macd_signal: float = 1.0,
) -> AnalysisSnapshot:
    indicators = IndicatorSnapshot(
        as_of=AS_OF,
        close=close,
        volume=volume,
        ma20=98.0,
        ma50=95.0,
        ma200=90.0,
        rsi=rsi,
        macd=macd,
        macd_signal=macd_signal,
        macd_histogram=macd - macd_signal,
        bollinger_percent_b=0.85,
        atr=2.5,
        obv=25_000_000,
        support=90.0,
        resistance=resistance,
    )

    return AnalysisSnapshot(
        symbol="TEST",
        display_name="Test Instrument",
        fetched_at_utc=AS_OF + timedelta(minutes=10),
        history_rows=260,
        indicators=indicators,
        quote_type="EQUITY",
        currency="USD",
        exchange="NMS",
    )


def make_context(
    *,
    range_resistance: float = 100.0,
    average_volume: float = 1_000_000,
    previous_close: float = 99.0,
) -> BreakoutContext:
    return BreakoutContext(
        range_resistance=range_resistance,
        average_volume=average_volume,
        previous_close=previous_close,
        range_sessions=20,
    )


def make_regime(
    regime: MarketRegime = MarketRegime.SIDEWAYS,
    *,
    confidence: float = 0.9,
) -> RegimeClassification:
    bullish_votes = 4 if regime is MarketRegime.BULLISH else 0
    bearish_votes = 4 if regime is MarketRegime.BEARISH else 0

    return RegimeClassification(
        regime=regime,
        confidence=confidence,
        bullish_votes=bullish_votes,
        bearish_votes=bearish_votes,
        inputs=RegimeInputs(
            short_trend_slope_pct=(
                0.20
                if regime is MarketRegime.BULLISH
                else -0.20
                if regime is MarketRegime.BEARISH
                else 0.0
            ),
            long_trend_slope_pct=(
                0.10
                if regime is MarketRegime.BULLISH
                else -0.10
                if regime is MarketRegime.BEARISH
                else 0.0
            ),
            price_vs_ma200_pct=(
                8.0
                if regime is MarketRegime.BULLISH
                else -8.0
                if regime is MarketRegime.BEARISH
                else 0.0
            ),
            short_vs_long_ma_pct=(
                2.0
                if regime is MarketRegime.BULLISH
                else -2.0
                if regime is MarketRegime.BEARISH
                else 0.0
            ),
            atr_pct=(
                5.0
                if regime is MarketRegime.HIGH_VOLATILITY
                else 1.5
            ),
        ),
        reasons=("Deterministic test fixture.",),
    )


def evidence_codes(result) -> set[str]:
    return {item.code for item in result.evidence}


def test_confirmed_breakout_returns_buy() -> None:
    result = evaluate_breakout(
        make_analysis(),
        make_regime(),
        make_context(),
    )

    assert result.signal is Signal.BUY
    assert result.score == 100.0
    assert result.vetoed is False
    assert "BREAKOUT_CONFIRMED" in evidence_codes(result)


def test_breakout_without_volume_confirmation_returns_watch() -> None:
    result = evaluate_breakout(
        make_analysis(volume=1_200_000),
        make_regime(),
        make_context(),
    )

    assert result.signal is Signal.WATCH
    assert result.vetoed is False
    assert "BREAKOUT_CONFIRMED" not in evidence_codes(result)
    assert "BREAKOUT_AWAITING_CONFIRMATION" in evidence_codes(result)

    volume_evidence = next(
        item
        for item in result.evidence
        if item.code == "VOLUME_CONFIRMATION"
    )

    assert volume_evidence.direction.value == "NEUTRAL"


def test_volume_without_close_confirmation_does_not_emit_breakout() -> None:
    result = evaluate_breakout(
        make_analysis(
            close=100.2,
            volume=2_000_000,
        ),
        make_regime(),
        make_context(),
    )

    assert result.signal is Signal.HOLD
    assert result.vetoed is False
    assert "BREAKOUT_CONFIRMED" not in evidence_codes(result)
    assert "BREAKOUT_ABSENT" in evidence_codes(result)


def test_stale_breakout_is_not_emitted_as_new_breakout() -> None:
    result = evaluate_breakout(
        make_analysis(),
        make_regime(),
        make_context(previous_close=101.0),
    )

    assert result.signal is Signal.HOLD
    assert result.vetoed is False
    assert "BREAKOUT_CONFIRMED" not in evidence_codes(result)


def test_failed_breakout_is_vetoed() -> None:
    result = evaluate_breakout(
        make_analysis(close=99.5),
        make_regime(),
        make_context(previous_close=102.0),
    )

    assert result.signal is Signal.HOLD
    assert result.score < 0
    assert result.vetoed is True
    assert "closed back below resistance" in result.veto_reason
    assert "BREAKOUT_INVALIDATED" in evidence_codes(result)


def test_overextended_breakout_is_vetoed() -> None:
    result = evaluate_breakout(
        make_analysis(close=110.0),
        make_regime(),
        make_context(),
    )

    assert result.signal is Signal.HOLD
    assert result.score < 0
    assert result.vetoed is True
    assert "excessively extended" in result.veto_reason


def test_bearish_regime_invalidates_breakout() -> None:
    result = evaluate_breakout(
        make_analysis(),
        make_regime(MarketRegime.BEARISH),
        make_context(),
    )

    assert result.signal is Signal.HOLD
    assert result.vetoed is True
    assert "Bearish regime" in result.veto_reason


def test_high_volatility_regime_invalidates_breakout() -> None:
    result = evaluate_breakout(
        make_analysis(),
        make_regime(MarketRegime.HIGH_VOLATILITY),
        make_context(),
    )

    assert result.signal is Signal.HOLD
    assert result.vetoed is True
    assert "High-volatility regime" in result.veto_reason


def test_negative_macd_prevents_confirmed_breakout() -> None:
    result = evaluate_breakout(
        make_analysis(
            macd=0.8,
            macd_signal=1.0,
        ),
        make_regime(),
        make_context(),
    )

    assert result.signal is Signal.WATCH
    assert "BREAKOUT_CONFIRMED" not in evidence_codes(result)


def test_evaluation_is_deterministic() -> None:
    analysis = make_analysis()
    regime = make_regime()
    context = make_context()

    first = evaluate_breakout(
        analysis,
        regime,
        context,
    )
    second = evaluate_breakout(
        analysis,
        regime,
        context,
    )

    assert first == second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_close_breakout_pct": 0},
        {
            "min_close_breakout_pct": 2,
            "max_close_extension_pct": 1,
        },
        {"min_volume_ratio": 1},
        {"rsi_min": 80, "rsi_max": 60},
        {"watch_score": 80, "buy_score": 70},
    ],
)
def test_thresholds_reject_invalid_configuration(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        BreakoutThresholds(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"range_resistance": 0},
        {"average_volume": 0},
        {"previous_close": -1},
        {"range_sessions": 4},
    ],
)
def test_context_rejects_invalid_values(
    kwargs: dict[str, float],
) -> None:
    base = {
        "range_resistance": 100.0,
        "average_volume": 1_000_000,
        "previous_close": 99.0,
        "range_sessions": 20,
    }
    base.update(kwargs)

    with pytest.raises(ValueError):
        BreakoutContext(**base)
