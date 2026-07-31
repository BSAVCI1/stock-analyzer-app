from datetime import datetime, timedelta, timezone

import pytest

from src.analysis import (
    AnalysisSnapshot,
    IndicatorSnapshot,
    MarketRegime,
    RegimeClassification,
    RegimeInputs,
    Signal,
    TrendPullbackThresholds,
    evaluate_trend_pullback,
)


AS_OF = datetime(2026, 7, 31, 20, 0, tzinfo=timezone.utc)


def make_analysis(
    *,
    close: float = 100.0,
    ma20: float = 101.0,
    ma50: float = 95.0,
    ma200: float = 90.0,
    rsi: float = 52.0,
    macd: float = 1.2,
    macd_signal: float = 1.0,
    support: float = 94.0,
    resistance: float = 115.0,
) -> AnalysisSnapshot:
    indicators = IndicatorSnapshot(
        as_of=AS_OF,
        close=close,
        volume=1_200_000,
        ma20=ma20,
        ma50=ma50,
        ma200=ma200,
        rsi=rsi,
        macd=macd,
        macd_signal=macd_signal,
        macd_histogram=macd - macd_signal,
        bollinger_percent_b=0.45,
        atr=2.5,
        obv=25_000_000,
        support=support,
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


def make_regime(
    regime: MarketRegime = MarketRegime.BULLISH,
    *,
    confidence: float = 1.0,
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


def test_confirmed_positive_fixture_returns_buy() -> None:
    result = evaluate_trend_pullback(
        make_analysis(),
        make_regime(),
    )

    assert result.signal is Signal.BUY
    assert result.score == 100.0
    assert result.confidence == 1.0
    assert result.vetoed is False
    assert "SETUP_CONFIRMED" in evidence_codes(result)


def test_setup_without_momentum_confirmation_returns_watch() -> None:
    analysis = make_analysis(
        macd=0.8,
        macd_signal=1.0,
    )

    result = evaluate_trend_pullback(
        analysis,
        make_regime(),
    )

    assert result.signal is Signal.WATCH
    assert result.score == 90.0
    assert result.vetoed is False
    assert "SETUP_FORMING" in evidence_codes(result)


def test_extended_price_is_ambiguous_hold() -> None:
    analysis = make_analysis(
        close=110.0,
        ma20=100.0,
        ma50=95.0,
        ma200=90.0,
        support=94.0,
        resistance=120.0,
    )

    result = evaluate_trend_pullback(
        analysis,
        make_regime(),
    )

    assert result.signal is Signal.HOLD
    assert result.vetoed is False
    assert "SETUP_ABSENT" in evidence_codes(result)


def test_close_below_support_invalidates_setup() -> None:
    analysis = make_analysis(
        close=93.0,
        ma20=100.0,
        ma50=95.0,
        ma200=90.0,
        support=94.0,
        resistance=115.0,
    )

    result = evaluate_trend_pullback(
        analysis,
        make_regime(),
    )

    assert result.signal is Signal.HOLD
    assert result.score < 0
    assert result.vetoed is True
    assert "support" in result.veto_reason.lower()
    assert "SETUP_INVALIDATED" in evidence_codes(result)


def test_bearish_regime_invalidates_setup() -> None:
    result = evaluate_trend_pullback(
        make_analysis(),
        make_regime(MarketRegime.BEARISH),
    )

    assert result.signal is Signal.HOLD
    assert result.vetoed is True
    assert result.score < 0
    assert "bearish regime" in result.veto_reason.lower()


def test_high_volatility_regime_invalidates_setup() -> None:
    result = evaluate_trend_pullback(
        make_analysis(),
        make_regime(MarketRegime.HIGH_VOLATILITY),
    )

    assert result.signal is Signal.HOLD
    assert result.vetoed is True
    assert "high-volatility" in result.veto_reason.lower()


def test_sideways_regime_returns_non_vetoed_hold() -> None:
    result = evaluate_trend_pullback(
        make_analysis(),
        make_regime(MarketRegime.SIDEWAYS),
    )

    assert result.signal is Signal.HOLD
    assert result.vetoed is False
    assert "SETUP_ABSENT" in evidence_codes(result)


def test_evaluation_is_deterministic() -> None:
    analysis = make_analysis()
    regime = make_regime()

    first = evaluate_trend_pullback(analysis, regime)
    second = evaluate_trend_pullback(analysis, regime)

    assert first == second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_close_below_ma20_pct": 0},
        {"rsi_min": 70, "rsi_max": 60},
        {"min_support_buffer_pct": -1},
        {"watch_score": 80, "buy_score": 70},
    ],
)
def test_thresholds_reject_invalid_configuration(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        TrendPullbackThresholds(**kwargs)
