from datetime import datetime, timedelta, timezone

import pytest

from src.analysis import (
    AnalysisSnapshot,
    IndicatorSnapshot,
    MarketRegime,
    MeanReversionContext,
    MeanReversionThresholds,
    RegimeClassification,
    RegimeInputs,
    Signal,
    evaluate_mean_reversion,
)


AS_OF = datetime(2026, 7, 31, 20, 0, tzinfo=timezone.utc)


def make_analysis(
    *,
    close: float = 101.5,
    support: float = 100.0,
    rsi: float = 40.0,
    macd: float = 0.5,
    macd_signal: float = 0.3,
    bollinger_percent_b: float = 0.20,
) -> AnalysisSnapshot:
    indicators = IndicatorSnapshot(
        as_of=AS_OF,
        close=close,
        volume=1_200_000,
        ma20=105.0,
        ma50=106.0,
        ma200=103.0,
        rsi=rsi,
        macd=macd,
        macd_signal=macd_signal,
        macd_histogram=macd - macd_signal,
        bollinger_percent_b=bollinger_percent_b,
        atr=2.0,
        obv=25_000_000,
        support=support,
        resistance=115.0,
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
    previous_close: float = 99.8,
) -> MeanReversionContext:
    return MeanReversionContext(
        previous_close=previous_close,
    )


def make_regime(
    regime: MarketRegime = MarketRegime.SIDEWAYS,
    *,
    confidence: float = 0.9,
    bearish_votes: int | None = None,
) -> RegimeClassification:
    resolved_bearish_votes = (
        bearish_votes
        if bearish_votes is not None
        else 4
        if regime is MarketRegime.BEARISH
        else 0
    )

    bullish_votes = (
        4
        if regime is MarketRegime.BULLISH
        else 0
    )

    return RegimeClassification(
        regime=regime,
        confidence=confidence,
        bullish_votes=bullish_votes,
        bearish_votes=resolved_bearish_votes,
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


def test_confirmed_mean_reversion_returns_buy() -> None:
    result = evaluate_mean_reversion(
        make_analysis(),
        make_regime(),
        make_context(),
    )

    assert result.signal is Signal.BUY
    assert result.score == 100.0
    assert result.confidence == 0.9
    assert result.vetoed is False
    assert "MEAN_REVERSION_CONFIRMED" in evidence_codes(result)


def test_reversal_without_momentum_returns_watch() -> None:
    result = evaluate_mean_reversion(
        make_analysis(
            macd=0.2,
            macd_signal=0.4,
        ),
        make_regime(),
        make_context(),
    )

    assert result.signal is Signal.WATCH
    assert result.score == 95.0
    assert result.vetoed is False
    assert (
        "MEAN_REVERSION_AWAITING_CONFIRMATION"
        in evidence_codes(result)
    )


def test_support_zone_without_reversal_returns_hold() -> None:
    result = evaluate_mean_reversion(
        make_analysis(close=100.5),
        make_regime(),
        make_context(previous_close=100.6),
    )

    assert result.signal is Signal.HOLD
    assert result.vetoed is False
    assert "MEAN_REVERSION_ABSENT" in evidence_codes(result)


def test_close_far_above_support_returns_hold() -> None:
    result = evaluate_mean_reversion(
        make_analysis(close=108.0),
        make_regime(),
        make_context(),
    )

    assert result.signal is Signal.HOLD
    assert result.vetoed is False
    assert "MEAN_REVERSION_CONFIRMED" not in evidence_codes(result)


def test_bullish_regime_disables_strategy_without_veto() -> None:
    result = evaluate_mean_reversion(
        make_analysis(),
        make_regime(MarketRegime.BULLISH),
        make_context(),
    )

    assert result.signal is Signal.HOLD
    assert result.vetoed is False
    assert "MEAN_REVERSION_ABSENT" in evidence_codes(result)


def test_strong_bearish_regime_vetoes_strategy() -> None:
    result = evaluate_mean_reversion(
        make_analysis(),
        make_regime(MarketRegime.BEARISH),
        make_context(),
    )

    assert result.signal is Signal.HOLD
    assert result.score < 0
    assert result.vetoed is True
    assert "strong bearish trend" in result.veto_reason.lower()
    assert (
        "MEAN_REVERSION_INVALIDATED"
        in evidence_codes(result)
    )


def test_high_volatility_regime_vetoes_strategy() -> None:
    result = evaluate_mean_reversion(
        make_analysis(),
        make_regime(MarketRegime.HIGH_VOLATILITY),
        make_context(),
    )

    assert result.signal is Signal.HOLD
    assert result.score < 0
    assert result.vetoed is True
    assert "high-volatility" in result.veto_reason.lower()


def test_material_support_break_vetoes_strategy() -> None:
    result = evaluate_mean_reversion(
        make_analysis(
            close=97.0,
            support=100.0,
        ),
        make_regime(),
        make_context(previous_close=99.5),
    )

    assert result.signal is Signal.HOLD
    assert result.score < 0
    assert result.vetoed is True
    assert "broken below local support" in result.veto_reason


def test_strategy_requires_previous_support_test() -> None:
    result = evaluate_mean_reversion(
        make_analysis(),
        make_regime(),
        make_context(previous_close=105.0),
    )

    assert result.signal is Signal.HOLD
    assert result.vetoed is False
    assert "MEAN_REVERSION_CONFIRMED" not in evidence_codes(result)


def test_evaluation_is_deterministic() -> None:
    analysis = make_analysis()
    regime = make_regime()
    context = make_context()

    first = evaluate_mean_reversion(
        analysis,
        regime,
        context,
    )
    second = evaluate_mean_reversion(
        analysis,
        regime,
        context,
    )

    assert first == second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_support_zone_above_pct": 0},
        {"prior_touch_tolerance_pct": -1},
        {"max_prior_support_breach_pct": -1},
        {"max_current_support_break_pct": 0},
        {"min_reversal_pct": 0},
        {"rsi_min": 70, "rsi_max": 60},
        {"bollinger_min": 0.5, "bollinger_max": 0.2},
        {"watch_score": 80, "buy_score": 70},
    ],
)
def test_thresholds_reject_invalid_configuration(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        MeanReversionThresholds(**kwargs)


@pytest.mark.parametrize(
    "previous_close",
    [
        0,
        -1,
        float("nan"),
        float("inf"),
    ],
)
def test_context_rejects_invalid_previous_close(
    previous_close: float,
) -> None:
    with pytest.raises(ValueError):
        MeanReversionContext(
            previous_close=previous_close,
        )
