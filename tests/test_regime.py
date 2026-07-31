import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    MarketRegime,
    RegimeInputs,
    RegimeThresholds,
    build_regime_inputs,
    classify_history,
    classify_market_regime,
)


def make_history(
    closes: np.ndarray,
    *,
    intraday_range: float = 1.0,
) -> pd.DataFrame:
    index = pd.date_range(
        "2025-01-01",
        periods=len(closes),
        freq="B",
        tz="UTC",
    )

    close = np.asarray(closes, dtype=float)

    return pd.DataFrame(
        {
            "Open": close,
            "High": close + intraday_range / 2,
            "Low": close - intraday_range / 2,
            "Close": close,
            "Volume": np.full(len(close), 1_000_000),
        },
        index=index,
    )


@pytest.fixture
def bullish_inputs() -> RegimeInputs:
    return RegimeInputs(
        short_trend_slope_pct=0.20,
        long_trend_slope_pct=0.10,
        price_vs_ma200_pct=8.0,
        short_vs_long_ma_pct=2.5,
        atr_pct=1.5,
    )


@pytest.fixture
def bearish_inputs() -> RegimeInputs:
    return RegimeInputs(
        short_trend_slope_pct=-0.20,
        long_trend_slope_pct=-0.10,
        price_vs_ma200_pct=-8.0,
        short_vs_long_ma_pct=-2.5,
        atr_pct=1.5,
    )


def test_market_regime_contains_required_values() -> None:
    assert [regime.value for regime in MarketRegime] == [
        "BULLISH",
        "BEARISH",
        "SIDEWAYS",
        "HIGH_VOLATILITY",
    ]


def test_bullish_fixture_is_classified_deterministically(
    bullish_inputs: RegimeInputs,
) -> None:
    first = classify_market_regime(bullish_inputs)
    second = classify_market_regime(bullish_inputs)

    assert first == second
    assert first.regime is MarketRegime.BULLISH
    assert first.bullish_votes == 4
    assert first.bearish_votes == 0
    assert first.confidence == 1.0


def test_bearish_fixture_is_classified_deterministically(
    bearish_inputs: RegimeInputs,
) -> None:
    result = classify_market_regime(bearish_inputs)

    assert result.regime is MarketRegime.BEARISH
    assert result.bearish_votes == 4
    assert result.bullish_votes == 0
    assert result.confidence == 1.0


def test_sideways_fixture_with_mixed_evidence() -> None:
    result = classify_market_regime(
        RegimeInputs(
            short_trend_slope_pct=0.12,
            long_trend_slope_pct=-0.06,
            price_vs_ma200_pct=3.0,
            short_vs_long_ma_pct=-0.8,
            atr_pct=1.2,
        )
    )

    assert result.regime is MarketRegime.SIDEWAYS
    assert result.bullish_votes == 2
    assert result.bearish_votes == 2
    assert result.confidence == 1.0


def test_sideways_fixture_with_weak_evidence() -> None:
    result = classify_market_regime(
        RegimeInputs(
            short_trend_slope_pct=0.01,
            long_trend_slope_pct=-0.01,
            price_vs_ma200_pct=0.5,
            short_vs_long_ma_pct=-0.2,
            atr_pct=1.0,
        )
    )

    assert result.regime is MarketRegime.SIDEWAYS
    assert result.bullish_votes == 0
    assert result.bearish_votes == 0


def test_high_volatility_takes_precedence_over_bullish_trend(
    bullish_inputs: RegimeInputs,
) -> None:
    volatile = RegimeInputs(
        short_trend_slope_pct=(
            bullish_inputs.short_trend_slope_pct
        ),
        long_trend_slope_pct=(
            bullish_inputs.long_trend_slope_pct
        ),
        price_vs_ma200_pct=(
            bullish_inputs.price_vs_ma200_pct
        ),
        short_vs_long_ma_pct=(
            bullish_inputs.short_vs_long_ma_pct
        ),
        atr_pct=5.0,
    )

    result = classify_market_regime(volatile)

    assert result.regime is MarketRegime.HIGH_VOLATILITY
    assert result.bullish_votes == 4
    assert result.confidence > 0.7


@pytest.mark.parametrize(
    "atr_pct",
    [-0.01, float("nan"), float("inf")],
)
def test_regime_inputs_reject_invalid_volatility(
    atr_pct: float,
) -> None:
    with pytest.raises(ValueError):
        RegimeInputs(
            short_trend_slope_pct=0,
            long_trend_slope_pct=0,
            price_vs_ma200_pct=0,
            short_vs_long_ma_pct=0,
            atr_pct=atr_pct,
        )


def test_thresholds_reject_non_positive_values() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        RegimeThresholds(
            high_volatility_atr_pct=0,
        )


def test_build_regime_inputs_from_rising_history() -> None:
    history = make_history(
        np.linspace(100, 160, 260),
        intraday_range=0.8,
    )

    inputs = build_regime_inputs(history)

    assert inputs.short_trend_slope_pct > 0
    assert inputs.long_trend_slope_pct > 0
    assert inputs.price_vs_ma200_pct > 0
    assert inputs.short_vs_long_ma_pct > 0
    assert 0 < inputs.atr_pct < 4


def test_rising_history_is_bullish() -> None:
    history = make_history(
        np.linspace(100, 160, 260),
        intraday_range=0.8,
    )

    result = classify_history(history)

    assert result.regime is MarketRegime.BULLISH
    assert result.bullish_votes >= 3


def test_falling_history_is_bearish() -> None:
    history = make_history(
        np.linspace(160, 100, 260),
        intraday_range=0.8,
    )

    result = classify_history(history)

    assert result.regime is MarketRegime.BEARISH
    assert result.bearish_votes >= 3


def test_history_builder_rejects_missing_columns() -> None:
    history = pd.DataFrame(
        {
            "Close": np.linspace(100, 120, 260),
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        build_regime_inputs(history)


def test_history_builder_requires_sufficient_rows() -> None:
    history = make_history(
        np.linspace(100, 120, 199),
    )

    with pytest.raises(
        ValueError,
        match="at least 200",
    ):
        build_regime_inputs(history)
