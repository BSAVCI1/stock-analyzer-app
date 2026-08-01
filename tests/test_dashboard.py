from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import (
    AnalysisSnapshot,
    IndicatorSnapshot,
    Signal,
    build_trading_expert_report,
    score_fundamentals,
)


def make_history() -> pd.DataFrame:
    index = pd.date_range(
        "2025-08-01",
        periods=260,
        freq="B",
        tz="UTC",
    )

    close = np.linspace(
        90.0,
        108.0,
        len(index),
    )

    volume = np.full(
        len(index),
        1_000_000.0,
    )

    volume[-1] = 2_000_000.0

    return pd.DataFrame(
        {
            "Open": close - 0.25,
            "High": close + 1.00,
            "Low": close - 1.00,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def make_analysis(
    history: pd.DataFrame,
) -> AnalysisSnapshot:
    as_of = history.index[-1].to_pydatetime()

    return AnalysisSnapshot(
        symbol="TEST",
        display_name="Test Instrument",
        fetched_at_utc=(
            as_of + timedelta(minutes=10)
        ),
        history_rows=len(history),
        indicators=IndicatorSnapshot(
            as_of=as_of,
            close=108.0,
            volume=2_000_000,
            ma20=106.5,
            ma50=104.0,
            ma200=98.0,
            rsi=61.0,
            macd=1.50,
            macd_signal=1.00,
            macd_histogram=0.50,
            bollinger_percent_b=0.80,
            atr=2.0,
            obv=30_000_000,
            support=100.0,
            resistance=125.0,
        ),
        quote_type="EQUITY",
        currency="USD",
        exchange="NMS",
    )


def make_metadata() -> dict[str, float]:
    return {
        "profitMargins": 0.12,
        "returnOnEquity": 0.18,
        "debtToEquity": 70.0,
        "trailingPE": 24.0,
    }


def test_dashboard_report_is_deterministic() -> None:
    history = make_history()
    analysis = make_analysis(history)
    metadata = make_metadata()

    first = build_trading_expert_report(
        analysis,
        history,
        metadata,
    )

    second = build_trading_expert_report(
        analysis,
        history,
        metadata,
    )

    assert first == second


def test_dashboard_runs_all_three_strategies() -> None:
    history = make_history()

    report = build_trading_expert_report(
        make_analysis(history),
        history,
        make_metadata(),
    )

    strategy_names = {
        result.strategy
        for result in report.strategy_results
    }

    assert strategy_names == {
        "trend_pullback",
        "breakout",
        "mean_reversion",
    }


def test_dashboard_has_six_traceable_components() -> None:
    history = make_history()

    report = build_trading_expert_report(
        make_analysis(history),
        history,
        make_metadata(),
    )

    assert len(report.component_traces) == 6

    trace_names = {
        trace.name
        for trace in report.component_traces
    }

    assert trace_names == {
        "Trend",
        "Setup",
        "Momentum",
        "Volume",
        "Volatility",
        "Fundamental",
    }

    for trace in report.component_traces:
        assert trace.explanation
        assert -100 <= trace.score <= 100


def test_final_recommendation_contains_calculation_trace() -> None:
    history = make_history()

    report = build_trading_expert_report(
        make_analysis(history),
        history,
        make_metadata(),
    )

    evidence_codes = {
        item.code
        for item in (
            report
            .risk_decision
            .recommendation
            .evidence
        )
    }

    assert {
        "TREND_SCORE",
        "SETUP_SCORE",
        "MOMENTUM_SCORE",
        "VOLUME_SCORE",
        "VOLATILITY_SCORE",
        "FUNDAMENTAL_SCORE",
        "WEIGHTED_COMPONENT_SCORE",
        "STRATEGY_CONSENSUS",
        "CONFLICT_RESOLUTION",
    }.issubset(evidence_codes)


def test_actionable_dashboard_decision_is_paper_only() -> None:
    history = make_history()

    report = build_trading_expert_report(
        make_analysis(history),
        history,
        make_metadata(),
    )

    recommendation = (
        report.risk_decision.recommendation
    )

    if recommendation.signal in {
        Signal.BUY,
        Signal.SELL,
    }:
        assert report.risk_decision.order is not None
        assert (
            report.risk_decision.order.paper_only
            is True
        )
        assert (
            report
            .risk_decision
            .order
            .invalidation_price
            > 0
        )
    else:
        assert report.risk_decision.order is None


def test_etf_fundamentals_are_neutral() -> None:
    score, explanation = score_fundamentals(
        {
            "profitMargins": -0.50,
            "returnOnEquity": -1.00,
        },
        "ETF",
    )

    assert score == 0.0
    assert "neutral" in explanation.lower()


def test_dashboard_module_has_no_execution_interface() -> None:
    source = Path(
        "src/analysis/dashboard.py"
    ).read_text(encoding="utf-8")

    forbidden_calls = (
        "place_order(",
        "submit_order(",
        "execute_order(",
        "send_order(",
    )

    assert not any(
        value in source
        for value in forbidden_calls
    )
