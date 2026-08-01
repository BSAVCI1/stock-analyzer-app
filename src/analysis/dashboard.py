"""Deterministic Trading Expert dashboard pipeline.

This module joins the completed P1 analysis layers:

- canonical analysis snapshot
- market-regime classification
- trend-pullback, breakout and mean-reversion strategies
- weighted scoring and conflict resolution
- risk management and paper-order generation

It contains no Streamlit code, broker integration or order-execution logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import pandas as pd

from .breakout import (
    BreakoutContext,
    evaluate_breakout,
)
from .mean_reversion import (
    MeanReversionContext,
    evaluate_mean_reversion,
)
from .model import (
    AnalysisSnapshot,
    StrategyResult,
)
from .regime import (
    RegimeClassification,
    classify_history,
)
from .risk import (
    RiskDecision,
    apply_risk_management,
)
from .scoring import (
    ScoreComponents,
    resolve_recommendation,
)
from .trend_pullback import evaluate_trend_pullback


def _finite_number(
    name: str,
    value: object,
) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a finite number."
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite number."
        ) from exc

    if not isfinite(result):
        raise ValueError(
            f"{name} must be a finite number."
        )

    return result


def _clip_score(value: object) -> float:
    """Clamp a finite score to the canonical -100 to 100 range."""

    result = _finite_number("score", value)
    return round(max(-100.0, min(100.0, result)), 4)


def _optional_number(
    metadata: Mapping[str, object],
    key: str,
) -> float | None:
    value = metadata.get(key)

    if value is None or isinstance(value, bool):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not isfinite(result):
        return None

    return result


@dataclass(frozen=True, slots=True)
class ComponentTrace:
    """Traceable explanation for one weighted score component."""

    name: str
    score: float
    explanation: str

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        explanation = str(
            self.explanation or ""
        ).strip()

        if not name:
            raise ValueError(
                "ComponentTrace.name cannot be blank."
            )

        if not explanation:
            raise ValueError(
                "ComponentTrace.explanation cannot be blank."
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "score",
            _clip_score(self.score),
        )
        object.__setattr__(
            self,
            "explanation",
            explanation,
        )


@dataclass(frozen=True, slots=True)
class TradingExpertReport:
    """Complete deterministic output consumed by the dashboard."""

    analysis: AnalysisSnapshot
    regime: RegimeClassification
    strategy_results: tuple[StrategyResult, ...]
    components: ScoreComponents
    component_traces: tuple[ComponentTrace, ...]
    recommendation: StrategyResult
    risk_decision: RiskDecision

    def __post_init__(self) -> None:
        if not isinstance(
            self.analysis,
            AnalysisSnapshot,
        ):
            raise ValueError(
                "analysis must be an AnalysisSnapshot."
            )

        if not isinstance(
            self.regime,
            RegimeClassification,
        ):
            raise ValueError(
                "regime must be a RegimeClassification."
            )

        strategies = tuple(self.strategy_results)

        if not strategies:
            raise ValueError(
                "At least one strategy result is required."
            )

        if not all(
            isinstance(result, StrategyResult)
            for result in strategies
        ):
            raise ValueError(
                "strategy_results must contain "
                "StrategyResult objects."
            )

        traces = tuple(self.component_traces)

        if len(traces) != 6:
            raise ValueError(
                "Exactly six component traces are required."
            )

        if not isinstance(
            self.components,
            ScoreComponents,
        ):
            raise ValueError(
                "components must be ScoreComponents."
            )

        if not isinstance(
            self.recommendation,
            StrategyResult,
        ):
            raise ValueError(
                "recommendation must be a StrategyResult."
            )

        if not isinstance(
            self.risk_decision,
            RiskDecision,
        ):
            raise ValueError(
                "risk_decision must be a RiskDecision."
            )

        object.__setattr__(
            self,
            "strategy_results",
            strategies,
        )
        object.__setattr__(
            self,
            "component_traces",
            traces,
        )


def score_fundamentals(
    metadata: Mapping[str, object] | None,
    quote_type: str,
) -> tuple[float, str]:
    """Return a deterministic fundamental-quality score.

    ETFs, funds and indexes receive a neutral score because company-level
    profitability and leverage measures are not directly comparable.
    """

    metadata = metadata or {}
    resolved_quote_type = str(
        quote_type or "UNKNOWN"
    ).strip().upper()

    if resolved_quote_type in {
        "ETF",
        "MUTUALFUND",
        "INDEX",
    }:
        return (
            0.0,
            (
                f"{resolved_quote_type} receives a neutral "
                "corporate-fundamental score."
            ),
        )

    factor_scores: list[float] = []
    reasons: list[str] = []

    margin = _optional_number(
        metadata,
        "profitMargins",
    )

    if margin is not None:
        if margin >= 0.15:
            score = 100.0
        elif margin >= 0.05:
            score = 60.0
        elif margin >= 0:
            score = 20.0
        else:
            score = -80.0

        factor_scores.append(score)
        reasons.append(
            f"Net margin {margin * 100:.1f}% "
            f"contributed {score:.0f}."
        )

    roe = _optional_number(
        metadata,
        "returnOnEquity",
    )

    if roe is not None:
        if roe >= 0.20:
            score = 100.0
        elif roe >= 0.10:
            score = 60.0
        elif roe >= 0:
            score = 20.0
        else:
            score = -80.0

        factor_scores.append(score)
        reasons.append(
            f"ROE {roe * 100:.1f}% "
            f"contributed {score:.0f}."
        )

    debt_to_equity = _optional_number(
        metadata,
        "debtToEquity",
    )

    if debt_to_equity is not None:
        if debt_to_equity <= 50:
            score = 80.0
        elif debt_to_equity <= 100:
            score = 40.0
        elif debt_to_equity <= 200:
            score = -20.0
        else:
            score = -80.0

        factor_scores.append(score)
        reasons.append(
            f"Debt/equity {debt_to_equity:.1f} "
            f"contributed {score:.0f}."
        )

    trailing_pe = _optional_number(
        metadata,
        "trailingPE",
    )

    if trailing_pe is not None:
        if 0 < trailing_pe <= 20:
            score = 80.0
        elif trailing_pe <= 35:
            score = 40.0
        elif trailing_pe <= 60:
            score = -20.0
        elif trailing_pe > 60:
            score = -60.0
        else:
            score = 0.0

        factor_scores.append(score)
        reasons.append(
            f"Trailing P/E {trailing_pe:.1f} "
            f"contributed {score:.0f}."
        )

    if not factor_scores:
        return (
            0.0,
            (
                "No comparable corporate fundamentals "
                "were available; score remains neutral."
            ),
        )

    final_score = _clip_score(
        sum(factor_scores)
        / len(factor_scores)
    )

    return final_score, " ".join(reasons)


def build_score_components(
    analysis: AnalysisSnapshot,
    regime: RegimeClassification,
    strategy_results: Sequence[StrategyResult],
    *,
    average_volume: float,
    fundamental_score: float,
    fundamental_explanation: str,
) -> tuple[
    ScoreComponents,
    tuple[ComponentTrace, ...],
]:
    """Build all six traceable weighted-score components."""

    if not isinstance(
        analysis,
        AnalysisSnapshot,
    ):
        raise ValueError(
            "analysis must be an AnalysisSnapshot."
        )

    if not isinstance(
        regime,
        RegimeClassification,
    ):
        raise ValueError(
            "regime must be a RegimeClassification."
        )

    strategies = tuple(strategy_results)

    if not strategies:
        raise ValueError(
            "At least one strategy result is required."
        )

    if not all(
        isinstance(result, StrategyResult)
        for result in strategies
    ):
        raise ValueError(
            "strategy_results must contain "
            "StrategyResult objects."
        )

    average_volume = _finite_number(
        "average_volume",
        average_volume,
    )

    if average_volume <= 0:
        raise ValueError(
            "average_volume must be greater than zero."
        )

    indicators = analysis.indicators

    trend_score = _clip_score(
        (
            regime.bullish_votes
            - regime.bearish_votes
        )
        * 25
    )

    setup_score = _clip_score(
        sum(
            result.score
            for result in strategies
        )
        / len(strategies)
    )

    rsi_score = _clip_score(
        (indicators.rsi - 50) * 2
    )

    macd_score = _clip_score(
        (
            indicators.macd_histogram
            / indicators.atr
        )
        * 100
    )

    momentum_score = _clip_score(
        (rsi_score + macd_score) / 2
    )

    volume_ratio = (
        indicators.volume
        / average_volume
    )

    volume_score = _clip_score(
        (volume_ratio - 1) * 100
    )

    atr_pct = (
        indicators.atr
        / indicators.close
    ) * 100

    volatility_score = _clip_score(
        60 - 25 * atr_pct
    )

    fundamental_score = _clip_score(
        fundamental_score
    )

    components = ScoreComponents(
        trend=trend_score,
        setup=setup_score,
        momentum=momentum_score,
        volume=volume_score,
        volatility=volatility_score,
        fundamental=fundamental_score,
    )

    traces = (
        ComponentTrace(
            name="Trend",
            score=trend_score,
            explanation=(
                "Trend score equals bullish votes minus "
                "bearish votes, multiplied by 25. "
                f"Votes: {regime.bullish_votes} bullish and "
                f"{regime.bearish_votes} bearish."
            ),
        ),
        ComponentTrace(
            name="Setup",
            score=setup_score,
            explanation=(
                "Setup score is the arithmetic mean of the "
                "three deterministic strategy scores: "
                + ", ".join(
                    (
                        f"{result.strategy} "
                        f"{result.score:.1f}"
                    )
                    for result in strategies
                )
                + "."
            ),
        ),
        ComponentTrace(
            name="Momentum",
            score=momentum_score,
            explanation=(
                "Momentum is the mean of the RSI-centred "
                f"score ({rsi_score:.1f}) and ATR-normalised "
                f"MACD histogram score ({macd_score:.1f})."
            ),
        ),
        ComponentTrace(
            name="Volume",
            score=volume_score,
            explanation=(
                "Volume score measures current completed-session "
                "volume against the preceding 20-session average. "
                f"Relative volume: {volume_ratio:.2f}x."
            ),
        ),
        ComponentTrace(
            name="Volatility",
            score=volatility_score,
            explanation=(
                "Volatility support is calculated as "
                "60 minus 25 times ATR as a percentage of price. "
                f"ATR/price: {atr_pct:.2f}%."
            ),
        ),
        ComponentTrace(
            name="Fundamental",
            score=fundamental_score,
            explanation=fundamental_explanation,
        ),
    )

    return components, traces


def build_trading_expert_report(
    analysis: AnalysisSnapshot,
    history: pd.DataFrame,
    metadata: Mapping[str, object] | None = None,
) -> TradingExpertReport:
    """Run the complete deterministic P1 Trading Expert pipeline."""

    if not isinstance(
        analysis,
        AnalysisSnapshot,
    ):
        raise ValueError(
            "analysis must be an AnalysisSnapshot."
        )

    if not isinstance(
        history,
        pd.DataFrame,
    ) or history.empty:
        raise ValueError(
            "history must be a non-empty DataFrame."
        )

    required_columns = (
        "High",
        "Low",
        "Close",
        "Volume",
    )

    missing = [
        column
        for column in required_columns
        if column not in history.columns
    ]

    if missing:
        raise ValueError(
            "history is missing required columns: "
            + ", ".join(missing)
        )

    clean = history.loc[
        :,
        list(required_columns),
    ].copy()

    for column in required_columns:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="coerce",
        )

    clean = clean.dropna(
        subset=list(required_columns)
    )

    if len(clean) < 200:
        raise ValueError(
            "Trading Expert requires at least "
            "200 completed sessions."
        )

    previous_close = float(
        clean["Close"].iloc[-2]
    )

    range_history = clean.iloc[-21:-1]

    if len(range_history) < 20:
        raise ValueError(
            "Breakout evaluation requires "
            "20 preceding sessions."
        )

    range_resistance = float(
        range_history["High"].max()
    )

    average_volume = float(
        range_history["Volume"].mean()
    )

    if average_volume <= 0:
        average_volume = max(
            float(clean["Volume"].iloc[-1]),
            1.0,
        )

    regime = classify_history(clean)

    trend_pullback = evaluate_trend_pullback(
        analysis,
        regime,
    )

    breakout = evaluate_breakout(
        analysis,
        regime,
        BreakoutContext(
            range_resistance=range_resistance,
            average_volume=average_volume,
            previous_close=previous_close,
            range_sessions=len(range_history),
        ),
    )

    mean_reversion = evaluate_mean_reversion(
        analysis,
        regime,
        MeanReversionContext(
            previous_close=previous_close,
        ),
    )

    strategy_results = (
        trend_pullback,
        breakout,
        mean_reversion,
    )

    fundamental_score, fundamental_explanation = (
        score_fundamentals(
            metadata,
            analysis.quote_type,
        )
    )

    components, component_traces = (
        build_score_components(
            analysis,
            regime,
            strategy_results,
            average_volume=average_volume,
            fundamental_score=fundamental_score,
            fundamental_explanation=(
                fundamental_explanation
            ),
        )
    )

    recommendation = resolve_recommendation(
        components,
        strategy_results,
    )

    risk_decision = apply_risk_management(
        analysis,
        recommendation,
    )

    return TradingExpertReport(
        analysis=analysis,
        regime=regime,
        strategy_results=strategy_results,
        components=components,
        component_traces=component_traces,
        recommendation=recommendation,
        risk_decision=risk_decision,
    )
