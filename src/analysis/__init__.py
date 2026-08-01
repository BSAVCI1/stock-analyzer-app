"""Trading-expert analysis domain models and classifiers."""

from .breakout import (
    BreakoutContext,
    BreakoutThresholds,
    evaluate_breakout,
)
from .dashboard import (
    ComponentTrace,
    TradingExpertReport,
    build_score_components,
    build_trading_expert_report,
    score_fundamentals,
)
from .mean_reversion import (
    MeanReversionContext,
    MeanReversionThresholds,
    evaluate_mean_reversion,
)
from .model import (
    AnalysisSnapshot,
    Evidence,
    EvidenceDirection,
    IndicatorSnapshot,
    Signal,
    StrategyResult,
)
from .regime import (
    MarketRegime,
    RegimeClassification,
    RegimeInputs,
    RegimeThresholds,
    build_regime_inputs,
    classify_history,
    classify_market_regime,
)
from .risk import (
    PaperOrder,
    RiskDecision,
    RiskThresholds,
    apply_risk_management,
)
from .scoring import (
    RecommendationThresholds,
    ScoreComponents,
    ScoreWeights,
    resolve_recommendation,
    weighted_component_score,
)
from .trend_pullback import (
    TrendPullbackThresholds,
    evaluate_trend_pullback,
)

__all__ = [
    "ComponentTrace",
    "TradingExpertReport",
    "build_score_components",
    "build_trading_expert_report",
    "score_fundamentals",
    "PaperOrder",
    "RiskDecision",
    "RiskThresholds",
    "apply_risk_management",
    "RecommendationThresholds",
    "ScoreComponents",
    "ScoreWeights",
    "resolve_recommendation",
    "weighted_component_score",
    "MeanReversionContext",
    "MeanReversionThresholds",
    "evaluate_mean_reversion",
    "BreakoutContext",
    "BreakoutThresholds",
    "evaluate_breakout",
    "AnalysisSnapshot",
    "Evidence",
    "EvidenceDirection",
    "IndicatorSnapshot",
    "MarketRegime",
    "RegimeClassification",
    "RegimeInputs",
    "RegimeThresholds",
    "Signal",
    "StrategyResult",
    "TrendPullbackThresholds",
    "build_regime_inputs",
    "classify_history",
    "classify_market_regime",
    "evaluate_trend_pullback",
]
