"""Trading-expert analysis domain models and classifiers."""

from .breakout import (
    BreakoutContext,
    BreakoutThresholds,
    evaluate_breakout,
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
from .trend_pullback import (
    TrendPullbackThresholds,
    evaluate_trend_pullback,
)

__all__ = [
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
