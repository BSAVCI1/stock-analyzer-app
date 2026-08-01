"""Deterministic order-candidate ranking."""

from __future__ import annotations

from .models import ScannerAnalysisOutcome


def calculate_candidate_rank_score(
    outcome: ScannerAnalysisOutcome,
) -> float:
    """Return a transparent score from zero to 100."""

    positive_score = max(
        0.0,
        min(100.0, outcome.score),
    )

    confidence = max(
        0.0,
        min(1.0, outcome.confidence),
    )

    regime_confidence = max(
        0.0,
        min(
            1.0,
            outcome.regime_confidence,
        ),
    )

    reward_to_risk = (
        outcome.order.reward_to_risk
        if outcome.order is not None
        else 0.0
    )

    reward_component = max(
        0.0,
        min(
            1.0,
            reward_to_risk / 4.0,
        ),
    )

    score = (
        positive_score * 0.50
        + confidence * 30.0
        + reward_component * 15.0
        + regime_confidence * 5.0
    )

    return round(score, 6)
