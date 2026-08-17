"""Deterministic, decomposed candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ScannerAnalysisOutcome


@dataclass(frozen=True, slots=True)
class CandidateRankScore:
    """Transparent weighted score from zero to 100."""

    analysis_score: float
    confidence: float
    reward_to_risk: float
    regime_confidence: float
    total: float

    def as_dict(
        self,
    ) -> dict[str, float]:
        return {
            "analysis_score":
            self.analysis_score,
            "confidence":
            self.confidence,
            "reward_to_risk":
            self.reward_to_risk,
            "regime_confidence":
            self.regime_confidence,
        }


def calculate_candidate_rank(
    outcome: ScannerAnalysisOutcome,
) -> CandidateRankScore:
    """Return the total and exact weighted components."""

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

    total = round(
        positive_score * 0.50
        + confidence * 30.0
        + reward_component * 15.0
        + regime_confidence * 5.0,
        6,
    )

    analysis_value = round(
        positive_score * 0.50,
        6,
    )
    confidence_value = round(
        confidence * 30.0,
        6,
    )
    reward_value = round(
        reward_component * 15.0,
        6,
    )
    regime_value = round(
        total
        - analysis_value
        - confidence_value
        - reward_value,
        6,
    )

    return CandidateRankScore(
        analysis_score=analysis_value,
        confidence=confidence_value,
        reward_to_risk=reward_value,
        regime_confidence=regime_value,
        total=total,
    )


def calculate_candidate_rank_score(
    outcome: ScannerAnalysisOutcome,
) -> float:
    """Backward-compatible scalar candidate score."""

    return calculate_candidate_rank(
        outcome
    ).total
