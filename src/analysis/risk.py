"""Deterministic risk management and paper-order generation.

The risk manager converts an actionable BUY or SELL recommendation into a
paper-only order containing:

- entry zone
- ATR/structure-based invalidation stop
- three target levels
- reward-to-risk calculation
- signal expiry
- deterministic risk vetoes

No broker connection or order submission exists in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import isfinite

from .model import (
    AnalysisSnapshot,
    Evidence,
    EvidenceDirection,
    Signal,
    StrategyResult,
)


RISK_STRATEGY_NAME = "risk_managed_recommendation"


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")

    return value.strip()


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc

    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number.")

    return result


def _positive_number(name: str, value: object) -> float:
    result = _finite_number(name, value)

    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return result


def _aware_datetime(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")

    return value


@dataclass(frozen=True, slots=True)
class RiskThresholds:
    """Configuration for deterministic paper-order risk management."""

    entry_zone_atr_fraction: float = 0.25
    atr_stop_multiple: float = 1.50
    structure_buffer_atr: float = 0.25
    target_atr_multiple: float = 3.00

    min_reward_to_risk: float = 2.00
    min_stop_distance_pct: float = 0.50
    max_stop_distance_pct: float = 15.00

    expiry_days: int = 5

    def __post_init__(self) -> None:
        for name in (
            "entry_zone_atr_fraction",
            "atr_stop_multiple",
            "structure_buffer_atr",
            "target_atr_multiple",
            "min_reward_to_risk",
            "min_stop_distance_pct",
            "max_stop_distance_pct",
        ):
            object.__setattr__(
                self,
                name,
                _positive_number(name, getattr(self, name)),
            )

        if (
            self.min_stop_distance_pct
            >= self.max_stop_distance_pct
        ):
            raise ValueError(
                "min_stop_distance_pct must be less than "
                "max_stop_distance_pct."
            )

        if (
            isinstance(self.expiry_days, bool)
            or not isinstance(self.expiry_days, int)
        ):
            raise ValueError("expiry_days must be an integer.")

        if self.expiry_days < 1:
            raise ValueError(
                "expiry_days must be at least 1."
            )


@dataclass(frozen=True, slots=True)
class PaperOrder:
    """Validated paper-only order produced by the risk manager."""

    symbol: str
    signal: Signal

    created_at: datetime
    expires_at: datetime

    entry_low: float
    entry_high: float
    stop_price: float
    targets: tuple[float, ...]

    risk_per_share: float
    reward_to_risk: float

    paper_only: bool = True

    def __post_init__(self) -> None:
        symbol = _required_text("symbol", self.symbol)

        if self.signal not in {
            Signal.BUY,
            Signal.SELL,
        }:
            raise ValueError(
                "A paper order requires BUY or SELL."
            )

        created_at = _aware_datetime(
            "created_at",
            self.created_at,
        )
        expires_at = _aware_datetime(
            "expires_at",
            self.expires_at,
        )

        if expires_at <= created_at:
            raise ValueError(
                "expires_at must be later than created_at."
            )

        entry_low = _positive_number(
            "entry_low",
            self.entry_low,
        )
        entry_high = _positive_number(
            "entry_high",
            self.entry_high,
        )
        stop_price = _positive_number(
            "stop_price",
            self.stop_price,
        )
        risk_per_share = _positive_number(
            "risk_per_share",
            self.risk_per_share,
        )
        reward_to_risk = _positive_number(
            "reward_to_risk",
            self.reward_to_risk,
        )

        if entry_low > entry_high:
            raise ValueError(
                "entry_low cannot exceed entry_high."
            )

        targets = tuple(
            _positive_number("target", value)
            for value in self.targets
        )

        if len(targets) < 1:
            raise ValueError(
                "A paper order requires at least one target."
            )

        if self.signal is Signal.BUY:
            if stop_price >= entry_low:
                raise ValueError(
                    "BUY stop must be below the entry zone."
                )

            if not all(
                target > entry_high
                for target in targets
            ):
                raise ValueError(
                    "BUY targets must be above the entry zone."
                )

            if any(
                later <= earlier
                for earlier, later in zip(
                    targets,
                    targets[1:],
                )
            ):
                raise ValueError(
                    "BUY targets must be strictly ascending."
                )

        if self.signal is Signal.SELL:
            if stop_price <= entry_high:
                raise ValueError(
                    "SELL stop must be above the entry zone."
                )

            if not all(
                target < entry_low
                for target in targets
            ):
                raise ValueError(
                    "SELL targets must be below the entry zone."
                )

            if any(
                later >= earlier
                for earlier, later in zip(
                    targets,
                    targets[1:],
                )
            ):
                raise ValueError(
                    "SELL targets must be strictly descending."
                )

        if self.paper_only is not True:
            raise ValueError(
                "Generated orders must remain paper-only."
            )

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "entry_low", entry_low)
        object.__setattr__(self, "entry_high", entry_high)
        object.__setattr__(self, "stop_price", stop_price)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(
            self,
            "risk_per_share",
            risk_per_share,
        )
        object.__setattr__(
            self,
            "reward_to_risk",
            reward_to_risk,
        )

    @property
    def invalidation_price(self) -> float:
        """Return the defined invalidation point."""

        return self.stop_price

    @property
    def entry_midpoint(self) -> float:
        """Return the middle of the entry zone."""

        return (
            self.entry_low
            + self.entry_high
        ) / 2


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Risk-managed recommendation and optional paper order."""

    recommendation: StrategyResult
    order: PaperOrder | None = None
    risk_vetoes: tuple[str, ...] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.recommendation,
            StrategyResult,
        ):
            raise ValueError(
                "recommendation must be a StrategyResult."
            )

        if (
            self.order is not None
            and not isinstance(self.order, PaperOrder)
        ):
            raise ValueError(
                "order must be a PaperOrder or None."
            )

        vetoes = tuple(
            _required_text("risk_veto", veto)
            for veto in self.risk_vetoes
        )

        actionable = (
            self.recommendation.signal
            in {Signal.BUY, Signal.SELL}
        )

        if actionable:
            if self.recommendation.vetoed:
                raise ValueError(
                    "An actionable recommendation cannot be vetoed."
                )

            if self.order is None:
                raise ValueError(
                    "Every BUY or SELL requires a paper order "
                    "and invalidation point."
                )

            if (
                self.order.signal
                is not self.recommendation.signal
            ):
                raise ValueError(
                    "Order direction must match the recommendation."
                )

        if self.order is not None and not actionable:
            raise ValueError(
                "A paper order is allowed only for BUY or SELL."
            )

        if self.order is not None and vetoes:
            raise ValueError(
                "A valid order cannot contain risk vetoes."
            )

        object.__setattr__(
            self,
            "risk_vetoes",
            vetoes,
        )


def _risk_evidence(
    *,
    code: str,
    message: str,
    direction: EvidenceDirection,
    observed_value: float | str,
    strength: float = 1.0,
) -> Evidence:
    return Evidence(
        code=code,
        message=message,
        direction=direction,
        strength=strength,
        observed_value=observed_value,
    )


def _copy_recommendation(
    recommendation: StrategyResult,
    *,
    signal: Signal | None = None,
    additional_evidence: tuple[Evidence, ...] = (),
    vetoed: bool | None = None,
    veto_reason: str | None = None,
) -> StrategyResult:
    resolved_signal = signal or recommendation.signal
    resolved_vetoed = (
        recommendation.vetoed
        if vetoed is None
        else vetoed
    )

    resolved_veto_reason = (
        recommendation.veto_reason
        if vetoed is None
        else veto_reason
    )

    return StrategyResult(
        strategy=RISK_STRATEGY_NAME,
        signal=resolved_signal,
        score=recommendation.score,
        confidence=recommendation.confidence,
        evidence=(
            tuple(recommendation.evidence)
            + tuple(additional_evidence)
        ),
        vetoed=resolved_vetoed,
        veto_reason=resolved_veto_reason,
    )


def apply_risk_management(
    analysis: AnalysisSnapshot,
    recommendation: StrategyResult,
    thresholds: RiskThresholds | None = None,
) -> RiskDecision:
    """Apply risk gates and generate a deterministic paper order.

    BUY or SELL survives only when:

    - the invalidation stop is valid
    - stop distance is inside configured limits
    - the first target is directionally valid
    - reward-to-risk meets the configured minimum

    Failed risk checks convert the decision to vetoed HOLD.
    """

    if not isinstance(analysis, AnalysisSnapshot):
        raise ValueError(
            "analysis must be an AnalysisSnapshot."
        )

    if not isinstance(
        recommendation,
        StrategyResult,
    ):
        raise ValueError(
            "recommendation must be a StrategyResult."
        )

    thresholds = thresholds or RiskThresholds()

    if not isinstance(thresholds, RiskThresholds):
        raise ValueError(
            "thresholds must be RiskThresholds."
        )

    if recommendation.signal not in {
        Signal.BUY,
        Signal.SELL,
    }:
        evidence = (
            _risk_evidence(
                code="PAPER_ORDER_NOT_APPLICABLE",
                message=(
                    "No paper order was generated because "
                    f"the recommendation is "
                    f"{recommendation.signal.value}."
                ),
                direction=EvidenceDirection.NEUTRAL,
                observed_value=recommendation.signal.value,
                strength=0.5,
            ),
        )

        copied = _copy_recommendation(
            recommendation,
            additional_evidence=evidence,
        )

        vetoes = (
            (recommendation.veto_reason,)
            if recommendation.vetoed
            and recommendation.veto_reason
            else ()
        )

        return RiskDecision(
            recommendation=copied,
            order=None,
            risk_vetoes=vetoes,
        )

    indicators = analysis.indicators

    close = _positive_number(
        "close",
        indicators.close,
    )
    atr = _positive_number(
        "atr",
        indicators.atr,
    )
    support = _positive_number(
        "support",
        indicators.support,
    )
    resistance = _positive_number(
        "resistance",
        indicators.resistance,
    )

    entry_half_width = (
        atr
        * thresholds.entry_zone_atr_fraction
    )

    entry_low = close - entry_half_width
    entry_high = close + entry_half_width

    if recommendation.signal is Signal.BUY:
        direction = 1.0

        atr_stop = (
            close
            - thresholds.atr_stop_multiple * atr
        )

        structure_stop = (
            support
            - thresholds.structure_buffer_atr * atr
        )

        stop_price = min(
            atr_stop,
            structure_stop,
        )

        primary_target = max(
            resistance,
            close
            + thresholds.target_atr_multiple * atr,
        )

        direction_text = "BUY"

    else:
        direction = -1.0

        atr_stop = (
            close
            + thresholds.atr_stop_multiple * atr
        )

        structure_stop = (
            resistance
            + thresholds.structure_buffer_atr * atr
        )

        stop_price = max(
            atr_stop,
            structure_stop,
        )

        primary_target = min(
            support,
            close
            - thresholds.target_atr_multiple * atr,
        )

        direction_text = "SELL"

    risk_per_share = abs(
        close - stop_price
    )

    reward_per_share = (
        direction
        * (primary_target - close)
    )

    stop_distance_pct = (
        risk_per_share / close
    ) * 100

    reward_to_risk = (
        reward_per_share / risk_per_share
        if risk_per_share > 0
        else 0.0
    )

    vetoes: list[str] = []

    if entry_low <= 0:
        vetoes.append(
            "The calculated entry zone is not valid."
        )

    if stop_price <= 0:
        vetoes.append(
            "The calculated invalidation stop is not valid."
        )

    if recommendation.signal is Signal.BUY:
        if stop_price >= entry_low:
            vetoes.append(
                "BUY invalidation stop is not below "
                "the entry zone."
            )

        if primary_target <= entry_high:
            vetoes.append(
                "BUY primary target is not above "
                "the entry zone."
            )

    if recommendation.signal is Signal.SELL:
        if stop_price <= entry_high:
            vetoes.append(
                "SELL invalidation stop is not above "
                "the entry zone."
            )

        if primary_target >= entry_low:
            vetoes.append(
                "SELL primary target is not below "
                "the entry zone."
            )

    if (
        stop_distance_pct
        < thresholds.min_stop_distance_pct
    ):
        vetoes.append(
            "Stop distance is below the configured "
            "minimum."
        )

    if (
        stop_distance_pct
        > thresholds.max_stop_distance_pct
    ):
        vetoes.append(
            "Stop distance exceeds the configured "
            "maximum."
        )

    if (
        reward_to_risk
        < thresholds.min_reward_to_risk
    ):
        vetoes.append(
            "Reward-to-risk "
            f"{reward_to_risk:.2f} is below the "
            f"minimum {thresholds.min_reward_to_risk:.2f}."
        )

    created_at = _aware_datetime(
        "indicator as_of",
        indicators.as_of,
    )

    expires_at = (
        created_at
        + timedelta(days=thresholds.expiry_days)
    )

    common_evidence = (
        _risk_evidence(
            code="ENTRY_ZONE",
            message=(
                f"{direction_text} entry zone is "
                f"{entry_low:.4f}–{entry_high:.4f}."
            ),
            direction=EvidenceDirection.NEUTRAL,
            observed_value=(
                f"{entry_low:.4f}-{entry_high:.4f}"
            ),
            strength=0.8,
        ),
        _risk_evidence(
            code="INVALIDATION_POINT",
            message=(
                "ATR/structure-based invalidation "
                f"point is {stop_price:.4f}."
            ),
            direction=EvidenceDirection.NEUTRAL,
            observed_value=round(stop_price, 4),
            strength=1.0,
        ),
        _risk_evidence(
            code="PRIMARY_TARGET",
            message=(
                f"Primary target is "
                f"{primary_target:.4f}."
            ),
            direction=(
                EvidenceDirection.BULLISH
                if recommendation.signal is Signal.BUY
                else EvidenceDirection.BEARISH
            ),
            observed_value=round(primary_target, 4),
            strength=0.8,
        ),
        _risk_evidence(
            code="REWARD_TO_RISK",
            message=(
                "Calculated reward-to-risk is "
                f"{reward_to_risk:.2f}."
            ),
            direction=(
                EvidenceDirection.BULLISH
                if reward_to_risk
                >= thresholds.min_reward_to_risk
                else EvidenceDirection.BEARISH
            ),
            observed_value=round(
                reward_to_risk,
                4,
            ),
            strength=1.0,
        ),
        _risk_evidence(
            code="SIGNAL_EXPIRY",
            message=(
                "Paper-order signal expires at "
                f"{expires_at.isoformat()}."
            ),
            direction=EvidenceDirection.NEUTRAL,
            observed_value=expires_at.isoformat(),
            strength=0.5,
        ),
    )

    if vetoes:
        veto_reason = " ".join(vetoes)

        veto_evidence = (
            _risk_evidence(
                code="RISK_VETO",
                message=veto_reason,
                direction=EvidenceDirection.BEARISH,
                observed_value=len(vetoes),
                strength=1.0,
            ),
        )

        adjusted = _copy_recommendation(
            recommendation,
            signal=Signal.HOLD,
            additional_evidence=(
                common_evidence
                + veto_evidence
            ),
            vetoed=True,
            veto_reason=veto_reason,
        )

        return RiskDecision(
            recommendation=adjusted,
            order=None,
            risk_vetoes=tuple(vetoes),
        )

    secondary_distance = max(
        reward_per_share + risk_per_share,
        2 * risk_per_share,
    )

    tertiary_distance = max(
        reward_per_share + 2 * risk_per_share,
        3 * risk_per_share,
    )

    targets = (
        primary_target,
        close + direction * secondary_distance,
        close + direction * tertiary_distance,
    )

    order = PaperOrder(
        symbol=analysis.symbol,
        signal=recommendation.signal,
        created_at=created_at,
        expires_at=expires_at,
        entry_low=round(entry_low, 6),
        entry_high=round(entry_high, 6),
        stop_price=round(stop_price, 6),
        targets=tuple(
            round(target, 6)
            for target in targets
        ),
        risk_per_share=round(
            risk_per_share,
            6,
        ),
        reward_to_risk=round(
            reward_to_risk,
            6,
        ),
        paper_only=True,
    )

    approval_evidence = (
        _risk_evidence(
            code="RISK_GATE_PASSED",
            message=(
                "Risk checks passed and a paper-only "
                f"{direction_text} order was generated."
            ),
            direction=(
                EvidenceDirection.BULLISH
                if recommendation.signal is Signal.BUY
                else EvidenceDirection.BEARISH
            ),
            observed_value=recommendation.signal.value,
            strength=1.0,
        ),
    )

    approved = _copy_recommendation(
        recommendation,
        additional_evidence=(
            common_evidence
            + approval_evidence
        ),
        vetoed=False,
        veto_reason=None,
    )

    return RiskDecision(
        recommendation=approved,
        order=order,
        risk_vetoes=(),
    )
