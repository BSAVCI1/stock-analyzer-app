"""Deterministic strategy acceptance reporting.

This module consolidates existing P2 performance and validation outputs into
one auditable accept/reject report.

It does not recalculate returns, drawdowns, trade results, walk-forward
performance or promotion decisions. It only compares existing deterministic
results against explicitly configured acceptance thresholds.

There is no provider, Streamlit, broker or live-execution integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping, Sequence

from src.strategy import (
    StrategyHorizon,
    coerce_strategy_horizon,
    normalise_strategy_version,
)

from .performance import PerformanceReport
from .validation import (
    PromotionDecision,
    WalkForwardValidationReport,
)


class AcceptanceStatus(str, Enum):
    """Final deterministic strategy acceptance status."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


class PerformanceScope(str, Enum):
    """Dimension represented by one performance slice."""

    INSTRUMENT = "INSTRUMENT"
    MARKET_REGIME = "MARKET_REGIME"


def _required_text(
    name: str,
    value: object,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    result = value.strip()

    if not result:
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return result


def _finite_float(
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


def _non_negative_integer(
    name: str,
    value: object,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer."
        )

    return value


@dataclass(frozen=True, slots=True)
class StrategyAcceptanceThresholds:
    """Required performance and validation criteria."""

    minimum_total_return: float = 0.0
    maximum_drawdown: float = 0.20
    minimum_trade_count: int = 20
    minimum_parameter_stability: float = 0.50

    def __post_init__(self) -> None:
        minimum_total_return = _finite_float(
            "minimum_total_return",
            self.minimum_total_return,
        )

        maximum_drawdown = _finite_float(
            "maximum_drawdown",
            self.maximum_drawdown,
        )

        minimum_trade_count = _non_negative_integer(
            "minimum_trade_count",
            self.minimum_trade_count,
        )

        minimum_parameter_stability = _finite_float(
            "minimum_parameter_stability",
            self.minimum_parameter_stability,
        )

        if not 0 <= maximum_drawdown <= 1:
            raise ValueError(
                "maximum_drawdown must be between 0 and 1."
            )

        if not 0 <= minimum_parameter_stability <= 1:
            raise ValueError(
                "minimum_parameter_stability must be "
                "between 0 and 1."
            )

        object.__setattr__(
            self,
            "minimum_total_return",
            minimum_total_return,
        )

        object.__setattr__(
            self,
            "maximum_drawdown",
            maximum_drawdown,
        )

        object.__setattr__(
            self,
            "minimum_trade_count",
            minimum_trade_count,
        )

        object.__setattr__(
            self,
            "minimum_parameter_stability",
            minimum_parameter_stability,
        )


@dataclass(frozen=True, slots=True)
class PerformanceSlice:
    """Read-only performance summary for an instrument or regime."""

    scope: PerformanceScope
    name: str

    total_return: float
    annualised_return: float
    maximum_drawdown: float
    trade_count: int
    win_rate: float
    sharpe_ratio: float

    benchmark_total_return: float
    strategy_excess_return: float

    def __post_init__(self) -> None:
        if not isinstance(
            self.scope,
            PerformanceScope,
        ):
            raise ValueError(
                "scope must be a PerformanceScope."
            )

        name = _required_text(
            "name",
            self.name,
        )

        for field_name in (
            "total_return",
            "annualised_return",
            "maximum_drawdown",
            "win_rate",
            "sharpe_ratio",
            "benchmark_total_return",
            "strategy_excess_return",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_float(
                    field_name,
                    getattr(self, field_name),
                ),
            )

        trade_count = _non_negative_integer(
            "trade_count",
            self.trade_count,
        )

        if not 0 <= self.maximum_drawdown <= 1:
            raise ValueError(
                "maximum_drawdown must be between 0 and 1."
            )

        if not 0 <= self.win_rate <= 1:
            raise ValueError(
                "win_rate must be between 0 and 1."
            )

        object.__setattr__(
            self,
            "name",
            name,
        )

        object.__setattr__(
            self,
            "trade_count",
            trade_count,
        )

    @classmethod
    def from_report(
        cls,
        scope: PerformanceScope,
        name: str,
        report: PerformanceReport,
    ) -> "PerformanceSlice":
        """Copy existing metrics without recalculating them."""

        if not isinstance(
            report,
            PerformanceReport,
        ):
            raise ValueError(
                "report must be a PerformanceReport."
            )

        return cls(
            scope=scope,
            name=name,
            total_return=report.total_return,
            annualised_return=(
                report.annualised_return
            ),
            maximum_drawdown=(
                report.max_drawdown
            ),
            trade_count=report.trade_count,
            win_rate=report.win_rate,
            sharpe_ratio=report.sharpe_ratio,
            benchmark_total_return=(
                report.benchmark.total_return
            ),
            strategy_excess_return=(
                report
                .benchmark
                .strategy_excess_return
            ),
        )


@dataclass(frozen=True, slots=True)
class AcceptanceCheck:
    """One auditable acceptance criterion."""

    code: str
    scope: str
    passed: bool

    observed_value: float
    criterion: str
    reason: str

    def __post_init__(self) -> None:
        code = _required_text(
            "code",
            self.code,
        )

        scope = _required_text(
            "scope",
            self.scope,
        )

        criterion = _required_text(
            "criterion",
            self.criterion,
        )

        reason = _required_text(
            "reason",
            self.reason,
        )

        if not isinstance(self.passed, bool):
            raise ValueError(
                "passed must be boolean."
            )

        observed_value = _finite_float(
            "observed_value",
            self.observed_value,
        )

        object.__setattr__(
            self,
            "code",
            code,
        )

        object.__setattr__(
            self,
            "scope",
            scope,
        )

        object.__setattr__(
            self,
            "criterion",
            criterion,
        )

        object.__setattr__(
            self,
            "reason",
            reason,
        )

        object.__setattr__(
            self,
            "observed_value",
            observed_value,
        )


@dataclass(frozen=True, slots=True)
class StrategyAcceptanceReport:
    """Complete P2.6 deterministic strategy acceptance result."""

    strategy: str
    status: AcceptanceStatus

    thresholds: StrategyAcceptanceThresholds

    instrument_performance: tuple[
        PerformanceSlice,
        ...,
    ]

    regime_performance: tuple[
        PerformanceSlice,
        ...,
    ]

    validation_report: WalkForwardValidationReport
    promotion_decision: PromotionDecision

    checks: tuple[AcceptanceCheck, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        strategy = _required_text(
            "strategy",
            self.strategy,
        )

        if not isinstance(
            self.status,
            AcceptanceStatus,
        ):
            raise ValueError(
                "status must be an AcceptanceStatus."
            )

        if not isinstance(
            self.thresholds,
            StrategyAcceptanceThresholds,
        ):
            raise ValueError(
                "thresholds must be "
                "StrategyAcceptanceThresholds."
            )

        instrument_performance = tuple(
            self.instrument_performance
        )

        regime_performance = tuple(
            self.regime_performance
        )

        if not all(
            isinstance(
                item,
                PerformanceSlice,
            )
            and item.scope
            is PerformanceScope.INSTRUMENT
            for item in instrument_performance
        ):
            raise ValueError(
                "instrument_performance contains "
                "an invalid performance slice."
            )

        if not all(
            isinstance(
                item,
                PerformanceSlice,
            )
            and item.scope
            is PerformanceScope.MARKET_REGIME
            for item in regime_performance
        ):
            raise ValueError(
                "regime_performance contains "
                "an invalid performance slice."
            )

        if not isinstance(
            self.validation_report,
            WalkForwardValidationReport,
        ):
            raise ValueError(
                "validation_report must be a "
                "WalkForwardValidationReport."
            )

        if not isinstance(
            self.promotion_decision,
            PromotionDecision,
        ):
            raise ValueError(
                "promotion_decision must be a "
                "PromotionDecision."
            )

        checks = tuple(self.checks)

        if not checks:
            raise ValueError(
                "checks cannot be empty."
            )

        if not all(
            isinstance(check, AcceptanceCheck)
            for check in checks
        ):
            raise ValueError(
                "checks contains an invalid object."
            )

        reasons = tuple(self.reasons)

        if not reasons:
            raise ValueError(
                "reasons cannot be empty."
            )

        if not all(
            isinstance(reason, str)
            and reason.strip()
            for reason in reasons
        ):
            raise ValueError(
                "reasons must contain non-empty strings."
            )

        expected_status = (
            AcceptanceStatus.ACCEPT
            if all(
                check.passed
                for check in checks
            )
            else AcceptanceStatus.REJECT
        )

        if self.status is not expected_status:
            raise ValueError(
                "status does not match the acceptance checks."
            )

        object.__setattr__(
            self,
            "strategy",
            strategy,
        )

        object.__setattr__(
            self,
            "instrument_performance",
            instrument_performance,
        )

        object.__setattr__(
            self,
            "regime_performance",
            regime_performance,
        )

        object.__setattr__(
            self,
            "checks",
            checks,
        )

        object.__setattr__(
            self,
            "reasons",
            reasons,
        )

    @property
    def accepted(self) -> bool:
        return (
            self.status
            is AcceptanceStatus.ACCEPT
        )


def _normalise_performance_reports(
    reports: Mapping[str, PerformanceReport],
    scope: PerformanceScope,
) -> tuple[PerformanceSlice, ...]:
    if not isinstance(reports, Mapping):
        raise ValueError(
            "Performance reports must be supplied "
            "as a mapping."
        )

    normalised: list[PerformanceSlice] = []
    names: set[str] = set()

    for raw_name, report in reports.items():
        name = _required_text(
            "performance report name",
            raw_name,
        )

        if name in names:
            raise ValueError(
                f"Duplicate performance report name: {name}."
            )

        names.add(name)

        normalised.append(
            PerformanceSlice.from_report(
                scope,
                name,
                report,
            )
        )

    return tuple(
        sorted(
            normalised,
            key=lambda item: item.name,
        )
    )


def _coverage_check(
    code: str,
    scope: str,
    available_count: int,
) -> AcceptanceCheck:
    passed = available_count > 0

    return AcceptanceCheck(
        code=code,
        scope=scope,
        passed=passed,
        observed_value=float(
            available_count
        ),
        criterion="At least one report is required.",
        reason=(
            f"{scope} performance coverage is available."
            if passed
            else (
                f"No {scope.lower()} performance "
                "reports were supplied."
            )
        ),
    )


def _performance_checks(
    performance: PerformanceSlice,
    thresholds: StrategyAcceptanceThresholds,
) -> tuple[AcceptanceCheck, ...]:
    scope_label = (
        f"{performance.scope.value}:"
        f"{performance.name}"
    )

    return_passed = (
        performance.total_return
        >= thresholds.minimum_total_return
    )

    drawdown_passed = (
        performance.maximum_drawdown
        <= thresholds.maximum_drawdown
    )

    trade_count_passed = (
        performance.trade_count
        >= thresholds.minimum_trade_count
    )

    return (
        AcceptanceCheck(
            code=(
                f"{scope_label}:"
                "minimum_total_return"
            ),
            scope=scope_label,
            passed=return_passed,
            observed_value=(
                performance.total_return
            ),
            criterion=(
                "Total return must be at least "
                f"{thresholds.minimum_total_return:.2%}."
            ),
            reason=(
                f"{scope_label} total return "
                f"{performance.total_return:.2%} "
                "met the required threshold."
                if return_passed
                else (
                    f"{scope_label} total return "
                    f"{performance.total_return:.2%} "
                    "was below the required "
                    f"{thresholds.minimum_total_return:.2%}."
                )
            ),
        ),
        AcceptanceCheck(
            code=(
                f"{scope_label}:"
                "maximum_drawdown"
            ),
            scope=scope_label,
            passed=drawdown_passed,
            observed_value=(
                performance.maximum_drawdown
            ),
            criterion=(
                "Maximum drawdown must not exceed "
                f"{thresholds.maximum_drawdown:.2%}."
            ),
            reason=(
                f"{scope_label} maximum drawdown "
                f"{performance.maximum_drawdown:.2%} "
                "met the required threshold."
                if drawdown_passed
                else (
                    f"{scope_label} maximum drawdown "
                    f"{performance.maximum_drawdown:.2%} "
                    "exceeded the permitted "
                    f"{thresholds.maximum_drawdown:.2%}."
                )
            ),
        ),
        AcceptanceCheck(
            code=(
                f"{scope_label}:"
                "minimum_trade_count"
            ),
            scope=scope_label,
            passed=trade_count_passed,
            observed_value=float(
                performance.trade_count
            ),
            criterion=(
                "Trade count must be at least "
                f"{thresholds.minimum_trade_count}."
            ),
            reason=(
                f"{scope_label} trade count "
                f"{performance.trade_count} "
                "met the required threshold."
                if trade_count_passed
                else (
                    f"{scope_label} trade count "
                    f"{performance.trade_count} "
                    "was below the required "
                    f"{thresholds.minimum_trade_count}."
                )
            ),
        ),
    )


def build_strategy_acceptance_report(
    strategy: str,
    *,
    instrument_performance: Mapping[
        str,
        PerformanceReport,
    ],
    regime_performance: Mapping[
        str,
        PerformanceReport,
    ],
    validation_report: WalkForwardValidationReport,
    promotion_decision: PromotionDecision,
    thresholds: (
        StrategyAcceptanceThresholds
        | None
    ) = None,
) -> StrategyAcceptanceReport:
    """Build one auditable P2.6 accept/reject report.

    Existing performance, validation and promotion values are copied and
    compared against the supplied thresholds. None of those source results
    are recalculated or overridden.
    """

    strategy_name = _required_text(
        "strategy",
        strategy,
    )

    thresholds = (
        thresholds
        or StrategyAcceptanceThresholds()
    )

    if not isinstance(
        thresholds,
        StrategyAcceptanceThresholds,
    ):
        raise ValueError(
            "thresholds must be "
            "StrategyAcceptanceThresholds."
        )

    if not isinstance(
        validation_report,
        WalkForwardValidationReport,
    ):
        raise ValueError(
            "validation_report must be a "
            "WalkForwardValidationReport."
        )

    if not isinstance(
        promotion_decision,
        PromotionDecision,
    ):
        raise ValueError(
            "promotion_decision must be a "
            "PromotionDecision."
        )

    instruments = _normalise_performance_reports(
        instrument_performance,
        PerformanceScope.INSTRUMENT,
    )

    regimes = _normalise_performance_reports(
        regime_performance,
        PerformanceScope.MARKET_REGIME,
    )

    checks: list[AcceptanceCheck] = [
        _coverage_check(
            "instrument_performance_coverage",
            "Instrument",
            len(instruments),
        ),
        _coverage_check(
            "regime_performance_coverage",
            "Market regime",
            len(regimes),
        ),
    ]

    for performance in (
        *instruments,
        *regimes,
    ):
        checks.extend(
            _performance_checks(
                performance,
                thresholds,
            )
        )

    stability_score = (
        validation_report
        .parameter_stability
        .overall_stability_score
    )

    stability_passed = (
        stability_score
        >= thresholds
        .minimum_parameter_stability
    )

    checks.append(
        AcceptanceCheck(
            code="parameter_stability",
            scope="Walk-forward validation",
            passed=stability_passed,
            observed_value=stability_score,
            criterion=(
                "Parameter stability must be at least "
                f"{thresholds.minimum_parameter_stability:.2%}."
            ),
            reason=(
                "Walk-forward parameter stability "
                f"{stability_score:.2%} met the "
                "required threshold."
                if stability_passed
                else (
                    "Walk-forward parameter stability "
                    f"{stability_score:.2%} was below "
                    "the required "
                    f"{thresholds.minimum_parameter_stability:.2%}."
                )
            ),
        )
    )

    promotion_reason = " ".join(
        promotion_decision.reasons
    )

    checks.append(
        AcceptanceCheck(
            code="out_of_sample_promotion",
            scope="Walk-forward validation",
            passed=promotion_decision.promoted,
            observed_value=(
                1.0
                if promotion_decision.promoted
                else 0.0
            ),
            criterion=(
                "The existing out-of-sample promotion "
                "decision must be approved."
            ),
            reason=promotion_reason,
        )
    )

    checks_tuple = tuple(checks)

    failed_reasons = tuple(
        check.reason
        for check in checks_tuple
        if not check.passed
    )

    if failed_reasons:
        status = AcceptanceStatus.REJECT
        reasons = failed_reasons
    else:
        status = AcceptanceStatus.ACCEPT
        reasons = (
            "Strategy met the required instrument, "
            "market-regime, return, drawdown, "
            "trade-count, parameter-stability and "
            "out-of-sample promotion criteria.",
        )

    return StrategyAcceptanceReport(
        strategy=strategy_name,
        status=status,
        thresholds=thresholds,
        instrument_performance=instruments,
        regime_performance=regimes,
        validation_report=validation_report,
        promotion_decision=promotion_decision,
        checks=checks_tuple,
        reasons=reasons,
    )

@dataclass(frozen=True, slots=True)
class HorizonAcceptanceEvidence:
    """One horizon's versioned acceptance evidence."""

    horizon: StrategyHorizon
    strategy_version: str
    acceptance_report: StrategyAcceptanceReport

    def __post_init__(self) -> None:
        horizon = coerce_strategy_horizon(
            self.horizon
        )

        if horizon is None:
            raise ValueError(
                "horizon is required."
            )

        version = normalise_strategy_version(
            self.strategy_version
        )

        if version is None:
            raise ValueError(
                "strategy_version is required."
            )

        if not isinstance(
            self.acceptance_report,
            StrategyAcceptanceReport,
        ):
            raise ValueError(
                "acceptance_report must be a "
                "StrategyAcceptanceReport."
            )

        object.__setattr__(
            self,
            "horizon",
            horizon,
        )
        object.__setattr__(
            self,
            "strategy_version",
            version,
        )

    @property
    def accepted(self) -> bool:
        return self.acceptance_report.accepted


@dataclass(frozen=True, slots=True)
class IndependentHorizonAcceptanceReport:
    """Independent, non-aggregated P4.3 horizon decisions."""

    evidence: tuple[
        HorizonAcceptanceEvidence,
        ...,
    ]

    def __post_init__(self) -> None:
        evidence = tuple(self.evidence)

        if not all(
            isinstance(
                item,
                HorizonAcceptanceEvidence,
            )
            for item in evidence
        ):
            raise ValueError(
                "evidence contains an invalid object."
            )

        expected = {
            StrategyHorizon.SWING,
            StrategyHorizon.MEDIUM_TERM,
        }
        actual = {
            item.horizon
            for item in evidence
        }

        if (
            actual != expected
            or len(evidence) != len(expected)
        ):
            raise ValueError(
                "Independent horizon evidence must "
                "contain exactly one SWING and one "
                "MEDIUM_TERM decision."
            )

        ordered = tuple(
            next(
                item
                for item in evidence
                if item.horizon is horizon
            )
            for horizon in (
                StrategyHorizon.SWING,
                StrategyHorizon.MEDIUM_TERM,
            )
        )

        object.__setattr__(
            self,
            "evidence",
            ordered,
        )

    def for_horizon(
        self,
        horizon: StrategyHorizon | str,
    ) -> HorizonAcceptanceEvidence:
        requested = coerce_strategy_horizon(
            horizon
        )

        return next(
            item
            for item in self.evidence
            if item.horizon is requested
        )

    @property
    def accepted_horizons(
        self,
    ) -> tuple[StrategyHorizon, ...]:
        return tuple(
            item.horizon
            for item in self.evidence
            if item.accepted
        )

    @property
    def rejected_horizons(
        self,
    ) -> tuple[StrategyHorizon, ...]:
        return tuple(
            item.horizon
            for item in self.evidence
            if not item.accepted
        )


def build_independent_horizon_acceptance_report(
    *,
    swing_report: StrategyAcceptanceReport,
    swing_strategy_version: str,
    medium_term_report: StrategyAcceptanceReport,
    medium_term_strategy_version: str,
) -> IndependentHorizonAcceptanceReport:
    """Build exact, independently decided horizon evidence."""

    return IndependentHorizonAcceptanceReport(
        evidence=(
            HorizonAcceptanceEvidence(
                horizon=StrategyHorizon.SWING,
                strategy_version=(
                    swing_strategy_version
                ),
                acceptance_report=swing_report,
            ),
            HorizonAcceptanceEvidence(
                horizon=(
                    StrategyHorizon.MEDIUM_TERM
                ),
                strategy_version=(
                    medium_term_strategy_version
                ),
                acceptance_report=(
                    medium_term_report
                ),
            ),
        )
    )

