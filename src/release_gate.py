"""Deterministic P3 operational release gate.

This module evaluates already-produced evidence. It does not run jobs,
contact brokers, submit orders, or enable live trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from src.backtest import RegressionEvidence


REQUIRED_P3_PHASES = (
    "P0",
    "P1",
    "P2",
    "P3",
)

REQUIRED_OPERATIONAL_CHECKS = (
    "account_reconciliation",
    "broker_reconciliation",
    "scans",
    "execution_runs",
    "scheduled_jobs",
    "notifications",
    "system_events",
)


def _required_text(
    field_name: str,
    value: object,
) -> str:
    text = str(value).strip()

    if not text:
        raise ValueError(
            f"{field_name} cannot be blank."
        )

    return text


def _non_negative_integer(
    field_name: str,
    value: object,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{field_name} must be a "
            "non-negative integer."
        )

    return value


def _aware_utc(
    field_name: str,
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} must be a "
            "timezone-aware datetime."
        )

    return value.astimezone(timezone.utc)


class OperationalCheckStatus(
    str,
    Enum,
):
    """Outcome of one persisted operational check."""

    PASS = "PASS"
    FAIL = "FAIL"
    NOT_OBSERVED = "NOT_OBSERVED"


@dataclass(frozen=True, slots=True)
class OperationalReliabilityCheck:
    """One auditable operational reliability result."""

    name: str
    status: OperationalCheckStatus

    observed_count: int
    failed_count: int

    details: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _required_text(
            "name",
            self.name,
        ).lower()

        if not isinstance(
            self.status,
            OperationalCheckStatus,
        ):
            raise ValueError(
                "status must be an "
                "OperationalCheckStatus."
            )

        observed_count = (
            _non_negative_integer(
                "observed_count",
                self.observed_count,
            )
        )

        failed_count = (
            _non_negative_integer(
                "failed_count",
                self.failed_count,
            )
        )

        if failed_count > observed_count:
            raise ValueError(
                "failed_count cannot exceed "
                "observed_count."
            )

        details = tuple(
            _required_text(
                "detail",
                detail,
            )
            for detail in self.details
        )

        if (
            self.status
            is OperationalCheckStatus.PASS
            and failed_count != 0
        ):
            raise ValueError(
                "A passing check cannot contain "
                "failed observations."
            )

        if (
            self.status
            is OperationalCheckStatus.FAIL
            and failed_count == 0
        ):
            raise ValueError(
                "A failed check must contain at "
                "least one failed observation."
            )

        if (
            self.status
            is OperationalCheckStatus.NOT_OBSERVED
            and (
                observed_count != 0
                or failed_count != 0
            )
        ):
            raise ValueError(
                "A not-observed check cannot "
                "contain observations."
            )

        if (
            self.status
            is not OperationalCheckStatus.PASS
            and not details
        ):
            raise ValueError(
                "A non-passing check must explain "
                "its status."
            )

        object.__setattr__(
            self,
            "name",
            name,
        )

        object.__setattr__(
            self,
            "observed_count",
            observed_count,
        )

        object.__setattr__(
            self,
            "failed_count",
            failed_count,
        )

        object.__setattr__(
            self,
            "details",
            details,
        )


@dataclass(frozen=True, slots=True)
class OperationalReliabilityReport:
    """Complete persisted operational evidence for P3."""

    generated_at: datetime
    account_id: str

    checks: tuple[
        OperationalReliabilityCheck,
        ...,
    ]

    broker_reconciliation_run_id: (
        str | None
    )

    unresolved_broker_differences: int
    live_trading_enabled: bool

    broker_reconciliation_required: bool = True

    def __post_init__(self) -> None:
        generated_at = _aware_utc(
            "generated_at",
            self.generated_at,
        )

        account_id = _required_text(
            "account_id",
            self.account_id,
        )

        checks = tuple(self.checks)

        if not checks:
            raise ValueError(
                "checks cannot be empty."
            )

        if not all(
            isinstance(
                check,
                OperationalReliabilityCheck,
            )
            for check in checks
        ):
            raise ValueError(
                "Every check must be an "
                "OperationalReliabilityCheck."
            )

        names = tuple(
            check.name
            for check in checks
        )

        if len(names) != len(set(names)):
            raise ValueError(
                "Operational check names must "
                "be unique."
            )

        missing_checks = tuple(
            name
            for name
            in REQUIRED_OPERATIONAL_CHECKS
            if name not in names
        )

        if missing_checks:
            raise ValueError(
                "Operational report is missing "
                "required checks: "
                + ", ".join(missing_checks)
                + "."
            )

        broker_run_id = (
            _required_text(
                "broker_reconciliation_run_id",
                self
                .broker_reconciliation_run_id,
            )
            if (
                self
                .broker_reconciliation_run_id
                is not None
            )
            else None
        )

        unresolved = (
            _non_negative_integer(
                "unresolved_broker_differences",
                self
                .unresolved_broker_differences,
            )
        )

        if not isinstance(
            self.live_trading_enabled,
            bool,
        ):
            raise ValueError(
                "live_trading_enabled must "
                "be boolean."
            )

        if not isinstance(
            self.broker_reconciliation_required,
            bool,
        ):
            raise ValueError(
                "broker_reconciliation_required must "
                "be boolean."
            )

        broker_check = next(
            check
            for check in checks
            if (
                check.name
                == "broker_reconciliation"
            )
        )

        if (
            broker_check.status
            is OperationalCheckStatus.PASS
            and broker_run_id is None
        ):
            raise ValueError(
                "A passing broker reconciliation "
                "requires a persisted run ID."
            )

        if (
            unresolved > 0
            and broker_check.status
            is OperationalCheckStatus.PASS
        ):
            raise ValueError(
                "Broker reconciliation cannot pass "
                "with unresolved differences."
            )

        object.__setattr__(
            self,
            "generated_at",
            generated_at,
        )

        object.__setattr__(
            self,
            "account_id",
            account_id,
        )

        object.__setattr__(
            self,
            "checks",
            checks,
        )

        object.__setattr__(
            self,
            "broker_reconciliation_run_id",
            broker_run_id,
        )

        object.__setattr__(
            self,
            "unresolved_broker_differences",
            unresolved,
        )

    def check_for(
        self,
        name: str,
    ) -> OperationalReliabilityCheck:
        normalised = _required_text(
            "name",
            name,
        ).lower()

        for check in self.checks:
            if check.name == normalised:
                return check

        raise KeyError(normalised)

    @property
    def non_passing_checks(
        self,
    ) -> tuple[
        OperationalReliabilityCheck,
        ...,
    ]:
        return tuple(
            check
            for check in self.checks
            if (
                check.status
                is not
                OperationalCheckStatus.PASS
            )
        )

    @property
    def blocking_checks(
        self,
    ) -> tuple[
        OperationalReliabilityCheck,
        ...,
    ]:
        return tuple(
            check
            for check in self.checks
            if not (
                check.name
                == "broker_reconciliation"
                and (
                    check.status
                    is OperationalCheckStatus
                    .NOT_OBSERVED
                )
                and not (
                    self
                    .broker_reconciliation_required
                )
            )
            and (
                check.status
                is not
                OperationalCheckStatus.PASS
            )
        )

    @property
    def passed(self) -> bool:
        broker_evidence_satisfied = (
            (
                self
                .broker_reconciliation_run_id
                is not None
            )
            if (
                self
                .broker_reconciliation_required
            )
            else True
        )

        return (
            not self.blocking_checks
            and broker_evidence_satisfied
            and (
                self
                .unresolved_broker_differences
                == 0
            )
            and not self.live_trading_enabled
        )


class P3ReleaseStatus(
    str,
    Enum,
):
    """Final P3 release decision."""

    READY = "READY"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class P3ReleaseGateReport:
    """Auditable final P3 release decision."""

    status: P3ReleaseStatus
    release_ready: bool

    regression_evidence: RegressionEvidence

    operational_reliability: (
        OperationalReliabilityReport
    )

    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.status,
            P3ReleaseStatus,
        ):
            raise ValueError(
                "status must be a "
                "P3ReleaseStatus."
            )

        if not isinstance(
            self.release_ready,
            bool,
        ):
            raise ValueError(
                "release_ready must be boolean."
            )

        if not isinstance(
            self.regression_evidence,
            RegressionEvidence,
        ):
            raise ValueError(
                "regression_evidence must be "
                "RegressionEvidence."
            )

        if not isinstance(
            self.operational_reliability,
            OperationalReliabilityReport,
        ):
            raise ValueError(
                "operational_reliability must be "
                "OperationalReliabilityReport."
            )

        reasons = tuple(
            _required_text(
                "release-gate reason",
                reason,
            )
            for reason in self.reasons
        )

        if not reasons:
            raise ValueError(
                "reasons cannot be empty."
            )

        expected_status = (
            P3ReleaseStatus.READY
            if self.release_ready
            else P3ReleaseStatus.BLOCKED
        )

        if self.status is not expected_status:
            raise ValueError(
                "status does not match "
                "release_ready."
            )

        object.__setattr__(
            self,
            "reasons",
            reasons,
        )


def evaluate_p3_release_gate(
    *,
    regression_evidence: RegressionEvidence,
    operational_reliability:
    OperationalReliabilityReport,
) -> P3ReleaseGateReport:
    """Evaluate complete P0-P3 and operational evidence."""

    if not isinstance(
        regression_evidence,
        RegressionEvidence,
    ):
        raise ValueError(
            "regression_evidence must be "
            "RegressionEvidence."
        )

    if not isinstance(
        operational_reliability,
        OperationalReliabilityReport,
    ):
        raise ValueError(
            "operational_reliability must be "
            "OperationalReliabilityReport."
        )

    reasons: list[str] = []

    if not regression_evidence.passed:
        reasons.append(
            "The complete P0-P3 regression "
            "suite did not pass."
        )

    covered_phases = set(
        regression_evidence.covered_phases
    )

    missing_phases = tuple(
        phase
        for phase in REQUIRED_P3_PHASES
        if phase not in covered_phases
    )

    if missing_phases:
        reasons.append(
            "Regression evidence is missing "
            "phases: "
            + ", ".join(missing_phases)
            + "."
        )

    non_passing = (
        operational_reliability
        .blocking_checks
    )

    if non_passing:
        reasons.append(
            "Operational checks are not "
            "passing: "
            + ", ".join(
                (
                    f"{check.name}="
                    f"{check.status.value}"
                )
                for check in non_passing
            )
            + "."
        )

    if (
        operational_reliability
        .broker_reconciliation_required
        and (
            operational_reliability
            .broker_reconciliation_run_id
            is None
        )
    ):
        reasons.append(
            "No persisted broker-paper "
            "reconciliation run is available."
        )

    if (
        operational_reliability
        .unresolved_broker_differences
        > 0
    ):
        reasons.append(
            "Broker-paper reconciliation has "
            f"{operational_reliability.unresolved_broker_differences} "
            "unresolved difference(s)."
        )

    if (
        operational_reliability
        .live_trading_enabled
    ):
        reasons.append(
            "Live trading is enabled."
        )

    release_ready = not reasons

    if release_ready:
        reasons = [
            "Complete P0-P3 regression evidence "
            "and all required operational checks "
            "passed. Broker-paper reconciliation "
            "has no unresolved differences and "
            "live trading remains disabled."
        ]

    return P3ReleaseGateReport(
        status=(
            P3ReleaseStatus.READY
            if release_ready
            else P3ReleaseStatus.BLOCKED
        ),
        release_ready=release_ready,
        regression_evidence=(
            regression_evidence
        ),
        operational_reliability=(
            operational_reliability
        ),
        reasons=tuple(reasons),
    )


__all__ = [
    "OperationalCheckStatus",
    "OperationalReliabilityCheck",
    "OperationalReliabilityReport",
    "P3ReleaseGateReport",
    "P3ReleaseStatus",
    "REQUIRED_OPERATIONAL_CHECKS",
    "REQUIRED_P3_PHASES",
    "evaluate_p3_release_gate",
]


def _operational_check_from_metric(
    name: str,
    metric,
) -> OperationalReliabilityCheck:
    """Convert one persisted dashboard metric into a gate check."""

    total = _non_negative_integer(
        f"{name}.total",
        metric.total,
    )

    successful = _non_negative_integer(
        f"{name}.successful",
        metric.successful,
    )

    failed = _non_negative_integer(
        f"{name}.failed",
        metric.failed,
    )

    pending_or_other = (
        _non_negative_integer(
            f"{name}.pending_or_other",
            metric.pending_or_other,
        )
    )

    if (
        successful
        + failed
        + pending_or_other
        != total
    ):
        raise ValueError(
            f"{name} reliability counts do "
            "not reconcile to total."
        )

    source_tables = tuple(
        metric.provenance.source_tables
    )

    details = (
        f"source_tables={','.join(source_tables)}",
        f"successful={successful}",
        f"failed={failed}",
        (
            "pending_or_other="
            f"{pending_or_other}"
        ),
    )

    if total == 0:
        return OperationalReliabilityCheck(
            name=name,
            status=(
                OperationalCheckStatus
                .NOT_OBSERVED
            ),
            observed_count=0,
            failed_count=0,
            details=(
                *details,
                "No persisted records were observed.",
            ),
        )

    non_passing = (
        failed
        + pending_or_other
    )

    if non_passing:
        return OperationalReliabilityCheck(
            name=name,
            status=(
                OperationalCheckStatus.FAIL
            ),
            observed_count=total,
            failed_count=non_passing,
            details=details,
        )

    return OperationalReliabilityCheck(
        name=name,
        status=OperationalCheckStatus.PASS,
        observed_count=total,
        failed_count=0,
        details=details,
    )


def build_operational_reliability_report(
    snapshot,
    *,
    execution_descriptor,
    broker_reconciliation_required:
    bool = True,
) -> OperationalReliabilityReport:
    """Build P3 operational evidence from one persisted snapshot.

    The function performs no database writes, broker calls, job execution,
    notification delivery or order submission.
    """

    account_reconciliation = (
        snapshot.reconciliation
    )

    if account_reconciliation.reconciled:
        account_check = (
            OperationalReliabilityCheck(
                name="account_reconciliation",
                status=(
                    OperationalCheckStatus
                    .PASS
                ),
                observed_count=1,
                failed_count=0,
                details=(
                    "Stored and ledger cash "
                    "balances reconcile.",
                ),
            )
        )
    else:
        account_check = (
            OperationalReliabilityCheck(
                name="account_reconciliation",
                status=(
                    OperationalCheckStatus
                    .FAIL
                ),
                observed_count=1,
                failed_count=1,
                details=(
                    "Stored and ledger cash "
                    "balances differ by "
                    f"{account_reconciliation.difference}.",
                ),
            )
        )

    broker_run = (
        snapshot
        .broker_reconciliation_run
    )

    if broker_run is None:
        broker_check = (
            OperationalReliabilityCheck(
                name="broker_reconciliation",
                status=(
                    OperationalCheckStatus
                    .NOT_OBSERVED
                ),
                observed_count=0,
                failed_count=0,
                details=(
                    "No persisted broker-paper "
                    "reconciliation run exists.",
                ),
            )
        )

        broker_run_id = None
        unresolved_broker_differences = 0
    else:
        broker_run_id = (
            broker_run
            .reconciliation_run_id
        )

        unresolved_broker_differences = (
            broker_run
            .unresolved_item_count
        )

        comparison_count = (
            broker_run.account_item_count
            + broker_run.order_item_count
            + broker_run.position_item_count
        )

        observed_count = max(
            1,
            comparison_count,
            unresolved_broker_differences,
        )

        broker_details = [
            (
                "reconciliation_run_id="
                f"{broker_run_id}"
            ),
            (
                "provider="
                f"{broker_run.provider}"
            ),
            (
                "status="
                f"{broker_run.status.value}"
            ),
            (
                "unresolved="
                f"{unresolved_broker_differences}"
            ),
        ]

        if broker_run.error_message:
            broker_details.append(
                "error="
                + broker_run.error_message
            )

        if (
            broker_run.reconciled
            and (
                unresolved_broker_differences
                == 0
            )
        ):
            broker_check = (
                OperationalReliabilityCheck(
                    name=(
                        "broker_reconciliation"
                    ),
                    status=(
                        OperationalCheckStatus
                        .PASS
                    ),
                    observed_count=(
                        observed_count
                    ),
                    failed_count=0,
                    details=tuple(
                        broker_details
                    ),
                )
            )
        else:
            broker_check = (
                OperationalReliabilityCheck(
                    name=(
                        "broker_reconciliation"
                    ),
                    status=(
                        OperationalCheckStatus
                        .FAIL
                    ),
                    observed_count=(
                        observed_count
                    ),
                    failed_count=max(
                        1,
                        unresolved_broker_differences,
                    ),
                    details=tuple(
                        broker_details
                    ),
                )
            )

    reliability = snapshot.reliability

    metric_checks = tuple(
        _operational_check_from_metric(
            name,
            metric,
        )
        for name, metric in (
            (
                "scans",
                reliability.scans,
            ),
            (
                "execution_runs",
                reliability.execution_runs,
            ),
            (
                "scheduled_jobs",
                reliability.scheduled_jobs,
            ),
            (
                "notifications",
                reliability.notifications,
            ),
            (
                "system_events",
                reliability.system_events,
            ),
        )
    )

    return OperationalReliabilityReport(
        generated_at=snapshot.generated_at,
        account_id=(
            snapshot.account.account_id
        ),
        checks=(
            account_check,
            broker_check,
            *metric_checks,
        ),
        broker_reconciliation_run_id=(
            broker_run_id
        ),
        unresolved_broker_differences=(
            unresolved_broker_differences
        ),
        live_trading_enabled=(
            execution_descriptor
            .live_trading_enabled
        ),
        broker_reconciliation_required=(
            broker_reconciliation_required
        ),
    )


__all__.append(
    "build_operational_reliability_report"
)
