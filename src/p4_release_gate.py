"""Deterministic P4 release-gate contract.

The evaluator consumes already-produced evidence. It never runs a job,
contacts a provider or enables execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping

from src.backtest import RegressionEvidence


P4_GATE_SCHEMA_VERSION = 1
REQUIRED_P4_PHASES = ("P0", "P1", "P2", "P3", "P4")
REQUIRED_P4_CHECKS = (
    "paper_only_invariants",
    "eur_portfolio_policy",
    "scheduler_deployment",
    "email_delivery",
    "telegram_delivery",
    "recovery_controls",
    "kill_switch",
    "strategy_horizon_acceptance",
)


def _text(name: str, value: object) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} cannot be blank.")
    return result


def _utc(name: str, value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")
    return value.astimezone(timezone.utc)


class P4CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_OBSERVED = "NOT_OBSERVED"


class P4ReleaseStatus(str, Enum):
    READY = "READY"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class P4GateCheck:
    name: str
    status: P4CheckStatus
    evidence_ids: tuple[str, ...]
    details: tuple[str, ...]

    def __post_init__(self) -> None:
        name = _text("name", self.name).lower()
        if not isinstance(self.status, P4CheckStatus):
            raise ValueError("status must be a P4CheckStatus.")
        evidence_ids = tuple(_text("evidence_id", x) for x in self.evidence_ids)
        details = tuple(_text("detail", x) for x in self.details)
        if self.status is P4CheckStatus.PASS and not evidence_ids:
            raise ValueError("A passing P4 check requires evidence IDs.")
        if self.status is not P4CheckStatus.PASS and not details:
            raise ValueError("A non-passing P4 check requires details.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "details", details)


@dataclass(frozen=True, slots=True)
class P4ReleaseEvidence:
    schema_version: int
    release_id: str
    generated_at: datetime
    account_id: str
    regression: RegressionEvidence
    checks: tuple[P4GateCheck, ...]
    execution_mode: str
    live_execution_enabled: bool
    unresolved_operational_failures: int

    def __post_init__(self) -> None:
        if self.schema_version != P4_GATE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {P4_GATE_SCHEMA_VERSION}."
            )
        if not isinstance(self.regression, RegressionEvidence):
            raise ValueError("regression must be RegressionEvidence.")
        checks = tuple(self.checks)
        if not all(isinstance(x, P4GateCheck) for x in checks):
            raise ValueError("Every check must be a P4GateCheck.")
        names = tuple(x.name for x in checks)
        if len(names) != len(set(names)):
            raise ValueError("P4 check names must be unique.")
        missing = tuple(x for x in REQUIRED_P4_CHECKS if x not in names)
        if missing:
            raise ValueError("P4 evidence is missing checks: " + ", ".join(missing) + ".")
        unknown = tuple(x for x in names if x not in REQUIRED_P4_CHECKS)
        if unknown:
            raise ValueError("P4 evidence contains unknown checks: " + ", ".join(unknown) + ".")
        if not isinstance(self.live_execution_enabled, bool):
            raise ValueError("live_execution_enabled must be boolean.")
        if (
            isinstance(self.unresolved_operational_failures, bool)
            or not isinstance(self.unresolved_operational_failures, int)
            or self.unresolved_operational_failures < 0
        ):
            raise ValueError("unresolved_operational_failures must be non-negative.")
        object.__setattr__(self, "release_id", _text("release_id", self.release_id))
        object.__setattr__(self, "generated_at", _utc("generated_at", self.generated_at))
        object.__setattr__(self, "account_id", _text("account_id", self.account_id))
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "execution_mode", _text("execution_mode", self.execution_mode).upper())


@dataclass(frozen=True, slots=True)
class P4ReleaseGateReport:
    status: P4ReleaseStatus
    release_ready: bool
    evidence: P4ReleaseEvidence
    blocking_checks: tuple[str, ...]
    reasons: tuple[str, ...]


def evaluate_p4_release_gate(evidence: P4ReleaseEvidence) -> P4ReleaseGateReport:
    if not isinstance(evidence, P4ReleaseEvidence):
        raise ValueError("evidence must be P4ReleaseEvidence.")
    reasons = []
    blocking = []
    if not evidence.regression.passed:
        reasons.append("The P0-P4 regression suite did not pass.")
    missing_phases = tuple(
        phase for phase in REQUIRED_P4_PHASES
        if phase not in set(evidence.regression.covered_phases)
    )
    if missing_phases:
        reasons.append("Regression evidence is missing phases: " + ", ".join(missing_phases) + ".")
    for check in evidence.checks:
        if check.name in REQUIRED_P4_CHECKS and check.status is not P4CheckStatus.PASS:
            blocking.append(check.name)
    if blocking:
        reasons.append("Required P4 checks are not passing: " + ", ".join(blocking) + ".")
    if evidence.execution_mode != "PAPER":
        reasons.append("Execution mode is not PAPER.")
    if evidence.live_execution_enabled:
        reasons.append("Live execution capability is enabled.")
    if evidence.unresolved_operational_failures:
        reasons.append(
            f"There are {evidence.unresolved_operational_failures} unresolved operational failure(s)."
        )
    ready = not reasons
    if ready:
        reasons = [
            "P0-P4 regression evidence and every required P4 release check passed; "
            "execution remains paper-only and no operational failures are unresolved."
        ]
    return P4ReleaseGateReport(
        status=P4ReleaseStatus.READY if ready else P4ReleaseStatus.BLOCKED,
        release_ready=ready, evidence=evidence,
        blocking_checks=tuple(blocking), reasons=tuple(reasons),
    )


def p4_evidence_from_mapping(payload: Mapping[str, object]) -> P4ReleaseEvidence:
    regression = payload.get("regression")
    if not isinstance(regression, Mapping):
        raise ValueError("regression must be an object.")
    raw_checks = payload.get("checks")
    if not isinstance(raw_checks, list):
        raise ValueError("checks must be a list.")
    checks = tuple(
        P4GateCheck(
            name=item["name"], status=P4CheckStatus(item["status"]),
            evidence_ids=tuple(item.get("evidence_ids", ())),
            details=tuple(item.get("details", ())),
        )
        for item in raw_checks
        if isinstance(item, Mapping)
    )
    if len(checks) != len(raw_checks):
        raise ValueError("Every checks item must be an object.")
    return P4ReleaseEvidence(
        schema_version=payload.get("schema_version"),
        release_id=payload.get("release_id"),
        generated_at=datetime.fromisoformat(str(payload.get("generated_at"))),
        account_id=payload.get("account_id"),
        regression=RegressionEvidence(
            passed=regression.get("passed"),
            test_count=regression.get("test_count"),
            covered_phases=tuple(regression.get("covered_phases", ())),
            workflow=regression.get("workflow", "Automated tests"),
        ),
        checks=checks,
        execution_mode=payload.get("execution_mode"),
        live_execution_enabled=payload.get("live_execution_enabled"),
        unresolved_operational_failures=payload.get("unresolved_operational_failures"),
    )


__all__ = [
    "P4CheckStatus", "P4GateCheck", "P4ReleaseEvidence", "P4ReleaseGateReport",
    "P4ReleaseStatus", "P4_GATE_SCHEMA_VERSION", "REQUIRED_P4_CHECKS",
    "REQUIRED_P4_PHASES", "evaluate_p4_release_gate", "p4_evidence_from_mapping",
]
