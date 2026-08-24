"""Assemble all P4 release evidence without producing observations."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import re

from src.backtest import RegressionEvidence
from src.p4_horizon_evidence import build_strategy_horizon_check
from src.p4_notification_evidence import build_notification_delivery_checks
from src.p4_policy_evidence import build_policy_gate_checks
from src.p4_recovery_evidence import build_recovery_control_checks
from src.p4_release_gate import P4ReleaseEvidence
from src.p4_scheduler_evidence import build_scheduler_deployment_check


_PHASES = ("P0", "P1", "P2", "P3", "P4")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _aware(value: object) -> datetime:
    try:
        result = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp must be timezone-aware ISO format.") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware ISO format.")
    return result


def build_p4_regression_evidence(
    evidence: Mapping[str, object],
) -> RegressionEvidence:
    """Validate traceable GitHub Actions regression metadata."""
    if not isinstance(evidence, Mapping):
        raise ValueError("regression evidence must be an object.")
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != 1
    ):
        raise ValueError("regression schema_version must be 1.")
    count = evidence.get("test_count")
    if type(count) is not int or count < 1:
        raise ValueError("regression test_count must be a positive integer.")
    raw_phases = evidence.get("covered_phases")
    phases = (
        tuple(str(item).strip().upper() for item in raw_phases)
        if isinstance(raw_phases, list)
        else ()
    )
    if not phases or any(not phase for phase in phases):
        raise ValueError("regression covered_phases must be a non-empty list.")
    if len(phases) != len(set(phases)):
        raise ValueError("regression covered_phases must be unique.")
    _aware(evidence.get("completed_at"))

    conclusion = str(evidence.get("conclusion", "")).strip().lower()
    run_id = str(evidence.get("workflow_run_id", "")).strip()
    run_url = str(evidence.get("workflow_url", "")).strip()
    commit_sha = str(evidence.get("commit_sha", "")).strip().lower()
    expected_url = (
        "https://github.com/BSAVCI1/stock-analyzer-app/actions/runs/" + run_id
    )
    traceable = (
        run_id.isdigit()
        and run_url == expected_url
        and _COMMIT.fullmatch(commit_sha) is not None
    )
    passed = (
        conclusion == "success"
        and set(phases) == set(_PHASES)
        and len(phases) == len(_PHASES)
        and traceable
    )
    workflow = (
        f"GitHub Actions run {run_id or 'NOT_OBSERVED'}; "
        f"commit {commit_sha or 'NOT_OBSERVED'}; {run_url or 'NO_URL'}"
    )
    return RegressionEvidence(
        passed=passed,
        test_count=count,
        covered_phases=phases,
        workflow=workflow,
    )


def assemble_p4_release_evidence(
    *,
    release: Mapping[str, object],
    policy: Mapping[str, object],
    regression: Mapping[str, object],
    scheduler: Mapping[str, object],
    notifications: Mapping[str, object],
    recovery: Mapping[str, object],
    horizons: Mapping[str, object],
) -> P4ReleaseEvidence:
    """Assemble one complete gate object from independent evidence sources."""
    if not isinstance(release, Mapping):
        raise ValueError("release metadata must be an object.")
    if (
        type(release.get("schema_version")) is not int
        or release.get("schema_version") != 1
    ):
        raise ValueError("release schema_version must be 1.")
    unresolved = release.get("unresolved_operational_failures")
    if type(unresolved) is not int or unresolved < 0:
        raise ValueError(
            "unresolved_operational_failures must be a non-negative integer."
        )
    live_enabled = release.get("live_execution_enabled")
    if not isinstance(live_enabled, bool):
        raise ValueError("live_execution_enabled must be boolean.")

    release_id = str(release.get("release_id", "")).strip()
    account_id = str(release.get("account_id", "")).strip()
    if not release_id or not account_id:
        raise ValueError("release_id and account_id are required.")
    identity_sources = (
        ("scheduler", scheduler, True),
        ("notifications", notifications, True),
        ("recovery", recovery, True),
        ("horizons", horizons, False),
    )
    for label, source, requires_account in identity_sources:
        if source.get("release_id") != release_id:
            raise ValueError(f"{label} release_id does not match release metadata.")
        if requires_account and source.get("account_id") != account_id:
            raise ValueError(f"{label} account_id does not match release metadata.")

    policy_checks = build_policy_gate_checks(policy)
    notification_checks = build_notification_delivery_checks(notifications)
    recovery_checks = build_recovery_control_checks(recovery)
    checks = (
        *policy_checks,
        build_scheduler_deployment_check(scheduler),
        *notification_checks,
        *recovery_checks,
        build_strategy_horizon_check(horizons),
    )
    return P4ReleaseEvidence(
        schema_version=1,
        release_id=release_id,
        generated_at=_aware(release.get("generated_at")),
        account_id=account_id,
        regression=build_p4_regression_evidence(regression),
        checks=checks,
        execution_mode=release.get("execution_mode"),
        live_execution_enabled=live_enabled,
        unresolved_operational_failures=unresolved,
    )


__all__ = ["assemble_p4_release_evidence", "build_p4_regression_evidence"]
