"""Fail-closed P4 recovery-controls and kill-switch evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime

from src.p4_release_gate import P4CheckStatus, P4GateCheck


def _fingerprint(value: Mapping[str, object]) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _aware(value: object) -> bool:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        return ()
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return ()
    return tuple(item.strip() for item in value)


def _result(
    name: str,
    failures: list[str],
    evidence_ids: list[str],
    *,
    release_id: str,
    fingerprint: str,
    details: str,
) -> P4GateCheck:
    if failures:
        return P4GateCheck(
            name=name, status=P4CheckStatus.FAIL,
            evidence_ids=(), details=tuple(failures),
        )
    return P4GateCheck(
        name=name, status=P4CheckStatus.PASS,
        evidence_ids=(
            f"CONTROL:{release_id}:{name}:sha256:{fingerprint}",
            *tuple(dict.fromkeys(evidence_ids)),
        ),
        details=(details,),
    )


def _recovery_check(
    evidence: Mapping[str, object],
    *,
    release_id: str,
    context: Mapping[str, object],
) -> P4GateCheck:
    failures: list[str] = []
    truths = (
        "restart_recovery_verified",
        "idempotent_replay_verified",
        "stale_data_breaker_verified",
        "reconciliation_breaker_verified",
        "loss_limit_pause_verified",
        "provider_outage_verified",
        "incidents_closed",
    )
    for key in truths:
        if evidence.get(key) is not True:
            failures.append(f"recovery_controls.{key} must be true.")
    unresolved = evidence.get("unresolved_critical_incidents")
    if type(unresolved) is not int or unresolved != 0:
        failures.append(
            "recovery_controls.unresolved_critical_incidents must be integer 0."
        )

    collected: list[str] = []
    for key in (
        "restart_evidence_ids",
        "replay_evidence_ids",
        "circuit_breaker_evidence_ids",
        "provider_outage_evidence_ids",
        "incident_evidence_ids",
    ):
        values = _ids(evidence.get(key))
        if not values:
            failures.append(
                f"recovery_controls.{key} must contain evidence IDs."
            )
        collected.extend(values)
    return _result(
        "recovery_controls", failures, collected,
        release_id=release_id, fingerprint=_fingerprint(context),
        details=(
            "Restart, replay, circuit-breaker, outage and incident-recovery "
            "evidence verified with no unresolved critical incident."
        ),
    )


def _kill_switch_check(
    evidence: Mapping[str, object],
    *,
    release_id: str,
    context: Mapping[str, object],
) -> P4GateCheck:
    failures: list[str] = []
    for key in (
        "activation_verified",
        "new_orders_blocked",
        "pending_order_policy_verified",
        "audit_trail_persisted",
        "recovery_verified",
    ):
        if evidence.get(key) is not True:
            failures.append(f"kill_switch.{key} must be true.")
    if str(evidence.get("final_state", "")).strip().upper() != "INACTIVE":
        failures.append("kill_switch.final_state must be INACTIVE after recovery.")
    operator = evidence.get("operator")
    reason = evidence.get("reason")
    if not isinstance(operator, str) or not operator.strip():
        failures.append("kill_switch.operator is required.")
    if not isinstance(reason, str) or not reason.strip():
        failures.append("kill_switch.reason is required.")

    collected: list[str] = []
    for key in (
        "activation_evidence_ids",
        "blocked_order_evidence_ids",
        "audit_evidence_ids",
        "recovery_evidence_ids",
    ):
        values = _ids(evidence.get(key))
        if not values:
            failures.append(f"kill_switch.{key} must contain evidence IDs.")
        collected.extend(values)
    return _result(
        "kill_switch", failures, collected,
        release_id=release_id, fingerprint=_fingerprint(context),
        details=(
            "Kill-switch activation, order blocking, audit and verified "
            "recovery to an inactive final state passed."
        ),
    )


def build_recovery_control_checks(
    evidence: Mapping[str, object],
) -> tuple[P4GateCheck, P4GateCheck]:
    """Build independent recovery-controls and kill-switch checks."""
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must be an object.")
    common: list[str] = []
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != 1
    ):
        common.append("schema_version must be 1.")
    release_id = str(evidence.get("release_id", "")).strip()
    if not release_id:
        common.append("release_id is required.")
    account_id = str(evidence.get("account_id", "")).strip()
    if not account_id:
        common.append("account_id is required.")
    if not _aware(evidence.get("observed_at")):
        common.append("observed_at must be timezone-aware.")
    if str(evidence.get("execution_mode", "")).strip().upper() != "PAPER":
        common.append("execution_mode must be PAPER.")

    recovery_raw = evidence.get("recovery_controls")
    kill_raw = evidence.get("kill_switch")
    recovery = recovery_raw if isinstance(recovery_raw, Mapping) else {}
    kill_switch = kill_raw if isinstance(kill_raw, Mapping) else {}
    header = {
        "schema_version": evidence.get("schema_version"),
        "release_id": evidence.get("release_id"),
        "account_id": evidence.get("account_id"),
        "observed_at": evidence.get("observed_at"),
        "execution_mode": evidence.get("execution_mode"),
    }
    checks = (
        _recovery_check(
            recovery, release_id=release_id,
            context={**header, "recovery_controls": recovery},
        ),
        _kill_switch_check(
            kill_switch, release_id=release_id,
            context={**header, "kill_switch": kill_switch},
        ),
    )
    if not common:
        return checks
    return tuple(
        P4GateCheck(
            name=check.name, status=P4CheckStatus.FAIL,
            evidence_ids=(), details=tuple(common) + check.details,
        )
        for check in checks
    )


__all__ = ["build_recovery_control_checks"]
