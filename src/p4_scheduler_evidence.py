"""Fail-closed P4 scheduler and deployment evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime

from src.p4_release_gate import P4CheckStatus, P4GateCheck


_RUNTIME_TARGETS = {"LOCAL_DEVICE", "EXTERNAL_ALWAYS_ON"}


def _fingerprint(evidence: Mapping[str, object]) -> str:
    canonical = json.dumps(
        evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _nonempty_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    if any(not isinstance(item, str) for item in value):
        return ()
    result = tuple(item.strip() for item in value)
    return result if result and all(result) else ()


def build_scheduler_deployment_check(
    evidence: Mapping[str, object],
) -> P4GateCheck:
    """Evaluate supplied deployment observations without running the service."""
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must be an object.")

    failures: list[str] = []
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != 1
    ):
        failures.append("schema_version must be 1.")

    deployment_id = str(evidence.get("deployment_id", "")).strip()
    if not deployment_id:
        failures.append("deployment_id is required.")

    observed_at = evidence.get("observed_at")
    try:
        timestamp = datetime.fromisoformat(str(observed_at))
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError
    except (TypeError, ValueError):
        failures.append("observed_at must be a timezone-aware ISO timestamp.")

    target = str(evidence.get("runtime_target", "")).strip().upper()
    if target not in _RUNTIME_TARGETS:
        failures.append(
            "runtime_target must be LOCAL_DEVICE or EXTERNAL_ALWAYS_ON."
        )
    if str(evidence.get("execution_mode", "")).strip().upper() != "PAPER":
        failures.append("execution_mode must be PAPER.")

    required_truths = (
        "scheduler_enabled",
        "managed_cycle_enabled",
        "container_healthy",
        "liveness_ok",
        "readiness_ok",
        "worker_heartbeat_current",
        "restart_verified",
        "persistent_storage_verified",
    )
    for key in required_truths:
        if evidence.get(key) is not True:
            failures.append(f"{key} must be true.")

    evidence_ids: list[str] = []
    for key in (
        "health_evidence_ids",
        "completed_cycle_ids",
        "restart_evidence_ids",
        "storage_evidence_ids",
    ):
        values = _nonempty_strings(evidence.get(key))
        if not values:
            failures.append(f"{key} must contain at least one evidence ID.")
        else:
            evidence_ids.extend(values)

    if failures:
        return P4GateCheck(
            name="scheduler_deployment",
            status=P4CheckStatus.FAIL,
            evidence_ids=(),
            details=tuple(failures),
        )

    fingerprint = _fingerprint(evidence)
    unique_ids = tuple(dict.fromkeys(evidence_ids))
    return P4GateCheck(
        name="scheduler_deployment",
        status=P4CheckStatus.PASS,
        evidence_ids=(
            f"DEPLOYMENT:{deployment_id}:sha256:{fingerprint}",
            *unique_ids,
        ),
        details=(
            f"{target} paper scheduler, health, restart and storage evidence verified.",
        ),
    )


__all__ = ["build_scheduler_deployment_check"]
