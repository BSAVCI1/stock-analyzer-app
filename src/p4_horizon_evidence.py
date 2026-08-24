"""Fail-closed P4 strategy-horizon acceptance evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime

from src.p4_release_gate import P4CheckStatus, P4GateCheck


_EXPECTED_VERSIONS = {
    "swing": "p4.3-swing-v1",
    "medium_term": "p4.3-medium-term-v1",
}


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


def build_strategy_horizon_check(
    evidence: Mapping[str, object],
) -> P4GateCheck:
    """Require independent accepted evidence for both configured horizons."""
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must be an object.")
    failures: list[str] = []
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != 1
    ):
        failures.append("schema_version must be 1.")
    release_id = str(evidence.get("release_id", "")).strip()
    if not release_id:
        failures.append("release_id is required.")
    if not _aware(evidence.get("observed_at")):
        failures.append("observed_at must be timezone-aware.")

    raw_horizons = evidence.get("horizons")
    horizons = raw_horizons if isinstance(raw_horizons, list) else []
    if len(horizons) != 2:
        failures.append("horizons must contain exactly two decisions.")

    names: list[str] = []
    evidence_ids: list[str] = []
    for index, raw in enumerate(horizons):
        path = f"horizons[{index}]"
        if not isinstance(raw, Mapping):
            failures.append(f"{path} must be an object.")
            continue
        horizon = str(raw.get("horizon", "")).strip().lower()
        names.append(horizon)
        expected_version = _EXPECTED_VERSIONS.get(horizon)
        if expected_version is None:
            failures.append(f"{path}.horizon is not enabled.")
        elif raw.get("strategy_version") != expected_version:
            failures.append(
                f"{path}.strategy_version must be {expected_version!r}."
            )
        if str(raw.get("decision", "")).strip().upper() != "ACCEPT":
            failures.append(f"{path}.decision must be ACCEPT.")
        for key in (
            "out_of_sample_passed",
            "walk_forward_passed",
            "costs_included",
            "minimum_trade_count_met",
            "parameter_stability_passed",
        ):
            if raw.get(key) is not True:
                failures.append(f"{path}.{key} must be true.")
        for key in (
            "acceptance_report_id",
            "validation_report_id",
            "threshold_manifest_id",
        ):
            value = raw.get(key)
            if not isinstance(value, str) or not value.strip():
                failures.append(f"{path}.{key} is required.")
            else:
                evidence_ids.append(f"{key.upper()}:{value.strip()}")

    if len(names) != len(set(names)):
        failures.append("horizon decisions must be unique.")
    if set(names) != set(_EXPECTED_VERSIONS):
        failures.append("horizons must contain swing and medium_term independently.")

    if failures:
        return P4GateCheck(
            name="strategy_horizon_acceptance",
            status=P4CheckStatus.FAIL,
            evidence_ids=(),
            details=tuple(failures),
        )
    return P4GateCheck(
        name="strategy_horizon_acceptance",
        status=P4CheckStatus.PASS,
        evidence_ids=(
            f"HORIZON-ACCEPTANCE:{release_id}:sha256:{_fingerprint(evidence)}",
            *tuple(dict.fromkeys(evidence_ids)),
        ),
        details=(
            "Swing and medium-term horizons independently passed versioned, "
            "out-of-sample, walk-forward and cost-aware acceptance.",
        ),
    )


__all__ = ["build_strategy_horizon_check"]
