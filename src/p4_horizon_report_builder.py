"""Build P4 horizon evidence from independent validation reports.

The builder does not run a backtest and cannot turn a rejected report into an
accepted one.  It validates two independently produced report payloads,
derives every gate boolean from their observations and fingerprints the source
reports and approved threshold manifest for traceability.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import hashlib
import json


_EXPECTED_VERSIONS = {
    "swing": "p4.3-swing-v1",
    "medium_term": "p4.3-medium-term-v1",
}


def _fingerprint(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _aware(value: object, label: str) -> str:
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a timezone-aware ISO timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be a timezone-aware ISO timestamp.")
    return text


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required.")
    return value.strip()


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean.")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    return float(value)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _build_horizon(
    report: Mapping[str, object],
    *,
    expected_horizon: str,
    threshold_manifest_id: str,
) -> dict[str, object]:
    if report.get("schema_version") != 1:
        raise ValueError(f"{expected_horizon} report schema_version must be 1.")
    horizon = str(report.get("horizon", "")).strip().lower()
    if horizon != expected_horizon:
        raise ValueError(
            f"Expected {expected_horizon} report, received {horizon or 'blank'}."
        )
    expected_version = _EXPECTED_VERSIONS[horizon]
    if report.get("strategy_version") != expected_version:
        raise ValueError(
            f"{horizon} strategy_version must be {expected_version!r}."
        )
    _aware(report.get("generated_at"), f"{horizon}.generated_at")
    _required_text(report.get("dataset_id"), f"{horizon}.dataset_id")
    _required_text(report.get("cost_model_id"), f"{horizon}.cost_model_id")

    raw = report.get("validation")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{horizon}.validation must be an object.")
    out_of_sample = _bool(
        raw.get("out_of_sample_passed"),
        f"{horizon}.validation.out_of_sample_passed",
    )
    walk_forward = _bool(
        raw.get("walk_forward_passed"),
        f"{horizon}.validation.walk_forward_passed",
    )
    costs = _bool(
        raw.get("costs_included"), f"{horizon}.validation.costs_included"
    )
    observed_trades = _positive_int(
        raw.get("observed_trade_count"),
        f"{horizon}.validation.observed_trade_count",
    )
    minimum_trades = _positive_int(
        raw.get("minimum_trade_count"),
        f"{horizon}.validation.minimum_trade_count",
    )
    stability = _number(
        raw.get("parameter_stability"),
        f"{horizon}.validation.parameter_stability",
    )
    minimum_stability = _number(
        raw.get("minimum_parameter_stability"),
        f"{horizon}.validation.minimum_parameter_stability",
    )
    if not 0 <= stability <= 1 or not 0 <= minimum_stability <= 1:
        raise ValueError(f"{horizon} parameter stability values must be 0..1.")

    trade_count_met = observed_trades >= minimum_trades
    stability_passed = stability >= minimum_stability
    accepted = all(
        (out_of_sample, walk_forward, costs, trade_count_met, stability_passed)
    )
    report_id = f"sha256:{_fingerprint(report)}"
    validation_id = f"sha256:{_fingerprint(raw)}"
    return {
        "horizon": horizon,
        "strategy_version": expected_version,
        "decision": "ACCEPT" if accepted else "REJECT",
        "out_of_sample_passed": out_of_sample,
        "walk_forward_passed": walk_forward,
        "costs_included": costs,
        "minimum_trade_count_met": trade_count_met,
        "parameter_stability_passed": stability_passed,
        "acceptance_report_id": report_id,
        "validation_report_id": validation_id,
        "threshold_manifest_id": threshold_manifest_id,
    }


def build_horizon_evidence(
    *,
    release_id: str,
    observed_at: object,
    swing_report: Mapping[str, object],
    medium_term_report: Mapping[str, object],
    threshold_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Return gate-compatible evidence derived from source observations."""

    release = _required_text(release_id, "release_id")
    observed = _aware(observed_at, "observed_at")
    if threshold_manifest.get("schema_version") != 1:
        raise ValueError("threshold manifest schema_version must be 1.")
    if threshold_manifest.get("approval_status") != "APPROVED_FOR_P2_RELEASE":
        raise ValueError("threshold manifest must be approved for P2 release.")
    manifest_id = f"sha256:{_fingerprint(threshold_manifest)}"
    return {
        "schema_version": 1,
        "release_id": release,
        "observed_at": observed,
        "horizons": [
            _build_horizon(
                swing_report,
                expected_horizon="swing",
                threshold_manifest_id=manifest_id,
            ),
            _build_horizon(
                medium_term_report,
                expected_horizon="medium_term",
                threshold_manifest_id=manifest_id,
            ),
        ],
    }


__all__ = ["build_horizon_evidence"]
