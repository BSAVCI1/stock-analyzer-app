"""P4.11.1 deterministic release-gate contract tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json

import pytest

from src.backtest import RegressionEvidence
from src.jobs.cli import main
from src.p4_release_gate import (
    P4CheckStatus,
    P4GateCheck,
    P4ReleaseEvidence,
    P4ReleaseStatus,
    REQUIRED_P4_CHECKS,
    evaluate_p4_release_gate,
)


T0 = datetime(2026, 8, 24, 20, 0, tzinfo=timezone.utc)


def _checks():
    return tuple(
        P4GateCheck(
            name=name, status=P4CheckStatus.PASS,
            evidence_ids=(f"EVIDENCE-{name}",),
            details=("Verified persisted evidence.",),
        )
        for name in REQUIRED_P4_CHECKS
    )


def _evidence():
    return P4ReleaseEvidence(
        schema_version=1,
        release_id="P4-RELEASE-2026-08-24",
        generated_at=T0,
        account_id="ACC-PAPER",
        regression=RegressionEvidence(
            passed=True, test_count=665,
            covered_phases=("P0", "P1", "P2", "P3", "P4"),
            workflow="Automated tests #191",
        ),
        checks=_checks(), execution_mode="PAPER",
        live_execution_enabled=False,
        unresolved_operational_failures=0,
    )


def test_complete_paper_only_evidence_is_ready() -> None:
    report = evaluate_p4_release_gate(_evidence())
    assert report.status is P4ReleaseStatus.READY
    assert report.release_ready is True
    assert report.blocking_checks == ()


def test_live_capability_and_operational_failures_always_block() -> None:
    report = evaluate_p4_release_gate(replace(
        _evidence(), live_execution_enabled=True,
        unresolved_operational_failures=2,
    ))
    assert report.status is P4ReleaseStatus.BLOCKED
    assert any("Live execution" in reason for reason in report.reasons)
    assert any("2 unresolved" in reason for reason in report.reasons)


def test_nonpassing_required_check_remains_visible_and_blocking() -> None:
    checks = list(_checks())
    checks[3] = P4GateCheck(
        name="email_delivery", status=P4CheckStatus.NOT_OBSERVED,
        evidence_ids=(), details=("No sent email evidence.",),
    )
    report = evaluate_p4_release_gate(replace(_evidence(), checks=tuple(checks)))
    assert report.release_ready is False
    assert report.blocking_checks == ("email_delivery",)


def test_contract_rejects_missing_or_unevidenced_checks() -> None:
    with pytest.raises(ValueError, match="missing checks"):
        replace(_evidence(), checks=_checks()[:-1])
    with pytest.raises(ValueError, match="requires evidence IDs"):
        P4GateCheck(
            name="kill_switch", status=P4CheckStatus.PASS,
            evidence_ids=(), details=("Claimed pass.",),
        )


def test_cli_evaluates_versioned_manifest(tmp_path, capsys) -> None:
    evidence = _evidence()
    manifest = {
        "schema_version": 1,
        "release_id": evidence.release_id,
        "generated_at": evidence.generated_at.isoformat(),
        "account_id": evidence.account_id,
        "regression": {
            "passed": True, "test_count": 665,
            "covered_phases": ["P0", "P1", "P2", "P3", "P4"],
            "workflow": "Automated tests #191",
        },
        "checks": [
            {
                "name": item.name, "status": item.status.value,
                "evidence_ids": list(item.evidence_ids),
                "details": list(item.details),
            }
            for item in evidence.checks
        ],
        "execution_mode": "PAPER",
        "live_execution_enabled": False,
        "unresolved_operational_failures": 0,
    }
    path = tmp_path / "p4-release.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert main(["p4-release-status", "--manifest", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "READY"
    assert result["release_ready"] is True

