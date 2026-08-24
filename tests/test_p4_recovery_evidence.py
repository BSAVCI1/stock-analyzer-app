"""P4.11.5 recovery and kill-switch evidence tests."""

from __future__ import annotations

from copy import deepcopy
import json

from src.jobs.cli import main
from src.p4_recovery_evidence import build_recovery_control_checks
from src.p4_release_gate import P4CheckStatus


def _evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "release_id": "P4-2026-08-24",
        "account_id": "ACC-PAPER",
        "observed_at": "2026-08-24T19:30:00+00:00",
        "execution_mode": "PAPER",
        "recovery_controls": {
            "restart_recovery_verified": True,
            "idempotent_replay_verified": True,
            "stale_data_breaker_verified": True,
            "reconciliation_breaker_verified": True,
            "loss_limit_pause_verified": True,
            "provider_outage_verified": True,
            "incidents_closed": True,
            "unresolved_critical_incidents": 0,
            "restart_evidence_ids": ["RESTART-1"],
            "replay_evidence_ids": ["REPLAY-1"],
            "circuit_breaker_evidence_ids": ["BREAKER-1"],
            "provider_outage_evidence_ids": ["OUTAGE-1"],
            "incident_evidence_ids": ["INCIDENT-1"],
        },
        "kill_switch": {
            "activation_verified": True,
            "new_orders_blocked": True,
            "pending_order_policy_verified": True,
            "audit_trail_persisted": True,
            "recovery_verified": True,
            "final_state": "INACTIVE",
            "operator": "release-operator",
            "reason": "P4 release drill",
            "activation_evidence_ids": ["KILL-ACTIVATE-1"],
            "blocked_order_evidence_ids": ["ORDER-BLOCKED-1"],
            "audit_evidence_ids": ["KILL-AUDIT-1"],
            "recovery_evidence_ids": ["KILL-RECOVER-1"],
        },
    }


def test_complete_recovery_and_kill_switch_evidence_passes() -> None:
    recovery, kill_switch = build_recovery_control_checks(_evidence())
    assert recovery.status is P4CheckStatus.PASS
    assert kill_switch.status is P4CheckStatus.PASS
    assert recovery.name == "recovery_controls"
    assert kill_switch.name == "kill_switch"
    assert "RESTART-1" in recovery.evidence_ids
    assert "KILL-ACTIVATE-1" in kill_switch.evidence_ids


def test_unresolved_incident_blocks_recovery_only() -> None:
    evidence = _evidence()
    evidence["recovery_controls"]["unresolved_critical_incidents"] = 1
    recovery, kill_switch = build_recovery_control_checks(evidence)
    assert recovery.status is P4CheckStatus.FAIL
    assert kill_switch.status is P4CheckStatus.PASS
    assert any("unresolved_critical" in item for item in recovery.details)


def test_incomplete_kill_switch_drill_fails_closed() -> None:
    evidence = _evidence()
    evidence["kill_switch"]["new_orders_blocked"] = False
    evidence["kill_switch"]["final_state"] = "ACTIVE"
    evidence["kill_switch"]["audit_evidence_ids"] = []
    _, kill_switch = build_recovery_control_checks(evidence)
    assert kill_switch.status is P4CheckStatus.FAIL
    assert kill_switch.evidence_ids == ()
    assert any("new_orders_blocked" in item for item in kill_switch.details)
    assert any("final_state" in item for item in kill_switch.details)
    assert any("audit_evidence_ids" in item for item in kill_switch.details)


def test_nonpaper_or_missing_identity_blocks_both_checks() -> None:
    evidence = _evidence()
    evidence["execution_mode"] = "LIVE"
    evidence["account_id"] = ""
    checks = build_recovery_control_checks(evidence)
    assert all(check.status is P4CheckStatus.FAIL for check in checks)
    assert all(any("execution_mode" in item for item in check.details) for check in checks)
    assert all(any("account_id" in item for item in check.details) for check in checks)


def test_missing_category_evidence_fails() -> None:
    evidence = _evidence()
    evidence["recovery_controls"]["replay_evidence_ids"] = []
    recovery, _ = build_recovery_control_checks(evidence)
    assert recovery.status is P4CheckStatus.FAIL
    assert any("replay_evidence_ids" in item for item in recovery.details)


def test_fingerprints_bind_account_and_control_evidence() -> None:
    evidence = _evidence()
    checks = build_recovery_control_checks(evidence)
    changed_account = deepcopy(evidence)
    changed_account["account_id"] = "ACC-OTHER"
    changed_checks = build_recovery_control_checks(changed_account)
    assert changed_checks[0].evidence_ids[0] != checks[0].evidence_ids[0]
    changed_kill = deepcopy(evidence)
    changed_kill["kill_switch"]["operator"] = "other-operator"
    assert (
        build_recovery_control_checks(changed_kill)[1].evidence_ids[0]
        != checks[1].evidence_ids[0]
    )


def test_cli_and_deliberately_blocked_example(tmp_path, capsys) -> None:
    path = tmp_path / "recovery.json"
    path.write_text(json.dumps(_evidence()), encoding="utf-8")
    assert main(["p4-recovery-evidence", "--evidence", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [item["status"] for item in result["checks"]] == ["PASS", "PASS"]
    assert main([
        "p4-recovery-evidence", "--evidence",
        "config/p4_recovery_evidence.example.json",
    ]) == 1
    blocked = json.loads(capsys.readouterr().out)
    assert [item["status"] for item in blocked["checks"]] == ["FAIL", "FAIL"]
