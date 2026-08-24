"""P4.11.7 regression evidence and assembly tests."""

from __future__ import annotations

from copy import deepcopy
import json

from src.jobs.cli import main
from src.p4_evidence_assembly import (
    assemble_p4_release_evidence,
    build_p4_regression_evidence,
)
from src.p4_release_gate import evaluate_p4_release_gate


def _regression() -> dict[str, object]:
    return {
        "schema_version": 1,
        "conclusion": "success",
        "test_count": 708,
        "covered_phases": ["P0", "P1", "P2", "P3", "P4"],
        "workflow_run_id": "32764865198",
        "workflow_url": (
            "https://github.com/BSAVCI1/stock-analyzer-app/actions/runs/"
            "32764865198"
        ),
        "commit_sha": "7ff3189985eea3a98743c86402231d0f87f86cb1",
        "completed_at": "2026-08-24T20:10:00+00:00",
    }


def test_traceable_complete_regression_passes() -> None:
    result = build_p4_regression_evidence(_regression())
    assert result.passed is True
    assert result.covered_phases == ("P0", "P1", "P2", "P3", "P4")
    assert "32764865198" in result.workflow


def test_success_without_traceable_run_or_all_phases_does_not_pass() -> None:
    evidence = _regression()
    evidence["workflow_url"] = "https://example.com/not-github"
    evidence["covered_phases"] = ["P0", "P1", "P2", "P3"]
    result = build_p4_regression_evidence(evidence)
    assert result.passed is False


def _load(path: str) -> dict[str, object]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _passing_inputs() -> dict[str, dict[str, object]]:
    release = _load("config/p4_release_metadata.example.json")
    release["release_id"] = "P4-2026-08-24"
    release["account_id"] = "ACC-PAPER"

    policy = _load("config/product_policy_v1.json")
    scheduler = _load("config/p4_scheduler_evidence.example.json")
    scheduler.update({
        "release_id": release["release_id"], "account_id": "ACC-PAPER",
        "deployment_id": "DEPLOY-1", "scheduler_enabled": True,
        "managed_cycle_enabled": True, "container_healthy": True,
        "liveness_ok": True, "readiness_ok": True,
        "worker_heartbeat_current": True, "restart_verified": True,
        "persistent_storage_verified": True,
        "health_evidence_ids": ["HEALTH-1"],
        "completed_cycle_ids": ["JOB-1"],
        "restart_evidence_ids": ["RESTART-1"],
        "storage_evidence_ids": ["STORE-1"],
    })
    notifications = _load("config/p4_notification_evidence.example.json")
    notifications.update({"release_id": release["release_id"], "account_id": "ACC-PAPER"})
    for channel in ("email", "telegram"):
        notifications[channel]["sent_records"] = [{
            "notification_id": f"NOTIFY-{channel}",
            "channel": channel.upper(), "status": "SENT",
            "reference_type": "SYSTEM_EVENT", "reference_id": "EVENT-1",
            "sent_at": "2026-08-24T20:00:00+00:00", "attempt_count": 1,
        }]
    recovery = _load("config/p4_recovery_evidence.example.json")
    recovery.update({"release_id": release["release_id"], "account_id": "ACC-PAPER"})
    for key in (
        "restart_recovery_verified", "idempotent_replay_verified",
        "stale_data_breaker_verified", "reconciliation_breaker_verified",
        "loss_limit_pause_verified", "provider_outage_verified", "incidents_closed",
    ):
        recovery["recovery_controls"][key] = True
    for key in (
        "restart_evidence_ids", "replay_evidence_ids",
        "circuit_breaker_evidence_ids", "provider_outage_evidence_ids",
        "incident_evidence_ids",
    ):
        recovery["recovery_controls"][key] = [key]
    recovery["kill_switch"].update({
        "activation_verified": True, "new_orders_blocked": True,
        "pending_order_policy_verified": True, "audit_trail_persisted": True,
        "recovery_verified": True, "final_state": "INACTIVE",
    })
    for key in (
        "activation_evidence_ids", "blocked_order_evidence_ids",
        "audit_evidence_ids", "recovery_evidence_ids",
    ):
        recovery["kill_switch"][key] = [key]
    horizons = _load("config/p4_horizon_evidence.example.json")
    horizons["release_id"] = release["release_id"]
    for item in horizons["horizons"]:
        item.update({
            "decision": "ACCEPT", "out_of_sample_passed": True,
            "walk_forward_passed": True, "costs_included": True,
            "minimum_trade_count_met": True, "parameter_stability_passed": True,
            "acceptance_report_id": f"ACCEPT-{item['horizon']}",
            "validation_report_id": f"VALIDATE-{item['horizon']}",
            "threshold_manifest_id": f"THRESHOLD-{item['horizon']}",
        })
    return {
        "release": release, "policy": policy, "regression": _regression(),
        "scheduler": scheduler, "notifications": notifications,
        "recovery": recovery, "horizons": horizons,
    }


def test_complete_evidence_assembles_ready_gate() -> None:
    evidence = assemble_p4_release_evidence(**_passing_inputs())
    report = evaluate_p4_release_gate(evidence)
    assert report.release_ready is True
    assert tuple(check.name for check in evidence.checks) == (
        "paper_only_invariants", "eur_portfolio_policy",
        "scheduler_deployment", "email_delivery", "telegram_delivery",
        "recovery_controls", "kill_switch", "strategy_horizon_acceptance",
    )


def test_one_component_failure_remains_visible_and_blocking() -> None:
    inputs = _passing_inputs()
    inputs["notifications"]["telegram"]["sent_records"] = []
    report = evaluate_p4_release_gate(assemble_p4_release_evidence(**inputs))
    assert report.release_ready is False
    assert report.blocking_checks == ("telegram_delivery",)


def test_live_capability_blocks_even_complete_evidence() -> None:
    inputs = _passing_inputs()
    inputs["release"]["live_execution_enabled"] = True
    report = evaluate_p4_release_gate(assemble_p4_release_evidence(**inputs))
    assert report.release_ready is False
    assert any("Live execution" in reason for reason in report.reasons)


def test_cross_release_or_account_evidence_is_rejected() -> None:
    inputs = _passing_inputs()
    inputs["scheduler"]["account_id"] = "ACC-OTHER"
    try:
        assemble_p4_release_evidence(**inputs)
    except ValueError as exc:
        assert "scheduler account_id" in str(exc)
    else:
        raise AssertionError("Cross-account evidence must be rejected.")

    inputs = _passing_inputs()
    inputs["horizons"]["release_id"] = "P4-OTHER"
    try:
        assemble_p4_release_evidence(**inputs)
    except ValueError as exc:
        assert "horizons release_id" in str(exc)
    else:
        raise AssertionError("Cross-release evidence must be rejected.")


def test_cli_assembles_repository_examples_as_blocked(capsys) -> None:
    result = main([
        "p4-assemble-evidence",
        "--release", "config/p4_release_metadata.example.json",
        "--policy", "config/product_policy_v1.json",
        "--regression", "config/p4_regression_evidence.example.json",
        "--scheduler", "config/p4_scheduler_evidence.example.json",
        "--notifications", "config/p4_notification_evidence.example.json",
        "--recovery", "config/p4_recovery_evidence.example.json",
        "--horizons", "config/p4_horizon_evidence.example.json",
    ])
    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "BLOCKED"
    assert payload["release_ready"] is False
    assert len(payload["blocking_checks"]) == 6
