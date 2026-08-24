"""P4.11.3 scheduler/deployment evidence tests."""

from __future__ import annotations

from copy import deepcopy
import json

from src.jobs.cli import main
from src.p4_release_gate import P4CheckStatus
from src.p4_scheduler_evidence import build_scheduler_deployment_check


def _evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "deployment_id": "MAC-LOCAL-2026-08-24",
        "observed_at": "2026-08-24T18:30:00+00:00",
        "runtime_target": "LOCAL_DEVICE",
        "execution_mode": "PAPER",
        "scheduler_enabled": True,
        "managed_cycle_enabled": True,
        "container_healthy": True,
        "liveness_ok": True,
        "readiness_ok": True,
        "worker_heartbeat_current": True,
        "restart_verified": True,
        "persistent_storage_verified": True,
        "health_evidence_ids": ["HEALTH-LIVE", "HEALTH-READY", "HEALTH-WORKER"],
        "completed_cycle_ids": ["JOB-123", "SCAN-123", "RUN-123"],
        "restart_evidence_ids": ["RESTART-STATUS-123"],
        "storage_evidence_ids": ["PERSISTENCE-COUNT-119"],
    }


def test_complete_local_device_evidence_passes() -> None:
    check = build_scheduler_deployment_check(_evidence())
    assert check.status is P4CheckStatus.PASS
    assert check.name == "scheduler_deployment"
    assert check.evidence_ids[0].startswith(
        "DEPLOYMENT:MAC-LOCAL-2026-08-24:sha256:"
    )
    assert "JOB-123" in check.evidence_ids


def test_external_always_on_target_is_supported() -> None:
    evidence = _evidence()
    evidence["runtime_target"] = "EXTERNAL_ALWAYS_ON"
    check = build_scheduler_deployment_check(evidence)
    assert check.status is P4CheckStatus.PASS
    assert "EXTERNAL_ALWAYS_ON" in check.details[0]


def test_missing_health_and_restart_evidence_fails_closed() -> None:
    evidence = _evidence()
    evidence["worker_heartbeat_current"] = False
    evidence["restart_evidence_ids"] = []
    check = build_scheduler_deployment_check(evidence)
    assert check.status is P4CheckStatus.FAIL
    assert check.evidence_ids == ()
    assert any("worker_heartbeat_current" in item for item in check.details)
    assert any("restart_evidence_ids" in item for item in check.details)


def test_nonstring_evidence_id_is_rejected() -> None:
    evidence = _evidence()
    evidence["health_evidence_ids"] = [123]
    check = build_scheduler_deployment_check(evidence)
    assert check.status is P4CheckStatus.FAIL
    assert any("health_evidence_ids" in item for item in check.details)


def test_nonpaper_or_disabled_scheduler_fails_closed() -> None:
    evidence = _evidence()
    evidence["execution_mode"] = "LIVE"
    evidence["scheduler_enabled"] = False
    check = build_scheduler_deployment_check(evidence)
    assert check.status is P4CheckStatus.FAIL
    assert any("execution_mode" in item for item in check.details)
    assert any("scheduler_enabled" in item for item in check.details)


def test_naive_timestamp_and_invalid_target_fail() -> None:
    evidence = _evidence()
    evidence["observed_at"] = "2026-08-24T18:30:00"
    evidence["runtime_target"] = "UNKNOWN"
    check = build_scheduler_deployment_check(evidence)
    assert check.status is P4CheckStatus.FAIL
    assert any("timezone-aware" in item for item in check.details)
    assert any("runtime_target" in item for item in check.details)


def test_fingerprint_is_deterministic_and_change_sensitive() -> None:
    evidence = _evidence()
    first = build_scheduler_deployment_check(evidence).evidence_ids[0]
    reordered = dict(reversed(tuple(evidence.items())))
    assert build_scheduler_deployment_check(reordered).evidence_ids[0] == first
    changed = deepcopy(evidence)
    changed["completed_cycle_ids"].append("JOB-456")
    assert build_scheduler_deployment_check(changed).evidence_ids[0] != first


def test_cli_outputs_manifest_ready_check(tmp_path, capsys) -> None:
    path = tmp_path / "scheduler.json"
    path.write_text(json.dumps(_evidence()), encoding="utf-8")
    assert main(["p4-scheduler-evidence", "--evidence", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["check"]["status"] == "PASS"


def test_blocked_example_returns_one(capsys) -> None:
    assert main([
        "p4-scheduler-evidence", "--evidence",
        "config/p4_scheduler_evidence.example.json",
    ]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["check"]["status"] == "FAIL"
