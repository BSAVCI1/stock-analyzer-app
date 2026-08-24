"""P4.11.4 email and Telegram delivery evidence tests."""

from __future__ import annotations

from copy import deepcopy
import json

from src.jobs.cli import main
from src.p4_notification_evidence import build_notification_delivery_checks
from src.p4_release_gate import P4CheckStatus


def _record(channel: str, suffix: str) -> dict[str, object]:
    return {
        "notification_id": f"NOTIFY-{suffix}",
        "channel": channel,
        "status": "SENT",
        "reference_type": "SYSTEM_EVENT",
        "reference_id": f"EVENT-{suffix}",
        "sent_at": "2026-08-24T18:55:00+00:00",
        "attempt_count": 1,
        "provider_message_id": f"PROVIDER-{suffix}",
    }


def _evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "release_id": "P4-2026-08-24",
        "account_id": "ACC-PAPER",
        "observed_at": "2026-08-24T19:00:00+00:00",
        "email": {
            "persisted": True, "deduplicated": True, "retryable": True,
            "pending_count": 0, "failed_count": 0,
            "sent_records": [_record("EMAIL", "EMAIL-1")],
        },
        "telegram": {
            "persisted": True, "deduplicated": True, "retryable": True,
            "pending_count": 0, "failed_count": 0,
            "sent_records": [_record("TELEGRAM", "TELEGRAM-1")],
        },
    }


def test_complete_channel_evidence_passes_independently() -> None:
    email, telegram = build_notification_delivery_checks(_evidence())
    assert email.status is P4CheckStatus.PASS
    assert telegram.status is P4CheckStatus.PASS
    assert email.name == "email_delivery"
    assert telegram.name == "telegram_delivery"
    assert "NOTIFICATION:NOTIFY-EMAIL-1" in email.evidence_ids


def test_configured_without_sent_record_does_not_pass() -> None:
    evidence = _evidence()
    evidence["telegram"]["sent_records"] = []
    email, telegram = build_notification_delivery_checks(evidence)
    assert email.status is P4CheckStatus.PASS
    assert telegram.status is P4CheckStatus.FAIL
    assert telegram.evidence_ids == ()


def test_pending_or_failed_delivery_blocks_channel() -> None:
    evidence = _evidence()
    evidence["email"]["pending_count"] = 1
    evidence["email"]["failed_count"] = 1
    email, telegram = build_notification_delivery_checks(evidence)
    assert email.status is P4CheckStatus.FAIL
    assert telegram.status is P4CheckStatus.PASS
    assert any("pending_count" in item for item in email.details)
    assert any("failed_count" in item for item in email.details)


def test_wrong_channel_status_or_missing_source_fails() -> None:
    evidence = _evidence()
    record = evidence["telegram"]["sent_records"][0]
    record["channel"] = "EMAIL"
    record["status"] = "PENDING"
    record["reference_id"] = ""
    _, telegram = build_notification_delivery_checks(evidence)
    assert telegram.status is P4CheckStatus.FAIL
    assert any("channel" in item for item in telegram.details)
    assert any("status" in item for item in telegram.details)
    assert any("reference_id" in item for item in telegram.details)


def test_duplicate_notification_ids_fail() -> None:
    evidence = _evidence()
    first = evidence["email"]["sent_records"][0]
    evidence["email"]["sent_records"].append(deepcopy(first))
    email, _ = build_notification_delivery_checks(evidence)
    assert email.status is P4CheckStatus.FAIL
    assert any("unique" in item for item in email.details)


def test_missing_common_identity_blocks_both_channels() -> None:
    evidence = _evidence()
    evidence["release_id"] = ""
    evidence["observed_at"] = "2026-08-24T19:00:00"
    checks = build_notification_delivery_checks(evidence)
    assert all(check.status is P4CheckStatus.FAIL for check in checks)
    assert all(any("release_id" in item for item in check.details) for check in checks)
    assert all(any("observed_at" in item for item in check.details) for check in checks)


def test_fingerprint_is_deterministic_and_change_sensitive() -> None:
    evidence = _evidence()
    first = build_notification_delivery_checks(evidence)[0].evidence_ids[0]
    reordered = dict(reversed(tuple(evidence.items())))
    assert build_notification_delivery_checks(reordered)[0].evidence_ids[0] == first
    changed = deepcopy(evidence)
    changed["email"]["sent_records"][0]["attempt_count"] = 2
    assert build_notification_delivery_checks(changed)[0].evidence_ids[0] != first
    changed_identity = deepcopy(evidence)
    changed_identity["account_id"] = "ACC-OTHER"
    assert (
        build_notification_delivery_checks(changed_identity)[0].evidence_ids[0]
        != first
    )


def test_cli_and_deliberately_blocked_example(tmp_path, capsys) -> None:
    path = tmp_path / "notifications.json"
    path.write_text(json.dumps(_evidence()), encoding="utf-8")
    assert main(["p4-notification-evidence", "--evidence", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [item["status"] for item in result["checks"]] == ["PASS", "PASS"]
    assert main([
        "p4-notification-evidence", "--evidence",
        "config/p4_notification_evidence.example.json",
    ]) == 1
    blocked = json.loads(capsys.readouterr().out)
    assert [item["status"] for item in blocked["checks"]] == ["FAIL", "FAIL"]
