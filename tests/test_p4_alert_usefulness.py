"""P4.10.5 alert usefulness and manual-copy journal tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import pytest

from src.jobs.cli import main
from src.paper import (
    AlertUsefulness,
    ManualAlertAction,
    NotificationChannel,
    PaperRepository,
)
from src.portfolio_dashboard import calculate_alert_usefulness


T0 = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


def _environment(tmp_path):
    path = tmp_path / "alert-feedback.db"
    paper = PaperRepository(path)
    account = paper.create_account(
        name="Alert Feedback", base_currency="EUR", starting_balance="10000",
        created_at=T0,
    )
    notification = paper.queue_notification(
        account_id=account.account_id,
        event_type="ORDER_CANDIDATE", reference_type="SIGNAL",
        reference_id="SIG-1", channel=NotificationChannel.EMAIL,
        payload={"symbol": "AAPL"}, created_at=T0,
    )
    notification = paper.mark_notification_sent(
        notification.notification_id, sent_at=T0 + timedelta(minutes=1),
        provider_message_id="provider-1",
    )
    return path, paper, account, notification


def test_alert_feedback_is_idempotent_immutable_and_copy_traceable(tmp_path) -> None:
    path, paper, account, notification = _environment(tmp_path)
    kwargs = dict(
        account_id=account.account_id,
        notification_id=notification.notification_id,
        usefulness=AlertUsefulness.USEFUL,
        manual_action=ManualAlertAction.COPIED_MODIFIED,
        operator="Salih AVCI",
        rationale="Copied after reducing quantity.",
        broker_reference="IBKR-PAPER-123",
        recorded_at=T0 + timedelta(minutes=5),
    )
    first = paper.record_alert_feedback(**kwargs)
    duplicate = PaperRepository(path).record_alert_feedback(**kwargs)
    assert duplicate == first
    assert paper.list_alert_feedback(account.account_id) == (first,)
    with pytest.raises(ValueError, match="conflicts"):
        paper.record_alert_feedback(
            **{**kwargs, "usefulness": AlertUsefulness.NOT_USEFUL}
        )


def test_copy_requires_broker_reference_and_sent_alert(tmp_path) -> None:
    _, paper, account, notification = _environment(tmp_path)
    with pytest.raises(ValueError, match="broker_reference"):
        paper.record_alert_feedback(
            account_id=account.account_id,
            notification_id=notification.notification_id,
            usefulness=AlertUsefulness.USEFUL,
            manual_action=ManualAlertAction.COPIED_AS_IS,
            operator="operator", rationale="Copied.", recorded_at=T0,
        )
    pending = paper.queue_notification(
        account_id=account.account_id, event_type="WATCH",
        reference_type="SIGNAL", reference_id="SIG-2",
        channel=NotificationChannel.EMAIL, payload={}, created_at=T0,
    )
    with pytest.raises(ValueError, match="sent notification"):
        paper.record_alert_feedback(
            account_id=account.account_id,
            notification_id=pending.notification_id,
            usefulness=AlertUsefulness.NOT_USEFUL,
            manual_action=ManualAlertAction.DISMISSED,
            operator="operator", rationale="Not actionable.", recorded_at=T0,
        )


def test_alert_usefulness_uses_assessed_sent_denominators(tmp_path) -> None:
    _, paper, account, first = _environment(tmp_path)
    second = paper.queue_notification(
        account_id=account.account_id, event_type="ORDER_CANDIDATE",
        reference_type="SIGNAL", reference_id="SIG-2",
        channel=NotificationChannel.EMAIL, payload={}, created_at=T0,
    )
    second = paper.mark_notification_sent(
        second.notification_id, sent_at=T0 + timedelta(minutes=2),
    )
    feedback = paper.record_alert_feedback(
        account_id=account.account_id, notification_id=first.notification_id,
        usefulness=AlertUsefulness.USEFUL,
        manual_action=ManualAlertAction.COPIED_AS_IS,
        operator="operator", rationale="Useful and copied.",
        broker_reference="PAPER-1", recorded_at=T0 + timedelta(minutes=3),
    )
    summary = calculate_alert_usefulness(
        (first, second), (feedback,),
    )
    assert summary.sent_alerts == 2
    assert summary.assessed_alerts == 1
    assert summary.assessment_coverage_pct == 50.0
    assert summary.usefulness_rate_pct == 100.0
    assert summary.manual_copy_rate_pct == 100.0


def test_alert_feedback_cli_records_and_lists(tmp_path, capsys) -> None:
    path, _, account, notification = _environment(tmp_path)
    common = ["--database", str(path), "--account-id", account.account_id]
    assert main([
        "alert-feedback", *common, "record", notification.notification_id,
        "--usefulness", "NOT_USEFUL", "--manual-action", "DISMISSED",
        "--operator", "Salih AVCI", "--rationale", "Too late to act.",
        "--recorded-at", (T0 + timedelta(minutes=5)).isoformat(),
    ]) == 0
    recorded = json.loads(capsys.readouterr().out)
    assert recorded["manual_action"] == "DISMISSED"
    assert main(["alert-feedback", *common, "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["total"] == 1
