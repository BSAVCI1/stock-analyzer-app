from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
import sqlite3

from src.notifications import (
    DeliveryResult,
    NotificationService,
)
from src.paper import (
    NotificationChannel,
    NotificationStatus,
    PaperRepository,
)


T0 = datetime(
    2026,
    8,
    1,
    20,
    0,
    tzinfo=timezone.utc,
)


class SuccessfulSender:
    def __init__(self) -> None:
        self.messages = []

    def send(self, notification):
        self.messages.append(notification)

        return DeliveryResult(
            provider_message_id="MSG-001",
            metadata={
                "provider": "test",
            },
        )


class FailingSender:
    def send(self, notification):
        raise RuntimeError(
            "Simulated delivery failure."
        )


def create_account(
    repository: PaperRepository,
):
    return repository.create_account(
        name="Notification Test",
        base_currency="USD",
        starting_balance="10000",
        created_at=T0,
    )


def queue_internal(
    repository: PaperRepository,
    account_id: str,
):
    return repository.queue_notification(
        account_id=account_id,
        event_type="PAPER_BUY_EXECUTED",
        reference_type="POSITION",
        reference_id="POSITION-001",
        channel=(
            NotificationChannel.INTERNAL
        ),
        payload={
            "symbol": "AAPL",
            "quantity": "10",
            "fill_price": "100",
            "stop_price": "95",
            "targets": ["110", "120"],
        },
        created_at=T0,
    )


def test_schema_version_four_and_job_table(
    tmp_path,
) -> None:
    repository = PaperRepository(
        tmp_path / "notifications.db"
    )

    connection = sqlite3.connect(
        repository.database_path
    )

    try:
        version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }

        notification_columns = {
            row[1]
            for row in connection.execute(
                """
                PRAGMA table_info(
                    paper_notifications
                )
                """
            )
        }
    finally:
        connection.close()

    assert version == 14

    assert "paper_job_runs" in tables

    assert {
        "attempt_count",
        "last_attempt_at",
        "provider_message_id",
        "delivery_metadata_json",
    }.issubset(notification_columns)


def test_internal_fanout_is_idempotent(
    tmp_path,
) -> None:
    repository = PaperRepository(
        tmp_path / "notifications.db"
    )

    account = create_account(repository)
    queue_internal(
        repository,
        account.account_id,
    )

    service = NotificationService(
        repository,
        senders={},
    )

    first = service.fan_out_internal(
        account.account_id,
        channels=(
            NotificationChannel.EMAIL,
            NotificationChannel.TELEGRAM,
        ),
        created_at=T0,
    )

    second = service.fan_out_internal(
        account.account_id,
        channels=(
            NotificationChannel.EMAIL,
            NotificationChannel.TELEGRAM,
        ),
        created_at=T0,
    )

    notifications = (
        repository.list_notifications(
            account.account_id
        )
    )

    assert first == 2
    assert second == 0
    assert len(notifications) == 3

    internal = next(
        item
        for item in notifications
        if item.channel
        is NotificationChannel.INTERNAL
    )

    assert (
        internal.status
        is NotificationStatus.SENT
    )


def test_successful_delivery_is_persisted(
    tmp_path,
) -> None:
    repository = PaperRepository(
        tmp_path / "notifications.db"
    )

    account = create_account(repository)

    repository.queue_notification(
        account_id=account.account_id,
        event_type="DAILY_SUMMARY",
        reference_type="JOB_RUN",
        reference_id="JOB-001",
        channel=NotificationChannel.EMAIL,
        payload={
            "subject": "Daily summary",
            "text": "Portfolio is reconciled.",
        },
        created_at=T0,
    )

    sender = SuccessfulSender()

    service = NotificationService(
        repository,
        senders={
            NotificationChannel.EMAIL:
            sender,
        },
    )

    report = service.dispatch_pending(
        account.account_id,
        attempted_at=T0,
    )

    notification = (
        repository.list_notifications(
            account.account_id
        )[0]
    )

    assert report.sent == 1
    assert report.failed == 0

    assert (
        notification.status
        is NotificationStatus.SENT
    )

    assert notification.sent_at == T0
    assert len(sender.messages) == 1


def test_failed_delivery_is_persisted(
    tmp_path,
) -> None:
    repository = PaperRepository(
        tmp_path / "notifications.db"
    )

    account = create_account(repository)

    repository.queue_notification(
        account_id=account.account_id,
        event_type="SYSTEM_FAILURE",
        reference_type="JOB_RUN",
        reference_id="JOB-FAILED",
        channel=(
            NotificationChannel.TELEGRAM
        ),
        payload={
            "subject": "System failure",
            "text": "Paper job failed.",
        },
        created_at=T0,
    )

    service = NotificationService(
        repository,
        senders={
            NotificationChannel.TELEGRAM:
            FailingSender(),
        },
    )

    report = service.dispatch_pending(
        account.account_id,
        attempted_at=T0,
    )

    notification = (
        repository.list_notifications(
            account.account_id
        )[0]
    )

    assert report.sent == 0
    assert report.failed == 1

    assert (
        notification.status
        is NotificationStatus.FAILED
    )

    assert "Simulated delivery failure" in (
        notification.error_message
    )


def test_missing_sender_is_recorded_as_failure(
    tmp_path,
) -> None:
    repository = PaperRepository(
        tmp_path / "notifications.db"
    )

    account = create_account(repository)

    repository.queue_notification(
        account_id=account.account_id,
        event_type="WEEKLY_REPORT",
        reference_type="JOB_RUN",
        reference_id="JOB-WEEKLY",
        channel=NotificationChannel.EMAIL,
        payload={
            "subject": "Weekly report",
            "text": "Weekly reliability report.",
        },
        created_at=T0,
    )

    service = NotificationService(
        repository,
        senders={},
    )

    report = service.dispatch_pending(
        account.account_id,
        attempted_at=T0,
    )

    notification = (
        repository.list_notifications(
            account.account_id
        )[0]
    )

    assert report.failed == 1

    assert (
        notification.status
        is NotificationStatus.FAILED
    )

    assert "No sender is configured" in (
        notification.error_message
    )
