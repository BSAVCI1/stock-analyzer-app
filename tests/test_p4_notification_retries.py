from datetime import (
    datetime,
    timedelta,
    timezone,
)

from src.notifications import (
    DeliveryResult,
    NOTIFICATION_RETRY_POLICY_VERSION,
    NotificationRetryPolicy,
    NotificationService,
    RetryEligibility,
    evaluate_notification_retry,
)
from src.paper import (
    NotificationChannel,
    NotificationStatus,
    PaperRepository,
)


T0 = datetime(
    2026,
    8,
    18,
    14,
    0,
    tzinfo=timezone.utc,
)


class AlwaysFail:
    def __init__(self):
        self.calls = 0

    def send(self, notification):
        self.calls += 1
        raise RuntimeError("temporary outage")


class FailOnce:
    def __init__(self):
        self.calls = 0

    def send(self, notification):
        self.calls += 1

        if self.calls == 1:
            raise RuntimeError("temporary outage")

        return DeliveryResult(
            provider_message_id="MSG-RETRY",
            metadata={"provider": "test"},
        )


def queued(repository):
    account = repository.create_account(
        name="Retry Test",
        base_currency="USD",
        starting_balance="1000",
        created_at=T0,
    )
    notification = (
        repository.queue_notification(
            account_id=account.account_id,
            event_type="SYSTEM_FAILURE",
            reference_type="JOB",
            reference_id="JOB-RETRY",
            channel=(
                NotificationChannel.TELEGRAM
            ),
            payload={"message": "Job failed."},
            created_at=T0,
        )
    )

    return account, notification


def test_delivery_evidence_is_hydrated(tmp_path):
    repository = PaperRepository(
        tmp_path / "evidence.db"
    )
    account, notification = queued(
        repository
    )
    sender = AlwaysFail()
    service = NotificationService(
        repository,
        senders={
            NotificationChannel.TELEGRAM:
                sender,
        },
    )

    service.dispatch_pending(
        account.account_id,
        attempted_at=T0,
    )
    stored = repository.get_notification(
        notification.notification_id
    )

    assert stored.attempt_count == 1
    assert stored.last_attempt_at == T0
    assert stored.provider_message_id is None
    assert (
        stored.delivery_metadata[
            "retry_policy_version"
        ]
        == NOTIFICATION_RETRY_POLICY_VERSION
    )
    assert (
        stored.delivery_metadata[
            "attempt_number"
        ]
        == 1
    )


def test_retry_waits_for_backoff(tmp_path):
    repository = PaperRepository(
        tmp_path / "backoff.db"
    )
    account, notification = queued(
        repository
    )
    sender = AlwaysFail()
    service = NotificationService(
        repository,
        senders={
            NotificationChannel.TELEGRAM:
                sender,
        },
    )

    service.dispatch_pending(
        account.account_id,
        attempted_at=T0,
    )
    report = service.dispatch_pending(
        account.account_id,
        include_failed=True,
        attempted_at=(
            T0 + timedelta(seconds=30)
        ),
    )
    stored = repository.get_notification(
        notification.notification_id
    )
    decision = evaluate_notification_retry(
        stored,
        evaluated_at=(
            T0 + timedelta(seconds=30)
        ),
    )

    assert report.skipped == 1
    assert sender.calls == 1
    assert (
        decision.eligibility
        is RetryEligibility.BACKOFF
    )
    assert decision.next_attempt_at == (
        T0 + timedelta(seconds=60)
    )


def test_retry_succeeds_after_delay(tmp_path):
    repository = PaperRepository(
        tmp_path / "success.db"
    )
    account, notification = queued(
        repository
    )
    sender = FailOnce()
    service = NotificationService(
        repository,
        senders={
            NotificationChannel.TELEGRAM:
                sender,
        },
    )

    first = service.dispatch_pending(
        account.account_id,
        attempted_at=T0,
    )
    second = service.dispatch_pending(
        account.account_id,
        include_failed=True,
        attempted_at=(
            T0 + timedelta(seconds=60)
        ),
    )
    stored = repository.get_notification(
        notification.notification_id
    )

    assert first.failed == 1
    assert second.sent == 1
    assert stored.status is NotificationStatus.SENT
    assert stored.attempt_count == 2
    assert (
        stored.provider_message_id
        == "MSG-RETRY"
    )


def test_retry_stops_after_maximum_attempts(
    tmp_path,
):
    repository = PaperRepository(
        tmp_path / "exhausted.db"
    )
    account, notification = queued(
        repository
    )
    sender = AlwaysFail()
    service = NotificationService(
        repository,
        senders={
            NotificationChannel.TELEGRAM:
                sender,
        },
        retry_policy=NotificationRetryPolicy(
            maximum_attempts=3,
            base_delay_seconds=1,
            maximum_delay_seconds=2,
        ),
    )

    service.dispatch_pending(
        account.account_id,
        attempted_at=T0,
    )
    service.dispatch_pending(
        account.account_id,
        include_failed=True,
        attempted_at=T0 + timedelta(seconds=1),
    )
    service.dispatch_pending(
        account.account_id,
        include_failed=True,
        attempted_at=T0 + timedelta(seconds=3),
    )
    exhausted = service.dispatch_pending(
        account.account_id,
        include_failed=True,
        attempted_at=T0 + timedelta(days=1),
    )
    stored = repository.get_notification(
        notification.notification_id
    )
    decision = evaluate_notification_retry(
        stored,
        evaluated_at=T0 + timedelta(days=1),
        policy=service.retry_policy,
    )

    assert sender.calls == 3
    assert exhausted.skipped == 1
    assert stored.attempt_count == 3
    assert (
        decision.eligibility
        is RetryEligibility.EXHAUSTED
    )
