from datetime import (
    datetime,
    timezone,
)
from decimal import Decimal

from src.notifications import (
    NOTIFICATION_ROUTING_POLICY_VERSION,
    NotificationPurpose,
    NotificationService,
    NotificationSeverity,
    route_notification_event,
)
from src.paper import (
    NotificationChannel,
    NotificationStatus,
    PaperPortfolioConfig,
    PaperRepository,
    PaperTradingService,
)


NOW = datetime(
    2026,
    8,
    18,
    12,
    0,
    tzinfo=timezone.utc,
)


def test_startup_status_routes_to_telegram_only():
    route = route_notification_event(
        "APPLICATION_STARTUP"
    )

    assert route.channels == (
        NotificationChannel.TELEGRAM,
    )
    assert (
        route.purpose
        is NotificationPurpose.CONCISE_STATUS
    )


def test_action_event_routes_to_both_channels():
    route = route_notification_event(
        "PAPER_BUY_EXECUTED"
    )

    assert route.channels == (
        NotificationChannel.TELEGRAM,
        NotificationChannel.EMAIL,
    )
    assert (
        route.severity
        is NotificationSeverity.ACTION
    )


def test_report_routes_to_email_only():
    route = route_notification_event(
        "WEEKLY_PERFORMANCE_REPORT"
    )

    assert route.channels == (
        NotificationChannel.EMAIL,
    )
    assert (
        route.purpose
        is NotificationPurpose.REPORT
    )


def test_failure_routes_to_both_with_detail():
    route = route_notification_event(
        "ENTRY_EXECUTION_ERROR"
    )

    assert route.channels == (
        NotificationChannel.TELEGRAM,
        NotificationChannel.EMAIL,
    )
    assert (
        route.severity
        is NotificationSeverity.CRITICAL
    )
    assert (
        route.policy_version
        == NOTIFICATION_ROUTING_POLICY_VERSION
    )


def test_unknown_event_is_email_audited():
    route = route_notification_event(
        "NEW_UNCLASSIFIED_EVENT"
    )

    assert route.channels == (
        NotificationChannel.EMAIL,
    )
    assert (
        route.severity
        is NotificationSeverity.WARNING
    )


def test_routed_fanout_is_persistent_and_idempotent(
    tmp_path,
):
    repository = PaperRepository(
        tmp_path / "paper.db"
    )
    trading = PaperTradingService(
        repository,
        config=PaperPortfolioConfig(
            starting_balance=Decimal("1000"),
            base_currency="EUR",
        ),
    )
    account = trading.create_account(
        created_at=NOW
    )
    internal = repository.queue_notification(
        account_id=account.account_id,
        event_type="PROVIDER_FAILURE",
        reference_type="JOB",
        reference_id="JOB-1",
        channel=NotificationChannel.INTERNAL,
        payload={"error": "provider unavailable"},
        created_at=NOW,
    )
    service = NotificationService(
        repository,
        senders={},
    )

    first = service.fan_out_routed(
        account.account_id,
        created_at=NOW,
    )
    second = service.fan_out_routed(
        account.account_id,
        created_at=NOW,
    )
    notifications = (
        repository.list_notifications(
            account.account_id
        )
    )
    external_channels = {
        item.channel
        for item in notifications
        if item.channel
        is not NotificationChannel.INTERNAL
    }

    assert first == 2
    assert second == 0
    assert external_channels == {
        NotificationChannel.TELEGRAM,
        NotificationChannel.EMAIL,
    }
    assert (
        repository.get_notification(
            internal.notification_id
        ).status
        is NotificationStatus.SENT
    )
