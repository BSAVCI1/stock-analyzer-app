from datetime import (
    datetime,
    timezone,
)

from src.notifications import (
    ChannelHealthStatus,
    notification_channel_health,
    render_notification,
)
from src.paper import (
    NotificationChannel,
    PaperRepository,
)


NOW = datetime(
    2026,
    8,
    18,
    13,
    0,
    tzinfo=timezone.utc,
)


def queued(
    tmp_path,
    *,
    channel,
    event_type,
    payload,
):
    repository = PaperRepository(
        tmp_path / (
            channel.value.lower()
            + ".db"
        )
    )
    account = repository.create_account(
        name="Notification Template Test",
        base_currency="USD",
        starting_balance="1000",
        created_at=NOW,
    )

    return repository.queue_notification(
        account_id=account.account_id,
        event_type=event_type,
        reference_type="TEST",
        reference_id="REF-1",
        channel=channel,
        payload=payload,
        created_at=NOW,
    )


def buy_payload():
    return {
        "symbol": "AAPL",
        "quantity": "0.9",
        "fill_price": "101",
        "stop_price": "95",
        "targets": ["120", "130"],
    }


def test_telegram_buy_is_concise(tmp_path):
    notification = queued(
        tmp_path,
        channel=NotificationChannel.TELEGRAM,
        event_type="PAPER_BUY_EXECUTED",
        payload=buy_payload(),
    )

    rendered = render_notification(
        notification
    )

    assert rendered.text.startswith(
        "[PAPER BUY] AAPL"
    )
    assert "\n" not in rendered.text
    assert "Stop 95" in rendered.text


def test_email_buy_is_detailed(tmp_path):
    notification = queued(
        tmp_path,
        channel=NotificationChannel.EMAIL,
        event_type="PAPER_BUY_EXECUTED",
        payload=buy_payload(),
    )

    rendered = render_notification(
        notification
    )

    assert "Paper BUY executed\n" in (
        rendered.text
    )
    assert "Quantity: 0.9" in rendered.text
    assert "Targets: 120, 130" in rendered.text


def test_generic_payload_redacts_sensitive_keys(
    tmp_path,
):
    notification = queued(
        tmp_path,
        channel=NotificationChannel.EMAIL,
        event_type="SYSTEM_FAILURE",
        payload={
            "message": "Provider failed.",
            "api_token": "must-not-appear",
            "nested": {
                "password": "also-secret",
            },
        },
    )

    rendered = render_notification(
        notification
    )

    assert "must-not-appear" not in (
        rendered.text
    )
    assert "also-secret" not in rendered.text
    assert rendered.text.count(
        "[REDACTED]"
    ) == 2


def test_health_reports_disabled_without_config():
    health = notification_channel_health({})

    assert all(
        item.status
        is ChannelHealthStatus.DISABLED
        for item in health
    )


def test_health_reports_ready_without_exposing_secrets():
    secret = "telegram-secret-value"
    password = "smtp-secret-value"
    health = notification_channel_health(
        {
            "PAPER_TELEGRAM_BOT_TOKEN":
                secret,
            "PAPER_TELEGRAM_CHAT_ID":
                "12345",
            "PAPER_SMTP_HOST":
                "smtp.example.test",
            "PAPER_EMAIL_FROM":
                "bot@example.test",
            "PAPER_EMAIL_TO":
                "owner@example.test",
            "PAPER_SMTP_USERNAME":
                "bot-user",
            "PAPER_SMTP_PASSWORD":
                password,
        }
    )

    assert all(
        item.status
        is ChannelHealthStatus.READY
        for item in health
    )
    assert secret not in str(health)
    assert password not in str(health)


def test_partial_configuration_is_misconfigured():
    health = notification_channel_health(
        {
            "PAPER_TELEGRAM_BOT_TOKEN":
                "token-only",
            "PAPER_SMTP_USERNAME":
                "username-only",
        }
    )

    assert all(
        item.status
        is ChannelHealthStatus.MISCONFIGURED
        for item in health
    )
