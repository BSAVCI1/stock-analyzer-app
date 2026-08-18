from datetime import (
    datetime,
    timezone,
)

import pytest

from src.notifications import (
    EmailConfig,
    EmailNotificationSender,
    NotificationService,
    RenderedNotification,
    TelegramConfig,
    TelegramNotificationSender,
)
from src.paper import (
    NotificationChannel,
    NotificationStatus,
    PaperRepository,
)


NOW = datetime(
    2026,
    8,
    18,
    15,
    0,
    tzinfo=timezone.utc,
)


class TelegramResponse:
    def __init__(self, *, ok=True):
        self.ok = ok

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "ok": self.ok,
            "result": {
                "message_id": 77,
            },
        }


class TelegramSession:
    def __init__(self):
        self.calls = []

    def post(self, url, *, json, timeout):
        self.calls.append(
            {
                "url": url,
                "json": json,
                "timeout": timeout,
            }
        )
        return TelegramResponse()


class FailingTelegramSession:
    def post(self, url, *, json, timeout):
        raise RuntimeError(
            "transport exposed " + url
        )


class SMTPClient:
    def __init__(
        self,
        *,
        fail_login=False,
    ):
        self.fail_login = fail_login
        self.started_tls = False
        self.login_values = None
        self.messages = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def starttls(self):
        self.started_tls = True

    def login(self, username, password):
        self.login_values = (
            username,
            password,
        )

        if self.fail_login:
            raise RuntimeError(
                "authentication exposed "
                + password
            )

    def send_message(self, message):
        self.messages.append(message)


class SMTPFactory:
    def __init__(self, client):
        self.client = client
        self.calls = []

    def __call__(
        self,
        host,
        port,
        *,
        timeout,
    ):
        self.calls.append(
            (host, port, timeout)
        )
        return self.client


def test_direct_telegram_sender_contract():
    session = TelegramSession()
    sender = TelegramNotificationSender(
        TelegramConfig(
            bot_token="test-token",
            chat_id="12345",
        ),
        session=session,
    )

    result = sender.send(
        RenderedNotification(
            subject="Test",
            text="Paper service started.",
        )
    )

    assert (
        result.provider_message_id == "77"
    )
    assert len(session.calls) == 1
    assert (
        session.calls[0]["json"]["chat_id"]
        == "12345"
    )
    assert (
        session.calls[0]["json"]["text"]
        == "Paper service started."
    )


def test_direct_email_sender_contract():
    client = SMTPClient()
    factory = SMTPFactory(client)
    sender = EmailNotificationSender(
        EmailConfig(
            host="smtp.example.test",
            port=587,
            sender="bot@example.test",
            recipients=(
                "owner@example.test",
            ),
            username="bot-user",
            password="test-password",
        ),
        smtp_factory=factory,
    )

    sender.send(
        RenderedNotification(
            subject="Action required",
            text="Review the paper order.",
        )
    )

    assert client.started_tls is True
    assert client.login_values == (
        "bot-user",
        "test-password",
    )
    assert len(client.messages) == 1
    assert (
        client.messages[0]["Subject"]
        == "Action required"
    )


def test_transport_errors_do_not_expose_secrets():
    telegram_secret = "telegram-secret"
    telegram = TelegramNotificationSender(
        TelegramConfig(
            bot_token=telegram_secret,
            chat_id="12345",
        ),
        session=FailingTelegramSession(),
    )

    with pytest.raises(
        RuntimeError,
        match="Telegram delivery failed",
    ) as telegram_error:
        telegram.send(
            RenderedNotification(
                subject="Test",
                text="Test",
            )
        )

    assert telegram_secret not in str(
        telegram_error.value
    )

    email_secret = "smtp-secret"
    client = SMTPClient(
        fail_login=True
    )
    email = EmailNotificationSender(
        EmailConfig(
            host="smtp.example.test",
            port=587,
            sender="bot@example.test",
            recipients=(
                "owner@example.test",
            ),
            username="bot-user",
            password=email_secret,
        ),
        smtp_factory=SMTPFactory(client),
    )

    with pytest.raises(
        RuntimeError,
        match="Email delivery failed",
    ) as email_error:
        email.send(
            RenderedNotification(
                subject="Test",
                text="Test",
            )
        )

    assert email_secret not in str(
        email_error.value
    )


def test_end_to_end_routing_rendering_and_delivery(
    tmp_path,
):
    repository = PaperRepository(
        tmp_path / "delivery.db"
    )
    account = repository.create_account(
        name="Delivery Contract",
        base_currency="USD",
        starting_balance="1000",
        created_at=NOW,
    )
    repository.queue_notification(
        account_id=account.account_id,
        event_type="PAPER_BUY_EXECUTED",
        reference_type="POSITION",
        reference_id="POS-1",
        channel=NotificationChannel.INTERNAL,
        payload={
            "symbol": "AAPL",
            "quantity": "0.9",
            "fill_price": "101",
            "stop_price": "95",
            "targets": ["120", "130"],
        },
        created_at=NOW,
    )

    telegram_session = TelegramSession()
    smtp_client = SMTPClient()
    service = NotificationService(
        repository,
        senders={
            NotificationChannel.TELEGRAM:
                TelegramNotificationSender(
                    TelegramConfig(
                        bot_token="test-token",
                        chat_id="12345",
                    ),
                    session=telegram_session,
                ),
            NotificationChannel.EMAIL:
                EmailNotificationSender(
                    EmailConfig(
                        host="smtp.example.test",
                        port=25,
                        sender=(
                            "bot@example.test"
                        ),
                        recipients=(
                            "owner@example.test",
                        ),
                        use_starttls=False,
                    ),
                    smtp_factory=SMTPFactory(
                        smtp_client
                    ),
                ),
        },
    )

    created = service.fan_out_routed(
        account.account_id,
        created_at=NOW,
    )
    report = service.dispatch_pending(
        account.account_id,
        attempted_at=NOW,
    )
    notifications = (
        repository.list_notifications(
            account.account_id
        )
    )
    external = [
        item
        for item in notifications
        if item.channel
        is not NotificationChannel.INTERNAL
    ]

    assert created == 2
    assert report.sent == 2
    assert report.failed == 0
    assert len(external) == 2
    assert all(
        item.status
        is NotificationStatus.SENT
        for item in external
    )
    assert (
        telegram_session.calls[0]["json"][
            "text"
        ].startswith("[PAPER BUY] AAPL")
    )
    assert "Paper BUY executed" in (
        smtp_client.messages[0]
        .get_content()
    )
