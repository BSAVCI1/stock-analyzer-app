"""Notification-delivery domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class RenderedNotification:
    subject: str
    text: str


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    provider_message_id: str | None = None
    metadata: Mapping[str, object] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class DispatchReport:
    processed: int
    sent: int
    failed: int
    skipped: int

    sent_notification_ids: tuple[str, ...]
    failed_notification_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    bot_token: str
    chat_id: str
    timeout_seconds: int = 15

    def __post_init__(self) -> None:
        if not str(self.bot_token).strip():
            raise ValueError(
                "Telegram bot token is required."
            )

        if not str(self.chat_id).strip():
            raise ValueError(
                "Telegram chat ID is required."
            )

        if (
            isinstance(
                self.timeout_seconds,
                bool,
            )
            or self.timeout_seconds < 1
        ):
            raise ValueError(
                "Telegram timeout must be "
                "positive."
            )


@dataclass(frozen=True, slots=True)
class EmailConfig:
    host: str
    port: int

    sender: str
    recipients: tuple[str, ...]

    username: str | None = None
    password: str | None = None

    use_starttls: bool = True
    use_ssl: bool = False

    timeout_seconds: int = 20

    def __post_init__(self) -> None:
        if not str(self.host).strip():
            raise ValueError(
                "SMTP host is required."
            )

        if (
            isinstance(self.port, bool)
            or self.port < 1
        ):
            raise ValueError(
                "SMTP port must be positive."
            )

        if not str(self.sender).strip():
            raise ValueError(
                "Email sender is required."
            )

        recipients = tuple(
            str(value).strip()
            for value in self.recipients
            if str(value).strip()
        )

        if not recipients:
            raise ValueError(
                "At least one email recipient "
                "is required."
            )

        if self.use_starttls and self.use_ssl:
            raise ValueError(
                "SMTP STARTTLS and SSL cannot "
                "both be enabled."
            )

        object.__setattr__(
            self,
            "recipients",
            recipients,
        )
