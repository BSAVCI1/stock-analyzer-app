"""Telegram, email and internal notification senders."""

from __future__ import annotations

from email.message import EmailMessage
import smtplib
from typing import Protocol

import requests

from .models import (
    DeliveryResult,
    EmailConfig,
    RenderedNotification,
    TelegramConfig,
)


class NotificationSender(Protocol):
    def send(
        self,
        notification: RenderedNotification,
    ) -> DeliveryResult:
        ...


class InternalNotificationSender:
    """Persisted in-app notification delivery."""

    def send(
        self,
        notification: RenderedNotification,
    ) -> DeliveryResult:
        return DeliveryResult(
            metadata={
                "delivery": "internal",
            }
        )


class TelegramNotificationSender:
    def __init__(
        self,
        config: TelegramConfig,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.session = (
            session or requests.Session()
        )

    def send(
        self,
        notification: RenderedNotification,
    ) -> DeliveryResult:
        text = notification.text.strip()

        if len(text) > 4096:
            text = text[:4093] + "..."

        try:
            response = self.session.post(
                (
                    "https://api.telegram.org/"
                    f"bot{self.config.bot_token}/"
                    "sendMessage"
                ),
                json={
                    "chat_id":
                    self.config.chat_id,
                    "text": text,
                    "disable_web_page_preview":
                    True,
                },
                timeout=(
                    self.config.timeout_seconds
                ),
            )

            response.raise_for_status()

            payload = response.json()

            if not payload.get("ok"):
                raise RuntimeError(
                    "Provider rejected request."
                )

            result = (
                payload.get("result") or {}
            )
            message_id = result.get(
                "message_id"
            )
        except Exception as exc:
            raise RuntimeError(
                "Telegram delivery failed."
            ) from exc

        return DeliveryResult(
            provider_message_id=(
                str(message_id)
                if message_id is not None
                else None
            ),
            metadata={
                "provider": "telegram",
                "chat_id":
                self.config.chat_id,
            },
        )


class EmailNotificationSender:
    def __init__(
        self,
        config: EmailConfig,
        *,
        smtp_factory=None,
        smtp_ssl_factory=None,
    ) -> None:
        self.config = config

        self.smtp_factory = (
            smtp_factory
            or smtplib.SMTP
        )

        self.smtp_ssl_factory = (
            smtp_ssl_factory
            or smtplib.SMTP_SSL
        )

    def send(
        self,
        notification: RenderedNotification,
    ) -> DeliveryResult:
        message = EmailMessage()

        message["Subject"] = (
            notification.subject
        )

        message["From"] = (
            self.config.sender
        )

        message["To"] = ", ".join(
            self.config.recipients
        )

        message.set_content(
            notification.text
        )

        factory = (
            self.smtp_ssl_factory
            if self.config.use_ssl
            else self.smtp_factory
        )

        try:
            with factory(
                self.config.host,
                self.config.port,
                timeout=(
                    self.config
                    .timeout_seconds
                ),
            ) as client:
                if self.config.use_starttls:
                    client.starttls()

                if self.config.username:
                    client.login(
                        self.config.username,
                        self.config.password or "",
                    )

                client.send_message(message)
        except Exception as exc:
            raise RuntimeError(
                "Email delivery failed."
            ) from exc

        return DeliveryResult(
            metadata={
                "provider": "smtp",
                "host": self.config.host,
                "recipient_count": len(
                    self.config.recipients
                ),
            }
        )
