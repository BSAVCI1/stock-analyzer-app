"""Environment-based notification configuration."""

from __future__ import annotations

import os
from typing import Mapping

from .models import (
    EmailConfig,
    TelegramConfig,
)


def _boolean(
    value: str | None,
    *,
    default: bool,
) -> bool:
    if value is None:
        return default

    normalised = value.strip().lower()

    if normalised in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if normalised in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    raise ValueError(
        f"Invalid boolean value: {value}."
    )


def load_telegram_config(
    environ: Mapping[
        str,
        str,
    ] | None = None,
) -> TelegramConfig | None:
    values = environ or os.environ

    token = values.get(
        "PAPER_TELEGRAM_BOT_TOKEN"
    )

    chat_id = values.get(
        "PAPER_TELEGRAM_CHAT_ID"
    )

    if not token and not chat_id:
        return None

    if not token or not chat_id:
        raise ValueError(
            "Both PAPER_TELEGRAM_BOT_TOKEN "
            "and PAPER_TELEGRAM_CHAT_ID "
            "are required."
        )

    return TelegramConfig(
        bot_token=token,
        chat_id=chat_id,
        timeout_seconds=int(
            values.get(
                "PAPER_TELEGRAM_TIMEOUT",
                "15",
            )
        ),
    )


def load_email_config(
    environ: Mapping[
        str,
        str,
    ] | None = None,
) -> EmailConfig | None:
    values = environ or os.environ

    host = values.get(
        "PAPER_SMTP_HOST"
    )

    recipients = values.get(
        "PAPER_EMAIL_TO"
    )

    if not host and not recipients:
        return None

    required = {
        "PAPER_SMTP_HOST": host,
        "PAPER_EMAIL_FROM":
        values.get("PAPER_EMAIL_FROM"),
        "PAPER_EMAIL_TO": recipients,
    }

    missing = [
        name
        for name, value
        in required.items()
        if not value
    ]

    if missing:
        raise ValueError(
            "Missing email configuration: "
            + ", ".join(missing)
            + "."
        )

    return EmailConfig(
        host=host,
        port=int(
            values.get(
                "PAPER_SMTP_PORT",
                "587",
            )
        ),
        sender=values[
            "PAPER_EMAIL_FROM"
        ],
        recipients=tuple(
            item.strip()
            for item
            in recipients.split(",")
            if item.strip()
        ),
        username=values.get(
            "PAPER_SMTP_USERNAME"
        ),
        password=values.get(
            "PAPER_SMTP_PASSWORD"
        ),
        use_starttls=_boolean(
            values.get(
                "PAPER_SMTP_STARTTLS"
            ),
            default=True,
        ),
        use_ssl=_boolean(
            values.get(
                "PAPER_SMTP_SSL"
            ),
            default=False,
        ),
        timeout_seconds=int(
            values.get(
                "PAPER_SMTP_TIMEOUT",
                "20",
            )
        ),
    )
