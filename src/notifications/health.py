"""Secret-safe notification configuration health checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from typing import Mapping

from src.paper import NotificationChannel

from .config import (
    load_email_config,
    load_telegram_config,
)


class ChannelHealthStatus(str, Enum):
    DISABLED = "DISABLED"
    READY = "READY"
    MISCONFIGURED = "MISCONFIGURED"


@dataclass(frozen=True, slots=True)
class ChannelHealth:
    channel: NotificationChannel
    status: ChannelHealthStatus
    reason: str


def _has_any(
    values: Mapping[str, str],
    prefixes: tuple[str, ...],
) -> bool:
    return any(
        key.startswith(prefix)
        and bool(str(value).strip())
        for key, value in values.items()
        for prefix in prefixes
    )


def notification_channel_health(
    environ: Mapping[str, str] | None = None,
) -> tuple[ChannelHealth, ...]:
    """Inspect readiness without network calls or secret values."""

    values = (
        os.environ
        if environ is None
        else environ
    )
    results: list[ChannelHealth] = []

    telegram_present = _has_any(
        values,
        ("PAPER_TELEGRAM_",),
    )

    if not telegram_present:
        results.append(
            ChannelHealth(
                channel=(
                    NotificationChannel.TELEGRAM
                ),
                status=(
                    ChannelHealthStatus.DISABLED
                ),
                reason=(
                    "Telegram environment "
                    "configuration is absent."
                ),
            )
        )
    else:
        try:
            configured = load_telegram_config(
                values
            )
        except (
            TypeError,
            ValueError,
        ):
            configured = None

        results.append(
            ChannelHealth(
                channel=(
                    NotificationChannel.TELEGRAM
                ),
                status=(
                    ChannelHealthStatus.READY
                    if configured is not None
                    else ChannelHealthStatus
                    .MISCONFIGURED
                ),
                reason=(
                    "Telegram required fields "
                    "are present."
                    if configured is not None
                    else "Telegram configuration "
                    "is incomplete or invalid."
                ),
            )
        )

    email_present = _has_any(
        values,
        (
            "PAPER_SMTP_",
            "PAPER_EMAIL_",
        ),
    )

    if not email_present:
        results.append(
            ChannelHealth(
                channel=NotificationChannel.EMAIL,
                status=(
                    ChannelHealthStatus.DISABLED
                ),
                reason=(
                    "Email environment "
                    "configuration is absent."
                ),
            )
        )
    else:
        try:
            configured = load_email_config(
                values
            )
        except (
            TypeError,
            ValueError,
        ):
            configured = None

        results.append(
            ChannelHealth(
                channel=NotificationChannel.EMAIL,
                status=(
                    ChannelHealthStatus.READY
                    if configured is not None
                    else ChannelHealthStatus
                    .MISCONFIGURED
                ),
                reason=(
                    "Email required fields "
                    "are present."
                    if configured is not None
                    else "Email configuration "
                    "is incomplete or invalid."
                ),
            )
        )

    return tuple(results)
