"""Versioned notification channel-routing policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.paper import NotificationChannel


NOTIFICATION_ROUTING_POLICY_VERSION = (
    "p4.7-routing-v1"
)


class NotificationSeverity(str, Enum):
    INFO = "INFO"
    ACTION = "ACTION"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class NotificationPurpose(str, Enum):
    CONCISE_STATUS = "CONCISE_STATUS"
    ACTION_TICKET = "ACTION_TICKET"
    REPORT = "REPORT"
    DETAILED_FAILURE = "DETAILED_FAILURE"


@dataclass(frozen=True, slots=True)
class NotificationRoute:
    policy_version: str
    severity: NotificationSeverity
    purpose: NotificationPurpose
    channels: tuple[
        NotificationChannel,
        ...,
    ]


def route_notification_event(
    event_type: str,
) -> NotificationRoute:
    """Return deterministic external channels for an event."""

    event = str(event_type).strip().upper()

    if not event:
        raise ValueError(
            "event_type is required."
        )

    failure_tokens = (
        "FAIL",
        "ERROR",
        "REJECTED",
        "CIRCUIT_BREAKER",
        "KILL_SWITCH",
    )
    report_tokens = (
        "DAILY_REPORT",
        "WEEKLY_REPORT",
        "PERFORMANCE_REPORT",
        "RECONCILIATION_REPORT",
    )
    action_tokens = (
        "OPPORTUNITY",
        "ORDER_CANDIDATE",
        "PAPER_BUY_EXECUTED",
        "PAPER_SELL_EXECUTED",
        "POSITION_EXIT",
        "ENTRY_BLOCKED",
    )
    status_tokens = (
        "STARTUP",
        "HEALTH",
        "HEARTBEAT",
        "SCHEDULER",
    )

    if any(
        token in event
        for token in failure_tokens
    ):
        return NotificationRoute(
            policy_version=(
                NOTIFICATION_ROUTING_POLICY_VERSION
            ),
            severity=(
                NotificationSeverity.CRITICAL
            ),
            purpose=(
                NotificationPurpose
                .DETAILED_FAILURE
            ),
            channels=(
                NotificationChannel.TELEGRAM,
                NotificationChannel.EMAIL,
            ),
        )

    if any(
        token in event
        for token in report_tokens
    ):
        return NotificationRoute(
            policy_version=(
                NOTIFICATION_ROUTING_POLICY_VERSION
            ),
            severity=(
                NotificationSeverity.INFO
            ),
            purpose=(
                NotificationPurpose.REPORT
            ),
            channels=(
                NotificationChannel.EMAIL,
            ),
        )

    if any(
        token in event
        for token in action_tokens
    ):
        return NotificationRoute(
            policy_version=(
                NOTIFICATION_ROUTING_POLICY_VERSION
            ),
            severity=(
                NotificationSeverity.ACTION
            ),
            purpose=(
                NotificationPurpose
                .ACTION_TICKET
            ),
            channels=(
                NotificationChannel.TELEGRAM,
                NotificationChannel.EMAIL,
            ),
        )

    if any(
        token in event
        for token in status_tokens
    ):
        return NotificationRoute(
            policy_version=(
                NOTIFICATION_ROUTING_POLICY_VERSION
            ),
            severity=(
                NotificationSeverity.INFO
            ),
            purpose=(
                NotificationPurpose
                .CONCISE_STATUS
            ),
            channels=(
                NotificationChannel.TELEGRAM,
            ),
        )

    # Unknown events remain auditable without
    # creating noisy Telegram alerts.
    return NotificationRoute(
        policy_version=(
            NOTIFICATION_ROUTING_POLICY_VERSION
        ),
        severity=NotificationSeverity.WARNING,
        purpose=(
            NotificationPurpose
            .DETAILED_FAILURE
        ),
        channels=(
            NotificationChannel.EMAIL,
        ),
    )
