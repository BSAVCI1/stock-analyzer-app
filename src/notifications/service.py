"""Notification fan-out and delivery orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from src.paper import (
    NotificationChannel,
    NotificationStatus,
    PaperRepository,
)

from .models import DispatchReport
from .retry import (
    NOTIFICATION_RETRY_POLICY_VERSION,
    NotificationRetryPolicy,
    RetryEligibility,
    evaluate_notification_retry,
)
from .routing import route_notification_event
from .senders import NotificationSender
from .templates import render_notification


class NotificationService:
    def __init__(
        self,
        repository: PaperRepository,
        *,
        senders: Mapping[
            NotificationChannel,
            NotificationSender,
        ],
        retry_policy: (
            NotificationRetryPolicy | None
        ) = None,
    ) -> None:
        self.repository = repository
        self.senders = dict(senders)
        self.retry_policy = (
            retry_policy
            or NotificationRetryPolicy()
        )

    def fan_out_internal(
        self,
        account_id: str,
        *,
        channels: tuple[
            NotificationChannel,
            ...,
        ],
        created_at: datetime | None = None,
    ) -> int:
        at = (
            created_at
            or datetime.now(timezone.utc)
        )

        external_channels = tuple(
            channel
            for channel in channels
            if channel
            is not NotificationChannel.INTERNAL
        )

        created_count = 0

        notifications = (
            self.repository
            .list_pending_notifications(
                account_id,
                channels=(
                    NotificationChannel.INTERNAL,
                ),
            )
        )

        for notification in notifications:
            for channel in external_channels:
                external = (
                    self.repository
                    .queue_notification(
                        account_id=(
                            notification
                            .account_id
                        ),
                        event_type=(
                            notification
                            .event_type
                        ),
                        reference_type=(
                            notification
                            .reference_type
                        ),
                        reference_id=(
                            notification
                            .reference_id
                        ),
                        channel=channel,
                        payload=(
                            notification.payload
                        ),
                        created_at=at,
                    )
                )

                if (
                    external.status
                    is NotificationStatus
                    .PENDING
                ):
                    created_count += 1

            self.repository.mark_notification_sent(
                notification.notification_id,
                sent_at=at,
                delivery_metadata={
                    "fanout_channels": [
                        channel.value
                        for channel
                        in external_channels
                    ],
                },
            )

        return created_count

    def fan_out_routed(
        self,
        account_id: str,
        *,
        created_at: datetime | None = None,
    ) -> int:
        """Fan out internal events using P4.7 routing."""

        at = (
            created_at
            or datetime.now(timezone.utc)
        )
        created_count = 0
        notifications = (
            self.repository
            .list_pending_notifications(
                account_id,
                channels=(
                    NotificationChannel.INTERNAL,
                ),
            )
        )

        for notification in notifications:
            route = route_notification_event(
                notification.event_type
            )

            for channel in route.channels:
                external = (
                    self.repository
                    .queue_notification(
                        account_id=(
                            notification.account_id
                        ),
                        event_type=(
                            notification.event_type
                        ),
                        reference_type=(
                            notification.reference_type
                        ),
                        reference_id=(
                            notification.reference_id
                        ),
                        channel=channel,
                        payload=notification.payload,
                        created_at=at,
                    )
                )

                if (
                    external.status
                    is NotificationStatus.PENDING
                ):
                    created_count += 1

            self.repository.mark_notification_sent(
                notification.notification_id,
                sent_at=at,
                delivery_metadata={
                    "routing_policy_version":
                        route.policy_version,
                    "severity":
                        route.severity.value,
                    "purpose":
                        route.purpose.value,
                    "fanout_channels": [
                        channel.value
                        for channel
                        in route.channels
                    ],
                },
            )

        return created_count

    def dispatch_pending(
        self,
        account_id: str,
        *,
        include_failed: bool = False,
        attempted_at: datetime | None = None,
    ) -> DispatchReport:
        at = (
            attempted_at
            or datetime.now(timezone.utc)
        )

        notifications = (
            self.repository
            .list_pending_notifications(
                account_id,
                include_failed=(
                    include_failed
                ),
            )
        )

        sent_ids: list[str] = []
        failed_ids: list[str] = []
        skipped = 0

        for notification in notifications:
            if (
                notification.status
                is NotificationStatus.FAILED
            ):
                retry = (
                    evaluate_notification_retry(
                        notification,
                        evaluated_at=at,
                        policy=self.retry_policy,
                    )
                )

                if (
                    retry.eligibility
                    is not RetryEligibility.ELIGIBLE
                ):
                    skipped += 1
                    continue

            sender = self.senders.get(
                notification.channel
            )

            if sender is None:
                self.repository.mark_notification_failed(
                    notification
                    .notification_id,
                    attempted_at=at,
                    error_message=(
                        "No sender is configured "
                        f"for "
                        f"{notification.channel.value}."
                    ),
                    delivery_metadata={
                        "retry_policy_version": (
                            NOTIFICATION_RETRY_POLICY_VERSION
                        ),
                        "attempt_number": (
                            notification.attempt_count
                            + 1
                        ),
                    },
                )

                failed_ids.append(
                    notification
                    .notification_id
                )

                continue

            try:
                rendered = (
                    render_notification(
                        notification
                    )
                )

                result = sender.send(
                    rendered
                )

                self.repository.mark_notification_sent(
                    notification
                    .notification_id,
                    sent_at=at,
                    provider_message_id=(
                        result
                        .provider_message_id
                    ),
                    delivery_metadata={
                        **dict(result.metadata),
                        "retry_policy_version": (
                            NOTIFICATION_RETRY_POLICY_VERSION
                        ),
                        "attempt_number": (
                            notification.attempt_count
                            + 1
                        ),
                    },
                )

                sent_ids.append(
                    notification
                    .notification_id
                )

            except Exception as exc:
                self.repository.mark_notification_failed(
                    notification
                    .notification_id,
                    attempted_at=at,
                    error_message=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                    delivery_metadata={
                        "channel":
                        notification
                        .channel.value,
                        "retry_policy_version": (
                            NOTIFICATION_RETRY_POLICY_VERSION
                        ),
                        "attempt_number": (
                            notification.attempt_count
                            + 1
                        ),
                    },
                )

                failed_ids.append(
                    notification
                    .notification_id
                )

        return DispatchReport(
            processed=len(notifications),
            sent=len(sent_ids),
            failed=len(failed_ids),
            skipped=skipped,
            sent_notification_ids=tuple(
                sent_ids
            ),
            failed_notification_ids=tuple(
                failed_ids
            ),
        )
