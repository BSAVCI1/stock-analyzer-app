"""Bounded notification retry policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import (
    datetime,
    timedelta,
)
from enum import Enum

from src.paper import (
    NotificationRecord,
    NotificationStatus,
)
from src.paper.models import aware_datetime


NOTIFICATION_RETRY_POLICY_VERSION = (
    "p4.7-retry-v1"
)


class RetryEligibility(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    BACKOFF = "BACKOFF"
    EXHAUSTED = "EXHAUSTED"
    NOT_FAILED = "NOT_FAILED"


@dataclass(frozen=True, slots=True)
class NotificationRetryPolicy:
    maximum_attempts: int = 3
    base_delay_seconds: int = 60
    maximum_delay_seconds: int = 3600

    def __post_init__(self) -> None:
        for name in (
            "maximum_attempts",
            "base_delay_seconds",
            "maximum_delay_seconds",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a "
                    "positive integer."
                )

        if (
            self.base_delay_seconds
            > self.maximum_delay_seconds
        ):
            raise ValueError(
                "base_delay_seconds cannot "
                "exceed maximum_delay_seconds."
            )


@dataclass(frozen=True, slots=True)
class RetryDecision:
    policy_version: str
    eligibility: RetryEligibility
    attempt_count: int
    maximum_attempts: int
    next_attempt_at: datetime | None
    reason: str


def evaluate_notification_retry(
    notification: NotificationRecord,
    *,
    evaluated_at: datetime,
    policy: NotificationRetryPolicy | None = None,
) -> RetryDecision:
    """Evaluate one persisted notification retry."""

    if not isinstance(
        notification,
        NotificationRecord,
    ):
        raise ValueError(
            "notification must be a "
            "NotificationRecord."
        )

    at = aware_datetime(
        "evaluated_at",
        evaluated_at,
    )
    resolved = (
        policy
        or NotificationRetryPolicy()
    )

    if (
        notification.status
        is not NotificationStatus.FAILED
    ):
        return RetryDecision(
            policy_version=(
                NOTIFICATION_RETRY_POLICY_VERSION
            ),
            eligibility=(
                RetryEligibility.NOT_FAILED
            ),
            attempt_count=(
                notification.attempt_count
            ),
            maximum_attempts=(
                resolved.maximum_attempts
            ),
            next_attempt_at=None,
            reason=(
                "Only failed notifications "
                "are retry candidates."
            ),
        )

    if (
        notification.attempt_count
        >= resolved.maximum_attempts
    ):
        return RetryDecision(
            policy_version=(
                NOTIFICATION_RETRY_POLICY_VERSION
            ),
            eligibility=(
                RetryEligibility.EXHAUSTED
            ),
            attempt_count=(
                notification.attempt_count
            ),
            maximum_attempts=(
                resolved.maximum_attempts
            ),
            next_attempt_at=None,
            reason=(
                "Maximum delivery attempts "
                "have been reached."
            ),
        )

    if notification.last_attempt_at is None:
        return RetryDecision(
            policy_version=(
                NOTIFICATION_RETRY_POLICY_VERSION
            ),
            eligibility=(
                RetryEligibility.ELIGIBLE
            ),
            attempt_count=(
                notification.attempt_count
            ),
            maximum_attempts=(
                resolved.maximum_attempts
            ),
            next_attempt_at=at,
            reason=(
                "Failed notification has no "
                "recorded attempt time."
            ),
        )

    delay_seconds = min(
        resolved.maximum_delay_seconds,
        resolved.base_delay_seconds
        * 2 ** max(
            notification.attempt_count - 1,
            0,
        ),
    )
    next_attempt = (
        notification.last_attempt_at
        + timedelta(seconds=delay_seconds)
    )

    eligibility = (
        RetryEligibility.ELIGIBLE
        if at >= next_attempt
        else RetryEligibility.BACKOFF
    )

    return RetryDecision(
        policy_version=(
            NOTIFICATION_RETRY_POLICY_VERSION
        ),
        eligibility=eligibility,
        attempt_count=(
            notification.attempt_count
        ),
        maximum_attempts=(
            resolved.maximum_attempts
        ),
        next_attempt_at=next_attempt,
        reason=(
            "Retry delay has elapsed."
            if eligibility
            is RetryEligibility.ELIGIBLE
            else "Retry is waiting for "
            "exponential backoff."
        ),
    )
