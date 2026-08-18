"""Versioned earnings and corporate-action risk policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .models import aware_datetime


EVENT_RISK_POLICY_VERSION = "p4.6-events-v1"


class SecurityEventType(str, Enum):
    EARNINGS = "EARNINGS"
    CASH_DIVIDEND = "CASH_DIVIDEND"
    STOCK_DIVIDEND = "STOCK_DIVIDEND"
    SPLIT = "SPLIT"
    SPINOFF = "SPINOFF"
    MERGER = "MERGER"
    DELISTING = "DELISTING"


class EventEvidenceStatus(str, Enum):
    CLEAR = "CLEAR"
    CONFIRMED = "CONFIRMED"
    UNKNOWN = "UNKNOWN"
    STALE = "STALE"


class EventRiskContext(str, Enum):
    NEW_ENTRY = "NEW_ENTRY"
    OPEN_POSITION = "OPEN_POSITION"


class EventRiskAction(str, Enum):
    ALLOW = "ALLOW"
    BLOCK_ENTRY = "BLOCK_ENTRY"
    EXIT_POSITION = "EXIT_POSITION"
    MANUAL_REVIEW = "MANUAL_REVIEW"


@dataclass(frozen=True, slots=True)
class SecurityEvent:
    """Provider-neutral confirmed security event."""

    symbol: str
    event_type: SecurityEventType
    effective_at: datetime
    source: str
    source_as_of: datetime

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        source = str(self.source).strip()

        if not symbol:
            raise ValueError("symbol is required.")

        if not source:
            raise ValueError("source is required.")

        if not isinstance(
            self.event_type,
            SecurityEventType,
        ):
            raise ValueError(
                "event_type must be a "
                "SecurityEventType."
            )

        effective_at = aware_datetime(
            "effective_at",
            self.effective_at,
        )
        source_as_of = aware_datetime(
            "source_as_of",
            self.source_as_of,
        )

        object.__setattr__(
            self,
            "symbol",
            symbol,
        )
        object.__setattr__(
            self,
            "source",
            source,
        )
        object.__setattr__(
            self,
            "effective_at",
            effective_at,
        )
        object.__setattr__(
            self,
            "source_as_of",
            source_as_of,
        )


@dataclass(frozen=True, slots=True)
class EventRiskDecision:
    policy_version: str
    context: EventRiskContext
    action: EventRiskAction
    reason: str
    event_type: SecurityEventType | None
    effective_at: datetime | None


def evaluate_event_risk(
    *,
    context: EventRiskContext,
    evidence_status: EventEvidenceStatus,
    evaluated_at: datetime,
    event: SecurityEvent | None = None,
) -> EventRiskDecision:
    """Return the conservative P4.6 event-risk decision."""

    if not isinstance(context, EventRiskContext):
        raise ValueError(
            "context must be an "
            "EventRiskContext."
        )

    if not isinstance(
        evidence_status,
        EventEvidenceStatus,
    ):
        raise ValueError(
            "evidence_status must be an "
            "EventEvidenceStatus."
        )

    at = aware_datetime(
        "evaluated_at",
        evaluated_at,
    )

    if (
        event is not None
        and event.source_as_of > at
    ):
        raise ValueError(
            "Event evidence cannot be dated "
            "after evaluated_at."
        )

    if event is None:
        if (
            evidence_status
            is EventEvidenceStatus.CONFIRMED
        ):
            raise ValueError(
                "CONFIRMED evidence requires "
                "an event."
            )

        if evidence_status in {
            EventEvidenceStatus.UNKNOWN,
            EventEvidenceStatus.STALE,
        }:
            action = (
                EventRiskAction.BLOCK_ENTRY
                if context
                is EventRiskContext.NEW_ENTRY
                else EventRiskAction
                .MANUAL_REVIEW
            )

            return EventRiskDecision(
                policy_version=(
                    EVENT_RISK_POLICY_VERSION
                ),
                context=context,
                action=action,
                reason=(
                    "Event data is "
                    + evidence_status.value.lower()
                    + "; fail-closed review required."
                ),
                event_type=None,
                effective_at=None,
            )

        return EventRiskDecision(
            policy_version=(
                EVENT_RISK_POLICY_VERSION
            ),
            context=context,
            action=EventRiskAction.ALLOW,
            reason=(
                "Verified event check is clear."
            ),
            event_type=None,
            effective_at=None,
        )

    if (
        evidence_status
        is not EventEvidenceStatus.CONFIRMED
    ):
        raise ValueError(
            "An event requires CONFIRMED "
            "evidence."
        )

    until_event = event.effective_at - at

    if until_event < timedelta(0):
        return EventRiskDecision(
            policy_version=(
                EVENT_RISK_POLICY_VERSION
            ),
            context=context,
            action=EventRiskAction.MANUAL_REVIEW,
            reason=(
                "Event effective time has passed; "
                "adjustment evidence is required."
            ),
            event_type=event.event_type,
            effective_at=event.effective_at,
        )

    action = EventRiskAction.ALLOW
    reason = "Confirmed event is outside its risk window."

    if (
        event.event_type
        is SecurityEventType.EARNINGS
    ):
        if (
            context
            is EventRiskContext.NEW_ENTRY
            and until_event <= timedelta(days=5)
        ):
            action = EventRiskAction.BLOCK_ENTRY
            reason = (
                "New entries are blocked within "
                "five days of earnings."
            )
        elif (
            context
            is EventRiskContext.OPEN_POSITION
            and until_event <= timedelta(days=1)
        ):
            action = (
                EventRiskAction.EXIT_POSITION
            )
            reason = (
                "Open position must exit before "
                "imminent earnings."
            )

    elif event.event_type in {
        SecurityEventType.SPLIT,
        SecurityEventType.STOCK_DIVIDEND,
        SecurityEventType.SPINOFF,
    } and until_event <= timedelta(days=5):
        action = (
            EventRiskAction.BLOCK_ENTRY
            if context
            is EventRiskContext.NEW_ENTRY
            else EventRiskAction.MANUAL_REVIEW
        )
        reason = (
            "Price or quantity adjustment event "
            "requires controlled processing."
        )

    elif event.event_type in {
        SecurityEventType.MERGER,
        SecurityEventType.DELISTING,
    } and until_event <= timedelta(days=30):
        action = (
            EventRiskAction.BLOCK_ENTRY
            if context
            is EventRiskContext.NEW_ENTRY
            else EventRiskAction.EXIT_POSITION
        )
        reason = (
            "Structural security event is inside "
            "the thirty-day risk window."
        )

    return EventRiskDecision(
        policy_version=(
            EVENT_RISK_POLICY_VERSION
        ),
        context=context,
        action=action,
        reason=reason,
        event_type=event.event_type,
        effective_at=event.effective_at,
    )
