from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest

from src.paper import (
    EVENT_RISK_POLICY_VERSION,
    EventEvidenceStatus,
    EventRiskAction,
    EventRiskContext,
    SecurityEvent,
    SecurityEventType,
    evaluate_event_risk,
)


NOW = datetime(
    2026,
    8,
    18,
    10,
    0,
    tzinfo=timezone.utc,
)


def event(
    event_type,
    *,
    after=timedelta(days=1),
):
    return SecurityEvent(
        symbol="AAPL",
        event_type=event_type,
        effective_at=NOW + after,
        source="verified-test-feed",
        source_as_of=NOW,
    )


def decide(
    context,
    event_type,
    *,
    after=timedelta(days=1),
):
    return evaluate_event_risk(
        context=context,
        evidence_status=(
            EventEvidenceStatus.CONFIRMED
        ),
        evaluated_at=NOW,
        event=event(
            event_type,
            after=after,
        ),
    )


def test_unknown_data_blocks_new_entries():
    decision = evaluate_event_risk(
        context=EventRiskContext.NEW_ENTRY,
        evidence_status=(
            EventEvidenceStatus.UNKNOWN
        ),
        evaluated_at=NOW,
    )

    assert (
        decision.action
        is EventRiskAction.BLOCK_ENTRY
    )
    assert (
        decision.policy_version
        == EVENT_RISK_POLICY_VERSION
    )


def test_stale_data_flags_open_position_for_review():
    decision = evaluate_event_risk(
        context=(
            EventRiskContext.OPEN_POSITION
        ),
        evidence_status=(
            EventEvidenceStatus.STALE
        ),
        evaluated_at=NOW,
    )

    assert (
        decision.action
        is EventRiskAction.MANUAL_REVIEW
    )


def test_verified_clear_check_allows_entry():
    decision = evaluate_event_risk(
        context=EventRiskContext.NEW_ENTRY,
        evidence_status=(
            EventEvidenceStatus.CLEAR
        ),
        evaluated_at=NOW,
    )

    assert decision.action is EventRiskAction.ALLOW


def test_entry_is_blocked_near_earnings():
    decision = decide(
        EventRiskContext.NEW_ENTRY,
        SecurityEventType.EARNINGS,
        after=timedelta(days=5),
    )

    assert (
        decision.action
        is EventRiskAction.BLOCK_ENTRY
    )


def test_open_position_exits_before_imminent_earnings():
    decision = decide(
        EventRiskContext.OPEN_POSITION,
        SecurityEventType.EARNINGS,
        after=timedelta(hours=12),
    )

    assert (
        decision.action
        is EventRiskAction.EXIT_POSITION
    )


def test_split_requires_controlled_position_review():
    decision = decide(
        EventRiskContext.OPEN_POSITION,
        SecurityEventType.SPLIT,
        after=timedelta(days=2),
    )

    assert (
        decision.action
        is EventRiskAction.MANUAL_REVIEW
    )


def test_merger_blocks_entry_and_exits_position():
    entry = decide(
        EventRiskContext.NEW_ENTRY,
        SecurityEventType.MERGER,
        after=timedelta(days=20),
    )
    position = decide(
        EventRiskContext.OPEN_POSITION,
        SecurityEventType.MERGER,
        after=timedelta(days=20),
    )

    assert (
        entry.action
        is EventRiskAction.BLOCK_ENTRY
    )
    assert (
        position.action
        is EventRiskAction.EXIT_POSITION
    )


def test_cash_dividend_does_not_force_action():
    decision = decide(
        EventRiskContext.OPEN_POSITION,
        SecurityEventType.CASH_DIVIDEND,
    )

    assert decision.action is EventRiskAction.ALLOW


def test_past_event_requires_adjustment_evidence():
    decision = decide(
        EventRiskContext.OPEN_POSITION,
        SecurityEventType.SPLIT,
        after=-timedelta(hours=1),
    )

    assert (
        decision.action
        is EventRiskAction.MANUAL_REVIEW
    )


def test_event_requires_confirmed_evidence():
    with pytest.raises(
        ValueError,
        match="requires CONFIRMED",
    ):
        evaluate_event_risk(
            context=EventRiskContext.NEW_ENTRY,
            evidence_status=(
                EventEvidenceStatus.CLEAR
            ),
            evaluated_at=NOW,
            event=event(
                SecurityEventType.EARNINGS
            ),
        )


def test_future_dated_evidence_is_rejected():
    future_evidence = SecurityEvent(
        symbol="AAPL",
        event_type=SecurityEventType.EARNINGS,
        effective_at=NOW + timedelta(days=2),
        source="verified-test-feed",
        source_as_of=NOW + timedelta(hours=1),
    )

    with pytest.raises(
        ValueError,
        match="after evaluated_at",
    ):
        evaluate_event_risk(
            context=EventRiskContext.NEW_ENTRY,
            evidence_status=(
                EventEvidenceStatus.CONFIRMED
            ),
            evaluated_at=NOW,
            event=future_evidence,
        )
