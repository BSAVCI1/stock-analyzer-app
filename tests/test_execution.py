from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.analysis import Signal
from src.backtest import (
    ExecutionStatus,
    FillRule,
    LifecycleEventType,
    OrderRecord,
    PositionSide,
    SignalRecord,
    execute_next_session,
)


T0 = datetime(
    2026,
    7,
    31,
    20,
    0,
    tzinfo=timezone.utc,
)

EXPIRY = T0 + timedelta(days=5)


def make_signal(
    *,
    side: PositionSide = PositionSide.LONG,
    expires_at: datetime = EXPIRY,
) -> SignalRecord:
    signal = (
        Signal.BUY
        if side is PositionSide.LONG
        else Signal.SELL
    )

    score = (
        80.0
        if side is PositionSide.LONG
        else -80.0
    )

    return SignalRecord(
        signal_id="SIG-001",
        symbol="TEST",
        strategy="final_recommendation",
        signal=signal,
        generated_at=T0,
        expires_at=expires_at,
        score=score,
        confidence=0.90,
    )


def make_order(
    *,
    side: PositionSide = PositionSide.LONG,
    expires_at: datetime = EXPIRY,
) -> OrderRecord:
    if side is PositionSide.LONG:
        stop_price = 95.0
        targets = (110.0, 120.0)
    else:
        stop_price = 105.0
        targets = (90.0, 80.0)

    return OrderRecord(
        order_id="ORD-001",
        signal_id="SIG-001",
        symbol="TEST",
        side=side,
        created_at=T0,
        expires_at=expires_at,
        entry_low=99.0,
        entry_high=101.0,
        stop_price=stop_price,
        targets=targets,
        quantity=10.0,
    )


def make_history(
    rows: list[
        tuple[
            datetime,
            float,
            float,
            float,
            float,
        ]
    ],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [row[1] for row in rows],
            "High": [row[2] for row in rows],
            "Low": [row[3] for row in rows],
            "Close": [row[4] for row in rows],
        },
        index=pd.DatetimeIndex(
            [row[0] for row in rows]
        ),
    )


def test_next_open_never_uses_same_bar() -> None:
    same_bar = T0 - timedelta(hours=20)
    next_session = T0 + timedelta(hours=4)

    history = make_history(
        [
            (same_bar, 100.0, 102.0, 98.0, 100.0),
            (next_session, 103.0, 104.0, 102.0, 103.0),
        ]
    )

    result = execute_next_session(
        make_signal(),
        make_order(),
        history,
        fill_rule=FillRule.NEXT_OPEN,
    )

    assert result.status is ExecutionStatus.NOT_FILLED
    assert result.fill is None
    assert result.evaluated_sessions == 1
    assert result.decision_at == next_session


def test_next_open_fills_at_first_eligible_open() -> None:
    next_session = T0 + timedelta(hours=4)

    history = make_history(
        [
            (
                T0 - timedelta(hours=20),
                103.0,
                104.0,
                98.0,
                100.0,
            ),
            (
                next_session,
                100.25,
                101.0,
                99.5,
                100.5,
            ),
        ]
    )

    result = execute_next_session(
        make_signal(),
        make_order(),
        history,
        fill_rule=FillRule.NEXT_OPEN,
    )

    assert result.status is ExecutionStatus.FILLED
    assert result.fill is not None
    assert result.fill.fill_price == 100.25
    assert result.fill.filled_at == next_session
    assert result.position is not None


def test_next_open_does_not_scan_later_sessions() -> None:
    first_session = T0 + timedelta(hours=4)
    second_session = T0 + timedelta(days=1, hours=4)

    history = make_history(
        [
            (
                first_session,
                103.0,
                104.0,
                102.0,
                103.5,
            ),
            (
                second_session,
                100.0,
                101.0,
                99.0,
                100.5,
            ),
        ]
    )

    result = execute_next_session(
        make_signal(),
        make_order(),
        history,
        fill_rule=FillRule.NEXT_OPEN,
    )

    assert result.status is ExecutionStatus.NOT_FILLED
    assert result.evaluated_sessions == 1
    assert result.decision_at == first_session


def test_limit_rule_skips_same_bar_and_fills_later() -> None:
    same_bar = T0 - timedelta(hours=20)
    first_session = T0 + timedelta(hours=4)
    second_session = T0 + timedelta(days=1, hours=4)

    history = make_history(
        [
            (same_bar, 100.0, 102.0, 98.0, 100.0),
            (
                first_session,
                103.0,
                104.0,
                102.0,
                103.0,
            ),
            (
                second_session,
                100.0,
                101.0,
                99.0,
                100.5,
            ),
        ]
    )

    result = execute_next_session(
        make_signal(),
        make_order(),
        history,
        fill_rule=FillRule.LIMIT,
    )

    assert result.status is ExecutionStatus.FILLED
    assert result.fill is not None
    assert result.fill.filled_at == second_session
    assert result.fill.fill_price == 100.0
    assert result.evaluated_sessions == 2


def test_limit_gap_below_zone_fills_at_lower_boundary() -> None:
    next_session = T0 + timedelta(hours=4)

    history = make_history(
        [
            (
                next_session,
                97.0,
                99.5,
                96.0,
                99.0,
            ),
        ]
    )

    result = execute_next_session(
        make_signal(),
        make_order(),
        history,
        fill_rule=FillRule.LIMIT,
    )

    assert result.status is ExecutionStatus.FILLED
    assert result.fill is not None
    assert result.fill.fill_price == 99.0


def test_limit_gap_above_zone_fills_at_upper_boundary() -> None:
    next_session = T0 + timedelta(hours=4)

    history = make_history(
        [
            (
                next_session,
                103.0,
                104.0,
                100.5,
                101.5,
            ),
        ]
    )

    result = execute_next_session(
        make_signal(),
        make_order(),
        history,
        fill_rule=FillRule.LIMIT,
    )

    assert result.status is ExecutionStatus.FILLED
    assert result.fill is not None
    assert result.fill.fill_price == 101.0


def test_limit_rule_supports_short_positions() -> None:
    next_session = T0 + timedelta(hours=4)

    history = make_history(
        [
            (
                next_session,
                103.0,
                104.0,
                100.5,
                101.5,
            ),
        ]
    )

    result = execute_next_session(
        make_signal(side=PositionSide.SHORT),
        make_order(side=PositionSide.SHORT),
        history,
        fill_rule=FillRule.LIMIT,
    )

    assert result.status is ExecutionStatus.FILLED
    assert result.fill is not None
    assert result.fill.fill_price == 101.0
    assert result.position is not None
    assert result.position.side is PositionSide.SHORT


def test_order_never_fills_after_expiry() -> None:
    expiry = T0 + timedelta(days=1)
    post_expiry = T0 + timedelta(days=2)

    history = make_history(
        [
            (
                post_expiry,
                100.0,
                101.0,
                99.0,
                100.5,
            ),
        ]
    )

    result = execute_next_session(
        make_signal(expires_at=expiry),
        make_order(expires_at=expiry),
        history,
        fill_rule=FillRule.LIMIT,
    )

    assert result.status is ExecutionStatus.EXPIRED
    assert result.fill is None
    assert result.position is None

    assert any(
        event.event_type
        is LifecycleEventType.ORDER_EXPIRED
        for event in result.lifecycle.events
    )


def test_order_remains_pending_when_history_ends_early() -> None:
    next_session = T0 + timedelta(hours=4)

    history = make_history(
        [
            (
                next_session,
                103.0,
                104.0,
                102.0,
                103.5,
            ),
        ]
    )

    result = execute_next_session(
        make_signal(),
        make_order(),
        history,
        fill_rule=FillRule.LIMIT,
    )

    assert result.status is ExecutionStatus.PENDING
    assert result.fill is None
    assert result.evaluated_sessions == 1


def test_filled_result_creates_linked_lifecycle() -> None:
    next_session = T0 + timedelta(hours=4)

    history = make_history(
        [
            (
                next_session,
                100.0,
                101.0,
                99.0,
                100.5,
            ),
        ]
    )

    result = execute_next_session(
        make_signal(),
        make_order(),
        history,
        fill_rule=FillRule.LIMIT,
    )

    lifecycle = result.lifecycle

    assert lifecycle.is_filled is True
    assert lifecycle.is_open is True
    assert lifecycle.is_closed is False

    assert lifecycle.fill is not None
    assert lifecycle.position is not None

    assert (
        lifecycle.position.fill_id
        == lifecycle.fill.fill_id
    )

    event_types = {
        event.event_type
        for event in lifecycle.events
    }

    assert event_types == {
        LifecycleEventType.ORDER_FILLED,
        LifecycleEventType.POSITION_OPENED,
    }


def test_execution_ids_are_deterministic() -> None:
    next_session = T0 + timedelta(hours=4)

    history = make_history(
        [
            (
                next_session,
                100.0,
                101.0,
                99.0,
                100.5,
            ),
        ]
    )

    first = execute_next_session(
        make_signal(),
        make_order(),
        history,
    )

    second = execute_next_session(
        make_signal(),
        make_order(),
        history,
    )

    assert first == second
    assert first.fill is not None
    assert first.position is not None
    assert first.fill.fill_id == "ORD-001:FILL"
    assert (
        first.position.position_id
        == "ORD-001:POSITION"
    )


def test_invalid_ohlc_geometry_is_rejected() -> None:
    next_session = T0 + timedelta(hours=4)

    history = make_history(
        [
            (
                next_session,
                100.0,
                99.0,
                98.0,
                100.0,
            ),
        ]
    )

    with pytest.raises(ValueError):
        execute_next_session(
            make_signal(),
            make_order(),
            history,
        )
