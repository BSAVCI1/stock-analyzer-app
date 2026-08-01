from datetime import datetime, timedelta, timezone

import pytest

from src.analysis import (
    Evidence,
    EvidenceDirection,
    PaperOrder,
    Signal,
    StrategyResult,
)
from src.backtest import (
    BacktestLifecycle,
    ClosedTradeRecord,
    ExitReason,
    FillRecord,
    LifecycleEvent,
    LifecycleEventType,
    OrderRecord,
    PositionRecord,
    PositionSide,
    SignalRecord,
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
FILL_TIME = T0 + timedelta(days=1)
CLOSE_TIME = T0 + timedelta(days=2)


def make_strategy_result(
    signal: Signal = Signal.BUY,
) -> StrategyResult:
    score = 80.0 if signal is Signal.BUY else -80.0
    direction = (
        EvidenceDirection.BULLISH
        if signal is Signal.BUY
        else EvidenceDirection.BEARISH
    )

    return StrategyResult(
        strategy="final_recommendation",
        signal=signal,
        score=score,
        confidence=0.90,
        evidence=(
            Evidence(
                code="TEST_SIGNAL",
                message="Deterministic fixture.",
                direction=direction,
                strength=1.0,
                observed_value=score,
            ),
        ),
    )


def make_signal(
    signal: Signal = Signal.BUY,
) -> SignalRecord:
    return SignalRecord.from_strategy_result(
        signal_id="SIG-001",
        symbol="TEST",
        generated_at=T0,
        expires_at=EXPIRY,
        result=make_strategy_result(signal),
    )


def make_order(
    side: PositionSide = PositionSide.LONG,
) -> OrderRecord:
    if side is PositionSide.LONG:
        return OrderRecord(
            order_id="ORD-001",
            signal_id="SIG-001",
            symbol="TEST",
            side=side,
            created_at=T0,
            expires_at=EXPIRY,
            entry_low=99.0,
            entry_high=101.0,
            stop_price=95.0,
            targets=(110.0, 120.0, 130.0),
            quantity=10,
        )

    return OrderRecord(
        order_id="ORD-001",
        signal_id="SIG-001",
        symbol="TEST",
        side=side,
        created_at=T0,
        expires_at=EXPIRY,
        entry_low=99.0,
        entry_high=101.0,
        stop_price=105.0,
        targets=(90.0, 80.0, 70.0),
        quantity=10,
    )


def make_fill(
    side: PositionSide = PositionSide.LONG,
) -> FillRecord:
    return FillRecord(
        fill_id="FILL-001",
        order_id="ORD-001",
        signal_id="SIG-001",
        symbol="TEST",
        side=side,
        filled_at=FILL_TIME,
        fill_price=100.0,
        quantity=10,
    )


def make_position(
    side: PositionSide = PositionSide.LONG,
) -> PositionRecord:
    if side is PositionSide.LONG:
        stop = 95.0
        targets = (110.0, 120.0, 130.0)
    else:
        stop = 105.0
        targets = (90.0, 80.0, 70.0)

    return PositionRecord(
        position_id="POS-001",
        order_id="ORD-001",
        fill_id="FILL-001",
        signal_id="SIG-001",
        symbol="TEST",
        side=side,
        opened_at=FILL_TIME,
        expires_at=EXPIRY,
        entry_price=100.0,
        quantity=10,
        stop_price=stop,
        targets=targets,
    )


def make_closed_trade(
    *,
    side: PositionSide = PositionSide.LONG,
    exit_reason: ExitReason = ExitReason.STOP,
    exit_price: float = 95.0,
    closed_at: datetime = CLOSE_TIME,
    target_index: int | None = None,
) -> ClosedTradeRecord:
    position = make_position(side)

    return ClosedTradeRecord(
        trade_id="TRADE-001",
        position_id=position.position_id,
        order_id=position.order_id,
        fill_id=position.fill_id,
        signal_id=position.signal_id,
        symbol=position.symbol,
        side=position.side,
        opened_at=position.opened_at,
        closed_at=closed_at,
        expires_at=position.expires_at,
        entry_price=position.entry_price,
        exit_price=exit_price,
        quantity=position.quantity,
        stop_price=position.stop_price,
        targets=position.targets,
        exit_reason=exit_reason,
        target_index=target_index,
    )


def test_entry_lifecycle_links_signal_order_fill_and_position() -> None:
    lifecycle = BacktestLifecycle(
        signal=make_signal(),
        order=make_order(),
        fill=make_fill(),
        position=make_position(),
        events=(
            LifecycleEvent(
                event_id="EVT-001",
                event_type=LifecycleEventType.SIGNAL_CREATED,
                occurred_at=T0,
                symbol="TEST",
                reference_id="SIG-001",
                message="Signal created.",
            ),
            LifecycleEvent(
                event_id="EVT-002",
                event_type=LifecycleEventType.ORDER_FILLED,
                occurred_at=FILL_TIME,
                symbol="TEST",
                reference_id="FILL-001",
                message="Order filled.",
            ),
        ),
    )

    assert lifecycle.is_filled is True
    assert lifecycle.is_open is True
    assert lifecycle.is_closed is False


def test_long_stop_exit_records_loss() -> None:
    trade = make_closed_trade(
        exit_reason=ExitReason.STOP,
        exit_price=95.0,
    )

    assert trade.exit_reason is ExitReason.STOP
    assert trade.gross_pnl == -50.0
    assert trade.return_pct == -5.0


def test_short_stop_exit_records_loss() -> None:
    trade = make_closed_trade(
        side=PositionSide.SHORT,
        exit_reason=ExitReason.STOP,
        exit_price=105.0,
    )

    assert trade.gross_pnl == -50.0
    assert trade.return_pct == -5.0


def test_long_target_exit_records_profit() -> None:
    trade = make_closed_trade(
        exit_reason=ExitReason.TARGET,
        exit_price=110.0,
        target_index=0,
    )

    assert trade.gross_pnl == 100.0
    assert trade.return_pct == 10.0
    assert trade.target_number == 1


def test_short_target_exit_records_profit() -> None:
    trade = make_closed_trade(
        side=PositionSide.SHORT,
        exit_reason=ExitReason.TARGET,
        exit_price=90.0,
        target_index=0,
    )

    assert trade.gross_pnl == 100.0
    assert trade.return_pct == 10.0
    assert trade.target_number == 1


def test_expiry_exit_cannot_occur_before_expiry() -> None:
    with pytest.raises(ValueError):
        make_closed_trade(
            exit_reason=ExitReason.EXPIRY,
            exit_price=102.0,
            closed_at=EXPIRY - timedelta(minutes=1),
        )


def test_expiry_exit_is_valid_at_expiry() -> None:
    trade = make_closed_trade(
        exit_reason=ExitReason.EXPIRY,
        exit_price=102.0,
        closed_at=EXPIRY,
    )

    assert trade.exit_reason is ExitReason.EXPIRY
    assert trade.gross_pnl == 20.0


def test_complete_closed_lifecycle_is_valid() -> None:
    trade = make_closed_trade(
        exit_reason=ExitReason.TARGET,
        exit_price=110.0,
        target_index=0,
    )

    lifecycle = BacktestLifecycle(
        signal=make_signal(),
        order=make_order(),
        fill=make_fill(),
        position=make_position(),
        closed_trade=trade,
    )

    assert lifecycle.is_filled is True
    assert lifecycle.is_open is False
    assert lifecycle.is_closed is True


def test_paper_order_conversion_preserves_risk_plan() -> None:
    paper_order = PaperOrder(
        symbol="TEST",
        signal=Signal.BUY,
        created_at=T0,
        expires_at=EXPIRY,
        entry_low=99.0,
        entry_high=101.0,
        stop_price=95.0,
        targets=(110.0, 120.0, 130.0),
        risk_per_share=5.0,
        reward_to_risk=2.0,
        paper_only=True,
    )

    order = OrderRecord.from_paper_order(
        order_id="ORD-001",
        signal_id="SIG-001",
        paper_order=paper_order,
        quantity=10,
    )

    assert order.side is PositionSide.LONG
    assert order.entry_low == paper_order.entry_low
    assert order.entry_high == paper_order.entry_high
    assert order.stop_price == paper_order.stop_price
    assert order.targets == paper_order.targets
    assert order.paper_only is True


def test_position_requires_matching_fill() -> None:
    with pytest.raises(ValueError):
        BacktestLifecycle(
            signal=make_signal(),
            order=make_order(),
            position=make_position(),
        )


def test_fill_must_occur_before_order_expiry() -> None:
    late_fill = FillRecord(
        fill_id="FILL-001",
        order_id="ORD-001",
        signal_id="SIG-001",
        symbol="TEST",
        side=PositionSide.LONG,
        filled_at=EXPIRY + timedelta(minutes=1),
        fill_price=100.0,
        quantity=10,
    )

    with pytest.raises(ValueError):
        BacktestLifecycle(
            signal=make_signal(),
            order=make_order(),
            fill=late_fill,
        )


def test_target_exit_requires_valid_target_index() -> None:
    with pytest.raises(ValueError):
        make_closed_trade(
            exit_reason=ExitReason.TARGET,
            exit_price=110.0,
            target_index=5,
        )


def test_order_cannot_outlive_signal() -> None:
    signal = make_signal()

    order = OrderRecord(
        order_id="ORD-001",
        signal_id="SIG-001",
        symbol="TEST",
        side=PositionSide.LONG,
        created_at=T0,
        expires_at=EXPIRY + timedelta(days=1),
        entry_low=99.0,
        entry_high=101.0,
        stop_price=95.0,
        targets=(110.0, 120.0),
        quantity=10,
    )

    with pytest.raises(ValueError):
        BacktestLifecycle(
            signal=signal,
            order=order,
        )


def test_duplicate_event_ids_are_rejected() -> None:
    event = LifecycleEvent(
        event_id="EVT-001",
        event_type=LifecycleEventType.ORDER_CREATED,
        occurred_at=T0,
        symbol="TEST",
        reference_id="ORD-001",
        message="Order created.",
    )

    with pytest.raises(ValueError):
        BacktestLifecycle(
            signal=make_signal(),
            order=make_order(),
            events=(event, event),
        )
