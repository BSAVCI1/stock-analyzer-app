"""Backtesting domain records and lifecycle validation."""

from .economics import (
    BindingConstraint,
    ExecutionCostModel,
    PositionSizeDecision,
    PositionSizingConstraints,
    PositionSizingError,
    TradeSettlement,
    apply_entry_slippage,
    apply_exit_slippage,
    apply_position_size,
    calculate_fee,
    calculate_position_size,
    settle_trade,
    validate_order_quantity,
)

from .execution import (
    ExecutionResult,
    ExecutionStatus,
    FillRule,
    execute_next_session,
)

from .model import (
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
    side_from_signal,
)

__all__ = [
    "BindingConstraint",
    "ExecutionCostModel",
    "PositionSizeDecision",
    "PositionSizingConstraints",
    "PositionSizingError",
    "TradeSettlement",
    "apply_entry_slippage",
    "apply_exit_slippage",
    "apply_position_size",
    "calculate_fee",
    "calculate_position_size",
    "settle_trade",
    "validate_order_quantity",
    "ExecutionResult",
    "ExecutionStatus",
    "FillRule",
    "execute_next_session",
    "BacktestLifecycle",
    "ClosedTradeRecord",
    "ExitReason",
    "FillRecord",
    "LifecycleEvent",
    "LifecycleEventType",
    "OrderRecord",
    "PositionRecord",
    "PositionSide",
    "SignalRecord",
    "side_from_signal",
]
