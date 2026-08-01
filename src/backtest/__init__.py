"""Backtesting domain records and lifecycle validation."""

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
