"""Backtesting domain records and lifecycle validation."""

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
