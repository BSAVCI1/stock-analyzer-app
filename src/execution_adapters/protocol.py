"""Common execution-adapter protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from src.paper import (
    ClosedPaperTrade,
    PaperExitReason,
    PaperFillRecord,
    PaperOrderRecord,
    PaperPositionRecord,
)

from .models import ExecutionAdapterDescriptor


@runtime_checkable
class ExecutionAdapter(Protocol):
    """Lifecycle operations used by automated execution."""

    @property
    def descriptor(
        self,
    ) -> ExecutionAdapterDescriptor:
        ...

    def record_buy_fill(
        self,
        *,
        order_id: str,
        fill_price: object,
        fees: object = 0,
        slippage: object = 0,
        filled_at: datetime | None = None,
    ) -> tuple[
        PaperFillRecord,
        PaperPositionRecord,
    ]:
        ...

    def cancel_order(
        self,
        *,
        order_id: str,
        reason: str,
        cancelled_at: datetime | None = None,
    ) -> PaperOrderRecord:
        ...

    def expire_order(
        self,
        order_id: str,
        *,
        expired_at: datetime,
        reason: str,
    ) -> PaperOrderRecord:
        ...

    def close_position(
        self,
        *,
        position_id: str,
        exit_price: object,
        exit_reason: PaperExitReason,
        exit_fees: object = 0,
        exit_slippage: object = 0,
        closed_at: datetime | None = None,
    ) -> ClosedPaperTrade:
        ...
