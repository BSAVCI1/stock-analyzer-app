"""Internal execution adapter backed by existing paper services."""

from __future__ import annotations

from datetime import datetime

from src.paper import (
    ClosedPaperTrade,
    PaperExitReason,
    PaperFillRecord,
    PaperOrderRecord,
    PaperPositionRecord,
    PaperRepository,
    PaperTradingService,
)

from .models import (
    ExecutionAdapterDescriptor,
    ExecutionAdapterType,
    ExecutionEnvironment,
)
from .safety import validate_paper_only_descriptor


class InternalPaperExecutionAdapter:
    """Delegate execution to the existing internal paper ledger."""

    def __init__(
        self,
        *,
        paper_repository: PaperRepository,
        paper_service: PaperTradingService,
        descriptor:
        ExecutionAdapterDescriptor | None = None,
    ) -> None:
        self.paper_repository = paper_repository
        self.paper_service = paper_service

        self._descriptor = (
            validate_paper_only_descriptor(
                descriptor
                or ExecutionAdapterDescriptor(
                    adapter_id="internal-paper",
                    adapter_type=(
                        ExecutionAdapterType.INTERNAL
                    ),
                    environment=(
                        ExecutionEnvironment
                        .INTERNAL_PAPER
                    ),
                    provider="internal-sqlite",
                    live_trading_enabled=False,
                    supports_account_reconciliation=True,
                    supports_order_reconciliation=True,
                    supports_position_reconciliation=True,
                    metadata={
                        "persistence":
                        "paper_trading_sqlite",
                        "paper_only": True,
                    },
                )
            )
        )

    @property
    def descriptor(
        self,
    ) -> ExecutionAdapterDescriptor:
        return self._descriptor

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
        return (
            self.paper_service
            .record_automatic_buy_fill(
                order_id=order_id,
                fill_price=fill_price,
                fees=fees,
                slippage=slippage,
                filled_at=filled_at,
            )
        )

    def cancel_order(
        self,
        *,
        order_id: str,
        reason: str,
        cancelled_at: datetime | None = None,
    ) -> PaperOrderRecord:
        return (
            self.paper_service
            .cancel_pending_order(
                order_id=order_id,
                reason=reason,
                cancelled_at=cancelled_at,
            )
        )

    def expire_order(
        self,
        order_id: str,
        *,
        expired_at: datetime,
        reason: str,
    ) -> PaperOrderRecord:
        return (
            self.paper_repository
            .expire_order(
                order_id,
                expired_at=expired_at,
                reason=reason,
            )
        )

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
        return (
            self.paper_service
            .close_automatic_position(
                position_id=position_id,
                exit_price=exit_price,
                exit_reason=exit_reason,
                exit_fees=exit_fees,
                exit_slippage=exit_slippage,
                closed_at=closed_at,
            )
        )
