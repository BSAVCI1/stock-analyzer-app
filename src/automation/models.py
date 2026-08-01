"""Models for automated paper execution and monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Mapping

from src.backtest import (
    ExecutionCostModel,
    FillRule,
)
from src.paper import (
    AccountReconciliation,
    PaperExitReason,
    money,
)


class ExecutionRunStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_ERRORS = (
        "COMPLETED_WITH_ERRORS"
    )
    FAILED = "FAILED"


class ExitRequestStatus(str, Enum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class PortfolioControl:
    account_id: str

    kill_switch_active: bool = False
    kill_switch_reason: str | None = None

    maximum_daily_loss_fraction: Decimal = Decimal(
        "0.03"
    )

    maximum_drawdown_fraction: Decimal = Decimal(
        "0.10"
    )

    maximum_new_orders_per_run: int = 3
    maximum_stale_market_days: int = 7

    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in (
            "maximum_daily_loss_fraction",
            "maximum_drawdown_fraction",
        ):
            value = money(
                getattr(self, name)
            )

            if not 0 < value <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        for name in (
            "maximum_new_orders_per_run",
            "maximum_stale_market_days",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be positive."
                )


@dataclass(frozen=True, slots=True)
class ExitRequest:
    request_id: str
    account_id: str
    position_id: str

    reason: PaperExitReason
    triggered_at: datetime

    status: ExitRequestStatus
    created_at: datetime
    executed_at: datetime | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class EquitySnapshot:
    snapshot_id: str
    run_id: str
    account_id: str

    captured_at: datetime

    cash_balance: Decimal
    reserved_cash: Decimal
    market_value: Decimal
    equity: Decimal


@dataclass(frozen=True, slots=True)
class ExecutionRun:
    run_id: str
    account_id: str
    run_key: str
    scan_id: str | None

    status: ExecutionRunStatus

    started_at: datetime
    completed_at: datetime | None

    created_orders: int
    filled_orders: int
    expired_orders: int
    cancelled_orders: int
    closed_positions: int
    rejected_entries: int
    error_count: int

    entry_block_reasons: tuple[str, ...]

    configuration: Mapping[str, object]
    app_version: str
    error_message: str | None


@dataclass(frozen=True, slots=True)
class AutomatedExecutionConfig:
    fill_rule: FillRule = FillRule.LIMIT

    costs: ExecutionCostModel = field(
        default_factory=ExecutionCostModel
    )

    enable_signal_reversal: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.fill_rule,
            FillRule,
        ):
            raise ValueError(
                "fill_rule must be a FillRule."
            )

        if not isinstance(
            self.costs,
            ExecutionCostModel,
        ):
            raise ValueError(
                "costs must be an ExecutionCostModel."
            )


@dataclass(frozen=True, slots=True)
class ExecutionRunReport:
    run: ExecutionRun

    entries_enabled: bool
    entry_block_reasons: tuple[str, ...]

    reconciliation: AccountReconciliation
    equity_snapshot: EquitySnapshot | None
