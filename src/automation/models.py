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
from src.costs import (
    IBKRFXMode,
    IBKRPricingPlan,
)

from src.paper import (
    AccountReconciliation,
    FixedNotionalSizingPolicy,
    PaperExitReason,
    PositionSizingMode,
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

    sizing_mode: PositionSizingMode | None = None
    portfolio_currency: str | None = None

    target_order_value: Decimal | None = None
    maximum_order_value: Decimal | None = None
    maximum_planned_loss: Decimal | None = None

    maximum_open_positions: int | None = None
    maximum_invested_exposure: Decimal | None = None

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

        fixed_values = (
            self.portfolio_currency,
            self.target_order_value,
            self.maximum_order_value,
            self.maximum_planned_loss,
            self.maximum_open_positions,
            self.maximum_invested_exposure,
        )

        if self.sizing_mode is None:
            if any(
                value is not None
                for value in fixed_values
            ):
                raise ValueError(
                    "Fixed-notional controls require "
                    "a sizing_mode."
                )

            return

        try:
            sizing_mode = PositionSizingMode(
                self.sizing_mode
            )
        except ValueError as exc:
            raise ValueError(
                "Unsupported portfolio sizing mode."
            ) from exc

        if any(
            value is None
            for value in fixed_values
        ):
            raise ValueError(
                "Fixed-notional sizing controls "
                "must be configured together."
            )

        policy = FixedNotionalSizingPolicy(
            mode=sizing_mode,
            portfolio_currency=str(
                self.portfolio_currency
            ),
            target_order_value=(
                self.target_order_value
            ),
            maximum_order_value=(
                self.maximum_order_value
            ),
            maximum_planned_loss=(
                self.maximum_planned_loss
            ),
            maximum_open_positions=(
                self.maximum_open_positions
            ),
            maximum_invested_exposure=(
                self.maximum_invested_exposure
            ),
        )

        object.__setattr__(
            self,
            "sizing_mode",
            policy.mode,
        )

        object.__setattr__(
            self,
            "portfolio_currency",
            policy.portfolio_currency,
        )

        object.__setattr__(
            self,
            "target_order_value",
            policy.target_order_value,
        )

        object.__setattr__(
            self,
            "maximum_order_value",
            policy.maximum_order_value,
        )

        object.__setattr__(
            self,
            "maximum_planned_loss",
            policy.maximum_planned_loss,
        )

        object.__setattr__(
            self,
            "maximum_open_positions",
            policy.maximum_open_positions,
        )

        object.__setattr__(
            self,
            "maximum_invested_exposure",
            policy.maximum_invested_exposure,
        )


@dataclass(frozen=True, slots=True)
class StrategyPause:
    """Persistent entry pause for one named strategy."""

    account_id: str
    strategy: str
    active: bool
    reason: str | None
    changed_by: str
    changed_at: datetime


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


    ibkr_cost_gate_enabled: bool = False

    ibkr_pricing_plan: (
        IBKRPricingPlan | None
    ) = None

    ibkr_fx_mode: (
        IBKRFXMode | None
    ) = None

    ibkr_include_entry_fx_conversion: bool = False

    ibkr_include_exit_fx_conversion: bool = False

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

        for name in (
            "ibkr_cost_gate_enabled",
            "ibkr_include_entry_fx_conversion",
            "ibkr_include_exit_fx_conversion",
        ):
            if not isinstance(
                getattr(self, name),
                bool,
            ):
                raise ValueError(
                    f"{name} must be boolean."
                )

        if (
            self.ibkr_pricing_plan
            is not None
            and not isinstance(
                self.ibkr_pricing_plan,
                IBKRPricingPlan,
            )
        ):
            raise ValueError(
                "ibkr_pricing_plan must be an "
                "IBKRPricingPlan or None."
            )

        if (
            self.ibkr_fx_mode
            is not None
            and not isinstance(
                self.ibkr_fx_mode,
                IBKRFXMode,
            )
        ):
            raise ValueError(
                "ibkr_fx_mode must be an "
                "IBKRFXMode or None."
            )

        if (
            self.ibkr_cost_gate_enabled
            and self.ibkr_pricing_plan
            is None
        ):
            raise ValueError(
                "ibkr_pricing_plan is required "
                "when the IBKR cost gate is enabled."
            )

        if (
            (
                self
                .ibkr_include_entry_fx_conversion
                or self
                .ibkr_include_exit_fx_conversion
            )
            and self.ibkr_fx_mode is None
        ):
            raise ValueError(
                "ibkr_fx_mode is required when "
                "IBKR FX conversion costs are enabled."
            )


@dataclass(frozen=True, slots=True)
class ExecutionRunReport:
    run: ExecutionRun

    entries_enabled: bool
    entry_block_reasons: tuple[str, ...]

    reconciliation: AccountReconciliation
    equity_snapshot: EquitySnapshot | None
