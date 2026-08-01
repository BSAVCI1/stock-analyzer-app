"""Deterministic backtest costs, slippage and position sizing.

This module provides:

- configurable commissions, transaction fees and slippage
- account-risk-based position sizing
- maximum capital-allocation constraints
- exact Decimal-based trade settlement
- validation against zero, invalid or excessive quantities

It contains no Streamlit, provider, broker or live-execution code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
)
from enum import Enum

from .model import (
    ClosedTradeRecord,
    OrderRecord,
    PositionSide,
)


_BASIS_POINTS = Decimal("10000")
_MONEY_QUANTUM = Decimal("0.00000001")


class PositionSizingError(ValueError):
    """Raised when no valid position size can satisfy the constraints."""


class BindingConstraint(str, Enum):
    """Constraint that determined the approved position size."""

    RISK = "RISK"
    ALLOCATION = "ALLOCATION"
    BOTH = "BOTH"


def _decimal(
    name: str,
    value: object,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a finite number."
        )

    try:
        result = (
            value
            if isinstance(value, Decimal)
            else Decimal(str(value))
        )
    except (
        InvalidOperation,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{name} must be a finite number."
        ) from exc

    if not result.is_finite():
        raise ValueError(
            f"{name} must be a finite number."
        )

    return result


def _non_negative_decimal(
    name: str,
    value: object,
) -> Decimal:
    result = _decimal(name, value)

    if result < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return result


def _positive_decimal(
    name: str,
    value: object,
) -> Decimal:
    result = _decimal(name, value)

    if result <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )

    return result


def _money(value: object) -> Decimal:
    return _decimal("money", value).quantize(
        _MONEY_QUANTUM,
        rounding=ROUND_HALF_EVEN,
    )


def _floor_to_step(
    value: Decimal,
    step: Decimal,
) -> Decimal:
    units = (
        value / step
    ).to_integral_value(
        rounding=ROUND_FLOOR
    )

    return units * step


@dataclass(frozen=True, slots=True)
class ExecutionCostModel:
    """Configurable deterministic transaction-cost assumptions."""

    fixed_fee: Decimal = Decimal("0")
    commission_bps: Decimal = Decimal("0")
    transaction_fee_bps: Decimal = Decimal("0")
    minimum_fee: Decimal = Decimal("0")
    slippage_bps: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        fixed_fee = _non_negative_decimal(
            "fixed_fee",
            self.fixed_fee,
        )
        commission_bps = _non_negative_decimal(
            "commission_bps",
            self.commission_bps,
        )
        transaction_fee_bps = _non_negative_decimal(
            "transaction_fee_bps",
            self.transaction_fee_bps,
        )
        minimum_fee = _non_negative_decimal(
            "minimum_fee",
            self.minimum_fee,
        )
        slippage_bps = _non_negative_decimal(
            "slippage_bps",
            self.slippage_bps,
        )

        if commission_bps > _BASIS_POINTS:
            raise ValueError(
                "commission_bps cannot exceed 10000."
            )

        if transaction_fee_bps > _BASIS_POINTS:
            raise ValueError(
                "transaction_fee_bps cannot exceed 10000."
            )

        if slippage_bps >= _BASIS_POINTS:
            raise ValueError(
                "slippage_bps must be below 10000."
            )

        object.__setattr__(
            self,
            "fixed_fee",
            fixed_fee,
        )
        object.__setattr__(
            self,
            "commission_bps",
            commission_bps,
        )
        object.__setattr__(
            self,
            "transaction_fee_bps",
            transaction_fee_bps,
        )
        object.__setattr__(
            self,
            "minimum_fee",
            minimum_fee,
        )
        object.__setattr__(
            self,
            "slippage_bps",
            slippage_bps,
        )


@dataclass(frozen=True, slots=True)
class PositionSizingConstraints:
    """Account-risk and capital-allocation limits."""

    risk_fraction: Decimal = Decimal("0.01")
    max_allocation_fraction: Decimal = Decimal("0.25")
    minimum_cash_reserve: Decimal = Decimal("0")

    allow_fractional: bool = False
    quantity_step: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        risk_fraction = _positive_decimal(
            "risk_fraction",
            self.risk_fraction,
        )
        max_allocation_fraction = _positive_decimal(
            "max_allocation_fraction",
            self.max_allocation_fraction,
        )
        minimum_cash_reserve = _non_negative_decimal(
            "minimum_cash_reserve",
            self.minimum_cash_reserve,
        )
        quantity_step = _positive_decimal(
            "quantity_step",
            self.quantity_step,
        )

        if risk_fraction > 1:
            raise ValueError(
                "risk_fraction cannot exceed 1."
            )

        if max_allocation_fraction > 1:
            raise ValueError(
                "max_allocation_fraction cannot exceed 1."
            )

        if not isinstance(
            self.allow_fractional,
            bool,
        ):
            raise ValueError(
                "allow_fractional must be boolean."
            )

        if (
            not self.allow_fractional
            and quantity_step != Decimal("1")
        ):
            raise ValueError(
                "Non-fractional sizing requires "
                "quantity_step equal to 1."
            )

        object.__setattr__(
            self,
            "risk_fraction",
            risk_fraction,
        )
        object.__setattr__(
            self,
            "max_allocation_fraction",
            max_allocation_fraction,
        )
        object.__setattr__(
            self,
            "minimum_cash_reserve",
            minimum_cash_reserve,
        )
        object.__setattr__(
            self,
            "quantity_step",
            quantity_step,
        )


@dataclass(frozen=True, slots=True)
class PositionSizeDecision:
    """Approved deterministic position-size calculation."""

    order_id: str
    quantity: Decimal

    expected_entry_price: Decimal
    expected_stop_exit_price: Decimal
    risk_per_unit: Decimal

    risk_budget: Decimal
    allocation_budget: Decimal

    estimated_entry_fee: Decimal
    estimated_stop_exit_fee: Decimal
    estimated_position_risk: Decimal
    capital_required: Decimal

    binding_constraint: BindingConstraint

    def __post_init__(self) -> None:
        if (
            not isinstance(self.order_id, str)
            or not self.order_id.strip()
        ):
            raise ValueError(
                "order_id must be a non-empty string."
            )

        if not isinstance(
            self.binding_constraint,
            BindingConstraint,
        ):
            raise ValueError(
                "binding_constraint must be "
                "a BindingConstraint."
            )

        for name in (
            "quantity",
            "expected_entry_price",
            "expected_stop_exit_price",
            "risk_per_unit",
            "risk_budget",
            "allocation_budget",
            "capital_required",
        ):
            value = _positive_decimal(
                name,
                getattr(self, name),
            )
            object.__setattr__(
                self,
                name,
                value,
            )

        for name in (
            "estimated_entry_fee",
            "estimated_stop_exit_fee",
            "estimated_position_risk",
        ):
            value = _non_negative_decimal(
                name,
                getattr(self, name),
            )
            object.__setattr__(
                self,
                name,
                value,
            )

        object.__setattr__(
            self,
            "order_id",
            self.order_id.strip(),
        )


@dataclass(frozen=True, slots=True)
class TradeSettlement:
    """Exact account reconciliation for one completed test trade."""

    opening_balance: Decimal

    raw_entry_price: Decimal
    executed_entry_price: Decimal
    raw_exit_price: Decimal
    executed_exit_price: Decimal
    quantity: Decimal

    market_gross_pnl: Decimal
    gross_pnl_after_slippage: Decimal
    slippage_cost: Decimal

    entry_fee: Decimal
    exit_fee: Decimal
    total_fees: Decimal

    net_pnl: Decimal
    ending_balance: Decimal

    def __post_init__(self) -> None:
        for name in (
            "opening_balance",
            "raw_entry_price",
            "executed_entry_price",
            "raw_exit_price",
            "executed_exit_price",
            "quantity",
        ):
            value = _positive_decimal(
                name,
                getattr(self, name),
            )
            object.__setattr__(
                self,
                name,
                value,
            )

        for name in (
            "entry_fee",
            "exit_fee",
            "total_fees",
            "slippage_cost",
        ):
            value = _non_negative_decimal(
                name,
                getattr(self, name),
            )
            object.__setattr__(
                self,
                name,
                value,
            )

        for name in (
            "market_gross_pnl",
            "gross_pnl_after_slippage",
            "net_pnl",
            "ending_balance",
        ):
            value = _decimal(
                name,
                getattr(self, name),
            )
            object.__setattr__(
                self,
                name,
                value,
            )

        expected_total_fees = _money(
            self.entry_fee
            + self.exit_fee
        )

        if self.total_fees != expected_total_fees:
            raise ValueError(
                "total_fees does not reconcile."
            )

        expected_slippage_cost = _money(
            self.market_gross_pnl
            - self.gross_pnl_after_slippage
        )

        if self.slippage_cost != expected_slippage_cost:
            raise ValueError(
                "slippage_cost does not reconcile."
            )

        expected_net_pnl = _money(
            self.gross_pnl_after_slippage
            - self.total_fees
        )

        if self.net_pnl != expected_net_pnl:
            raise ValueError(
                "net_pnl does not reconcile."
            )

        expected_ending_balance = _money(
            self.opening_balance
            + self.net_pnl
        )

        if self.ending_balance != expected_ending_balance:
            raise ValueError(
                "ending_balance does not reconcile."
            )

    @property
    def reconciliation_delta(self) -> Decimal:
        """Return zero when the account equation balances exactly."""

        return _money(
            self.ending_balance
            - self.opening_balance
            - self.net_pnl
        )

    @property
    def reconciled(self) -> bool:
        return (
            self.reconciliation_delta
            == Decimal("0.00000000")
        )


def calculate_fee(
    notional: object,
    costs: ExecutionCostModel,
) -> Decimal:
    """Calculate one side of transaction fees."""

    if not isinstance(
        costs,
        ExecutionCostModel,
    ):
        raise ValueError(
            "costs must be an ExecutionCostModel."
        )

    absolute_notional = abs(
        _decimal("notional", notional)
    )

    variable_rate = (
        costs.commission_bps
        + costs.transaction_fee_bps
    ) / _BASIS_POINTS

    calculated = (
        costs.fixed_fee
        + absolute_notional * variable_rate
    )

    return _money(
        max(
            calculated,
            costs.minimum_fee,
        )
    )


def apply_entry_slippage(
    price: object,
    side: PositionSide,
    costs: ExecutionCostModel,
) -> Decimal:
    """Return an adverse entry price for the position side."""

    raw_price = _positive_decimal(
        "price",
        price,
    )

    if not isinstance(side, PositionSide):
        raise ValueError(
            "side must be a PositionSide."
        )

    rate = (
        costs.slippage_bps
        / _BASIS_POINTS
    )

    multiplier = (
        Decimal("1") + rate
        if side is PositionSide.LONG
        else Decimal("1") - rate
    )

    executed_price = _money(
        raw_price * multiplier
    )

    if executed_price <= 0:
        raise ValueError(
            "Entry slippage produced an invalid price."
        )

    return executed_price


def apply_exit_slippage(
    price: object,
    side: PositionSide,
    costs: ExecutionCostModel,
) -> Decimal:
    """Return an adverse exit price for the position side."""

    raw_price = _positive_decimal(
        "price",
        price,
    )

    if not isinstance(side, PositionSide):
        raise ValueError(
            "side must be a PositionSide."
        )

    rate = (
        costs.slippage_bps
        / _BASIS_POINTS
    )

    multiplier = (
        Decimal("1") - rate
        if side is PositionSide.LONG
        else Decimal("1") + rate
    )

    return _money(
        raw_price * multiplier
    )


def calculate_position_size(
    order: OrderRecord,
    account_balance: object,
    *,
    costs: ExecutionCostModel | None = None,
    constraints: PositionSizingConstraints | None = None,
) -> PositionSizeDecision:
    """Return the largest quantity satisfying risk and allocation limits."""

    if not isinstance(order, OrderRecord):
        raise ValueError(
            "order must be an OrderRecord."
        )

    costs = costs or ExecutionCostModel()
    constraints = (
        constraints
        or PositionSizingConstraints()
    )

    if not isinstance(
        costs,
        ExecutionCostModel,
    ):
        raise ValueError(
            "costs must be an ExecutionCostModel."
        )

    if not isinstance(
        constraints,
        PositionSizingConstraints,
    ):
        raise ValueError(
            "constraints must be "
            "PositionSizingConstraints."
        )

    balance = _positive_decimal(
        "account_balance",
        account_balance,
    )

    available_balance = (
        balance
        - constraints.minimum_cash_reserve
    )

    if available_balance <= 0:
        raise PositionSizingError(
            "Minimum cash reserve leaves no "
            "capital available."
        )

    risk_budget = _money(
        balance
        * constraints.risk_fraction
    )

    allocation_budget = _money(
        min(
            balance
            * constraints.max_allocation_fraction,
            available_balance,
        )
    )

    raw_entry_price = (
        order.entry_high
        if order.side is PositionSide.LONG
        else order.entry_low
    )

    expected_entry_price = apply_entry_slippage(
        raw_entry_price,
        order.side,
        costs,
    )

    expected_stop_exit_price = apply_exit_slippage(
        order.stop_price,
        order.side,
        costs,
    )

    risk_per_unit = _money(
        abs(
            expected_entry_price
            - expected_stop_exit_price
        )
    )

    if risk_per_unit <= 0:
        raise PositionSizingError(
            "Stop distance must be greater than zero."
        )

    maximum_by_risk = (
        risk_budget
        / risk_per_unit
    )

    maximum_by_allocation = (
        allocation_budget
        / expected_entry_price
    )

    theoretical_maximum = min(
        maximum_by_risk,
        maximum_by_allocation,
    )

    step = constraints.quantity_step

    maximum_units = int(
        (
            theoretical_maximum
            / step
        ).to_integral_value(
            rounding=ROUND_FLOOR
        )
    )

    if maximum_units < 1:
        raise PositionSizingError(
            "No positive position size satisfies "
            "the account constraints."
        )

    def evaluate_units(
        units: int,
    ) -> tuple[
        bool,
        Decimal,
        Decimal,
        Decimal,
        Decimal,
    ]:
        quantity = step * units

        entry_notional = _money(
            expected_entry_price
            * quantity
        )

        stop_exit_notional = _money(
            expected_stop_exit_price
            * quantity
        )

        entry_fee = calculate_fee(
            entry_notional,
            costs,
        )

        stop_exit_fee = calculate_fee(
            stop_exit_notional,
            costs,
        )

        capital_required = _money(
            entry_notional
            + entry_fee
        )

        estimated_position_risk = _money(
            risk_per_unit
            * quantity
            + entry_fee
            + stop_exit_fee
        )

        valid = (
            capital_required
            <= allocation_budget
            and estimated_position_risk
            <= risk_budget
        )

        return (
            valid,
            entry_fee,
            stop_exit_fee,
            capital_required,
            estimated_position_risk,
        )

    low = 1
    high = maximum_units
    approved_units = 0

    while low <= high:
        middle = (low + high) // 2

        valid, _, _, _, _ = evaluate_units(
            middle
        )

        if valid:
            approved_units = middle
            low = middle + 1
        else:
            high = middle - 1

    if approved_units < 1:
        raise PositionSizingError(
            "Fees and slippage leave no valid "
            "positive position size."
        )

    (
        _,
        entry_fee,
        stop_exit_fee,
        capital_required,
        estimated_position_risk,
    ) = evaluate_units(approved_units)

    quantity = (
        step
        * approved_units
    )

    risk_utilisation = (
        estimated_position_risk
        / risk_budget
    )

    allocation_utilisation = (
        capital_required
        / allocation_budget
    )

    utilisation_difference = abs(
        risk_utilisation
        - allocation_utilisation
    )

    if utilisation_difference <= Decimal(
        "0.000001"
    ):
        binding_constraint = (
            BindingConstraint.BOTH
        )
    elif risk_utilisation > allocation_utilisation:
        binding_constraint = (
            BindingConstraint.RISK
        )
    else:
        binding_constraint = (
            BindingConstraint.ALLOCATION
        )

    return PositionSizeDecision(
        order_id=order.order_id,
        quantity=quantity,
        expected_entry_price=expected_entry_price,
        expected_stop_exit_price=(
            expected_stop_exit_price
        ),
        risk_per_unit=risk_per_unit,
        risk_budget=risk_budget,
        allocation_budget=allocation_budget,
        estimated_entry_fee=entry_fee,
        estimated_stop_exit_fee=(
            stop_exit_fee
        ),
        estimated_position_risk=(
            estimated_position_risk
        ),
        capital_required=capital_required,
        binding_constraint=(
            binding_constraint
        ),
    )


def validate_order_quantity(
    order: OrderRecord,
    account_balance: object,
    *,
    costs: ExecutionCostModel | None = None,
    constraints: PositionSizingConstraints | None = None,
) -> PositionSizeDecision:
    """Reject an order whose quantity exceeds the valid maximum."""

    decision = calculate_position_size(
        order,
        account_balance,
        costs=costs,
        constraints=constraints,
    )

    constraints = (
        constraints
        or PositionSizingConstraints()
    )

    submitted_quantity = _positive_decimal(
        "order.quantity",
        order.quantity,
    )

    if (
        submitted_quantity
        % constraints.quantity_step
    ) != 0:
        raise PositionSizingError(
            "Order quantity does not match "
            "the configured quantity step."
        )

    if submitted_quantity > decision.quantity:
        raise PositionSizingError(
            "Order quantity exceeds the maximum "
            "risk-adjusted position size."
        )

    return decision


def apply_position_size(
    order: OrderRecord,
    decision: PositionSizeDecision,
) -> OrderRecord:
    """Return a new immutable OrderRecord with the approved quantity."""

    if not isinstance(order, OrderRecord):
        raise ValueError(
            "order must be an OrderRecord."
        )

    if not isinstance(
        decision,
        PositionSizeDecision,
    ):
        raise ValueError(
            "decision must be a PositionSizeDecision."
        )

    if order.order_id != decision.order_id:
        raise ValueError(
            "Sizing decision does not belong "
            "to this order."
        )

    return replace(
        order,
        quantity=float(decision.quantity),
    )


def settle_trade(
    trade: ClosedTradeRecord,
    opening_balance: object,
    *,
    costs: ExecutionCostModel | None = None,
) -> TradeSettlement:
    """Apply entry/exit slippage and fees and reconcile account equity.

    The ClosedTradeRecord stores the observed market prices. This function
    converts both entry and exit to adverse executed prices, calculates all
    fees, and returns an exact Decimal-based account reconciliation.
    """

    if not isinstance(
        trade,
        ClosedTradeRecord,
    ):
        raise ValueError(
            "trade must be a ClosedTradeRecord."
        )

    costs = costs or ExecutionCostModel()

    if not isinstance(
        costs,
        ExecutionCostModel,
    ):
        raise ValueError(
            "costs must be an ExecutionCostModel."
        )

    balance = _positive_decimal(
        "opening_balance",
        opening_balance,
    )

    quantity = _positive_decimal(
        "trade.quantity",
        trade.quantity,
    )

    raw_entry_price = _money(
        trade.entry_price
    )
    raw_exit_price = _money(
        trade.exit_price
    )

    executed_entry_price = apply_entry_slippage(
        raw_entry_price,
        trade.side,
        costs,
    )

    executed_exit_price = apply_exit_slippage(
        raw_exit_price,
        trade.side,
        costs,
    )

    direction = (
        Decimal("1")
        if trade.side is PositionSide.LONG
        else Decimal("-1")
    )

    market_gross_pnl = _money(
        (
            raw_exit_price
            - raw_entry_price
        )
        * quantity
        * direction
    )

    gross_pnl_after_slippage = _money(
        (
            executed_exit_price
            - executed_entry_price
        )
        * quantity
        * direction
    )

    slippage_cost = _money(
        market_gross_pnl
        - gross_pnl_after_slippage
    )

    entry_notional = _money(
        executed_entry_price
        * quantity
    )

    exit_notional = _money(
        executed_exit_price
        * quantity
    )

    entry_fee = calculate_fee(
        entry_notional,
        costs,
    )

    exit_fee = calculate_fee(
        exit_notional,
        costs,
    )

    total_fees = _money(
        entry_fee
        + exit_fee
    )

    net_pnl = _money(
        gross_pnl_after_slippage
        - total_fees
    )

    ending_balance = _money(
        balance
        + net_pnl
    )

    return TradeSettlement(
        opening_balance=_money(balance),
        raw_entry_price=raw_entry_price,
        executed_entry_price=(
            executed_entry_price
        ),
        raw_exit_price=raw_exit_price,
        executed_exit_price=(
            executed_exit_price
        ),
        quantity=quantity,
        market_gross_pnl=market_gross_pnl,
        gross_pnl_after_slippage=(
            gross_pnl_after_slippage
        ),
        slippage_cost=slippage_cost,
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        total_fees=total_fees,
        net_pnl=net_pnl,
        ending_balance=ending_balance,
    )
