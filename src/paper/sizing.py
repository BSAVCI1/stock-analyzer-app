"""Deterministic P4.1 fixed-notional position sizing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_FLOOR,
)
from enum import Enum

from .models import money


class PositionSizingMode(str, Enum):
    """Supported operational paper-sizing modes."""

    FIXED_NOTIONAL_WITH_RISK_CAP = (
        "FIXED_NOTIONAL_WITH_RISK_CAP"
    )


class PositionSizingConstraint(str, Enum):
    """Constraint preventing the next quantity step."""

    TARGET_NOTIONAL = "TARGET_NOTIONAL"
    HARD_NOTIONAL = "HARD_NOTIONAL"
    PLANNED_LOSS = "PLANNED_LOSS"
    AVAILABLE_CASH = "AVAILABLE_CASH"
    INVESTED_EXPOSURE = "INVESTED_EXPOSURE"
    QUANTITY_STEP = "QUANTITY_STEP"


class PositionSizingRejected(ValueError):
    """Raised when no valid positive order can be sized."""


def _decimal(
    name: str,
    value: object,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be numeric."
        )

    try:
        result = Decimal(str(value))
    except (
        InvalidOperation,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{name} must be numeric."
        ) from exc

    if not result.is_finite():
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def _positive(
    name: str,
    value: object,
) -> Decimal:
    result = _decimal(name, value)

    if result <= 0:
        raise ValueError(
            f"{name} must be positive."
        )

    return money(result)


def _non_negative(
    name: str,
    value: object,
) -> Decimal:
    result = _decimal(name, value)

    if result < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return money(result)


def _currency(
    name: str,
    value: object,
) -> str:
    result = str(value).strip().upper()

    if (
        len(result) != 3
        or not result.isalpha()
    ):
        raise ValueError(
            f"{name} must be a three-letter "
            "currency code."
        )

    return result


def _whole_number(
    name: str,
    value: object,
    *,
    minimum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer "
            f"greater than or equal to "
            f"{minimum}."
        )

    return value


@dataclass(frozen=True, slots=True)
class FixedNotionalSizingPolicy:
    """Approved P4.1 portfolio sizing limits."""

    mode: PositionSizingMode = (
        PositionSizingMode
        .FIXED_NOTIONAL_WITH_RISK_CAP
    )

    portfolio_currency: str = "EUR"

    target_order_value: Decimal = Decimal(
        "100"
    )

    maximum_order_value: Decimal = Decimal(
        "100"
    )

    maximum_planned_loss: Decimal = Decimal(
        "10"
    )

    maximum_open_positions: int = 5

    maximum_invested_exposure: Decimal = (
        Decimal("500")
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.mode,
            PositionSizingMode,
        ):
            raise ValueError(
                "mode must be a "
                "PositionSizingMode."
            )

        portfolio_currency = _currency(
            "portfolio_currency",
            self.portfolio_currency,
        )

        target_order_value = _positive(
            "target_order_value",
            self.target_order_value,
        )

        maximum_order_value = _positive(
            "maximum_order_value",
            self.maximum_order_value,
        )

        maximum_planned_loss = _positive(
            "maximum_planned_loss",
            self.maximum_planned_loss,
        )

        maximum_invested_exposure = _positive(
            "maximum_invested_exposure",
            self.maximum_invested_exposure,
        )

        maximum_open_positions = _whole_number(
            "maximum_open_positions",
            self.maximum_open_positions,
            minimum=1,
        )

        if (
            target_order_value
            > maximum_order_value
        ):
            raise ValueError(
                "target_order_value cannot "
                "exceed maximum_order_value."
            )

        if (
            maximum_order_value
            > maximum_invested_exposure
        ):
            raise ValueError(
                "maximum_order_value cannot "
                "exceed "
                "maximum_invested_exposure."
            )

        object.__setattr__(
            self,
            "portfolio_currency",
            portfolio_currency,
        )

        object.__setattr__(
            self,
            "target_order_value",
            target_order_value,
        )

        object.__setattr__(
            self,
            "maximum_order_value",
            maximum_order_value,
        )

        object.__setattr__(
            self,
            "maximum_planned_loss",
            maximum_planned_loss,
        )

        object.__setattr__(
            self,
            "maximum_open_positions",
            maximum_open_positions,
        )

        object.__setattr__(
            self,
            "maximum_invested_exposure",
            maximum_invested_exposure,
        )


@dataclass(frozen=True, slots=True)
class FixedNotionalSizingRequest:
    """Inputs required for one sizing decision."""

    quote_currency: str

    entry_price_quote: Decimal
    stop_price_quote: Decimal

    quote_to_portfolio_rate: Decimal

    available_cash_portfolio: Decimal
    invested_exposure_portfolio: Decimal

    current_position_count: int

    estimated_entry_fee_portfolio: Decimal = (
        Decimal("0")
    )

    estimated_exit_fee_portfolio: Decimal = (
        Decimal("0")
    )

    quantity_step: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        quote_currency = _currency(
            "quote_currency",
            self.quote_currency,
        )

        entry_price_quote = _positive(
            "entry_price_quote",
            self.entry_price_quote,
        )

        stop_price_quote = _positive(
            "stop_price_quote",
            self.stop_price_quote,
        )

        if stop_price_quote >= entry_price_quote:
            raise ValueError(
                "stop_price_quote must be below "
                "entry_price_quote."
            )

        quote_to_portfolio_rate = _positive(
            "quote_to_portfolio_rate",
            self.quote_to_portfolio_rate,
        )

        available_cash_portfolio = (
            _non_negative(
                "available_cash_portfolio",
                self.available_cash_portfolio,
            )
        )

        invested_exposure_portfolio = (
            _non_negative(
                "invested_exposure_portfolio",
                self
                .invested_exposure_portfolio,
            )
        )

        current_position_count = _whole_number(
            "current_position_count",
            self.current_position_count,
            minimum=0,
        )

        estimated_entry_fee_portfolio = (
            _non_negative(
                "estimated_entry_fee_portfolio",
                self
                .estimated_entry_fee_portfolio,
            )
        )

        estimated_exit_fee_portfolio = (
            _non_negative(
                "estimated_exit_fee_portfolio",
                self
                .estimated_exit_fee_portfolio,
            )
        )

        quantity_step = _positive(
            "quantity_step",
            self.quantity_step,
        )

        object.__setattr__(
            self,
            "quote_currency",
            quote_currency,
        )

        object.__setattr__(
            self,
            "entry_price_quote",
            entry_price_quote,
        )

        object.__setattr__(
            self,
            "stop_price_quote",
            stop_price_quote,
        )

        object.__setattr__(
            self,
            "quote_to_portfolio_rate",
            quote_to_portfolio_rate,
        )

        object.__setattr__(
            self,
            "available_cash_portfolio",
            available_cash_portfolio,
        )

        object.__setattr__(
            self,
            "invested_exposure_portfolio",
            invested_exposure_portfolio,
        )

        object.__setattr__(
            self,
            "current_position_count",
            current_position_count,
        )

        object.__setattr__(
            self,
            "estimated_entry_fee_portfolio",
            estimated_entry_fee_portfolio,
        )

        object.__setattr__(
            self,
            "estimated_exit_fee_portfolio",
            estimated_exit_fee_portfolio,
        )

        object.__setattr__(
            self,
            "quantity_step",
            quantity_step,
        )


@dataclass(frozen=True, slots=True)
class FixedNotionalSizingDecision:
    """Auditable result of the P4.1 sizing rule."""

    mode: PositionSizingMode

    portfolio_currency: str
    quote_currency: str

    quote_to_portfolio_rate: Decimal
    quantity_step: Decimal
    quantity: Decimal

    order_notional_quote: Decimal
    order_notional_portfolio: Decimal

    planned_loss_portfolio: Decimal
    capital_required_portfolio: Decimal

    exposure_before_portfolio: Decimal
    exposure_after_portfolio: Decimal

    binding_constraints: tuple[
        PositionSizingConstraint,
        ...,
    ]


def _floor_to_step(
    value: Decimal,
    step: Decimal,
) -> Decimal:
    units = (
        value / step
    ).to_integral_value(
        rounding=ROUND_FLOOR
    )

    return money(
        units * step
    )


def calculate_fixed_notional_size(
    request: FixedNotionalSizingRequest,
    *,
    policy: FixedNotionalSizingPolicy
    | None = None,
) -> FixedNotionalSizingDecision:
    """Return the largest valid quantity under P4.1."""

    if not isinstance(
        request,
        FixedNotionalSizingRequest,
    ):
        raise ValueError(
            "request must be a "
            "FixedNotionalSizingRequest."
        )

    policy = (
        policy
        or FixedNotionalSizingPolicy()
    )

    if not isinstance(
        policy,
        FixedNotionalSizingPolicy,
    ):
        raise ValueError(
            "policy must be a "
            "FixedNotionalSizingPolicy."
        )

    if (
        request.current_position_count
        >= policy.maximum_open_positions
    ):
        raise PositionSizingRejected(
            "Maximum open-position limit "
            "has been reached."
        )

    remaining_exposure = money(
        policy.maximum_invested_exposure
        - request
        .invested_exposure_portfolio
    )

    if remaining_exposure <= 0:
        raise PositionSizingRejected(
            "No invested-exposure capacity "
            "remains."
        )

    entry_price_portfolio = money(
        request.entry_price_quote
        * request.quote_to_portfolio_rate
    )

    stop_price_portfolio = money(
        request.stop_price_quote
        * request.quote_to_portfolio_rate
    )

    risk_per_unit_portfolio = money(
        entry_price_portfolio
        - stop_price_portfolio
    )

    if risk_per_unit_portfolio <= 0:
        raise PositionSizingRejected(
            "Converted stop distance must "
            "be positive."
        )

    fee_total = money(
        request
        .estimated_entry_fee_portfolio
        + request
        .estimated_exit_fee_portfolio
    )

    risk_capacity = money(
        policy.maximum_planned_loss
        - fee_total
    )

    if risk_capacity <= 0:
        raise PositionSizingRejected(
            "Estimated fees consume the "
            "planned-loss cap."
        )

    cash_notional_capacity = money(
        request.available_cash_portfolio
        - request
        .estimated_entry_fee_portfolio
    )

    if cash_notional_capacity <= 0:
        raise PositionSizingRejected(
            "No cash remains after estimated "
            "entry fees."
        )

    notional_budget = min(
        policy.target_order_value,
        policy.maximum_order_value,
        remaining_exposure,
        cash_notional_capacity,
    )

    if notional_budget <= 0:
        raise PositionSizingRejected(
            "No positive order-notional "
            "capacity remains."
        )

    maximum_by_notional = (
        notional_budget
        / entry_price_portfolio
    )

    maximum_by_risk = (
        risk_capacity
        / risk_per_unit_portfolio
    )

    quantity = _floor_to_step(
        min(
            maximum_by_notional,
            maximum_by_risk,
        ),
        request.quantity_step,
    )

    while quantity > 0:
        order_notional_quote = money(
            request.entry_price_quote
            * quantity
        )

        order_notional_portfolio = money(
            order_notional_quote
            * request.quote_to_portfolio_rate
        )

        planned_loss_portfolio = money(
            risk_per_unit_portfolio
            * quantity
            + fee_total
        )

        capital_required_portfolio = money(
            order_notional_portfolio
            + request
            .estimated_entry_fee_portfolio
        )

        exposure_after_portfolio = money(
            request
            .invested_exposure_portfolio
            + order_notional_portfolio
        )

        valid = (
            order_notional_portfolio
            <= policy.target_order_value
            and order_notional_portfolio
            <= policy.maximum_order_value
            and planned_loss_portfolio
            <= policy.maximum_planned_loss
            and capital_required_portfolio
            <= request.available_cash_portfolio
            and exposure_after_portfolio
            <= policy
            .maximum_invested_exposure
        )

        if valid:
            break

        quantity = money(
            quantity
            - request.quantity_step
        )

    if quantity <= 0:
        raise PositionSizingRejected(
            "No positive quantity step "
            "satisfies all P4.1 limits."
        )

    next_quantity = money(
        quantity
        + request.quantity_step
    )

    next_notional = money(
        request.entry_price_quote
        * next_quantity
        * request.quote_to_portfolio_rate
    )

    next_planned_loss = money(
        risk_per_unit_portfolio
        * next_quantity
        + fee_total
    )

    next_capital = money(
        next_notional
        + request
        .estimated_entry_fee_portfolio
    )

    next_exposure = money(
        request
        .invested_exposure_portfolio
        + next_notional
    )

    binding: list[
        PositionSizingConstraint
    ] = []

    if (
        next_notional
        > policy.target_order_value
    ):
        binding.append(
            PositionSizingConstraint
            .TARGET_NOTIONAL
        )

    if (
        next_notional
        > policy.maximum_order_value
    ):
        binding.append(
            PositionSizingConstraint
            .HARD_NOTIONAL
        )

    if (
        next_planned_loss
        > policy.maximum_planned_loss
    ):
        binding.append(
            PositionSizingConstraint
            .PLANNED_LOSS
        )

    if (
        next_capital
        > request.available_cash_portfolio
    ):
        binding.append(
            PositionSizingConstraint
            .AVAILABLE_CASH
        )

    if (
        next_exposure
        > policy.maximum_invested_exposure
    ):
        binding.append(
            PositionSizingConstraint
            .INVESTED_EXPOSURE
        )

    if not binding:
        binding.append(
            PositionSizingConstraint
            .QUANTITY_STEP
        )

    return FixedNotionalSizingDecision(
        mode=policy.mode,
        portfolio_currency=(
            policy.portfolio_currency
        ),
        quote_currency=(
            request.quote_currency
        ),
        quote_to_portfolio_rate=(
            request
            .quote_to_portfolio_rate
        ),
        quantity_step=(
            request.quantity_step
        ),
        quantity=quantity,
        order_notional_quote=(
            order_notional_quote
        ),
        order_notional_portfolio=(
            order_notional_portfolio
        ),
        planned_loss_portfolio=(
            planned_loss_portfolio
        ),
        capital_required_portfolio=(
            capital_required_portfolio
        ),
        exposure_before_portfolio=(
            request
            .invested_exposure_portfolio
        ),
        exposure_after_portfolio=(
            exposure_after_portfolio
        ),
        binding_constraints=tuple(binding),
    )

def fixed_notional_policy_from_product_policy(
    product_policy: Mapping[str, object],
) -> FixedNotionalSizingPolicy:
    """Build the P4.1 sizing policy from validated configuration."""

    if not isinstance(
        product_policy,
        Mapping,
    ):
        raise ValueError(
            "product_policy must be a mapping."
        )

    portfolio = product_policy.get(
        "portfolio"
    )

    if not isinstance(
        portfolio,
        Mapping,
    ):
        raise ValueError(
            "product_policy.portfolio must "
            "be a mapping."
        )

    required = {
        "currency",
        "sizing_mode",
        "target_order_value",
        "maximum_order_value",
        "maximum_planned_loss",
        "maximum_open_positions",
        "maximum_invested_exposure",
    }

    missing = sorted(
        required.difference(
            str(key)
            for key in portfolio
        )
    )

    if missing:
        raise ValueError(
            "product_policy.portfolio is "
            "missing: "
            + ", ".join(missing)
            + "."
        )

    try:
        mode = PositionSizingMode(
            str(
                portfolio[
                    "sizing_mode"
                ]
            )
        )
    except ValueError as exc:
        raise ValueError(
            "Unsupported product-policy "
            "sizing mode."
        ) from exc

    return FixedNotionalSizingPolicy(
        mode=mode,
        portfolio_currency=str(
            portfolio["currency"]
        ),
        target_order_value=(
            portfolio[
                "target_order_value"
            ]
        ),
        maximum_order_value=(
            portfolio[
                "maximum_order_value"
            ]
        ),
        maximum_planned_loss=(
            portfolio[
                "maximum_planned_loss"
            ]
        ),
        maximum_open_positions=(
            portfolio[
                "maximum_open_positions"
            ]
        ),
        maximum_invested_exposure=(
            portfolio[
                "maximum_invested_exposure"
            ]
        ),
    )
