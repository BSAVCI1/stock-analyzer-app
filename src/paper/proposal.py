"""Cost-aware P4.6 paper-order proposals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from src.costs import (
    IBKREconomicDecision,
    IBKRLongTradeEconomics,
    IBKRPricingPlan,
    calculate_us_long_trade_economics,
)

from .models import aware_datetime, money, positive_money
from .sizing import (
    FixedNotionalSizingDecision,
    FixedNotionalSizingPolicy,
    FixedNotionalSizingRequest,
    calculate_fixed_notional_size,
)


class OrderProposalRejected(ValueError):
    """Raised when a safe paper order cannot be proposed."""


@dataclass(frozen=True, slots=True)
class CostAwareOrderProposalRequest:
    """Complete inputs for one US long paper-order proposal."""

    symbol: str
    quote_currency: str
    entry_price: Decimal
    stop_price: Decimal
    target_price: Decimal
    expires_at: datetime
    proposed_at: datetime
    quote_to_portfolio_rate: Decimal
    available_cash_portfolio: Decimal
    invested_exposure_portfolio: Decimal
    current_position_count: int
    minimum_net_reward_to_risk: Decimal
    quantity_step: Decimal = Decimal("0.0001")
    fractional_eligible: bool = True

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()

        if not symbol:
            raise ValueError("symbol is required.")

        currency = str(
            self.quote_currency
        ).strip().upper()

        if currency != "USD":
            raise ValueError(
                "P4.6.1 supports USD-quoted "
                "instruments only."
            )

        entry = positive_money(
            "entry_price",
            self.entry_price,
        )
        stop = positive_money(
            "stop_price",
            self.stop_price,
        )
        target = positive_money(
            "target_price",
            self.target_price,
        )

        if stop >= entry:
            raise ValueError(
                "stop_price must be below "
                "entry_price."
            )

        if target <= entry:
            raise ValueError(
                "target_price must be above "
                "entry_price."
            )

        proposed_at = aware_datetime(
            "proposed_at",
            self.proposed_at,
        )
        expires_at = aware_datetime(
            "expires_at",
            self.expires_at,
        )

        if expires_at <= proposed_at:
            raise ValueError(
                "expires_at must be after "
                "proposed_at."
            )

        if not isinstance(
            self.fractional_eligible,
            bool,
        ):
            raise ValueError(
                "fractional_eligible must "
                "be boolean."
            )

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(
            self,
            "quote_currency",
            currency,
        )
        object.__setattr__(
            self,
            "entry_price",
            entry,
        )
        object.__setattr__(
            self,
            "stop_price",
            stop,
        )
        object.__setattr__(
            self,
            "target_price",
            target,
        )
        object.__setattr__(
            self,
            "proposed_at",
            proposed_at,
        )
        object.__setattr__(
            self,
            "expires_at",
            expires_at,
        )
        object.__setattr__(
            self,
            "quote_to_portfolio_rate",
            positive_money(
                "quote_to_portfolio_rate",
                self.quote_to_portfolio_rate,
            ),
        )
        object.__setattr__(
            self,
            "minimum_net_reward_to_risk",
            positive_money(
                "minimum_net_reward_to_risk",
                self.minimum_net_reward_to_risk,
            ),
        )
        object.__setattr__(
            self,
            "quantity_step",
            positive_money(
                "quantity_step",
                self.quantity_step,
            ),
        )


@dataclass(frozen=True, slots=True)
class CostAwareOrderProposal:
    """Auditable, non-executable paper-order proposal."""

    symbol: str
    quote_currency: str
    portfolio_currency: str
    quantity: Decimal
    entry_price: Decimal
    stop_price: Decimal
    target_price: Decimal
    proposed_at: datetime
    expires_at: datetime
    capital_required_portfolio: Decimal
    planned_loss_portfolio: Decimal
    net_reward_to_risk: Decimal
    sizing: FixedNotionalSizingDecision
    economics: IBKRLongTradeEconomics


def _economics(
    request: CostAwareOrderProposalRequest,
    quantity: Decimal,
) -> IBKRLongTradeEconomics:
    return calculate_us_long_trade_economics(
        quantity=quantity,
        entry_price_usd=request.entry_price,
        stop_price_usd=request.stop_price,
        target_price_usd=request.target_price,
        usd_to_portfolio_rate=(
            request.quote_to_portfolio_rate
        ),
        pricing_plan=IBKRPricingPlan.FIXED,
        minimum_net_reward_to_risk=(
            request.minimum_net_reward_to_risk
        ),
        fractional=(
            request.fractional_eligible
            and quantity % Decimal("1") != 0
        ),
        include_entry_fx_conversion=False,
        include_exit_fx_conversion=False,
    )


def propose_cost_aware_us_long_order(
    request: CostAwareOrderProposalRequest,
    *,
    policy: FixedNotionalSizingPolicy
    | None = None,
) -> CostAwareOrderProposal:
    """Size and economically validate a paper-only proposal."""

    if not isinstance(
        request,
        CostAwareOrderProposalRequest,
    ):
        raise ValueError(
            "request must be a "
            "CostAwareOrderProposalRequest."
        )

    resolved_policy = (
        policy
        or FixedNotionalSizingPolicy()
    )
    entry_fee = Decimal("0")
    exit_fee = Decimal("0")
    sizing = None

    for _ in range(8):
        sizing = calculate_fixed_notional_size(
            FixedNotionalSizingRequest(
                quote_currency=(
                    request.quote_currency
                ),
                entry_price_quote=(
                    request.entry_price
                ),
                stop_price_quote=(
                    request.stop_price
                ),
                quote_to_portfolio_rate=(
                    request
                    .quote_to_portfolio_rate
                ),
                available_cash_portfolio=(
                    request
                    .available_cash_portfolio
                ),
                invested_exposure_portfolio=(
                    request
                    .invested_exposure_portfolio
                ),
                current_position_count=(
                    request
                    .current_position_count
                ),
                estimated_entry_fee_portfolio=(
                    entry_fee
                ),
                estimated_exit_fee_portfolio=(
                    exit_fee
                ),
                quantity_step=(
                    request.quantity_step
                ),
            ),
            policy=resolved_policy,
        )
        economics = _economics(
            request,
            sizing.quantity,
        )
        next_entry_fee = money(
            economics.entry_stock_cost_usd
            * request.quote_to_portfolio_rate
        )
        next_exit_fee = money(
            economics.stop_exit_stock_cost_usd
            * request.quote_to_portfolio_rate
        )

        if (
            next_entry_fee == entry_fee
            and next_exit_fee == exit_fee
        ):
            break

        entry_fee = next_entry_fee
        exit_fee = next_exit_fee
    else:
        raise OrderProposalRejected(
            "Cost-aware sizing did not converge."
        )

    assert sizing is not None
    economics = _economics(
        request,
        sizing.quantity,
    )

    if (
        economics.decision
        is not IBKREconomicDecision.ACCEPT
    ):
        raise OrderProposalRejected(
            "Order is not economical after "
            "modeled costs: "
            + economics.decision.value
            + "."
        )

    return CostAwareOrderProposal(
        symbol=request.symbol,
        quote_currency=(
            request.quote_currency
        ),
        portfolio_currency=(
            resolved_policy.portfolio_currency
        ),
        quantity=sizing.quantity,
        entry_price=request.entry_price,
        stop_price=request.stop_price,
        target_price=request.target_price,
        proposed_at=request.proposed_at,
        expires_at=request.expires_at,
        capital_required_portfolio=(
            sizing.capital_required_portfolio
        ),
        planned_loss_portfolio=(
            sizing.planned_loss_portfolio
        ),
        net_reward_to_risk=(
            economics.net_reward_to_risk
        ),
        sizing=sizing,
        economics=economics,
    )
