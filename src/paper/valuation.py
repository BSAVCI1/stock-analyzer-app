"""Portfolio-currency economics for multi-currency paper trading."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .fx import QuoteToPortfolioFXRate
from .models import money


def _require_fx_rate(
    name: str,
    value: object,
) -> QuoteToPortfolioFXRate:
    if not isinstance(
        value,
        QuoteToPortfolioFXRate,
    ):
        raise ValueError(
            f"{name} must be a "
            "QuoteToPortfolioFXRate."
        )

    return value


@dataclass(frozen=True, slots=True)
class PortfolioEntryCash:
    """Entry economics expressed in both quote and portfolio currency."""

    quote_currency: str
    portfolio_currency: str

    entry_notional_quote: Decimal
    entry_fee_quote: Decimal

    entry_notional_portfolio: Decimal
    entry_fee_portfolio: Decimal
    cash_required_portfolio: Decimal

    quote_to_portfolio_rate: Decimal

    def __post_init__(self) -> None:
        for name in (
            "entry_notional_quote",
            "entry_fee_quote",
            "entry_notional_portfolio",
            "entry_fee_portfolio",
            "cash_required_portfolio",
            "quote_to_portfolio_rate",
        ):
            object.__setattr__(
                self,
                name,
                money(
                    getattr(self, name)
                ),
            )


@dataclass(frozen=True, slots=True)
class PortfolioLongTradeCalculation:
    """Closed long-trade economics in portfolio currency."""

    quote_currency: str
    portfolio_currency: str

    entry_notional_quote: Decimal
    exit_notional_quote: Decimal
    gross_pnl_quote: Decimal

    entry_notional_portfolio: Decimal
    exit_notional_portfolio: Decimal
    gross_pnl_portfolio: Decimal

    entry_fee_portfolio: Decimal
    exit_fee_portfolio: Decimal
    total_fees_portfolio: Decimal

    entry_slippage_portfolio: Decimal
    exit_slippage_portfolio: Decimal
    total_slippage_portfolio: Decimal

    net_pnl_portfolio: Decimal
    cash_proceeds_portfolio: Decimal

    entry_quote_to_portfolio_rate: Decimal
    exit_quote_to_portfolio_rate: Decimal

    return_pct: float

    def __post_init__(self) -> None:
        decimal_fields = (
            "entry_notional_quote",
            "exit_notional_quote",
            "gross_pnl_quote",
            "entry_notional_portfolio",
            "exit_notional_portfolio",
            "gross_pnl_portfolio",
            "entry_fee_portfolio",
            "exit_fee_portfolio",
            "total_fees_portfolio",
            "entry_slippage_portfolio",
            "exit_slippage_portfolio",
            "total_slippage_portfolio",
            "net_pnl_portfolio",
            "cash_proceeds_portfolio",
            "entry_quote_to_portfolio_rate",
            "exit_quote_to_portfolio_rate",
        )

        for name in decimal_fields:
            object.__setattr__(
                self,
                name,
                money(
                    getattr(self, name)
                ),
            )


def calculate_entry_cash_portfolio(
    *,
    price_quote: object,
    quantity: object,
    fee_quote: object = 0,
    fx_rate: QuoteToPortfolioFXRate,
) -> PortfolioEntryCash:
    """Calculate entry cash required in account portfolio currency."""

    rate = _require_fx_rate(
        "fx_rate",
        fx_rate,
    )

    price = money(price_quote)
    qty = money(quantity)
    fee = money(fee_quote)

    if price <= 0:
        raise ValueError(
            "price_quote must be positive."
        )

    if qty <= 0:
        raise ValueError(
            "quantity must be positive."
        )

    if fee < 0:
        raise ValueError(
            "fee_quote cannot be negative."
        )

    entry_notional_quote = money(
        price * qty
    )

    entry_notional_portfolio = (
        rate.convert_quote_to_portfolio(
            entry_notional_quote
        )
    )

    entry_fee_portfolio = (
        rate.convert_quote_to_portfolio(
            fee
        )
    )

    cash_required = money(
        entry_notional_portfolio
        + entry_fee_portfolio
    )

    return PortfolioEntryCash(
        quote_currency=rate.quote_currency,
        portfolio_currency=(
            rate.portfolio_currency
        ),
        entry_notional_quote=(
            entry_notional_quote
        ),
        entry_fee_quote=fee,
        entry_notional_portfolio=(
            entry_notional_portfolio
        ),
        entry_fee_portfolio=(
            entry_fee_portfolio
        ),
        cash_required_portfolio=(
            cash_required
        ),
        quote_to_portfolio_rate=(
            rate.rate
        ),
    )


def calculate_market_value_portfolio(
    *,
    price_quote: object,
    quantity: object,
    fx_rate: QuoteToPortfolioFXRate,
) -> Decimal:
    """Convert an instrument market value to portfolio currency."""

    rate = _require_fx_rate(
        "fx_rate",
        fx_rate,
    )

    price = money(price_quote)
    qty = money(quantity)

    if price < 0:
        raise ValueError(
            "price_quote cannot be negative."
        )

    if qty < 0:
        raise ValueError(
            "quantity cannot be negative."
        )

    quote_value = money(
        price * qty
    )

    return (
        rate.convert_quote_to_portfolio(
            quote_value
        )
    )


def calculate_long_trade_portfolio(
    *,
    entry_price_quote: object,
    exit_price_quote: object,
    quantity: object,
    entry_fee_quote: object = 0,
    exit_fee_quote: object = 0,
    entry_slippage_quote: object = 0,
    exit_slippage_quote: object = 0,
    entry_fx_rate: QuoteToPortfolioFXRate,
    exit_fx_rate: QuoteToPortfolioFXRate,
) -> PortfolioLongTradeCalculation:
    """Calculate a long trade with entry and exit FX conversion."""

    entry_fx = _require_fx_rate(
        "entry_fx_rate",
        entry_fx_rate,
    )

    exit_fx = _require_fx_rate(
        "exit_fx_rate",
        exit_fx_rate,
    )

    if (
        entry_fx.quote_currency
        != exit_fx.quote_currency
    ):
        raise ValueError(
            "Entry and exit FX rates must "
            "use the same quote currency."
        )

    if (
        entry_fx.portfolio_currency
        != exit_fx.portfolio_currency
    ):
        raise ValueError(
            "Entry and exit FX rates must "
            "use the same portfolio currency."
        )

    entry_price = money(
        entry_price_quote
    )

    exit_price = money(
        exit_price_quote
    )

    qty = money(quantity)

    entry_fee = money(
        entry_fee_quote
    )

    exit_fee = money(
        exit_fee_quote
    )

    entry_slippage = money(
        entry_slippage_quote
    )

    exit_slippage = money(
        exit_slippage_quote
    )

    if entry_price <= 0:
        raise ValueError(
            "entry_price_quote must be positive."
        )

    if exit_price <= 0:
        raise ValueError(
            "exit_price_quote must be positive."
        )

    if qty <= 0:
        raise ValueError(
            "quantity must be positive."
        )

    for name, value in (
        ("entry_fee_quote", entry_fee),
        ("exit_fee_quote", exit_fee),
        (
            "entry_slippage_quote",
            entry_slippage,
        ),
        (
            "exit_slippage_quote",
            exit_slippage,
        ),
    ):
        if value < 0:
            raise ValueError(
                f"{name} cannot be negative."
            )

    entry_notional_quote = money(
        entry_price * qty
    )

    exit_notional_quote = money(
        exit_price * qty
    )

    gross_pnl_quote = money(
        exit_notional_quote
        - entry_notional_quote
    )

    entry_notional_portfolio = (
        entry_fx.convert_quote_to_portfolio(
            entry_notional_quote
        )
    )

    exit_notional_portfolio = (
        exit_fx.convert_quote_to_portfolio(
            exit_notional_quote
        )
    )

    gross_pnl_portfolio = money(
        exit_notional_portfolio
        - entry_notional_portfolio
    )

    entry_fee_portfolio = (
        entry_fx.convert_quote_to_portfolio(
            entry_fee
        )
    )

    exit_fee_portfolio = (
        exit_fx.convert_quote_to_portfolio(
            exit_fee
        )
    )

    total_fees_portfolio = money(
        entry_fee_portfolio
        + exit_fee_portfolio
    )

    entry_slippage_portfolio = (
        entry_fx.convert_quote_to_portfolio(
            entry_slippage
        )
    )

    exit_slippage_portfolio = (
        exit_fx.convert_quote_to_portfolio(
            exit_slippage
        )
    )

    total_slippage_portfolio = money(
        entry_slippage_portfolio
        + exit_slippage_portfolio
    )

    # Executed prices already include slippage.
    # Therefore slippage is reported but not
    # deducted a second time from P&L.
    net_pnl_portfolio = money(
        gross_pnl_portfolio
        - total_fees_portfolio
    )

    cash_proceeds_portfolio = money(
        exit_notional_portfolio
        - exit_fee_portfolio
    )

    return_pct = (
        float(
            net_pnl_portfolio
            / entry_notional_portfolio
        )
        if entry_notional_portfolio > 0
        else 0.0
    )

    return PortfolioLongTradeCalculation(
        quote_currency=(
            entry_fx.quote_currency
        ),
        portfolio_currency=(
            entry_fx.portfolio_currency
        ),
        entry_notional_quote=(
            entry_notional_quote
        ),
        exit_notional_quote=(
            exit_notional_quote
        ),
        gross_pnl_quote=gross_pnl_quote,
        entry_notional_portfolio=(
            entry_notional_portfolio
        ),
        exit_notional_portfolio=(
            exit_notional_portfolio
        ),
        gross_pnl_portfolio=(
            gross_pnl_portfolio
        ),
        entry_fee_portfolio=(
            entry_fee_portfolio
        ),
        exit_fee_portfolio=(
            exit_fee_portfolio
        ),
        total_fees_portfolio=(
            total_fees_portfolio
        ),
        entry_slippage_portfolio=(
            entry_slippage_portfolio
        ),
        exit_slippage_portfolio=(
            exit_slippage_portfolio
        ),
        total_slippage_portfolio=(
            total_slippage_portfolio
        ),
        net_pnl_portfolio=(
            net_pnl_portfolio
        ),
        cash_proceeds_portfolio=(
            cash_proceeds_portfolio
        ),
        entry_quote_to_portfolio_rate=(
            entry_fx.rate
        ),
        exit_quote_to_portfolio_rate=(
            exit_fx.rate
        ),
        return_pct=return_pct,
    )
