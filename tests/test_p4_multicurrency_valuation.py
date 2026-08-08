from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.paper.fx import (
    QuoteToPortfolioFXRate,
    identity_fx_rate,
)
from src.paper.ledger import (
    calculate_entry_cash,
    calculate_long_trade,
)
from src.paper.valuation import (
    calculate_entry_cash_portfolio,
    calculate_long_trade_portfolio,
    calculate_market_value_portfolio,
)


T0 = datetime(
    2026,
    8,
    8,
    10,
    0,
    tzinfo=timezone.utc,
)

T1 = datetime(
    2026,
    8,
    15,
    10,
    0,
    tzinfo=timezone.utc,
)


def eur_per_usd(
    rate: str,
    *,
    as_of=T0,
):
    return QuoteToPortfolioFXRate(
        quote_currency="USD",
        portfolio_currency="EUR",
        rate=Decimal(rate),
        as_of=as_of,
        source="TEST",
    )


def test_cross_currency_entry_cash() -> None:
    result = (
        calculate_entry_cash_portfolio(
            price_quote="100",
            quantity="10",
            fee_quote="1",
            fx_rate=eur_per_usd(
                "0.90"
            ),
        )
    )

    assert result.quote_currency == "USD"
    assert (
        result.portfolio_currency
        == "EUR"
    )

    assert (
        result.entry_notional_quote
        == Decimal("1000.00000000")
    )

    assert (
        result.entry_notional_portfolio
        == Decimal("900.00000000")
    )

    assert (
        result.entry_fee_portfolio
        == Decimal("0.90000000")
    )

    assert (
        result.cash_required_portfolio
        == Decimal("900.90000000")
    )


def test_cross_currency_market_value() -> None:
    value = (
        calculate_market_value_portfolio(
            price_quote="110",
            quantity="3",
            fx_rate=eur_per_usd(
                "0.92"
            ),
        )
    )

    assert value == Decimal(
        "303.60000000"
    )


def test_fx_change_is_reflected_in_portfolio_pnl() -> None:
    result = (
        calculate_long_trade_portfolio(
            entry_price_quote="100",
            exit_price_quote="110",
            quantity="10",
            entry_fee_quote="1",
            exit_fee_quote="1",
            entry_slippage_quote="0.50",
            exit_slippage_quote="0.50",
            entry_fx_rate=eur_per_usd(
                "0.90",
                as_of=T0,
            ),
            exit_fx_rate=eur_per_usd(
                "0.95",
                as_of=T1,
            ),
        )
    )

    assert (
        result.entry_notional_quote
        == Decimal("1000.00000000")
    )

    assert (
        result.exit_notional_quote
        == Decimal("1100.00000000")
    )

    assert (
        result.gross_pnl_quote
        == Decimal("100.00000000")
    )

    assert (
        result.entry_notional_portfolio
        == Decimal("900.00000000")
    )

    assert (
        result.exit_notional_portfolio
        == Decimal("1045.00000000")
    )

    # Includes both security-price movement
    # and the change in USD/EUR conversion.
    assert (
        result.gross_pnl_portfolio
        == Decimal("145.00000000")
    )

    assert (
        result.total_fees_portfolio
        == Decimal("1.85000000")
    )

    assert (
        result.total_slippage_portfolio
        == Decimal("0.92500000")
    )

    assert (
        result.net_pnl_portfolio
        == Decimal("143.15000000")
    )

    assert (
        result.cash_proceeds_portfolio
        == Decimal("1044.05000000")
    )


def test_same_currency_entry_matches_legacy() -> None:
    fx = identity_fx_rate(
        "USD",
        as_of=T0,
    )

    modern = (
        calculate_entry_cash_portfolio(
            price_quote="100",
            quantity="10",
            fee_quote="1.25",
            fx_rate=fx,
        )
    )

    legacy = calculate_entry_cash(
        "100",
        "10",
        "1.25",
    )

    assert (
        modern.cash_required_portfolio
        == legacy
    )


def test_same_currency_trade_matches_legacy() -> None:
    fx = identity_fx_rate(
        "USD",
        as_of=T0,
    )

    modern = (
        calculate_long_trade_portfolio(
            entry_price_quote="100",
            exit_price_quote="110",
            quantity="10",
            entry_fee_quote="1",
            exit_fee_quote="2",
            entry_slippage_quote="0.5",
            exit_slippage_quote="0.75",
            entry_fx_rate=fx,
            exit_fx_rate=identity_fx_rate(
                "USD",
                as_of=T1,
            ),
        )
    )

    legacy = calculate_long_trade(
        entry_price="100",
        exit_price="110",
        quantity="10",
        entry_fees="1",
        exit_fees="2",
        entry_slippage="0.5",
        exit_slippage="0.75",
    )

    assert (
        modern.entry_notional_portfolio
        == legacy.entry_notional
    )

    assert (
        modern.exit_notional_portfolio
        == legacy.exit_notional
    )

    assert (
        modern.gross_pnl_portfolio
        == legacy.gross_pnl
    )

    assert (
        modern.total_fees_portfolio
        == legacy.total_fees
    )

    assert (
        modern.total_slippage_portfolio
        == legacy.total_slippage
    )

    assert (
        modern.net_pnl_portfolio
        == legacy.net_pnl
    )

    assert (
        modern.cash_proceeds_portfolio
        == legacy.cash_proceeds
    )

    assert modern.return_pct == (
        legacy.return_pct
    )


def test_mismatched_quote_currencies_rejected() -> None:
    entry = eur_per_usd(
        "0.90",
        as_of=T0,
    )

    exit_rate = (
        QuoteToPortfolioFXRate(
            quote_currency="GBP",
            portfolio_currency="EUR",
            rate=Decimal("1.15"),
            as_of=T1,
            source="TEST",
        )
    )

    with pytest.raises(
        ValueError,
        match="same quote currency",
    ):
        calculate_long_trade_portfolio(
            entry_price_quote="100",
            exit_price_quote="110",
            quantity="1",
            entry_fx_rate=entry,
            exit_fx_rate=exit_rate,
        )


def test_mismatched_portfolio_currencies_rejected() -> None:
    entry = eur_per_usd(
        "0.90",
        as_of=T0,
    )

    exit_rate = (
        QuoteToPortfolioFXRate(
            quote_currency="USD",
            portfolio_currency="GBP",
            rate=Decimal("0.78"),
            as_of=T1,
            source="TEST",
        )
    )

    with pytest.raises(
        ValueError,
        match="same portfolio currency",
    ):
        calculate_long_trade_portfolio(
            entry_price_quote="100",
            exit_price_quote="110",
            quantity="1",
            entry_fx_rate=entry,
            exit_fx_rate=exit_rate,
        )
