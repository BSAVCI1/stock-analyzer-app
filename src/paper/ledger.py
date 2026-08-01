"""Paper-account cash and trade calculations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .models import money


@dataclass(frozen=True, slots=True)
class LongTradeCalculation:
    entry_notional: Decimal
    exit_notional: Decimal
    gross_pnl: Decimal
    total_fees: Decimal
    total_slippage: Decimal
    net_pnl: Decimal
    cash_proceeds: Decimal
    return_pct: float


def calculate_entry_cash(
    price: object,
    quantity: object,
    fees: object = 0,
) -> Decimal:
    return money(
        money(price) * money(quantity)
        + money(fees)
    )


def calculate_long_trade(
    *,
    entry_price: object,
    exit_price: object,
    quantity: object,
    entry_fees: object = 0,
    exit_fees: object = 0,
    entry_slippage: object = 0,
    exit_slippage: object = 0,
) -> LongTradeCalculation:
    entry = money(entry_price)
    exit_value = money(exit_price)
    qty = money(quantity)

    entry_notional = money(
        entry * qty
    )

    exit_notional = money(
        exit_value * qty
    )

    gross_pnl = money(
        exit_notional
        - entry_notional
    )

    total_fees = money(
        money(entry_fees)
        + money(exit_fees)
    )

    total_slippage = money(
        money(entry_slippage)
        + money(exit_slippage)
    )

    # Slippage is already reflected in the executed entry and exit prices.
    net_pnl = money(
        gross_pnl
        - total_fees
    )

    cash_proceeds = money(
        exit_notional
        - money(exit_fees)
    )

    return_pct = (
        float(
            net_pnl / entry_notional
        )
        if entry_notional > 0
        else 0.0
    )

    return LongTradeCalculation(
        entry_notional=entry_notional,
        exit_notional=exit_notional,
        gross_pnl=gross_pnl,
        total_fees=total_fees,
        total_slippage=total_slippage,
        net_pnl=net_pnl,
        cash_proceeds=cash_proceeds,
        return_pct=return_pct,
    )
