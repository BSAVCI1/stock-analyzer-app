from decimal import Decimal

import pytest

from src.product_config import (
    load_product_policy,
)
from src.paper.sizing import (
    FixedNotionalSizingPolicy,
    FixedNotionalSizingRequest,
    PositionSizingConstraint,
    PositionSizingMode,
    PositionSizingRejected,
    calculate_fixed_notional_size,
    fixed_notional_policy_from_product_policy,
)


def request(
    *,
    quote_currency="EUR",
    entry_price_quote="25",
    stop_price_quote="22",
    quote_to_portfolio_rate="1",
    available_cash_portfolio="2000",
    invested_exposure_portfolio="0",
    current_position_count=0,
    estimated_entry_fee_portfolio="0",
    estimated_exit_fee_portfolio="0",
    quantity_step="1",
):
    return FixedNotionalSizingRequest(
        quote_currency=quote_currency,
        entry_price_quote=entry_price_quote,
        stop_price_quote=stop_price_quote,
        quote_to_portfolio_rate=(
            quote_to_portfolio_rate
        ),
        available_cash_portfolio=(
            available_cash_portfolio
        ),
        invested_exposure_portfolio=(
            invested_exposure_portfolio
        ),
        current_position_count=(
            current_position_count
        ),
        estimated_entry_fee_portfolio=(
            estimated_entry_fee_portfolio
        ),
        estimated_exit_fee_portfolio=(
            estimated_exit_fee_portfolio
        ),
        quantity_step=quantity_step,
    )


def test_default_policy_matches_p4_1() -> None:
    policy = FixedNotionalSizingPolicy()

    assert policy.mode is (
        PositionSizingMode
        .FIXED_NOTIONAL_WITH_RISK_CAP
    )

    assert policy.portfolio_currency == "EUR"

    assert policy.target_order_value == Decimal(
        "100.00000000"
    )

    assert policy.maximum_order_value == Decimal(
        "100.00000000"
    )

    assert policy.maximum_planned_loss == Decimal(
        "10.00000000"
    )

    assert policy.maximum_open_positions == 5

    assert (
        policy.maximum_invested_exposure
        == Decimal("500.00000000")
    )


def test_exact_target_notional_is_allowed() -> None:
    decision = calculate_fixed_notional_size(
        request()
    )

    assert decision.quantity == Decimal(
        "3.00000000"
    )

    assert (
        decision.order_notional_portfolio
        == Decimal("75.00000000")
    )

    assert (
        decision.planned_loss_portfolio
        == Decimal("9.00000000")
    )


def test_whole_share_rounding_never_exceeds_ceiling() -> None:
    decision = calculate_fixed_notional_size(
        request(
            entry_price_quote="60",
            stop_price_quote="55",
        )
    )

    assert decision.quantity == Decimal(
        "1.00000000"
    )

    assert (
        decision.order_notional_portfolio
        == Decimal("60.00000000")
    )

    assert (
        decision.order_notional_portfolio
        <= Decimal("100")
    )

    assert (
        PositionSizingConstraint
        .TARGET_NOTIONAL
        in decision.binding_constraints
    )


def test_planned_loss_cap_reduces_order() -> None:
    decision = calculate_fixed_notional_size(
        request(
            entry_price_quote="20",
            stop_price_quote="15",
        )
    )

    assert decision.quantity == Decimal(
        "2.00000000"
    )

    assert (
        decision.planned_loss_portfolio
        == Decimal("10.00000000")
    )

    assert (
        PositionSizingConstraint
        .PLANNED_LOSS
        in decision.binding_constraints
    )


def test_usd_quote_is_converted_to_eur() -> None:
    decision = calculate_fixed_notional_size(
        request(
            quote_currency="USD",
            entry_price_quote="50",
            stop_price_quote="47",
            quote_to_portfolio_rate="0.90",
        )
    )

    assert decision.quote_currency == "USD"

    assert (
        decision.quote_to_portfolio_rate
        == Decimal("0.90000000")
    )

    assert decision.quantity == Decimal(
        "2.00000000"
    )

    assert (
        decision.order_notional_quote
        == Decimal("100.00000000")
    )

    assert (
        decision.order_notional_portfolio
        == Decimal("90.00000000")
    )


def test_exposure_cap_reduces_order() -> None:
    decision = calculate_fixed_notional_size(
        request(
            entry_price_quote="20",
            stop_price_quote="19",
            invested_exposure_portfolio="450",
        )
    )

    assert decision.quantity == Decimal(
        "2.00000000"
    )

    assert (
        decision.order_notional_portfolio
        == Decimal("40.00000000")
    )

    assert (
        decision.exposure_after_portfolio
        == Decimal("490.00000000")
    )

    assert (
        PositionSizingConstraint
        .INVESTED_EXPOSURE
        in decision.binding_constraints
    )


def test_available_cash_reduces_order() -> None:
    decision = calculate_fixed_notional_size(
        request(
            entry_price_quote="20",
            stop_price_quote="19",
            available_cash_portfolio="55",
            estimated_entry_fee_portfolio="5",
        )
    )

    assert decision.quantity == Decimal(
        "2.00000000"
    )

    assert (
        decision.capital_required_portfolio
        == Decimal("45.00000000")
    )

    assert (
        PositionSizingConstraint
        .AVAILABLE_CASH
        in decision.binding_constraints
    )


def test_fractional_step_supports_smaller_order() -> None:
    decision = calculate_fixed_notional_size(
        request(
            entry_price_quote="120",
            stop_price_quote="115",
            quantity_step="0.1",
        )
    )

    assert decision.quantity == Decimal(
        "0.80000000"
    )

    assert (
        decision.order_notional_portfolio
        == Decimal("96.00000000")
    )

    assert (
        decision.order_notional_portfolio
        <= Decimal("100")
    )


def test_position_cap_rejects_new_order() -> None:
    with pytest.raises(
        PositionSizingRejected,
        match="open-position limit",
    ):
        calculate_fixed_notional_size(
            request(
                current_position_count=5,
            )
        )


def test_fees_are_included_in_planned_loss() -> None:
    decision = calculate_fixed_notional_size(
        request(
            entry_price_quote="20",
            stop_price_quote="16",
            estimated_entry_fee_portfolio="1",
            estimated_exit_fee_portfolio="1",
        )
    )

    assert decision.quantity == Decimal(
        "2.00000000"
    )

    assert (
        decision.planned_loss_portfolio
        == Decimal("10.00000000")
    )


def test_rejects_when_one_whole_share_breaks_risk_cap() -> None:
    with pytest.raises(
        PositionSizingRejected,
        match="No positive quantity step",
    ):
        calculate_fixed_notional_size(
            request(
                entry_price_quote="100",
                stop_price_quote="80",
            )
        )

def test_versioned_product_policy_builds_sizing_policy() -> None:
    policy = fixed_notional_policy_from_product_policy(
        load_product_policy()
    )

    assert policy.mode is (
        PositionSizingMode
        .FIXED_NOTIONAL_WITH_RISK_CAP
    )

    assert policy.portfolio_currency == "EUR"

    assert (
        policy.target_order_value
        == Decimal("100.00000000")
    )

    assert (
        policy.maximum_order_value
        == Decimal("100.00000000")
    )

    assert (
        policy.maximum_planned_loss
        == Decimal("10.00000000")
    )

    assert policy.maximum_open_positions == 5

    assert (
        policy.maximum_invested_exposure
        == Decimal("500.00000000")
    )
