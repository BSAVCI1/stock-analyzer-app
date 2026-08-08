"""P4.2 broker-disconnected IBKR reference-cost tests."""

from copy import deepcopy
from decimal import Decimal

import pytest

from src.costs import (
    IBKRCostProfileError,
    IBKREconomicDecision,
    IBKRFXMode,
    IBKRPricingPlan,
    IBKRTradeSide,
    calculate_us_long_trade_economics,
    calculate_europe_eur_reference_fees,
    calculate_fx_reference_cost,
    calculate_net_reward_to_risk,
    calculate_us_stock_commission,
    calculate_us_stock_reference_fees,
    load_ibkr_reference_profile,
    validate_ibkr_reference_profile,
)


def test_reference_profile_is_disconnected_and_unselected() -> None:
    profile = (
        load_ibkr_reference_profile()
    )

    assert (
        profile["provider"]
        == "IBKR"
    )

    assert (
        profile[
            "api_connection_enabled"
        ]
        is False
    )

    assert (
        profile[
            "active_pricing_plan"
        ]
        is None
    )

    assert (
        profile[
            "active_fx_mode"
        ]
        is None
    )


def test_profile_rejects_sensitive_keys() -> None:
    profile = deepcopy(
        load_ibkr_reference_profile()
    )

    profile["api_token"] = (
        "must-never-be-valid"
    )

    with pytest.raises(
        IBKRCostProfileError,
        match="Sensitive",
    ):
        validate_ibkr_reference_profile(
            profile
        )


def test_us_tiered_whole_share_commission_minimum() -> None:
    commission = (
        calculate_us_stock_commission(
            quantity="10",
            trade_value_usd="100",
            pricing_plan="TIERED",
        )
    )

    assert commission == Decimal(
        "0.35000000"
    )


def test_us_fixed_whole_share_commission_minimum() -> None:
    commission = (
        calculate_us_stock_commission(
            quantity="10",
            trade_value_usd="100",
            pricing_plan="FIXED",
        )
    )

    assert commission == Decimal(
        "1.00000000"
    )


def test_us_fractional_commission_uses_one_percent_rule() -> None:
    commission = (
        calculate_us_stock_commission(
            quantity="0.5",
            trade_value_usd="5",
            pricing_plan="TIERED",
            fractional=True,
        )
    )

    assert commission == Decimal(
        "0.05000000"
    )


def test_us_tiered_buy_separates_route_dependent_fees() -> None:
    estimate = (
        calculate_us_stock_reference_fees(
            quantity="10",
            trade_value_usd="100",
            pricing_plan=(
                IBKRPricingPlan.TIERED
            ),
            side=IBKRTradeSide.BUY,
        )
    )

    assert estimate.commission == Decimal(
        "0.35000000"
    )

    assert (
        estimate.regulatory_fees
        == Decimal("0.00003000")
    )

    assert (
        estimate.clearing_fees
        == Decimal("0.00200000")
    )

    assert (
        estimate.total_known_cost
        == Decimal("0.35203000")
    )

    assert estimate.complete is False


def test_us_tiered_becomes_complete_with_route_fee() -> None:
    estimate = (
        calculate_us_stock_reference_fees(
            quantity="10",
            trade_value_usd="100",
            pricing_plan="TIERED",
            side="BUY",
            route_dependent_fee_usd="0.01",
        )
    )

    assert (
        estimate.total_known_cost
        == Decimal("0.36203000")
    )

    assert estimate.complete is True


def test_us_fixed_sell_includes_known_regulatory_fees() -> None:
    estimate = (
        calculate_us_stock_reference_fees(
            quantity="10",
            trade_value_usd="100",
            pricing_plan="FIXED",
            side="SELL",
        )
    )

    assert estimate.commission == Decimal(
        "1.00000000"
    )

    assert (
        estimate.regulatory_fees
        == Decimal("0.00404000")
    )

    assert (
        estimate.clearing_fees
        == Decimal("0.00000000")
    )

    assert (
        estimate.total_known_cost
        == Decimal("1.00404000")
    )

    assert estimate.complete is True


def test_europe_reference_shows_eur_100_minimum_effect() -> None:
    tiered = (
        calculate_europe_eur_reference_fees(
            trade_value_eur="100",
            pricing_plan="TIERED",
        )
    )

    fixed = (
        calculate_europe_eur_reference_fees(
            trade_value_eur="100",
            pricing_plan="FIXED",
        )
    )

    assert tiered.commission == Decimal(
        "1.25000000"
    )

    assert fixed.commission == Decimal(
        "3.00000000"
    )

    # Europe is deliberately not marked
    # complete until venue-specific rules
    # are selected.
    assert tiered.complete is False
    assert fixed.complete is False


def test_spot_fx_minimum_dominates_small_conversion() -> None:
    estimate = (
        calculate_fx_reference_cost(
            trade_value_usd="100",
            mode=IBKRFXMode.SPOT_FX,
        )
    )

    assert (
        estimate.estimated_cost
        == Decimal("2.00000000")
    )

    assert (
        estimate.separate_commission
        is True
    )


def test_auto_conversion_reference_cost_is_three_basis_points() -> None:
    estimate = (
        calculate_fx_reference_cost(
            trade_value_usd="100",
            mode="AUTO_CONVERSION",
        )
    )

    assert (
        estimate.estimated_cost
        == Decimal("0.03000000")
    )

    assert (
        estimate.separate_commission
        is False
    )


def test_cost_adjusted_reward_to_risk_matches_hand_calculation() -> None:
    result = (
        calculate_net_reward_to_risk(
            gross_reward_portfolio="20",
            gross_risk_portfolio="10",
            round_trip_cost_portfolio="2",
        )
    )

    assert result.net_reward == Decimal(
        "18.00000000"
    )

    assert (
        result.cost_adjusted_risk
        == Decimal("12.00000000")
    )

    assert (
        result.net_reward_to_risk
        == Decimal("1.50000000")
    )

def test_cost_adjusted_trade_matches_hand_calculation() -> None:
    result = (
        calculate_us_long_trade_economics(
            quantity="10",
            entry_price_usd="10",
            stop_price_usd="9",
            target_price_usd="13",
            usd_to_portfolio_rate="0.90",
            pricing_plan="FIXED",
            minimum_net_reward_to_risk="2",
            fx_mode="AUTO_CONVERSION",
            include_entry_fx_conversion=True,
            include_exit_fx_conversion=True,
        )
    )

    assert (
        result.gross_reward_portfolio
        == Decimal("27.00000000")
    )

    assert (
        result.gross_risk_portfolio
        == Decimal("9.00000000")
    )

    assert (
        result.reward_path_cost_portfolio
        == Decimal("1.86631920")
    )

    assert (
        result.risk_path_cost_portfolio
        == Decimal("1.76477760")
    )

    assert (
        result.net_reward_portfolio
        == Decimal("25.13368080")
    )

    assert (
        result.cost_adjusted_risk_portfolio
        == Decimal("10.76477760")
    )

    assert (
        result.gross_reward_to_risk
        == Decimal("3.00000000")
    )

    assert (
        result.net_reward_to_risk
        == Decimal("2.33480725")
    )

    assert (
        result.decision
        is IBKREconomicDecision.ACCEPT
    )

    assert result.complete is True


def test_cost_adjusted_trade_rejects_grossly_good_but_net_uneconomic_trade() -> None:
    result = (
        calculate_us_long_trade_economics(
            quantity="11",
            entry_price_usd="10",
            stop_price_usd="9.5",
            target_price_usd="11.5",
            usd_to_portfolio_rate="0.90",
            pricing_plan="FIXED",
            minimum_net_reward_to_risk="2",
            fx_mode="AUTO_CONVERSION",
            include_entry_fx_conversion=True,
            include_exit_fx_conversion=True,
        )
    )

    # Entry notional is USD 110 =
    # EUR 99 at the supplied FX rate.
    assert (
        result.entry_notional_usd
        == Decimal("110.00000000")
    )

    assert (
        result.gross_reward_to_risk
        == Decimal("3.00000000")
    )

    assert (
        result.net_reward_to_risk
        == Decimal("1.90577074")
    )

    assert (
        result.decision
        is IBKREconomicDecision
        .UNECONOMIC_AFTER_COSTS
    )

    assert result.complete is True


def test_tiered_trade_fails_closed_when_route_cost_is_unknown() -> None:
    result = (
        calculate_us_long_trade_economics(
            quantity="10",
            entry_price_usd="10",
            stop_price_usd="9",
            target_price_usd="13",
            usd_to_portfolio_rate="0.90",
            pricing_plan="TIERED",
            minimum_net_reward_to_risk="2",
        )
    )

    assert result.complete is False

    assert (
        result.decision
        is IBKREconomicDecision
        .INCOMPLETE_COST_ESTIMATE
    )


def test_tiered_trade_can_be_complete_with_explicit_route_costs() -> None:
    result = (
        calculate_us_long_trade_economics(
            quantity="10",
            entry_price_usd="10",
            stop_price_usd="9",
            target_price_usd="13",
            usd_to_portfolio_rate="0.90",
            pricing_plan="TIERED",
            minimum_net_reward_to_risk="2",
            entry_route_dependent_fee_usd="0.01",
            stop_exit_route_dependent_fee_usd="0.01",
            target_exit_route_dependent_fee_usd="0.01",
        )
    )

    assert result.complete is True

    assert (
        result.decision
        is IBKREconomicDecision.ACCEPT
    )


def test_fx_cost_cannot_be_silently_assumed() -> None:
    with pytest.raises(
        ValueError,
        match="fx_mode is required",
    ):
        calculate_us_long_trade_economics(
            quantity="10",
            entry_price_usd="10",
            stop_price_usd="9",
            target_price_usd="13",
            usd_to_portfolio_rate="0.90",
            pricing_plan="FIXED",
            minimum_net_reward_to_risk="2",
            include_entry_fx_conversion=True,
        )
