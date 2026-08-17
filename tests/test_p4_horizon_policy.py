from __future__ import annotations

from copy import deepcopy

import pytest

from src.product_config import (
    ProductPolicyError,
    load_product_policy,
    validate_product_policy,
)
from src.strategy import (
    HORIZON_POLICY_VERSION,
    HorizonPolicy,
    StrategyConfirmationPolicy,
    StrategyEntryTiming,
    StrategyExitPolicy,
    StrategyHorizon,
    coerce_strategy_horizon,
    horizon_policies_from_product_policy,
)


EXPECTED_EXITS = (
    StrategyExitPolicy.STOP_LOSS,
    StrategyExitPolicy.TARGET,
    StrategyExitPolicy.TIME_EXIT,
    StrategyExitPolicy.SIGNAL_REVERSAL,
    StrategyExitPolicy.REGIME_INVALIDATION,
    StrategyExitPolicy.PORTFOLIO_RISK,
)


def test_default_policy_has_versioned_independent_horizons(
) -> None:
    policy = load_product_policy()

    assert (
        policy["policy_version"]
        == "p4.3-1"
    )

    strategies = policy[
        "strategies"
    ]

    assert (
        strategies[
            "horizon_policy_version"
        ]
        == HORIZON_POLICY_VERSION
    )

    parsed = (
        horizon_policies_from_product_policy(
            policy
        )
    )

    assert set(parsed) == {
        StrategyHorizon.SWING,
        StrategyHorizon.MEDIUM_TERM,
    }

    swing = parsed[
        StrategyHorizon.SWING
    ]

    assert isinstance(
        swing,
        HorizonPolicy,
    )

    assert (
        swing.strategy_version
        == "p4.3-swing-v1"
    )

    assert (
        swing.market_data_period
        == "2y"
    )

    assert (
        swing.market_data_interval
        == "1d"
    )

    assert (
        swing.signal_validity_sessions
        == 5
    )

    assert (
        swing.maximum_holding_sessions
        == 20
    )

    assert (
        swing.confirmation_policy
        is StrategyConfirmationPolicy
        .STRATEGY_CONFIRMATION
    )

    assert (
        swing.entry_timing
        is StrategyEntryTiming
        .NEXT_ELIGIBLE_SESSION
    )

    assert (
        swing.intraday_entries_allowed
        is False
    )

    assert (
        swing.exit_policies
        == EXPECTED_EXITS
    )

    medium = parsed[
        StrategyHorizon.MEDIUM_TERM
    ]

    assert (
        medium.strategy_version
        == "p4.3-medium-term-v1"
    )

    assert (
        medium.market_data_period
        == "5y"
    )

    assert (
        medium.market_data_interval
        == "1wk"
    )

    assert (
        medium.signal_validity_sessions
        == 10
    )

    assert (
        medium.maximum_holding_sessions
        == 65
    )

    assert (
        medium.confirmation_policy
        is StrategyConfirmationPolicy
        .WEEKLY_CLOSE_PLUS_STRATEGY_CONFIRMATION
    )

    assert (
        medium.entry_timing
        is StrategyEntryTiming
        .NEXT_ELIGIBLE_SESSION
    )

    assert (
        medium.intraday_entries_allowed
        is False
    )

    assert (
        medium.exit_policies
        == EXPECTED_EXITS
    )


def test_horizon_policy_parsing_is_deterministic(
) -> None:
    policy = load_product_policy()

    first = (
        horizon_policies_from_product_policy(
            policy
        )
    )

    second = (
        horizon_policies_from_product_policy(
            policy
        )
    )

    assert first == second


def test_signal_validity_and_holding_are_separate_contracts(
) -> None:
    policies = (
        horizon_policies_from_product_policy(
            load_product_policy()
        )
    )

    for horizon_policy in (
        policies.values()
    ):
        assert (
            horizon_policy
            .maximum_holding_sessions
            > horizon_policy
            .signal_validity_sessions
        )


def test_no_intraday_horizon_is_representable(
) -> None:
    with pytest.raises(
        ValueError,
        match="SWING or MEDIUM_TERM",
    ):
        coerce_strategy_horizon(
            "intraday"
        )

    with pytest.raises(
        ValueError,
        match="SWING or MEDIUM_TERM",
    ):
        coerce_strategy_horizon(
            "day_trading"
        )


def test_product_policy_rejects_intraday_interval(
) -> None:
    policy = deepcopy(
        load_product_policy()
    )

    policy[
        "strategies"
    ][
        "horizon_policies"
    ][
        "swing"
    ][
        "market_data_interval"
    ] = "1h"

    with pytest.raises(
        ProductPolicyError,
        match="market_data_interval",
    ):
        validate_product_policy(
            policy
        )


def test_product_policy_rejects_intraday_entry_flag(
) -> None:
    policy = deepcopy(
        load_product_policy()
    )

    policy[
        "strategies"
    ][
        "horizon_policies"
    ][
        "medium_term"
    ][
        "intraday_entries_allowed"
    ] = True

    with pytest.raises(
        ProductPolicyError,
        match="intraday_entries_allowed",
    ):
        validate_product_policy(
            policy
        )


def test_domain_rejects_invalid_holding_contract(
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "maximum_holding_sessions"
        ),
    ):
        HorizonPolicy(
            policy_version=(
                HORIZON_POLICY_VERSION
            ),
            horizon=(
                StrategyHorizon.SWING
            ),
            strategy_version=(
                "fixture-v1"
            ),
            market_data_period="2y",
            market_data_interval="1d",
            signal_validity_sessions=5,
            maximum_holding_sessions=5,
            confirmation_policy=(
                StrategyConfirmationPolicy
                .STRATEGY_CONFIRMATION
            ),
            entry_timing=(
                StrategyEntryTiming
                .NEXT_ELIGIBLE_SESSION
            ),
            intraday_entries_allowed=False,
            exit_policies=EXPECTED_EXITS,
        )


def test_domain_rejects_intraday_market_interval(
) -> None:
    with pytest.raises(
        ValueError,
        match="end-of-day or weekly",
    ):
        HorizonPolicy(
            policy_version=(
                HORIZON_POLICY_VERSION
            ),
            horizon=(
                StrategyHorizon.SWING
            ),
            strategy_version=(
                "fixture-v1"
            ),
            market_data_period="2y",
            market_data_interval="1h",
            signal_validity_sessions=5,
            maximum_holding_sessions=20,
            confirmation_policy=(
                StrategyConfirmationPolicy
                .STRATEGY_CONFIRMATION
            ),
            entry_timing=(
                StrategyEntryTiming
                .NEXT_ELIGIBLE_SESSION
            ),
            intraday_entries_allowed=False,
            exit_policies=EXPECTED_EXITS,
        )
