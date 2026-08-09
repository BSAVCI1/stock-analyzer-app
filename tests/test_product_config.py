from __future__ import annotations

from copy import deepcopy
import json

import pytest

from src.jobs.cli import main
from src.product_config import (
    DEFAULT_PRODUCT_POLICY_PATH,
    ProductPolicyError,
    load_product_policy,
    validate_product_policy,
)


def test_default_policy_matches_p4_direction() -> None:
    policy = load_product_policy()

    assert policy["policy_version"] == "p4.2-3"

    assert (
        policy["portfolio"][
            "currency"
        ]
        == "EUR"
    )

    assert (
        policy["portfolio"][
            "starting_balance"
        ]
        == 2000
    )

    assert (
        policy["portfolio"][
            "maximum_order_value"
        ]
        == 100
    )

    assert (
        policy["portfolio"][
            "sizing_mode"
        ]
        == "FIXED_NOTIONAL_WITH_RISK_CAP"
    )

    assert (
        policy["portfolio"][
            "maximum_planned_loss"
        ]
        == 10
    )

    assert (
        policy["portfolio"][
            "maximum_open_positions"
        ]
        == 5
    )

    assert (
        policy["portfolio"][
            "maximum_invested_exposure"
        ]
        == 500
    )

    assert (
        policy["execution"][
            "paper_only"
        ]
        is True
    )

    assert (
        policy["execution"][
            "live_execution_enabled"
        ]
        is False
    )

    assert (
        policy["cost_model"][
            "api_connection_enabled"
        ]
        is False
    )


def test_policy_rejects_live_execution() -> None:
    policy = deepcopy(
        load_product_policy()
    )

    policy["execution"][
        "live_execution_enabled"
    ] = True

    with pytest.raises(
        ProductPolicyError,
        match="live_execution_enabled",
    ):
        validate_product_policy(policy)


def test_policy_rejects_sensitive_keys() -> None:
    policy = deepcopy(
        load_product_policy()
    )

    policy["notifications"][
        "api_token"
    ] = "must-not-be-printed"

    with pytest.raises(
        ProductPolicyError,
        match="Sensitive configuration key",
    ):
        validate_product_policy(policy)


def test_product_config_cli_needs_no_runtime_account(
    capsys,
) -> None:
    result = main(
        [
            "product-config",
            "--config",
            str(
                DEFAULT_PRODUCT_POLICY_PATH
            ),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(
        captured.out
    )

    assert result == 0
    assert captured.err == ""

    assert payload["status"] == "VALID"

    assert (
        payload["safety"]["secret_free"]
        is True
    )

    assert (
        payload["safety"][
            "live_execution_enabled"
        ]
        is False
    )

    assert (
        payload["safety"][
            "broker_api_connection_enabled"
        ]
        is False
    )


def test_policy_rejects_unknown_keys() -> None:
    policy = deepcopy(
        load_product_policy()
    )

    policy["execution"][
        "future_live_mode"
    ] = False

    with pytest.raises(
        ProductPolicyError,
        match="unexpected=future_live_mode",
    ):
        validate_product_policy(policy)
