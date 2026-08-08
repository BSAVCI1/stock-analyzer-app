from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.automation import (
    AutomatedExecutionConfig,
)
from src.costs import (
    load_ibkr_reference_profile,
)
from src.product_config import (
    ProductPolicyError,
    load_product_policy,
    validate_product_policy,
)


def test_p4_2_policy_pins_exact_ibkr_profile() -> None:
    policy = load_product_policy()

    assert (
        policy["policy_version"]
        == "p4.2-1"
    )

    cost_model = policy["cost_model"]

    profile = (
        load_ibkr_reference_profile()
    )

    assert (
        cost_model["reference_provider"]
        == profile["provider"]
        == "IBKR"
    )

    assert (
        cost_model[
            "reference_profile_version"
        ]
        == profile["profile_version"]
        == "ibkr-reference-2026-08-08-v1"
    )

    assert (
        cost_model[
            "api_connection_enabled"
        ]
        is False
    )

    assert (
        profile[
            "api_connection_enabled"
        ]
        is False
    )

    assert (
        cost_model[
            "ibkr_pricing_plan"
        ]
        is None
    )

    assert (
        profile[
            "active_pricing_plan"
        ]
        is None
    )

    assert (
        cost_model[
            "ibkr_fx_mode"
        ]
        is None
    )

    assert (
        profile[
            "active_fx_mode"
        ]
        is None
    )


def test_p4_2_profile_and_workflow_paths_exist() -> None:
    policy = load_product_policy()

    cost_model = policy["cost_model"]

    profile_path = Path(
        cost_model[
            "reference_profile_path"
        ]
    )

    workflow_path = Path(
        cost_model[
            "manual_update_workflow"
        ]
    )

    assert profile_path.is_file()
    assert workflow_path.is_file()


def test_p4_2_runtime_defaults_match_inactive_policy() -> None:
    policy = load_product_policy()

    cost_model = policy["cost_model"]

    runtime = AutomatedExecutionConfig()

    assert (
        runtime.ibkr_cost_gate_enabled
        == cost_model[
            "ibkr_cost_gate_enabled"
        ]
        is False
    )

    assert (
        runtime.ibkr_pricing_plan
        == cost_model[
            "ibkr_pricing_plan"
        ]
        is None
    )

    assert (
        runtime.ibkr_fx_mode
        == cost_model[
            "ibkr_fx_mode"
        ]
        is None
    )

    assert (
        runtime
        .ibkr_include_entry_fx_conversion
        == cost_model[
            "ibkr_include_entry_fx_conversion"
        ]
        is False
    )

    assert (
        runtime
        .ibkr_include_exit_fx_conversion
        == cost_model[
            "ibkr_include_exit_fx_conversion"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (
            "api_connection_enabled",
            True,
        ),
        (
            "ibkr_cost_gate_enabled",
            True,
        ),
        (
            "ibkr_pricing_plan",
            "FIXED",
        ),
        (
            "ibkr_fx_mode",
            "AUTO_CONVERSION",
        ),
        (
            "ibkr_include_entry_fx_conversion",
            True,
        ),
        (
            "ibkr_include_exit_fx_conversion",
            True,
        ),
    ],
)
def test_p4_2_policy_rejects_unapproved_activation(
    key: str,
    value: object,
) -> None:
    policy = deepcopy(
        load_product_policy()
    )

    policy["cost_model"][key] = value

    with pytest.raises(
        ProductPolicyError,
        match=key,
    ):
        validate_product_policy(
            policy
        )


def test_p4_2_manual_update_workflow_has_safety_gates() -> None:
    policy = load_product_policy()

    path = Path(
        policy["cost_model"][
            "manual_update_workflow"
        ]
    )

    text = path.read_text(
        encoding="utf-8"
    )

    required = (
        "Create a new version",
        "official IBKR",
        "hand-calculated",
        "full regression",
        "pricing plan",
        "FX mode",
        "credentials",
        "API connection",
        "cost gate remains disabled",
    )

    for marker in required:
        assert marker in text
