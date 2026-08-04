"""Versioned, secret-free product policy for P4."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path


DEFAULT_PRODUCT_POLICY_PATH = Path(
    "config/product_policy_v1.json"
)

_SENSITIVE_KEY_FRAGMENTS = (
    "secret",
    "password",
    "token",
    "api_key",
    "apikey",
    "credential",
)


class ProductPolicyError(ValueError):
    """Raised when product policy violates an invariant."""


def _mapping(
    value: object,
    path: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProductPolicyError(
            f"{path} must be an object."
        )

    return value


def _expect(
    mapping: Mapping[str, object],
    key: str,
    expected: object,
    path: str,
) -> None:
    if key not in mapping:
        raise ProductPolicyError(
            f"{path}.{key} is required."
        )

    actual = mapping[key]

    if (
        type(actual) is not type(expected)
        or actual != expected
    ):
        raise ProductPolicyError(
            f"{path}.{key} must be "
            f"{expected!r}; received "
            f"{actual!r}."
        )


def _expect_keys(
    mapping: Mapping[str, object],
    expected_keys: set[str],
    path: str,
) -> None:
    actual_keys = {
        str(key)
        for key in mapping
    }

    missing = sorted(
        expected_keys - actual_keys
    )

    unexpected = sorted(
        actual_keys - expected_keys
    )

    if missing or unexpected:
        details = []

        if missing:
            details.append(
                "missing="
                + ", ".join(missing)
            )

        if unexpected:
            details.append(
                "unexpected="
                + ", ".join(unexpected)
            )

        raise ProductPolicyError(
            f"{path} contains invalid keys: "
            + "; ".join(details)
            + "."
        )


def _validate_secret_free(
    value: object,
    path: str = "$",
) -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            normalised = (
                key.strip()
                .lower()
                .replace("-", "_")
            )

            if any(
                fragment in normalised
                for fragment
                in _SENSITIVE_KEY_FRAGMENTS
            ):
                raise ProductPolicyError(
                    "Sensitive configuration key "
                    f"is prohibited at "
                    f"{path}.{key}."
                )

            _validate_secret_free(
                nested,
                f"{path}.{key}",
            )

    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_secret_free(
                nested,
                f"{path}[{index}]",
            )


def validate_product_policy(
    policy: Mapping[str, object],
) -> None:
    """Enforce the approved P4.0 direction."""

    _validate_secret_free(policy)

    _expect_keys(
        policy,
        {
            "schema_version",
            "policy_version",
            "product",
            "portfolio",
            "strategies",
            "instruments",
            "cost_model",
            "scheduling",
            "notifications",
            "execution",
        },
        "$",
    )

    _expect(
        policy,
        "schema_version",
        1,
        "$",
    )

    _expect(
        policy,
        "policy_version",
        "p4.0-1",
        "$",
    )

    product = _mapping(
        policy.get("product"),
        "$.product",
    )

    _expect_keys(
        product,
        {
            "name",
            "autonomy",
            "mode",
            "official_performance_source",
            "manual_ibkr_copy_allowed",
            "decision_outputs",
        },
        "$.product",
    )

    _expect(
        product,
        "name",
        "BSAVCI Smart Investment Bot",
        "$.product",
    )

    _expect(
        product,
        "autonomy",
        "always_on",
        "$.product",
    )

    _expect(
        product,
        "mode",
        "paper_only",
        "$.product",
    )

    _expect(
        product,
        "official_performance_source",
        "paper_portfolio",
        "$.product",
    )

    _expect(
        product,
        "manual_ibkr_copy_allowed",
        True,
        "$.product",
    )

    _expect(
        product,
        "decision_outputs",
        [
            "ranked_watchlist",
            "opportunity_tickets",
            "rejection_reasons",
            "near_qualifiers",
        ],
        "$.product",
    )

    portfolio = _mapping(
        policy.get("portfolio"),
        "$.portfolio",
    )

    _expect_keys(
        portfolio,
        {
            "currency",
            "starting_balance",
            "target_order_value",
            "maximum_order_value",
            "historical_account",
        },
        "$.portfolio",
    )

    for key, expected in (
        ("currency", "EUR"),
        ("starting_balance", 2000),
        ("target_order_value", 100),
        ("maximum_order_value", 100),
    ):
        _expect(
            portfolio,
            key,
            expected,
            "$.portfolio",
        )

    historical = _mapping(
        portfolio.get(
            "historical_account"
        ),
        "$.portfolio.historical_account",
    )

    _expect_keys(
        historical,
        {
            "account_id",
            "currency",
            "starting_balance",
            "preserve",
        },
        "$.portfolio.historical_account",
    )

    for key, expected in (
        (
            "account_id",
            "ACC-495a2ae778834fc4a2c14d24e66ef41e",
        ),
        ("currency", "USD"),
        ("starting_balance", 10000),
        ("preserve", True),
    ):
        _expect(
            historical,
            key,
            expected,
            "$.portfolio.historical_account",
        )

    strategies = _mapping(
        policy.get("strategies"),
        "$.strategies",
    )

    _expect_keys(
        strategies,
        {
            "enabled_horizons",
            "prohibited_horizons",
        },
        "$.strategies",
    )

    _expect(
        strategies,
        "enabled_horizons",
        [
            "swing",
            "medium_term",
        ],
        "$.strategies",
    )

    _expect(
        strategies,
        "prohibited_horizons",
        [
            "intraday",
            "day_trading",
        ],
        "$.strategies",
    )

    instruments = _mapping(
        policy.get("instruments"),
        "$.instruments",
    )

    _expect_keys(
        instruments,
        {
            "leverage",
            "shorts",
            "options",
            "cfds",
            "crypto",
        },
        "$.instruments",
    )

    for key in (
        "leverage",
        "shorts",
        "options",
        "cfds",
        "crypto",
    ):
        _expect(
            instruments,
            key,
            False,
            "$.instruments",
        )

    cost_model = _mapping(
        policy.get("cost_model"),
        "$.cost_model",
    )

    _expect_keys(
        cost_model,
        {
            "reference_provider",
            "api_connection_enabled",
        },
        "$.cost_model",
    )

    _expect(
        cost_model,
        "reference_provider",
        "IBKR",
        "$.cost_model",
    )

    _expect(
        cost_model,
        "api_connection_enabled",
        False,
        "$.cost_model",
    )

    scheduling = _mapping(
        policy.get("scheduling"),
        "$.scheduling",
    )

    _expect_keys(
        scheduling,
        {
            "always_on",
            "deployment_target",
            "swing_cycle",
            "medium_term_cycle",
        },
        "$.scheduling",
    )

    for key, expected in (
        ("always_on", True),
        (
            "deployment_target",
            "external_always_on",
        ),
        (
            "swing_cycle",
            "exchange_aware_windows",
        ),
        (
            "medium_term_cycle",
            "after_close",
        ),
    ):
        _expect(
            scheduling,
            key,
            expected,
            "$.scheduling",
        )

    notifications = _mapping(
        policy.get("notifications"),
        "$.notifications",
    )

    _expect_keys(
        notifications,
        {
            "channels",
            "delivery_settings_source",
        },
        "$.notifications",
    )

    _expect(
        notifications,
        "channels",
        [
            "email",
            "telegram",
        ],
        "$.notifications",
    )

    _expect(
        notifications,
        "delivery_settings_source",
        "environment",
        "$.notifications",
    )

    execution = _mapping(
        policy.get("execution"),
        "$.execution",
    )

    _expect_keys(
        execution,
        {
            "paper_only",
            "live_execution_enabled",
            "deny_by_default",
            "broker_api_connection_enabled",
        },
        "$.execution",
    )

    for key, expected in (
        ("paper_only", True),
        ("live_execution_enabled", False),
        ("deny_by_default", True),
        (
            "broker_api_connection_enabled",
            False,
        ),
    ):
        _expect(
            execution,
            key,
            expected,
            "$.execution",
        )


def load_product_policy(
    path: str | Path = (
        DEFAULT_PRODUCT_POLICY_PATH
    ),
) -> dict[str, object]:
    """Load and validate a versioned policy file."""

    resolved = Path(path)

    try:
        payload = json.loads(
            resolved.read_text(
                encoding="utf-8"
            )
        )
    except FileNotFoundError as exc:
        raise ProductPolicyError(
            "Product policy file does not "
            f"exist: {resolved}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ProductPolicyError(
            "Product policy contains invalid "
            f"JSON: {exc}."
        ) from exc

    if not isinstance(payload, dict):
        raise ProductPolicyError(
            "Product policy root must be "
            "an object."
        )

    validate_product_policy(payload)

    return payload


def safe_product_policy_payload(
    policy: Mapping[str, object],
    *,
    source_path: str | Path,
) -> dict[str, object]:
    """Build a printable secret-free policy report."""

    validate_product_policy(policy)

    execution = _mapping(
        policy["execution"],
        "$.execution",
    )

    return {
        "status": "VALID",
        "configuration_path":
        str(Path(source_path)),
        "policy": dict(policy),
        "safety": {
            "secret_free": True,
            "paper_only":
            execution["paper_only"],
            "live_execution_enabled":
            execution[
                "live_execution_enabled"
            ],
            "deny_by_default":
            execution["deny_by_default"],
            "broker_api_connection_enabled":
            execution[
                "broker_api_connection_enabled"
            ],
        },
    }
