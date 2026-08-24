"""Read-only P4 release evidence for product and portfolio policy."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from src.p4_release_gate import P4CheckStatus, P4GateCheck


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _fingerprint(policy: Mapping[str, object]) -> str:
    canonical = json.dumps(
        policy, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _matches(actual: object, expected: object) -> bool:
    return type(actual) is type(expected) and actual == expected


def _check(
    name: str,
    failures: list[str],
    *,
    policy_version: object,
    fingerprint: str,
) -> P4GateCheck:
    if failures:
        return P4GateCheck(
            name=name,
            status=P4CheckStatus.FAIL,
            evidence_ids=(),
            details=tuple(failures),
        )
    version = str(policy_version).strip()
    return P4GateCheck(
        name=name,
        status=P4CheckStatus.PASS,
        evidence_ids=(f"PRODUCT-POLICY:{version}:sha256:{fingerprint}",),
        details=("Approved versioned product policy invariants verified.",),
    )


def build_policy_gate_checks(
    policy: Mapping[str, object],
) -> tuple[P4GateCheck, P4GateCheck]:
    """Build evidence for the two static P4 policy checks.

    The function only evaluates supplied configuration. Invalid or incomplete
    policy becomes explicit FAIL evidence rather than an exception or pass.
    """
    if not isinstance(policy, Mapping):
        raise ValueError("policy must be an object.")

    fingerprint = _fingerprint(policy)
    version = policy.get("policy_version")
    product = _mapping(policy.get("product"))
    execution = _mapping(policy.get("execution"))
    instruments = _mapping(policy.get("instruments"))
    cost_model = _mapping(policy.get("cost_model"))
    portfolio = _mapping(policy.get("portfolio"))

    common_failures: list[str] = []
    if not _matches(policy.get("schema_version"), 1):
        common_failures.append("schema_version must be 1.")
    if not _matches(version, "p4.3-1"):
        common_failures.append("policy_version must be 'p4.3-1'.")

    paper_failures: list[str] = list(common_failures)
    expected_paper = (
        (product, "mode", "paper_only"),
        (execution, "paper_only", True),
        (execution, "live_execution_enabled", False),
        (execution, "deny_by_default", True),
        (execution, "broker_api_connection_enabled", False),
        (cost_model, "api_connection_enabled", False),
    )
    for section, key, expected in expected_paper:
        if not _matches(section.get(key), expected):
            paper_failures.append(
                f"{key} must be {expected!r}; received {section.get(key)!r}."
            )
    for key in ("leverage", "shorts", "options", "cfds", "crypto"):
        if instruments.get(key) is not False:
            paper_failures.append(f"instruments.{key} must be False.")

    portfolio_failures: list[str] = list(common_failures)
    expected_portfolio = (
        ("currency", "EUR"),
        ("starting_balance", 2000),
        ("sizing_mode", "FIXED_NOTIONAL_WITH_RISK_CAP"),
        ("target_order_value", 100),
        ("maximum_order_value", 100),
        ("maximum_planned_loss", 10),
        ("maximum_open_positions", 5),
        ("maximum_invested_exposure", 500),
    )
    for key, expected in expected_portfolio:
        if not _matches(portfolio.get(key), expected):
            portfolio_failures.append(
                f"portfolio.{key} must be {expected!r}; "
                f"received {portfolio.get(key)!r}."
            )

    return (
        _check(
            "paper_only_invariants", paper_failures,
            policy_version=version, fingerprint=fingerprint,
        ),
        _check(
            "eur_portfolio_policy", portfolio_failures,
            policy_version=version, fingerprint=fingerprint,
        ),
    )


__all__ = ["build_policy_gate_checks"]
