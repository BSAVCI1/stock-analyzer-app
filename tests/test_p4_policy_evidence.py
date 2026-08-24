"""P4.11.2 static policy-evidence tests."""

from __future__ import annotations

from copy import deepcopy
import json

from src.jobs.cli import main
from src.p4_policy_evidence import build_policy_gate_checks
from src.p4_release_gate import P4CheckStatus
from src.product_config import load_product_policy


def test_approved_policy_produces_two_traceable_passes() -> None:
    checks = build_policy_gate_checks(load_product_policy())
    assert tuple(check.name for check in checks) == (
        "paper_only_invariants", "eur_portfolio_policy",
    )
    assert all(check.status is P4CheckStatus.PASS for check in checks)
    assert checks[0].evidence_ids == checks[1].evidence_ids
    assert checks[0].evidence_ids[0].startswith(
        "PRODUCT-POLICY:p4.3-1:sha256:"
    )


def test_live_or_broker_capability_fails_closed() -> None:
    policy = deepcopy(load_product_policy())
    policy["execution"]["live_execution_enabled"] = True
    policy["cost_model"]["api_connection_enabled"] = True
    paper_check, portfolio_check = build_policy_gate_checks(policy)
    assert paper_check.status is P4CheckStatus.FAIL
    assert paper_check.evidence_ids == ()
    assert any("live_execution_enabled" in x for x in paper_check.details)
    assert any("api_connection_enabled" in x for x in paper_check.details)
    assert portfolio_check.status is P4CheckStatus.PASS


def test_wrong_currency_or_limits_fail_only_portfolio_check() -> None:
    policy = deepcopy(load_product_policy())
    policy["portfolio"]["currency"] = "USD"
    policy["portfolio"]["maximum_order_value"] = 101
    paper_check, portfolio_check = build_policy_gate_checks(policy)
    assert paper_check.status is P4CheckStatus.PASS
    assert portfolio_check.status is P4CheckStatus.FAIL
    assert portfolio_check.evidence_ids == ()
    assert any("currency" in x for x in portfolio_check.details)
    assert any("maximum_order_value" in x for x in portfolio_check.details)


def test_missing_or_wrong_contract_version_fails_both_checks() -> None:
    policy = deepcopy(load_product_policy())
    del policy["policy_version"]
    policy["schema_version"] = True
    checks = build_policy_gate_checks(policy)
    assert all(check.status is P4CheckStatus.FAIL for check in checks)
    assert all(check.evidence_ids == () for check in checks)
    assert all(any("policy_version" in x for x in check.details) for check in checks)
    assert all(any("schema_version" in x for x in check.details) for check in checks)


def test_policy_fingerprint_is_deterministic_and_change_sensitive() -> None:
    policy = load_product_policy()
    first = build_policy_gate_checks(policy)[0].evidence_ids
    reordered = dict(reversed(tuple(policy.items())))
    assert build_policy_gate_checks(reordered)[0].evidence_ids == first
    changed = deepcopy(policy)
    changed["product"]["name"] = "Changed name"
    assert build_policy_gate_checks(changed)[0].evidence_ids != first


def test_cli_outputs_manifest_ready_checks(tmp_path, capsys) -> None:
    policy = load_product_policy()
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    assert main(["p4-policy-evidence", "--policy", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [item["status"] for item in result["checks"]] == ["PASS", "PASS"]


def test_cli_returns_blocked_exit_for_policy_violation(tmp_path, capsys) -> None:
    policy = deepcopy(load_product_policy())
    policy["execution"]["paper_only"] = False
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    assert main(["p4-policy-evidence", "--policy", str(path)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["checks"][0]["status"] == "FAIL"
