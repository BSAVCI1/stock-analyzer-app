"""P4.11.6 strategy-horizon acceptance evidence tests."""

from __future__ import annotations

from copy import deepcopy
import json

from src.jobs.cli import main
from src.p4_horizon_evidence import build_strategy_horizon_check
from src.p4_release_gate import P4CheckStatus


def _horizon(name: str, version: str) -> dict[str, object]:
    return {
        "horizon": name,
        "strategy_version": version,
        "decision": "ACCEPT",
        "out_of_sample_passed": True,
        "walk_forward_passed": True,
        "costs_included": True,
        "minimum_trade_count_met": True,
        "parameter_stability_passed": True,
        "acceptance_report_id": f"ACCEPT-{name}",
        "validation_report_id": f"VALIDATE-{name}",
        "threshold_manifest_id": f"THRESHOLD-{name}",
    }


def _evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "release_id": "P4-2026-08-24",
        "observed_at": "2026-08-24T20:00:00+00:00",
        "horizons": [
            _horizon("swing", "p4.3-swing-v1"),
            _horizon("medium_term", "p4.3-medium-term-v1"),
        ],
    }


def test_both_independent_horizons_pass() -> None:
    check = build_strategy_horizon_check(_evidence())
    assert check.status is P4CheckStatus.PASS
    assert check.name == "strategy_horizon_acceptance"
    assert "ACCEPTANCE_REPORT_ID:ACCEPT-swing" in check.evidence_ids
    assert "ACCEPTANCE_REPORT_ID:ACCEPT-medium_term" in check.evidence_ids


def test_one_rejected_horizon_blocks_combined_gate() -> None:
    evidence = _evidence()
    evidence["horizons"][1]["decision"] = "REJECT"
    check = build_strategy_horizon_check(evidence)
    assert check.status is P4CheckStatus.FAIL
    assert check.evidence_ids == ()
    assert any("decision" in item for item in check.details)


def test_version_mismatch_blocks_without_aggregation() -> None:
    evidence = _evidence()
    evidence["horizons"][0]["strategy_version"] = "p4.3-swing-v2"
    check = build_strategy_horizon_check(evidence)
    assert check.status is P4CheckStatus.FAIL
    assert any("strategy_version" in item for item in check.details)


def test_duplicate_or_missing_horizon_blocks() -> None:
    evidence = _evidence()
    evidence["horizons"][1] = deepcopy(evidence["horizons"][0])
    check = build_strategy_horizon_check(evidence)
    assert check.status is P4CheckStatus.FAIL
    assert any("unique" in item for item in check.details)
    assert any("swing and medium_term" in item for item in check.details)


def test_missing_validation_dimension_or_ids_blocks() -> None:
    evidence = _evidence()
    evidence["horizons"][0]["costs_included"] = False
    evidence["horizons"][0]["validation_report_id"] = ""
    check = build_strategy_horizon_check(evidence)
    assert check.status is P4CheckStatus.FAIL
    assert any("costs_included" in item for item in check.details)
    assert any("validation_report_id" in item for item in check.details)


def test_fingerprint_is_order_stable_but_evidence_sensitive() -> None:
    evidence = _evidence()
    first = build_strategy_horizon_check(evidence).evidence_ids[0]
    reordered = dict(reversed(tuple(evidence.items())))
    assert build_strategy_horizon_check(reordered).evidence_ids[0] == first
    changed = deepcopy(evidence)
    changed["horizons"][0]["acceptance_report_id"] = "ACCEPT-NEW"
    assert build_strategy_horizon_check(changed).evidence_ids[0] != first


def test_cli_and_deliberately_blocked_example(tmp_path, capsys) -> None:
    path = tmp_path / "horizons.json"
    path.write_text(json.dumps(_evidence()), encoding="utf-8")
    assert main(["p4-horizon-evidence", "--evidence", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["check"]["status"] == "PASS"
    assert main([
        "p4-horizon-evidence", "--evidence",
        "config/p4_horizon_evidence.example.json",
    ]) == 1
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["check"]["status"] == "FAIL"
