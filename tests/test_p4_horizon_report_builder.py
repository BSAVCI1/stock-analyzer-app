"""P4.11.11 derived horizon-evidence builder tests."""

from __future__ import annotations

import copy
import json

from src.jobs.cli import main
from src.p4_horizon_evidence import build_strategy_horizon_check
from src.p4_horizon_report_builder import build_horizon_evidence


def _report(horizon: str) -> dict[str, object]:
    version = (
        "p4.3-swing-v1"
        if horizon == "swing"
        else "p4.3-medium-term-v1"
    )
    return {
        "schema_version": 1,
        "horizon": horizon,
        "strategy_version": version,
        "generated_at": "2026-08-24T22:00:00+00:00",
        "dataset_id": f"DATA-{horizon}",
        "cost_model_id": "IBKR-FIXED-EUR-V1",
        "validation": {
            "out_of_sample_passed": True,
            "walk_forward_passed": True,
            "costs_included": True,
            "observed_trade_count": 25,
            "minimum_trade_count": 20,
            "parameter_stability": 0.75,
            "minimum_parameter_stability": 0.5,
        },
    }


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "approval_status": "APPROVED_FOR_P2_RELEASE",
        "profiles": {"trend_pullback": {"values": {"buy_score": 75}}},
    }


def test_builder_derives_passing_independent_evidence() -> None:
    evidence = build_horizon_evidence(
        release_id="P4-RELEASE",
        observed_at="2026-08-24T22:30:00+00:00",
        swing_report=_report("swing"),
        medium_term_report=_report("medium_term"),
        threshold_manifest=_manifest(),
    )
    assert build_strategy_horizon_check(evidence).status.value == "PASS"
    swing, medium = evidence["horizons"]
    assert swing["decision"] == "ACCEPT"
    assert medium["decision"] == "ACCEPT"
    assert swing["acceptance_report_id"].startswith("sha256:")
    assert swing["threshold_manifest_id"] == medium["threshold_manifest_id"]
    assert swing["acceptance_report_id"] != medium["acceptance_report_id"]


def test_builder_cannot_promote_failed_observation() -> None:
    swing = _report("swing")
    swing["validation"]["costs_included"] = False
    evidence = build_horizon_evidence(
        release_id="P4-RELEASE",
        observed_at="2026-08-24T22:30:00+00:00",
        swing_report=swing,
        medium_term_report=_report("medium_term"),
        threshold_manifest=_manifest(),
    )
    assert evidence["horizons"][0]["decision"] == "REJECT"
    check = build_strategy_horizon_check(evidence)
    assert check.status.value == "FAIL"
    assert any("decision must be ACCEPT" in item for item in check.details)


def test_builder_derives_trade_count_and_stability_results() -> None:
    swing = _report("swing")
    swing["validation"].update({
        "observed_trade_count": 19,
        "parameter_stability": 0.49,
    })
    evidence = build_horizon_evidence(
        release_id="P4-RELEASE",
        observed_at="2026-08-24T22:30:00+00:00",
        swing_report=swing,
        medium_term_report=_report("medium_term"),
        threshold_manifest=_manifest(),
    )
    result = evidence["horizons"][0]
    assert result["minimum_trade_count_met"] is False
    assert result["parameter_stability_passed"] is False
    assert result["decision"] == "REJECT"


def test_builder_rejects_wrong_horizon_version() -> None:
    swing = _report("swing")
    swing["strategy_version"] = "wrong"
    try:
        build_horizon_evidence(
            release_id="P4-RELEASE",
            observed_at="2026-08-24T22:30:00+00:00",
            swing_report=swing,
            medium_term_report=_report("medium_term"),
            threshold_manifest=_manifest(),
        )
    except ValueError as exc:
        assert "strategy_version" in str(exc)
    else:
        raise AssertionError("wrong strategy version must fail")


def test_cli_writes_derived_evidence(tmp_path, capsys) -> None:
    paths = {}
    for name, value in (
        ("swing", _report("swing")),
        ("medium", _report("medium_term")),
        ("manifest", _manifest()),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path

    assert main([
        "p4-build-horizon-evidence",
        "--release-id", "P4-RELEASE",
        "--observed-at", "2026-08-24T22:30:00+00:00",
        "--swing-report", str(paths["swing"]),
        "--medium-term-report", str(paths["medium"]),
        "--threshold-manifest", str(paths["manifest"]),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["decision"] for item in payload["horizons"]] == [
        "ACCEPT", "ACCEPT"
    ]


def test_report_fingerprint_changes_with_source_observation() -> None:
    original = _report("swing")
    changed = copy.deepcopy(original)
    changed["validation"]["observed_trade_count"] = 26
    first = build_horizon_evidence(
        release_id="P4-RELEASE",
        observed_at="2026-08-24T22:30:00+00:00",
        swing_report=original,
        medium_term_report=_report("medium_term"),
        threshold_manifest=_manifest(),
    )
    second = build_horizon_evidence(
        release_id="P4-RELEASE",
        observed_at="2026-08-24T22:30:00+00:00",
        swing_report=changed,
        medium_term_report=_report("medium_term"),
        threshold_manifest=_manifest(),
    )
    assert (
        first["horizons"][0]["acceptance_report_id"]
        != second["horizons"][0]["acceptance_report_id"]
    )
