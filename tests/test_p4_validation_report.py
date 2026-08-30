"""P4 cost-aware walk-forward validation report tests."""

import copy
import json

from src.jobs.cli import main
from src.p4_validation_report import build_validation_report


def _build(value):
    return build_validation_report(
        value,
        approved_cost_model_id="ibkr-reference-2026-08-09-v2",
    )


def _observation() -> dict[str, object]:
    return {
        "schema_version": 1,
        "horizon": "swing",
        "strategy_version": "p4.3-swing-v1",
        "generated_at": "2026-08-30T09:00:00+00:00",
        "dataset_id": "sha256:" + "a" * 64,
        "cost_model_id": "ibkr-reference-2026-08-09-v2",
        "costs_included": True,
        "requirements": {
            "minimum_trade_count": 20,
            "minimum_parameter_stability": 0.5,
            "minimum_net_expectancy": 0.01,
        },
        "folds": [
            {
                "train_start": "2020-01-01T00:00:00+00:00",
                "train_end": "2021-01-01T00:00:00+00:00",
                "test_start": "2021-01-02T00:00:00+00:00",
                "test_end": "2021-06-30T00:00:00+00:00",
                "trade_count": 10,
                "gross_pnl": 20.0,
                "execution_costs": 5.0,
                "net_pnl": 15.0,
                "selected_parameters": {"buy_score": 75, "rsi_max": 65},
            },
            {
                "train_start": "2020-07-01T00:00:00+00:00",
                "train_end": "2021-06-30T00:00:00+00:00",
                "test_start": "2021-07-01T00:00:00+00:00",
                "test_end": "2021-12-31T00:00:00+00:00",
                "trade_count": 12,
                "gross_pnl": 10.0,
                "execution_costs": 4.0,
                "net_pnl": 6.0,
                "selected_parameters": {"buy_score": 75, "rsi_max": 60},
            },
        ],
    }


def test_report_derives_costed_out_of_sample_results() -> None:
    report = _build(_observation())
    assert report["validation"] == {
        "out_of_sample_passed": True,
        "walk_forward_passed": True,
        "costs_included": True,
        "observed_trade_count": 22,
        "minimum_trade_count": 20,
        "parameter_stability": 0.5,
        "minimum_parameter_stability": 0.5,
    }
    assert report["observations"]["net_pnl"] == 21.0
    assert report["observations"]["net_expectancy"] == 21.0 / 22


def test_insufficient_trades_remain_rejected() -> None:
    value = _observation()
    value["requirements"]["minimum_trade_count"] = 30
    assert _build(value)["validation"]["out_of_sample_passed"] is False


def test_negative_cost_adjusted_expectancy_remains_rejected() -> None:
    value = _observation()
    value["folds"][1].update({"gross_pnl": -20.0, "execution_costs": 4.0, "net_pnl": -24.0})
    assert _build(value)["validation"]["out_of_sample_passed"] is False


def test_inconsistent_cost_arithmetic_fails_closed() -> None:
    value = _observation()
    value["folds"][0]["net_pnl"] = 20.0
    try:
        _build(value)
    except ValueError as exc:
        assert "gross_pnl minus costs" in str(exc)
    else:
        raise AssertionError("inconsistent costs must fail")


def test_overlapping_out_of_sample_folds_fail_closed() -> None:
    value = _observation()
    value["folds"][0]["test_end"] = "2021-07-02T00:00:00+00:00"
    try:
        _build(value)
    except ValueError as exc:
        assert "must not overlap" in str(exc)
    else:
        raise AssertionError("overlapping folds must fail")


def test_dataset_requires_cryptographic_identity() -> None:
    value = copy.deepcopy(_observation())
    value["dataset_id"] = "DATA-SWING"
    try:
        _build(value)
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("unbound datasets must fail")


def test_cli_writes_derived_report(tmp_path, capsys) -> None:
    path = tmp_path / "observation.json"
    policy = tmp_path / "policy.json"
    path.write_text(json.dumps(_observation()), encoding="utf-8")
    policy.write_text(json.dumps({
        "cost_model": {
            "reference_profile_version": "ibkr-reference-2026-08-09-v2"
        }
    }), encoding="utf-8")
    assert main([
        "p4-build-validation-report", "--observation", str(path),
        "--policy", str(policy),
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["validation"]["observed_trade_count"] == 22


def test_unapproved_cost_model_fails_closed() -> None:
    try:
        build_validation_report(
            _observation(), approved_cost_model_id="some-other-cost-model"
        )
    except ValueError as exc:
        assert "approved product policy" in str(exc)
    else:
        raise AssertionError("unapproved cost assumptions must fail")
