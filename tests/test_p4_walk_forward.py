"""P4 independent walk-forward orchestration tests."""

from datetime import datetime, timezone

import pandas as pd

from src.p4_validation_report import build_validation_report
from src.p4_walk_forward import run_walk_forward_study


def _dataset(horizon="swing", periods=520):
    index = pd.date_range("2020-01-01", periods=periods, freq="D", tz="UTC")
    rows = [{"at": at.isoformat()} for at in index]
    return {
        "schema_version": 2,
        "horizon": horizon,
        "strategy_version": (
            "p4.3-swing-v1" if horizon == "swing" else "p4.3-medium-term-v1"
        ),
        "dataset_id": "sha256:" + "b" * 64,
        "instruments": [{"symbol": "TEST", "rows": rows}],
    }


def _candidates():
    return ({"rsi_max": 60}, {"rsi_max": 65}, {"rsi_max": 70})


def test_selection_uses_training_before_one_untouched_test_replay() -> None:
    calls = []

    def replay(dataset, *, parameters, test_start, test_end):
        days = (test_end - test_start).days
        calls.append((parameters["rsi_max"], days, test_start, test_end))
        is_training = days > 70
        net = parameters["rsi_max"] if is_training else 5.0
        return {
            "trade_count": 10,
            "gross_pnl": net + 2,
            "execution_costs": 2,
            "net_pnl": net,
            "trades": [],
        }

    study = run_walk_forward_study(
        _dataset(),
        parameter_candidates=_candidates(),
        generated_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        cost_model_id="ibkr-reference-2026-08-09-v2",
        replay_runner=replay,
    )
    assert len(study["folds"]) >= 2
    assert all(fold["selected_parameters"] == {"rsi_max": 70} for fold in study["folds"])
    for offset in range(0, len(calls), 4):
        training = calls[offset:offset + 3]
        test = calls[offset + 3]
        assert [item[0] for item in training] == [60, 65, 70]
        assert test[0] == 70
        assert max(item[3] for item in training) < test[2]


def test_study_output_builds_a_cost_aware_validation_report() -> None:
    def replay(dataset, *, parameters, test_start, test_end):
        return {
            "trade_count": 10,
            "gross_pnl": 12.0,
            "execution_costs": 2.0,
            "net_pnl": 10.0,
            "trades": [],
        }

    study = run_walk_forward_study(
        _dataset(),
        parameter_candidates=_candidates(),
        generated_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        cost_model_id="ibkr-reference-2026-08-09-v2",
        replay_runner=replay,
    )
    report = build_validation_report(
        study, approved_cost_model_id="ibkr-reference-2026-08-09-v2"
    )
    assert report["validation"]["out_of_sample_passed"] is True
    assert report["validation"]["parameter_stability"] == 1.0


def test_zero_trade_study_remains_a_rejected_observation() -> None:
    def replay(dataset, *, parameters, test_start, test_end):
        return {
            "trade_count": 0, "gross_pnl": 0.0,
            "execution_costs": 0.0, "net_pnl": 0.0, "trades": [],
        }

    study = run_walk_forward_study(
        _dataset(),
        parameter_candidates=_candidates(),
        generated_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        cost_model_id="ibkr-reference-2026-08-09-v2",
        replay_runner=replay,
    )
    report = build_validation_report(
        study, approved_cost_model_id="ibkr-reference-2026-08-09-v2"
    )
    assert report["validation"]["out_of_sample_passed"] is False
    assert report["validation"]["observed_trade_count"] == 0


def test_medium_term_requires_two_independent_test_folds() -> None:
    def replay(dataset, *, parameters, test_start, test_end):
        return {
            "trade_count": 1, "gross_pnl": 1.0,
            "execution_costs": 0.5, "net_pnl": 0.5, "trades": [],
        }

    study = run_walk_forward_study(
        _dataset("medium_term", periods=260),
        parameter_candidates=_candidates(),
        generated_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        cost_model_id="ibkr-reference-2026-08-09-v2",
        replay_runner=replay,
    )
    assert len(study["folds"]) >= 2
    assert all(
        fold["train_end"] < fold["test_start"] for fold in study["folds"]
    )
