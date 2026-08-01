from __future__ import annotations

import pandas as pd
import pytest

from src.backtest import (
    PromotionThresholds,
    WalkForwardConfig,
    WindowMode,
    assess_strategy_promotion,
    chronological_train_test_split,
    create_walk_forward_folds,
    run_walk_forward_validation,
)


def make_history(
    values: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": values},
        index=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
            tz="UTC",
        ),
    )


def target_evaluator(
    frame: pd.DataFrame,
    parameters,
):
    target = float(parameters["target"])
    mean_value = float(
        frame["Close"].mean()
    )

    return {
        "score": (
            100.0
            - abs(
                mean_value
                - target
            )
        ),
        "mean_close": mean_value,
    }


def test_chronological_split_separates_periods() -> None:
    history = make_history(
        list(range(10))
    )

    split = chronological_train_test_split(
        history,
        train_size=6,
        test_size=3,
        gap_size=1,
    )

    assert len(split.in_sample) == 6
    assert len(split.purge_gap) == 1
    assert len(split.out_of_sample) == 3

    assert (
        split.in_sample.index.max()
        < split.purge_gap.index.min()
        < split.out_of_sample.index.min()
    )


def test_expanding_walk_forward_boundaries() -> None:
    history = make_history(
        list(range(14))
    )

    config = WalkForwardConfig(
        train_size=6,
        test_size=2,
        step_size=2,
        mode=WindowMode.EXPANDING,
    )

    folds = create_walk_forward_folds(
        history,
        config,
    )

    assert len(folds) == 4

    assert [
        fold.train_size
        for fold in folds
    ] == [6, 8, 10, 12]

    assert [
        fold.test_size
        for fold in folds
    ] == [2, 2, 2, 2]

    assert all(
        fold.train_end
        < fold.test_start
        for fold in folds
    )


def test_rolling_walk_forward_keeps_train_size() -> None:
    history = make_history(
        list(range(14))
    )

    config = WalkForwardConfig(
        train_size=6,
        test_size=2,
        step_size=2,
        mode=WindowMode.ROLLING,
    )

    folds = create_walk_forward_folds(
        history,
        config,
    )

    assert len(folds) == 4

    assert all(
        fold.train_size == 6
        for fold in folds
    )

    assert [
        fold.train_start_position
        for fold in folds
    ] == [0, 2, 4, 6]


def test_gap_prevents_training_test_overlap() -> None:
    history = make_history(
        list(range(16))
    )

    config = WalkForwardConfig(
        train_size=6,
        test_size=2,
        step_size=2,
        gap_size=2,
    )

    folds = create_walk_forward_folds(
        history,
        config,
    )

    assert all(
        fold.gap_size == 2
        for fold in folds
    )

    assert all(
        fold.train_end_position
        + 2
        == fold.test_start_position
        for fold in folds
    )


def test_selection_uses_in_sample_only() -> None:
    base_history = make_history(
        [
            1, 1, 1, 1,
            100, 100,
        ]
    )

    changed_test_history = make_history(
        [
            1, 1, 1, 1,
            10_000, 10_000,
        ]
    )

    config = WalkForwardConfig(
        train_size=4,
        test_size=2,
        max_folds=1,
    )

    candidates = (
        {"target": 1},
        {"target": 100},
    )

    first = run_walk_forward_validation(
        base_history,
        candidates,
        target_evaluator,
        config=config,
    )

    second = run_walk_forward_validation(
        changed_test_history,
        candidates,
        target_evaluator,
        config=config,
    )

    assert dict(
        first
        .fold_results[0]
        .selected_parameters
    ) == {"target": 1}

    assert dict(
        second
        .fold_results[0]
        .selected_parameters
    ) == {"target": 1}

    assert (
        first.mean_out_of_sample_metric
        != second.mean_out_of_sample_metric
    )


def test_walk_forward_result_separates_metrics() -> None:
    history = make_history(
        [
            1, 1, 1, 1,
            2, 2,
            3, 3,
            4, 4,
        ]
    )

    report = run_walk_forward_validation(
        history,
        (
            {"target": 1},
            {"target": 4},
        ),
        target_evaluator,
        config=WalkForwardConfig(
            train_size=4,
            test_size=2,
            step_size=2,
        ),
    )

    assert report.fold_count == 3

    for result in report.fold_results:
        assert "score" in (
            result.in_sample_metrics
        )

        assert "score" in (
            result.out_of_sample_metrics
        )

        assert (
            result.fold.train_end
            < result.fold.test_start
        )


def test_parameter_stability_is_one_when_unchanged() -> None:
    history = make_history(
        [1] * 10
    )

    report = run_walk_forward_validation(
        history,
        (
            {"target": 1},
            {"target": 10},
        ),
        target_evaluator,
        config=WalkForwardConfig(
            train_size=4,
            test_size=2,
            step_size=2,
        ),
    )

    stability = (
        report
        .parameter_stability
    )

    assert (
        stability.overall_stability_score
        == pytest.approx(1.0)
    )

    assert (
        stability.entries[0].change_count
        == 0
    )


def test_parameter_stability_detects_drift() -> None:
    history = make_history(
        [
            1, 1, 1, 1,
            10, 10,
            10, 10,
            10, 10,
        ]
    )

    report = run_walk_forward_validation(
        history,
        (
            {"target": 1},
            {"target": 10},
        ),
        target_evaluator,
        config=WalkForwardConfig(
            train_size=4,
            test_size=2,
            step_size=2,
            mode=WindowMode.ROLLING,
        ),
    )

    entry = (
        report
        .parameter_stability
        .entries[0]
    )

    assert entry.unique_value_count == 2
    assert entry.change_count >= 1
    assert entry.stability_score < 1


def test_ties_are_resolved_deterministically() -> None:
    history = make_history(
        [5] * 6
    )

    def tied_evaluator(frame, parameters):
        return {"score": 1.0}

    report = run_walk_forward_validation(
        history,
        (
            {"target": 2},
            {"target": 1},
        ),
        tied_evaluator,
        config=WalkForwardConfig(
            train_size=4,
            test_size=2,
            max_folds=1,
        ),
    )

    assert dict(
        report
        .fold_results[0]
        .selected_parameters
    ) == {"target": 1}


def test_promotion_requires_enough_oos_folds() -> None:
    history = make_history(
        [1] * 6
    )

    report = run_walk_forward_validation(
        history,
        (
            {"target": 1},
            {"target": 10},
        ),
        target_evaluator,
        config=WalkForwardConfig(
            train_size=4,
            test_size=2,
            max_folds=1,
        ),
    )

    decision = assess_strategy_promotion(
        report,
        thresholds=PromotionThresholds(
            minimum_folds=3,
            minimum_mean_out_of_sample_metric=0,
            maximum_generalisation_gap=100,
            minimum_parameter_stability=0,
        ),
    )

    assert decision.promoted is False
    assert any(
        "out-of-sample folds"
        in reason
        for reason in decision.reasons
    )


def test_promotion_rejects_weak_oos_result() -> None:
    history = make_history(
        [
            1, 1, 1, 1,
            100, 100,
            100, 100,
            100, 100,
        ]
    )

    report = run_walk_forward_validation(
        history,
        (
            {"target": 1},
            {"target": 100},
        ),
        target_evaluator,
        config=WalkForwardConfig(
            train_size=4,
            test_size=2,
            step_size=2,
            mode=WindowMode.EXPANDING,
        ),
    )

    decision = assess_strategy_promotion(
        report,
        thresholds=PromotionThresholds(
            minimum_folds=3,
            minimum_mean_out_of_sample_metric=50,
            maximum_generalisation_gap=10,
            minimum_parameter_stability=0,
        ),
    )

    assert decision.promoted is False

    combined_reasons = " ".join(
        decision.reasons
    ).lower()

    assert (
        "out-of-sample"
        in combined_reasons
        or "generalisation"
        in combined_reasons
    )


def test_promotion_accepts_stable_oos_result() -> None:
    history = make_history(
        [1] * 10
    )

    report = run_walk_forward_validation(
        history,
        (
            {"target": 1},
            {"target": 10},
        ),
        target_evaluator,
        config=WalkForwardConfig(
            train_size=4,
            test_size=2,
            step_size=2,
        ),
    )

    decision = assess_strategy_promotion(
        report,
        thresholds=PromotionThresholds(
            minimum_folds=3,
            minimum_mean_out_of_sample_metric=99,
            maximum_generalisation_gap=0,
            minimum_parameter_stability=1,
        ),
    )

    assert decision.promoted is True


def test_duplicate_parameter_candidates_rejected() -> None:
    with pytest.raises(ValueError):
        run_walk_forward_validation(
            make_history(
                [1] * 6
            ),
            (
                {"target": 1},
                {"target": 1},
            ),
            target_evaluator,
            config=WalkForwardConfig(
                train_size=4,
                test_size=2,
            ),
        )


def test_random_or_unsorted_input_is_normalised() -> None:
    history = make_history(
        list(range(8))
    ).sort_index(
        ascending=False
    )

    folds = create_walk_forward_folds(
        history,
        WalkForwardConfig(
            train_size=4,
            test_size=2,
        ),
    )

    assert folds[0].train_start < (
        folds[0].train_end
    )

    assert folds[0].train_end < (
        folds[0].test_start
    )
