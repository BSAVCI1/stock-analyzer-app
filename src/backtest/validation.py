"""Chronological train/test and walk-forward validation.

This module provides:

- strict chronological train/test separation
- expanding and rolling walk-forward windows
- optional purge gaps between training and testing
- deterministic parameter selection using in-sample data only
- separate out-of-sample evaluation
- parameter-stability reporting
- promotion decisions that require out-of-sample evidence

It contains no random splitting, provider integration, Streamlit code,
broker connection or live execution.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from types import MappingProxyType
from typing import Callable, Mapping, Sequence, TypeAlias

import pandas as pd


ParameterValue: TypeAlias = str | int | float | bool | None
ParameterMapping: TypeAlias = Mapping[str, ParameterValue]
MetricMapping: TypeAlias = Mapping[str, float]

MetricEvaluator: TypeAlias = Callable[
    [pd.DataFrame, ParameterMapping],
    MetricMapping,
]


class WindowMode(str, Enum):
    """Training-window behaviour between validation folds."""

    EXPANDING = "EXPANDING"
    ROLLING = "ROLLING"


def _positive_integer(
    name: str,
    value: object,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return value


def _non_negative_integer(
    name: str,
    value: object,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer."
        )

    return value


def _finite_float(
    name: str,
    value: object,
) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a finite number."
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite number."
        ) from exc

    if not isfinite(result):
        raise ValueError(
            f"{name} must be a finite number."
        )

    return result


def _normalise_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame):
        raise ValueError(
            "history must be a pandas DataFrame."
        )

    if history.empty:
        raise ValueError(
            "history cannot be empty."
        )

    frame = history.copy()

    index = pd.to_datetime(
        frame.index,
        errors="coerce",
        utc=True,
    )

    if index.isna().any():
        raise ValueError(
            "history contains invalid timestamps."
        )

    if index.duplicated().any():
        raise ValueError(
            "history contains duplicate timestamps."
        )

    frame.index = index
    frame = frame.sort_index()

    return frame


def _normalise_parameters(
    parameters: ParameterMapping,
) -> Mapping[str, ParameterValue]:
    if not isinstance(parameters, Mapping):
        raise ValueError(
            "parameters must be a mapping."
        )

    if not parameters:
        raise ValueError(
            "parameters cannot be empty."
        )

    result: dict[str, ParameterValue] = {}

    for raw_name, value in parameters.items():
        if (
            not isinstance(raw_name, str)
            or not raw_name.strip()
        ):
            raise ValueError(
                "Parameter names must be non-empty strings."
            )

        name = raw_name.strip()

        if not isinstance(
            value,
            (str, int, float, bool, type(None)),
        ):
            raise ValueError(
                f"Unsupported value for parameter {name}."
            )

        if (
            isinstance(value, float)
            and not isfinite(value)
        ):
            raise ValueError(
                f"Parameter {name} must be finite."
            )

        result[name] = value

    return MappingProxyType(
        dict(sorted(result.items()))
    )


def _normalise_metrics(
    metrics: MetricMapping,
) -> Mapping[str, float]:
    if not isinstance(metrics, Mapping):
        raise ValueError(
            "Evaluator must return a metric mapping."
        )

    if not metrics:
        raise ValueError(
            "Evaluator metrics cannot be empty."
        )

    result: dict[str, float] = {}

    for raw_name, raw_value in metrics.items():
        if (
            not isinstance(raw_name, str)
            or not raw_name.strip()
        ):
            raise ValueError(
                "Metric names must be non-empty strings."
            )

        name = raw_name.strip()

        result[name] = _finite_float(
            name,
            raw_value,
        )

    return MappingProxyType(
        dict(sorted(result.items()))
    )


def _parameter_key(
    parameters: ParameterMapping,
) -> str:
    """Return a deterministic tie-breaking representation."""

    return json.dumps(
        dict(parameters),
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """One strictly chronological in/out-of-sample split."""

    in_sample: pd.DataFrame
    purge_gap: pd.DataFrame
    out_of_sample: pd.DataFrame

    def __post_init__(self) -> None:
        if self.in_sample.empty:
            raise ValueError(
                "in_sample cannot be empty."
            )

        if self.out_of_sample.empty:
            raise ValueError(
                "out_of_sample cannot be empty."
            )

        if (
            self.in_sample.index.max()
            >= self.out_of_sample.index.min()
        ):
            raise ValueError(
                "In-sample observations must precede "
                "out-of-sample observations."
            )

        if not self.purge_gap.empty:
            if (
                self.in_sample.index.max()
                >= self.purge_gap.index.min()
            ):
                raise ValueError(
                    "Purge gap must follow in-sample data."
                )

            if (
                self.purge_gap.index.max()
                >= self.out_of_sample.index.min()
            ):
                raise ValueError(
                    "Purge gap must precede out-of-sample data."
                )


def chronological_train_test_split(
    history: pd.DataFrame,
    *,
    train_size: int,
    test_size: int | None = None,
    gap_size: int = 0,
) -> ChronologicalSplit:
    """Create one deterministic chronological train/test split."""

    frame = _normalise_history(history)

    train_size = _positive_integer(
        "train_size",
        train_size,
    )

    gap_size = _non_negative_integer(
        "gap_size",
        gap_size,
    )

    available_test_size = (
        len(frame)
        - train_size
        - gap_size
    )

    if test_size is None:
        test_size = available_test_size
    else:
        test_size = _positive_integer(
            "test_size",
            test_size,
        )

    required_rows = (
        train_size
        + gap_size
        + test_size
    )

    if len(frame) < required_rows:
        raise ValueError(
            "history does not contain enough rows "
            "for the requested split."
        )

    train_end = train_size
    gap_end = train_end + gap_size
    test_end = gap_end + test_size

    return ChronologicalSplit(
        in_sample=frame.iloc[:train_end].copy(),
        purge_gap=frame.iloc[
            train_end:gap_end
        ].copy(),
        out_of_sample=frame.iloc[
            gap_end:test_end
        ].copy(),
    )


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Configuration for deterministic walk-forward folds."""

    train_size: int
    test_size: int

    step_size: int | None = None
    gap_size: int = 0

    mode: WindowMode = WindowMode.EXPANDING
    max_folds: int | None = None

    selection_metric: str = "score"
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        train_size = _positive_integer(
            "train_size",
            self.train_size,
        )

        test_size = _positive_integer(
            "test_size",
            self.test_size,
        )

        step_size = (
            test_size
            if self.step_size is None
            else _positive_integer(
                "step_size",
                self.step_size,
            )
        )

        gap_size = _non_negative_integer(
            "gap_size",
            self.gap_size,
        )

        if not isinstance(self.mode, WindowMode):
            raise ValueError(
                "mode must be a WindowMode."
            )

        max_folds = self.max_folds

        if max_folds is not None:
            max_folds = _positive_integer(
                "max_folds",
                max_folds,
            )

        if (
            not isinstance(self.selection_metric, str)
            or not self.selection_metric.strip()
        ):
            raise ValueError(
                "selection_metric must be non-empty."
            )

        if not isinstance(
            self.higher_is_better,
            bool,
        ):
            raise ValueError(
                "higher_is_better must be boolean."
            )

        object.__setattr__(
            self,
            "train_size",
            train_size,
        )
        object.__setattr__(
            self,
            "test_size",
            test_size,
        )
        object.__setattr__(
            self,
            "step_size",
            step_size,
        )
        object.__setattr__(
            self,
            "gap_size",
            gap_size,
        )
        object.__setattr__(
            self,
            "max_folds",
            max_folds,
        )
        object.__setattr__(
            self,
            "selection_metric",
            self.selection_metric.strip(),
        )


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """Auditable chronological boundaries for one fold."""

    fold_number: int

    train_start_position: int
    train_end_position: int

    test_start_position: int
    test_end_position: int

    train_start: pd.Timestamp
    train_end: pd.Timestamp

    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        _positive_integer(
            "fold_number",
            self.fold_number,
        )

        for name in (
            "train_start_position",
            "train_end_position",
            "test_start_position",
            "test_end_position",
        ):
            _non_negative_integer(
                name,
                getattr(self, name),
            )

        if (
            self.train_start_position
            >= self.train_end_position
        ):
            raise ValueError(
                "Training positions are invalid."
            )

        if (
            self.test_start_position
            >= self.test_end_position
        ):
            raise ValueError(
                "Testing positions are invalid."
            )

        if (
            self.train_end_position
            > self.test_start_position
        ):
            raise ValueError(
                "Training and testing windows overlap."
            )

        if self.train_end >= self.test_start:
            raise ValueError(
                "Training timestamps must precede testing."
            )

    @property
    def train_size(self) -> int:
        return (
            self.train_end_position
            - self.train_start_position
        )

    @property
    def test_size(self) -> int:
        return (
            self.test_end_position
            - self.test_start_position
        )

    @property
    def gap_size(self) -> int:
        return (
            self.test_start_position
            - self.train_end_position
        )


def create_walk_forward_folds(
    history: pd.DataFrame,
    config: WalkForwardConfig,
) -> tuple[WalkForwardFold, ...]:
    """Create chronological expanding or rolling folds."""

    frame = _normalise_history(history)

    if not isinstance(
        config,
        WalkForwardConfig,
    ):
        raise ValueError(
            "config must be a WalkForwardConfig."
        )

    folds: list[WalkForwardFold] = []
    fold_number = 1

    train_end = config.train_size

    while True:
        test_start = (
            train_end
            + config.gap_size
        )

        test_end = (
            test_start
            + config.test_size
        )

        if test_end > len(frame):
            break

        if config.mode is WindowMode.EXPANDING:
            train_start = 0
        else:
            train_start = (
                train_end
                - config.train_size
            )

        fold = WalkForwardFold(
            fold_number=fold_number,
            train_start_position=train_start,
            train_end_position=train_end,
            test_start_position=test_start,
            test_end_position=test_end,
            train_start=frame.index[train_start],
            train_end=frame.index[train_end - 1],
            test_start=frame.index[test_start],
            test_end=frame.index[test_end - 1],
        )

        folds.append(fold)

        if (
            config.max_folds is not None
            and len(folds) >= config.max_folds
        ):
            break

        train_end += config.step_size
        fold_number += 1

    if not folds:
        raise ValueError(
            "history does not contain enough rows "
            "for one walk-forward fold."
        )

    return tuple(folds)


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """One candidate's in-sample evaluation."""

    parameters: Mapping[str, ParameterValue]
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameters",
            _normalise_parameters(
                self.parameters
            ),
        )

        object.__setattr__(
            self,
            "metrics",
            _normalise_metrics(
                self.metrics
            ),
        )


@dataclass(frozen=True, slots=True)
class FoldValidationResult:
    """Selected parameters and separated fold performance."""

    fold: WalkForwardFold

    selected_parameters: Mapping[
        str,
        ParameterValue,
    ]

    in_sample_metrics: Mapping[str, float]
    out_of_sample_metrics: Mapping[str, float]

    candidate_evaluations: tuple[
        CandidateEvaluation,
        ...,
    ]

    def __post_init__(self) -> None:
        if not isinstance(
            self.fold,
            WalkForwardFold,
        ):
            raise ValueError(
                "fold must be a WalkForwardFold."
            )

        selected_parameters = (
            _normalise_parameters(
                self.selected_parameters
            )
        )

        in_sample_metrics = _normalise_metrics(
            self.in_sample_metrics
        )

        out_of_sample_metrics = (
            _normalise_metrics(
                self.out_of_sample_metrics
            )
        )

        candidate_evaluations = tuple(
            self.candidate_evaluations
        )

        if not candidate_evaluations:
            raise ValueError(
                "candidate_evaluations cannot be empty."
            )

        if not all(
            isinstance(
                evaluation,
                CandidateEvaluation,
            )
            for evaluation in candidate_evaluations
        ):
            raise ValueError(
                "candidate_evaluations contains "
                "an invalid object."
            )

        if not any(
            dict(evaluation.parameters)
            == dict(selected_parameters)
            for evaluation in candidate_evaluations
        ):
            raise ValueError(
                "Selected parameters were not evaluated "
                "in-sample."
            )

        object.__setattr__(
            self,
            "selected_parameters",
            selected_parameters,
        )
        object.__setattr__(
            self,
            "in_sample_metrics",
            in_sample_metrics,
        )
        object.__setattr__(
            self,
            "out_of_sample_metrics",
            out_of_sample_metrics,
        )
        object.__setattr__(
            self,
            "candidate_evaluations",
            candidate_evaluations,
        )


@dataclass(frozen=True, slots=True)
class ParameterStabilityEntry:
    """Stability details for one selected parameter."""

    parameter: str
    selected_values: tuple[ParameterValue, ...]

    most_common_value: ParameterValue
    most_common_share: float

    unique_value_count: int
    change_count: int
    stability_score: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parameter, str)
            or not self.parameter.strip()
        ):
            raise ValueError(
                "parameter must be non-empty."
            )

        if not self.selected_values:
            raise ValueError(
                "selected_values cannot be empty."
            )

        if not 0 <= self.most_common_share <= 1:
            raise ValueError(
                "most_common_share must be between 0 and 1."
            )

        if not 0 <= self.stability_score <= 1:
            raise ValueError(
                "stability_score must be between 0 and 1."
            )


@dataclass(frozen=True, slots=True)
class ParameterStabilityReport:
    """Aggregate parameter stability across validation folds."""

    entries: tuple[ParameterStabilityEntry, ...]
    overall_stability_score: float

    def __post_init__(self) -> None:
        entries = tuple(self.entries)

        if not entries:
            raise ValueError(
                "entries cannot be empty."
            )

        if not all(
            isinstance(
                entry,
                ParameterStabilityEntry,
            )
            for entry in entries
        ):
            raise ValueError(
                "entries contains an invalid object."
            )

        if not 0 <= self.overall_stability_score <= 1:
            raise ValueError(
                "overall_stability_score must be "
                "between 0 and 1."
            )

        object.__setattr__(
            self,
            "entries",
            entries,
        )


def calculate_parameter_stability(
    fold_results: Sequence[
        FoldValidationResult
    ],
) -> ParameterStabilityReport:
    """Summarise selected-parameter drift across folds."""

    results = tuple(fold_results)

    if not results:
        raise ValueError(
            "At least one fold result is required."
        )

    parameter_names = tuple(
        results[0]
        .selected_parameters
        .keys()
    )

    expected_names = set(parameter_names)

    for result in results:
        if set(
            result.selected_parameters.keys()
        ) != expected_names:
            raise ValueError(
                "Selected parameter keys must remain "
                "consistent across folds."
            )

    entries: list[ParameterStabilityEntry] = []

    for parameter in parameter_names:
        values = tuple(
            result.selected_parameters[
                parameter
            ]
            for result in results
        )

        counts = Counter(values)

        most_common_value, frequency = sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                repr(item[0]),
            ),
        )[0]

        change_count = sum(
            current != previous
            for previous, current in zip(
                values,
                values[1:],
            )
        )

        stability_score = (
            1.0
            if len(values) == 1
            else (
                1.0
                - change_count
                / (len(values) - 1)
            )
        )

        entries.append(
            ParameterStabilityEntry(
                parameter=parameter,
                selected_values=values,
                most_common_value=(
                    most_common_value
                ),
                most_common_share=(
                    frequency / len(values)
                ),
                unique_value_count=len(counts),
                change_count=change_count,
                stability_score=stability_score,
            )
        )

    overall_stability_score = (
        sum(
            entry.stability_score
            for entry in entries
        )
        / len(entries)
    )

    return ParameterStabilityReport(
        entries=tuple(entries),
        overall_stability_score=(
            overall_stability_score
        ),
    )


@dataclass(frozen=True, slots=True)
class WalkForwardValidationReport:
    """Complete separated walk-forward validation output."""

    config: WalkForwardConfig
    fold_results: tuple[FoldValidationResult, ...]

    mean_in_sample_metric: float
    mean_out_of_sample_metric: float
    generalisation_gap: float

    parameter_stability: ParameterStabilityReport

    def __post_init__(self) -> None:
        if not isinstance(
            self.config,
            WalkForwardConfig,
        ):
            raise ValueError(
                "config must be a WalkForwardConfig."
            )

        fold_results = tuple(
            self.fold_results
        )

        if not fold_results:
            raise ValueError(
                "fold_results cannot be empty."
            )

        if not all(
            isinstance(
                result,
                FoldValidationResult,
            )
            for result in fold_results
        ):
            raise ValueError(
                "fold_results contains an invalid object."
            )

        for name in (
            "mean_in_sample_metric",
            "mean_out_of_sample_metric",
            "generalisation_gap",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    name,
                    getattr(self, name),
                ),
            )

        if not isinstance(
            self.parameter_stability,
            ParameterStabilityReport,
        ):
            raise ValueError(
                "parameter_stability must be a "
                "ParameterStabilityReport."
            )

        object.__setattr__(
            self,
            "fold_results",
            fold_results,
        )

    @property
    def fold_count(self) -> int:
        return len(self.fold_results)


def run_walk_forward_validation(
    history: pd.DataFrame,
    parameter_candidates: Sequence[
        ParameterMapping
    ],
    evaluator: MetricEvaluator,
    *,
    config: WalkForwardConfig,
) -> WalkForwardValidationReport:
    """Select parameters in-sample and evaluate them out-of-sample.

    The out-of-sample frame is never passed to the evaluator until after
    the winning candidate has been selected using only the training frame.
    """

    frame = _normalise_history(history)

    if not isinstance(
        config,
        WalkForwardConfig,
    ):
        raise ValueError(
            "config must be a WalkForwardConfig."
        )

    if not callable(evaluator):
        raise ValueError(
            "evaluator must be callable."
        )

    candidates = tuple(
        _normalise_parameters(candidate)
        for candidate in parameter_candidates
    )

    if not candidates:
        raise ValueError(
            "parameter_candidates cannot be empty."
        )

    expected_keys = set(candidates[0].keys())

    if any(
        set(candidate.keys()) != expected_keys
        for candidate in candidates
    ):
        raise ValueError(
            "All parameter candidates must use "
            "the same keys."
        )

    canonical_keys = [
        _parameter_key(candidate)
        for candidate in candidates
    ]

    if len(canonical_keys) != len(
        set(canonical_keys)
    ):
        raise ValueError(
            "parameter_candidates contains duplicates."
        )

    folds = create_walk_forward_folds(
        frame,
        config,
    )

    fold_results: list[
        FoldValidationResult
    ] = []

    metric_name = config.selection_metric

    for fold in folds:
        train_frame = frame.iloc[
            fold.train_start_position:
            fold.train_end_position
        ].copy()

        test_frame = frame.iloc[
            fold.test_start_position:
            fold.test_end_position
        ].copy()

        candidate_evaluations: list[
            CandidateEvaluation
        ] = []

        for candidate in candidates:
            metrics = _normalise_metrics(
                evaluator(
                    train_frame.copy(),
                    candidate,
                )
            )

            if metric_name not in metrics:
                raise ValueError(
                    f"Evaluator did not return "
                    f"selection metric {metric_name!r}."
                )

            candidate_evaluations.append(
                CandidateEvaluation(
                    parameters=candidate,
                    metrics=metrics,
                )
            )

        if config.higher_is_better:
            ordered_candidates = sorted(
                candidate_evaluations,
                key=lambda evaluation: (
                    -evaluation.metrics[
                        metric_name
                    ],
                    _parameter_key(
                        evaluation.parameters
                    ),
                ),
            )
        else:
            ordered_candidates = sorted(
                candidate_evaluations,
                key=lambda evaluation: (
                    evaluation.metrics[
                        metric_name
                    ],
                    _parameter_key(
                        evaluation.parameters
                    ),
                ),
            )

        selected = ordered_candidates[0]

        # This is the first point where the selected candidate sees
        # the out-of-sample frame.
        out_of_sample_metrics = (
            _normalise_metrics(
                evaluator(
                    test_frame.copy(),
                    selected.parameters,
                )
            )
        )

        if metric_name not in out_of_sample_metrics:
            raise ValueError(
                f"Evaluator did not return "
                f"selection metric {metric_name!r} "
                "for out-of-sample data."
            )

        fold_results.append(
            FoldValidationResult(
                fold=fold,
                selected_parameters=(
                    selected.parameters
                ),
                in_sample_metrics=(
                    selected.metrics
                ),
                out_of_sample_metrics=(
                    out_of_sample_metrics
                ),
                candidate_evaluations=tuple(
                    sorted(
                        candidate_evaluations,
                        key=lambda evaluation:
                        _parameter_key(
                            evaluation.parameters
                        ),
                    )
                ),
            )
        )

    mean_in_sample_metric = sum(
        result.in_sample_metrics[
            metric_name
        ]
        for result in fold_results
    ) / len(fold_results)

    mean_out_of_sample_metric = sum(
        result.out_of_sample_metrics[
            metric_name
        ]
        for result in fold_results
    ) / len(fold_results)

    if config.higher_is_better:
        generalisation_gap = (
            mean_in_sample_metric
            - mean_out_of_sample_metric
        )
    else:
        generalisation_gap = (
            mean_out_of_sample_metric
            - mean_in_sample_metric
        )

    stability = calculate_parameter_stability(
        fold_results
    )

    return WalkForwardValidationReport(
        config=config,
        fold_results=tuple(fold_results),
        mean_in_sample_metric=(
            mean_in_sample_metric
        ),
        mean_out_of_sample_metric=(
            mean_out_of_sample_metric
        ),
        generalisation_gap=generalisation_gap,
        parameter_stability=stability,
    )


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    """Minimum out-of-sample requirements for promotion."""

    minimum_folds: int = 3
    minimum_mean_out_of_sample_metric: float = 0.0
    maximum_generalisation_gap: float = 0.25
    minimum_parameter_stability: float = 0.50

    def __post_init__(self) -> None:
        minimum_folds = _positive_integer(
            "minimum_folds",
            self.minimum_folds,
        )

        minimum_oos = _finite_float(
            "minimum_mean_out_of_sample_metric",
            self.minimum_mean_out_of_sample_metric,
        )

        maximum_gap = _finite_float(
            "maximum_generalisation_gap",
            self.maximum_generalisation_gap,
        )

        minimum_stability = _finite_float(
            "minimum_parameter_stability",
            self.minimum_parameter_stability,
        )

        if maximum_gap < 0:
            raise ValueError(
                "maximum_generalisation_gap cannot "
                "be negative."
            )

        if not 0 <= minimum_stability <= 1:
            raise ValueError(
                "minimum_parameter_stability must "
                "be between 0 and 1."
            )

        object.__setattr__(
            self,
            "minimum_folds",
            minimum_folds,
        )
        object.__setattr__(
            self,
            "minimum_mean_out_of_sample_metric",
            minimum_oos,
        )
        object.__setattr__(
            self,
            "maximum_generalisation_gap",
            maximum_gap,
        )
        object.__setattr__(
            self,
            "minimum_parameter_stability",
            minimum_stability,
        )


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Accept/reject result based on out-of-sample evidence."""

    promoted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.promoted, bool):
            raise ValueError(
                "promoted must be boolean."
            )

        reasons = tuple(self.reasons)

        if not reasons:
            raise ValueError(
                "reasons cannot be empty."
            )

        if not all(
            isinstance(reason, str)
            and reason.strip()
            for reason in reasons
        ):
            raise ValueError(
                "reasons must contain non-empty strings."
            )

        object.__setattr__(
            self,
            "reasons",
            reasons,
        )


def assess_strategy_promotion(
    report: WalkForwardValidationReport,
    *,
    thresholds: PromotionThresholds | None = None,
) -> PromotionDecision:
    """Require acceptable out-of-sample performance for promotion."""

    if not isinstance(
        report,
        WalkForwardValidationReport,
    ):
        raise ValueError(
            "report must be a "
            "WalkForwardValidationReport."
        )

    thresholds = (
        thresholds
        or PromotionThresholds()
    )

    if not isinstance(
        thresholds,
        PromotionThresholds,
    ):
        raise ValueError(
            "thresholds must be PromotionThresholds."
        )

    reasons: list[str] = []

    if report.fold_count < thresholds.minimum_folds:
        reasons.append(
            "Insufficient out-of-sample folds: "
            f"{report.fold_count} available, "
            f"{thresholds.minimum_folds} required."
        )

    if (
        report.mean_out_of_sample_metric
        < thresholds
        .minimum_mean_out_of_sample_metric
    ):
        reasons.append(
            "Mean out-of-sample metric is below "
            "the promotion threshold."
        )

    if (
        report.generalisation_gap
        > thresholds.maximum_generalisation_gap
    ):
        reasons.append(
            "In-sample to out-of-sample "
            "generalisation gap is too large."
        )

    if (
        report
        .parameter_stability
        .overall_stability_score
        < thresholds.minimum_parameter_stability
    ):
        reasons.append(
            "Selected parameters are not stable "
            "across walk-forward folds."
        )

    if reasons:
        return PromotionDecision(
            promoted=False,
            reasons=tuple(reasons),
        )

    return PromotionDecision(
        promoted=True,
        reasons=(
            "Strategy met the required "
            "out-of-sample performance, "
            "generalisation and parameter-stability "
            "criteria.",
        ),
    )
