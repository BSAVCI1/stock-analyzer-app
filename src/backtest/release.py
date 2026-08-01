"""Deterministic P2 release-gate eligibility.

This module does not schedule alerts, connect to brokers or execute orders.
It only determines whether a strategy is eligible for a future alert-
scheduling layer after passing the documented P2 release requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .acceptance import StrategyAcceptanceReport


APPROVED_THRESHOLD_STATUS = (
    "APPROVED_FOR_P2_RELEASE"
)

REQUIRED_REGRESSION_PHASES = (
    "P0",
    "P1",
    "P2",
)

DEFAULT_P2_LIMITATIONS = (
    "No OpenAI or other AI-platform connection.",
    "No broker connection or live order execution.",
    "No background alerts or scheduled watchlist scans.",
)


class ReleaseGateStatus(str, Enum):
    """Final eligibility status after the P2 release gate."""

    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"


def _required_text(
    name: str,
    value: object,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    result = value.strip()

    if not result:
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return result


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


def _validate_json_value(
    name: str,
    value: object,
) -> None:
    if value is None:
        return

    if isinstance(value, bool):
        return

    if isinstance(value, int):
        return

    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(
                f"{name} must be finite."
            )
        return

    if isinstance(value, str):
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(
                f"{name}[{index}]",
                item,
            )
        return

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = _required_text(
                f"{name} key",
                raw_key,
            )
            _validate_json_value(
                f"{name}.{key}",
                item,
            )
        return

    raise ValueError(
        f"{name} contains unsupported value "
        f"{type(value).__name__}."
    )


def _freeze_mapping(
    value: Mapping[str, object],
) -> Mapping[str, object]:
    frozen: dict[str, object] = {}

    for raw_key, item in sorted(
        value.items(),
        key=lambda pair: str(pair[0]),
    ):
        key = _required_text(
            "mapping key",
            raw_key,
        )

        if isinstance(item, Mapping):
            frozen[key] = _freeze_mapping(item)
        elif isinstance(item, list):
            frozen[key] = tuple(item)
        else:
            frozen[key] = item

    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class RegressionEvidence:
    """Evidence that the complete P0–P2 suite passed."""

    passed: bool
    test_count: int
    covered_phases: tuple[str, ...]
    workflow: str = "Automated tests"

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ValueError(
                "passed must be boolean."
            )

        test_count = _positive_integer(
            "test_count",
            self.test_count,
        )

        phases = tuple(
            _required_text(
                "covered phase",
                phase,
            ).upper()
            for phase in self.covered_phases
        )

        if not phases:
            raise ValueError(
                "covered_phases cannot be empty."
            )

        if len(phases) != len(set(phases)):
            raise ValueError(
                "covered_phases contains duplicates."
            )

        workflow = _required_text(
            "workflow",
            self.workflow,
        )

        object.__setattr__(
            self,
            "test_count",
            test_count,
        )
        object.__setattr__(
            self,
            "covered_phases",
            phases,
        )
        object.__setattr__(
            self,
            "workflow",
            workflow,
        )


@dataclass(frozen=True, slots=True)
class ApprovedThresholdManifest:
    """Versioned collection of explicitly approved thresholds."""

    schema_version: int
    approval_status: str
    profiles: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        schema_version = _positive_integer(
            "schema_version",
            self.schema_version,
        )

        approval_status = _required_text(
            "approval_status",
            self.approval_status,
        )

        if not isinstance(self.profiles, Mapping):
            raise ValueError(
                "profiles must be a mapping."
            )

        if not self.profiles:
            raise ValueError(
                "profiles cannot be empty."
            )

        normalised: dict[
            str,
            Mapping[str, object],
        ] = {}

        for raw_name, raw_profile in self.profiles.items():
            name = _required_text(
                "profile name",
                raw_name,
            )

            if not isinstance(
                raw_profile,
                Mapping,
            ):
                raise ValueError(
                    f"Profile {name} must be a mapping."
                )

            profile_class = _required_text(
                f"{name}.class",
                raw_profile.get("class"),
            )

            values = raw_profile.get("values")

            if not isinstance(values, Mapping):
                raise ValueError(
                    f"{name}.values must be a mapping."
                )

            if not values:
                raise ValueError(
                    f"{name}.values cannot be empty."
                )

            _validate_json_value(
                f"{name}.values",
                values,
            )

            normalised[name] = _freeze_mapping(
                {
                    "class": profile_class,
                    "values": dict(values),
                }
            )

        object.__setattr__(
            self,
            "schema_version",
            schema_version,
        )
        object.__setattr__(
            self,
            "approval_status",
            approval_status,
        )
        object.__setattr__(
            self,
            "profiles",
            MappingProxyType(
                dict(sorted(normalised.items()))
            ),
        )

    @property
    def approved(self) -> bool:
        return (
            self.approval_status
            == APPROVED_THRESHOLD_STATUS
        )


def load_approved_threshold_manifest(
    path: str | Path,
) -> ApprovedThresholdManifest:
    """Load and validate a committed threshold manifest."""

    manifest_path = Path(path)

    if not manifest_path.is_file():
        raise ValueError(
            f"Threshold manifest does not exist: "
            f"{manifest_path}."
        )

    try:
        raw = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            "Threshold manifest could not be read."
        ) from exc

    if not isinstance(raw, Mapping):
        raise ValueError(
            "Threshold manifest must contain "
            "a JSON object."
        )

    return ApprovedThresholdManifest(
        schema_version=raw.get(
            "schema_version"
        ),
        approval_status=raw.get(
            "approval_status"
        ),
        profiles=raw.get("profiles"),
    )


@dataclass(frozen=True, slots=True)
class P2ReleaseGateReport:
    """Auditable final P2 eligibility decision."""

    strategy: str
    status: ReleaseGateStatus

    alert_scheduling_eligible: bool

    acceptance_report: StrategyAcceptanceReport
    regression_evidence: RegressionEvidence
    threshold_manifest: ApprovedThresholdManifest

    documented_limitations: tuple[str, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        strategy = _required_text(
            "strategy",
            self.strategy,
        )

        if not isinstance(
            self.status,
            ReleaseGateStatus,
        ):
            raise ValueError(
                "status must be a ReleaseGateStatus."
            )

        if not isinstance(
            self.alert_scheduling_eligible,
            bool,
        ):
            raise ValueError(
                "alert_scheduling_eligible "
                "must be boolean."
            )

        if not isinstance(
            self.acceptance_report,
            StrategyAcceptanceReport,
        ):
            raise ValueError(
                "acceptance_report must be a "
                "StrategyAcceptanceReport."
            )

        if not isinstance(
            self.regression_evidence,
            RegressionEvidence,
        ):
            raise ValueError(
                "regression_evidence must be "
                "RegressionEvidence."
            )

        if not isinstance(
            self.threshold_manifest,
            ApprovedThresholdManifest,
        ):
            raise ValueError(
                "threshold_manifest must be an "
                "ApprovedThresholdManifest."
            )

        limitations = tuple(
            _required_text(
                "documented limitation",
                limitation,
            )
            for limitation
            in self.documented_limitations
        )

        if not limitations:
            raise ValueError(
                "documented_limitations cannot "
                "be empty."
            )

        reasons = tuple(
            _required_text(
                "release-gate reason",
                reason,
            )
            for reason in self.reasons
        )

        if not reasons:
            raise ValueError(
                "reasons cannot be empty."
            )

        expected_status = (
            ReleaseGateStatus.ELIGIBLE
            if self.alert_scheduling_eligible
            else ReleaseGateStatus.INELIGIBLE
        )

        if self.status is not expected_status:
            raise ValueError(
                "status does not match eligibility."
            )

        object.__setattr__(
            self,
            "strategy",
            strategy,
        )
        object.__setattr__(
            self,
            "documented_limitations",
            limitations,
        )
        object.__setattr__(
            self,
            "reasons",
            reasons,
        )


def evaluate_p2_release_gate(
    acceptance_report: StrategyAcceptanceReport,
    *,
    regression_evidence: RegressionEvidence,
    threshold_manifest: ApprovedThresholdManifest,
    documented_limitations: tuple[
        str,
        ...,
    ] = DEFAULT_P2_LIMITATIONS,
) -> P2ReleaseGateReport:
    """Determine eligibility for a future alert-scheduling layer.

    This function does not schedule an alert. It only marks a validated
    strategy as eligible for such functionality in a later phase.
    """

    if not isinstance(
        acceptance_report,
        StrategyAcceptanceReport,
    ):
        raise ValueError(
            "acceptance_report must be a "
            "StrategyAcceptanceReport."
        )

    if not isinstance(
        regression_evidence,
        RegressionEvidence,
    ):
        raise ValueError(
            "regression_evidence must be "
            "RegressionEvidence."
        )

    if not isinstance(
        threshold_manifest,
        ApprovedThresholdManifest,
    ):
        raise ValueError(
            "threshold_manifest must be an "
            "ApprovedThresholdManifest."
        )

    reasons: list[str] = []

    if not acceptance_report.accepted:
        reasons.append(
            "Strategy acceptance report is rejected."
        )

    if not regression_evidence.passed:
        reasons.append(
            "The P0–P2 regression suite did not pass."
        )

    covered = set(
        regression_evidence.covered_phases
    )

    missing_phases = [
        phase
        for phase in REQUIRED_REGRESSION_PHASES
        if phase not in covered
    ]

    if missing_phases:
        reasons.append(
            "Regression evidence is missing phases: "
            + ", ".join(missing_phases)
            + "."
        )

    if not threshold_manifest.approved:
        reasons.append(
            "Signal-threshold manifest is not approved "
            "for the P2 release."
        )

    limitations = tuple(
        documented_limitations
    )

    if not limitations:
        raise ValueError(
            "documented_limitations cannot be empty."
        )

    eligible = not reasons

    if eligible:
        reasons = [
            "Strategy passed deterministic acceptance, "
            "P0–P2 regression and approved-threshold "
            "requirements. It is eligible for a future "
            "alert-scheduling layer, but no alert has "
            "been scheduled."
        ]

    return P2ReleaseGateReport(
        strategy=acceptance_report.strategy,
        status=(
            ReleaseGateStatus.ELIGIBLE
            if eligible
            else ReleaseGateStatus.INELIGIBLE
        ),
        alert_scheduling_eligible=eligible,
        acceptance_report=acceptance_report,
        regression_evidence=regression_evidence,
        threshold_manifest=threshold_manifest,
        documented_limitations=limitations,
        reasons=tuple(reasons),
    )
