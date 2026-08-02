from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from src.backtest import RegressionEvidence
from src.release_gate import (
    OperationalCheckStatus,
    OperationalReliabilityCheck,
    OperationalReliabilityReport,
    P3ReleaseStatus,
    REQUIRED_OPERATIONAL_CHECKS,
    evaluate_p3_release_gate,
)


T0 = datetime(
    2026,
    8,
    2,
    20,
    0,
    tzinfo=timezone.utc,
)


def make_regression(
    *,
    passed: bool = True,
    phases=(
        "P0",
        "P1",
        "P2",
        "P3",
    ),
) -> RegressionEvidence:
    return RegressionEvidence(
        passed=passed,
        test_count=360,
        covered_phases=tuple(phases),
        workflow="Automated tests #43",
    )


def make_checks():
    return tuple(
        OperationalReliabilityCheck(
            name=name,
            status=(
                OperationalCheckStatus.PASS
            ),
            observed_count=1,
            failed_count=0,
            details=(
                "Persisted evidence verified.",
            ),
        )
        for name
        in REQUIRED_OPERATIONAL_CHECKS
    )


def make_reliability(
    *,
    replacement_check:
    OperationalReliabilityCheck | None = None,
    broker_run_id: str | None = (
        "BRR-P3-RELEASE"
    ),
    unresolved: int = 0,
    live_trading_enabled: bool = False,
) -> OperationalReliabilityReport:
    checks = list(make_checks())

    if replacement_check is not None:
        checks = [
            (
                replacement_check
                if (
                    check.name
                    == replacement_check.name
                )
                else check
            )
            for check in checks
        ]

    return OperationalReliabilityReport(
        generated_at=T0,
        account_id="ACC-P3-RELEASE",
        checks=tuple(checks),
        broker_reconciliation_run_id=(
            broker_run_id
        ),
        unresolved_broker_differences=(
            unresolved
        ),
        live_trading_enabled=(
            live_trading_enabled
        ),
    )


def failed_check(
    name: str,
) -> OperationalReliabilityCheck:
    return OperationalReliabilityCheck(
        name=name,
        status=OperationalCheckStatus.FAIL,
        observed_count=1,
        failed_count=1,
        details=(
            "One failed persisted record.",
        ),
    )


def not_observed_check(
    name: str,
) -> OperationalReliabilityCheck:
    return OperationalReliabilityCheck(
        name=name,
        status=(
            OperationalCheckStatus
            .NOT_OBSERVED
        ),
        observed_count=0,
        failed_count=0,
        details=(
            "No persisted evidence exists.",
        ),
    )


def test_operational_report_passes() -> None:
    report = make_reliability()

    assert report.passed is True

    assert (
        report
        .check_for("broker_reconciliation")
        .status
        is OperationalCheckStatus.PASS
    )

    assert report.non_passing_checks == ()


def test_complete_p3_evidence_is_ready() -> None:
    report = evaluate_p3_release_gate(
        regression_evidence=(
            make_regression()
        ),
        operational_reliability=(
            make_reliability()
        ),
    )

    assert (
        report.status
        is P3ReleaseStatus.READY
    )

    assert report.release_ready is True


def test_failed_regression_blocks_release() -> None:
    report = evaluate_p3_release_gate(
        regression_evidence=(
            make_regression(
                passed=False
            )
        ),
        operational_reliability=(
            make_reliability()
        ),
    )

    assert (
        report.status
        is P3ReleaseStatus.BLOCKED
    )

    assert report.release_ready is False

    assert any(
        "regression" in reason.lower()
        for reason in report.reasons
    )


def test_missing_p3_phase_blocks_release() -> None:
    report = evaluate_p3_release_gate(
        regression_evidence=(
            make_regression(
                phases=(
                    "P0",
                    "P1",
                    "P2",
                )
            )
        ),
        operational_reliability=(
            make_reliability()
        ),
    )

    assert report.release_ready is False

    assert any(
        "P3" in reason
        for reason in report.reasons
    )


def test_failed_operational_check_blocks() -> None:
    report = evaluate_p3_release_gate(
        regression_evidence=(
            make_regression()
        ),
        operational_reliability=(
            make_reliability(
                replacement_check=(
                    failed_check(
                        "notifications"
                    )
                )
            )
        ),
    )

    assert report.release_ready is False

    assert any(
        "notifications=FAIL" in reason
        for reason in report.reasons
    )


def test_unresolved_broker_difference_blocks() -> None:
    report = evaluate_p3_release_gate(
        regression_evidence=(
            make_regression()
        ),
        operational_reliability=(
            make_reliability(
                replacement_check=(
                    failed_check(
                        "broker_reconciliation"
                    )
                ),
                unresolved=1,
            )
        ),
    )

    assert report.release_ready is False

    assert any(
        "unresolved" in reason.lower()
        for reason in report.reasons
    )


def test_missing_broker_run_blocks() -> None:
    report = evaluate_p3_release_gate(
        regression_evidence=(
            make_regression()
        ),
        operational_reliability=(
            make_reliability(
                replacement_check=(
                    not_observed_check(
                        "broker_reconciliation"
                    )
                ),
                broker_run_id=None,
            )
        ),
    )

    assert report.release_ready is False

    assert any(
        "no persisted broker-paper" in
        reason.lower()
        for reason in report.reasons
    )


def test_live_trading_blocks_release() -> None:
    report = evaluate_p3_release_gate(
        regression_evidence=(
            make_regression()
        ),
        operational_reliability=(
            make_reliability(
                live_trading_enabled=True
            )
        ),
    )

    assert report.release_ready is False

    assert any(
        "live trading" in reason.lower()
        for reason in report.reasons
    )


def test_required_checks_cannot_be_omitted() -> None:
    checks = tuple(
        check
        for check in make_checks()
        if check.name != "system_events"
    )

    with pytest.raises(
        ValueError,
        match="system_events",
    ):
        OperationalReliabilityReport(
            generated_at=T0,
            account_id="ACC-P3",
            checks=checks,
            broker_reconciliation_run_id=(
                "BRR-P3"
            ),
            unresolved_broker_differences=0,
            live_trading_enabled=False,
        )


def test_release_gate_is_deterministic() -> None:
    arguments = {
        "regression_evidence":
        make_regression(),
        "operational_reliability":
        make_reliability(),
    }

    first = evaluate_p3_release_gate(
        **arguments
    )

    second = evaluate_p3_release_gate(
        **arguments
    )

    assert first == second
