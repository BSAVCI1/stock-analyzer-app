from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.backtest import RegressionEvidence
from src.release_gate import (
    OperationalCheckStatus,
    P3ReleaseStatus,
    build_operational_reliability_report,
    evaluate_p3_release_gate,
)


T0 = datetime(
    2026,
    8,
    2,
    20,
    30,
    tzinfo=timezone.utc,
)


def metric(
    name: str,
    *,
    total: int = 1,
    successful: int = 1,
    failed: int = 0,
    pending_or_other: int = 0,
):
    return SimpleNamespace(
        name=name,
        total=total,
        successful=successful,
        failed=failed,
        pending_or_other=(
            pending_or_other
        ),
        provenance=SimpleNamespace(
            source_tables=(
                f"paper_{name}",
            ),
        ),
    )


def reliability(
    *,
    notifications=None,
    jobs=None,
):
    return SimpleNamespace(
        scans=metric("scans"),
        execution_runs=metric(
            "execution_runs"
        ),
        scheduled_jobs=(
            jobs
            or metric("scheduled_jobs")
        ),
        notifications=(
            notifications
            or metric("notifications")
        ),
        system_events=metric(
            "system_events"
        ),
    )


def broker_run(
    *,
    reconciled: bool = True,
    unresolved: int = 0,
):
    return SimpleNamespace(
        reconciliation_run_id=(
            "BRR-P3-PERSISTED"
        ),
        provider="deterministic-paper",
        status=SimpleNamespace(
            value=(
                "MATCHED"
                if reconciled
                else "DIFFERENCES"
            ),
        ),
        reconciled=reconciled,
        unresolved_item_count=unresolved,
        account_item_count=1,
        order_item_count=1,
        position_item_count=1,
        error_message=None,
    )


def snapshot(
    *,
    account_reconciled: bool = True,
    broker=None,
    reliability_summary=None,
):
    difference = (
        Decimal("0")
        if account_reconciled
        else Decimal("10")
    )

    return SimpleNamespace(
        generated_at=T0,
        account=SimpleNamespace(
            account_id="ACC-P3"
        ),
        reconciliation=SimpleNamespace(
            reconciled=account_reconciled,
            difference=difference,
        ),
        broker_reconciliation_run=(
            broker
            if broker is not None
            else broker_run()
        ),
        reliability=(
            reliability_summary
            or reliability()
        ),
    )


def descriptor(
    *,
    live: bool = False,
):
    return SimpleNamespace(
        live_trading_enabled=live
    )


def test_persisted_snapshot_builds_passing_report():
    report = (
        build_operational_reliability_report(
            snapshot(),
            execution_descriptor=(
                descriptor()
            ),
        )
    )

    assert report.passed is True

    assert (
        report
        .check_for(
            "account_reconciliation"
        )
        .status
        is OperationalCheckStatus.PASS
    )

    assert (
        report
        .check_for(
            "broker_reconciliation"
        )
        .status
        is OperationalCheckStatus.PASS
    )


def test_zero_records_are_not_observed():
    empty_notifications = metric(
        "notifications",
        total=0,
        successful=0,
    )

    report = (
        build_operational_reliability_report(
            snapshot(
                reliability_summary=(
                    reliability(
                        notifications=(
                            empty_notifications
                        )
                    )
                )
            ),
            execution_descriptor=(
                descriptor()
            ),
        )
    )

    check = report.check_for(
        "notifications"
    )

    assert (
        check.status
        is OperationalCheckStatus
        .NOT_OBSERVED
    )

    assert report.passed is False


def test_failed_metric_blocks_report():
    failed_notifications = metric(
        "notifications",
        total=2,
        successful=1,
        failed=1,
    )

    report = (
        build_operational_reliability_report(
            snapshot(
                reliability_summary=(
                    reliability(
                        notifications=(
                            failed_notifications
                        )
                    )
                )
            ),
            execution_descriptor=(
                descriptor()
            ),
        )
    )

    check = report.check_for(
        "notifications"
    )

    assert (
        check.status
        is OperationalCheckStatus.FAIL
    )

    assert check.failed_count == 1


def test_pending_metric_blocks_report():
    pending_jobs = metric(
        "scheduled_jobs",
        total=2,
        successful=1,
        pending_or_other=1,
    )

    report = (
        build_operational_reliability_report(
            snapshot(
                reliability_summary=(
                    reliability(
                        jobs=pending_jobs
                    )
                )
            ),
            execution_descriptor=(
                descriptor()
            ),
        )
    )

    assert (
        report
        .check_for("scheduled_jobs")
        .status
        is OperationalCheckStatus.FAIL
    )


def test_internal_reconciliation_failure_is_reported():
    report = (
        build_operational_reliability_report(
            snapshot(
                account_reconciled=False
            ),
            execution_descriptor=(
                descriptor()
            ),
        )
    )

    assert (
        report
        .check_for(
            "account_reconciliation"
        )
        .status
        is OperationalCheckStatus.FAIL
    )


def test_missing_broker_run_is_not_observed():
    source = snapshot()
    source.broker_reconciliation_run = None

    report = (
        build_operational_reliability_report(
            source,
            execution_descriptor=(
                descriptor()
            ),
        )
    )

    assert (
        report
        .check_for(
            "broker_reconciliation"
        )
        .status
        is OperationalCheckStatus
        .NOT_OBSERVED
    )

    assert (
        report
        .broker_reconciliation_run_id
        is None
    )


def test_broker_differences_are_reported():
    report = (
        build_operational_reliability_report(
            snapshot(
                broker=broker_run(
                    reconciled=False,
                    unresolved=2,
                )
            ),
            execution_descriptor=(
                descriptor()
            ),
        )
    )

    assert (
        report
        .unresolved_broker_differences
        == 2
    )

    assert (
        report
        .check_for(
            "broker_reconciliation"
        )
        .status
        is OperationalCheckStatus.FAIL
    )


def test_built_report_drives_ready_release():
    operational = (
        build_operational_reliability_report(
            snapshot(),
            execution_descriptor=(
                descriptor()
            ),
        )
    )

    release = evaluate_p3_release_gate(
        regression_evidence=(
            RegressionEvidence(
                passed=True,
                test_count=370,
                covered_phases=(
                    "P0",
                    "P1",
                    "P2",
                    "P3",
                ),
                workflow=(
                    "Automated tests"
                ),
            )
        ),
        operational_reliability=(
            operational
        ),
    )

    assert (
        release.status
        is P3ReleaseStatus.READY
    )

    assert release.release_ready is True


def test_live_descriptor_blocks_built_report():
    operational = (
        build_operational_reliability_report(
            snapshot(),
            execution_descriptor=(
                descriptor(live=True)
            ),
        )
    )

    release = evaluate_p3_release_gate(
        regression_evidence=(
            RegressionEvidence(
                passed=True,
                test_count=370,
                covered_phases=(
                    "P0",
                    "P1",
                    "P2",
                    "P3",
                ),
            )
        ),
        operational_reliability=(
            operational
        ),
    )

    assert operational.passed is False
    assert release.release_ready is False


def test_metric_counts_must_reconcile():
    malformed = metric(
        "notifications",
        total=1,
        successful=1,
        failed=1,
    )

    with pytest.raises(
        ValueError,
        match="do not reconcile",
    ):
        build_operational_reliability_report(
            snapshot(
                reliability_summary=(
                    reliability(
                        notifications=malformed
                    )
                )
            ),
            execution_descriptor=(
                descriptor()
            ),
        )


def test_internal_only_profile_allows_missing_broker():
    source = snapshot()
    source.broker_reconciliation_run = None

    report = (
        build_operational_reliability_report(
            source,
            execution_descriptor=(
                descriptor()
            ),
            broker_reconciliation_required=(
                False
            ),
        )
    )

    assert (
        report
        .check_for(
            "broker_reconciliation"
        )
        .status
        is OperationalCheckStatus
        .NOT_OBSERVED
    )

    assert (
        report
        .broker_reconciliation_required
        is False
    )

    assert report.passed is True


def test_broker_profile_requires_broker_run():
    source = snapshot()
    source.broker_reconciliation_run = None

    report = (
        build_operational_reliability_report(
            source,
            execution_descriptor=(
                descriptor()
            ),
            broker_reconciliation_required=(
                True
            ),
        )
    )

    assert report.passed is False

    assert (
        report
        .broker_reconciliation_required
        is True
    )
