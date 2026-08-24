"""P4.10.6 reproducible operational-reliability analytics tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.jobs import JobStatus, JobType
from src.paper import NotificationStatus
from src.portfolio_dashboard import (
    calculate_operational_reliability,
    operational_reliability_rows,
)


T0 = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


def _job(job_id, status, delay_seconds, *, scan_id="SCAN", run_id="RUN"):
    return SimpleNamespace(
        job_run_id=job_id,
        job_type=JobType.MARKET_CYCLE,
        status=status,
        scheduled_for=T0 + timedelta(hours=int(job_id[-1])),
        started_at=T0 + timedelta(
            hours=int(job_id[-1]), seconds=delay_seconds,
        ),
        scan_id=scan_id,
        execution_run_id=run_id,
    )


def test_operational_rates_use_explicit_persisted_denominators() -> None:
    jobs = (
        _job("JOB-1", JobStatus.COMPLETED, 60),
        _job("JOB-2", JobStatus.COMPLETED, 600, run_id=None),
        _job("JOB-3", JobStatus.FAILED, 120, scan_id=None, run_id=None),
        _job("JOB-4", JobStatus.RUNNING, 30, scan_id=None, run_id=None),
    )
    notifications = (
        SimpleNamespace(notification_id="N1", status=NotificationStatus.SENT),
        SimpleNamespace(notification_id="N2", status=NotificationStatus.FAILED),
        SimpleNamespace(notification_id="N3", status=NotificationStatus.PENDING),
    )
    events = (
        SimpleNamespace(event_id="E1", severity="INFO"),
        SimpleNamespace(event_id="E2", severity="CRITICAL"),
    )
    summary = calculate_operational_reliability(
        jobs=jobs, notifications=notifications, system_events=events,
        start_tolerance_seconds=300,
    )
    assert summary.job_count == 4
    assert summary.terminal_job_count == 3
    assert summary.successful_job_count == 2
    assert summary.job_success_rate_pct == pytest.approx(66.6666666667)
    assert summary.on_time_job_count == 3
    assert summary.on_time_start_rate_pct == 75.0
    assert summary.average_start_delay_seconds == 202.5
    assert summary.maximum_start_delay_seconds == 600
    assert summary.completed_market_cycles == 2
    assert summary.evidence_complete_cycles == 1
    assert summary.cycle_evidence_rate_pct == 50.0
    assert summary.terminal_notifications == 2
    assert summary.notification_delivery_rate_pct == 50.0
    assert summary.critical_system_events == 1
    rows = operational_reliability_rows(
        SimpleNamespace(operational_reliability=summary)
    )
    assert rows[2]["metric"] == "Cycle evidence complete"
    assert rows[2]["denominator"] == 2


def test_operational_empty_evidence_does_not_invent_perfect_rates() -> None:
    summary = calculate_operational_reliability(
        jobs=(), notifications=(), system_events=(),
    )
    assert summary.window_started_at is None
    assert summary.job_success_rate_pct is None
    assert summary.on_time_start_rate_pct is None
    assert summary.cycle_evidence_rate_pct is None
    assert summary.notification_delivery_rate_pct is None


def test_start_tolerance_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_operational_reliability(
            jobs=(), notifications=(), system_events=(),
            start_tolerance_seconds=-1,
        )

