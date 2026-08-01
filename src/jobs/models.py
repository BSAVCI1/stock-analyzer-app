"""Domain models for scheduled paper-trading jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Mapping


class JobType(str, Enum):
    MARKET_CYCLE = "MARKET_CYCLE"
    WEEKLY_REPORT = "WEEKLY_REPORT"


class JobStatus(str, Enum):
    RUNNING = "RUNNING"
    SKIPPED = "SKIPPED"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_ERRORS = (
        "COMPLETED_WITH_ERRORS"
    )
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ExchangeSession:
    exchange_code: str
    session_date: date

    opens_at: datetime
    closes_at: datetime


@dataclass(frozen=True, slots=True)
class JobRun:
    job_run_id: str
    account_id: str

    job_key: str
    job_type: JobType

    scheduled_for: datetime
    exchange_code: str

    status: JobStatus

    started_at: datetime
    completed_at: datetime | None

    scan_id: str | None
    execution_run_id: str | None

    queued_notifications: int
    sent_notifications: int
    failed_notifications: int

    metadata: Mapping[str, object] = field(
        default_factory=dict
    )

    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class ScheduledJobReport:
    job: JobRun

    session: ExchangeSession | None

    duplicate: bool
    skipped_reason: str | None = None
