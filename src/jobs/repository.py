"""SQLite persistence for scheduled job runs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import sqlite3
from typing import Mapping
from uuid import uuid4

from src.paper import (
    DEFAULT_DATABASE_PATH,
    connect_database,
    initialize_database,
    transaction,
)

from .models import (
    JobRun,
    JobStatus,
    JobType,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _timestamp(
    value: datetime,
) -> str:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            "Timestamp must be "
            "timezone-aware."
        )

    return value.astimezone(
        timezone.utc
    ).isoformat()


def _datetime(
    value: str | None,
) -> datetime | None:
    if value is None:
        return None

    return datetime.fromisoformat(value)


def _json_default(
    value: object,
) -> object:
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Enum):
        return value.value

    raise TypeError(
        f"Unsupported JSON type: "
        f"{type(value).__name__}."
    )


def _json_dump(
    value: object,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


class JobRepository:
    """Persistent and idempotent job-run store."""

    def __init__(
        self,
        database_path: str | Path = (
            DEFAULT_DATABASE_PATH
        ),
    ) -> None:
        self.database_path = Path(
            database_path
        )

        initialize_database(
            self.database_path
        )

    @staticmethod
    def _from_row(
        row: sqlite3.Row,
    ) -> JobRun:
        return JobRun(
            job_run_id=row["job_run_id"],
            account_id=row["account_id"],
            job_key=row["job_key"],
            job_type=JobType(
                row["job_type"]
            ),
            scheduled_for=_datetime(
                row["scheduled_for"]
            ),
            exchange_code=(
                row["exchange_code"]
            ),
            status=JobStatus(
                row["status"]
            ),
            started_at=_datetime(
                row["started_at"]
            ),
            completed_at=_datetime(
                row["completed_at"]
            ),
            scan_id=row["scan_id"],
            execution_run_id=(
                row["execution_run_id"]
            ),
            queued_notifications=int(
                row[
                    "queued_notifications"
                ]
            ),
            sent_notifications=int(
                row[
                    "sent_notifications"
                ]
            ),
            failed_notifications=int(
                row[
                    "failed_notifications"
                ]
            ),
            metadata=json.loads(
                row["metadata_json"]
            ),
            error_message=(
                row["error_message"]
            ),
        )

    def start_job(
        self,
        *,
        account_id: str,
        job_key: str,
        job_type: JobType,
        scheduled_for: datetime,
        exchange_code: str,
        metadata: Mapping[
            str,
            object,
        ] | None = None,
        started_at: datetime | None = None,
    ) -> tuple[JobRun, bool]:
        key = str(job_key).strip()

        if not key:
            raise ValueError(
                "job_key cannot be empty."
            )

        if not isinstance(
            job_type,
            JobType,
        ):
            raise ValueError(
                "job_type must be a JobType."
            )

        at = started_at or scheduled_for

        with transaction(
            self.database_path
        ) as connection:
            account = connection.execute(
                """
                SELECT account_id
                FROM paper_accounts
                WHERE account_id = ?
                """,
                (account_id,),
            ).fetchone()

            if account is None:
                raise ValueError(
                    f"Unknown account: "
                    f"{account_id}."
                )

            existing = connection.execute(
                """
                SELECT *
                FROM paper_job_runs
                WHERE account_id = ?
                  AND job_key = ?
                """,
                (
                    account_id,
                    key,
                ),
            ).fetchone()

            if existing is not None:
                return (
                    self._from_row(
                        existing
                    ),
                    False,
                )

            job_run_id = _new_id("JOB")

            connection.execute(
                """
                INSERT INTO paper_job_runs(
                    job_run_id,
                    account_id,
                    job_key,
                    job_type,
                    scheduled_for,
                    exchange_code,
                    status,
                    started_at,
                    completed_at,
                    scan_id,
                    execution_run_id,
                    queued_notifications,
                    sent_notifications,
                    failed_notifications,
                    metadata_json,
                    error_message
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    NULL, NULL, NULL,
                    0, 0, 0, ?, NULL
                )
                """,
                (
                    job_run_id,
                    account_id,
                    key,
                    job_type.value,
                    _timestamp(
                        scheduled_for
                    ),
                    (
                        str(exchange_code)
                        .strip()
                        .upper()
                    ),
                    JobStatus.RUNNING.value,
                    _timestamp(at),
                    _json_dump(
                        dict(metadata or {})
                    ),
                ),
            )

        return (
            self.get_job(job_run_id),
            True,
        )

    def get_job(
        self,
        job_run_id: str,
    ) -> JobRun:
        connection = connect_database(
            self.database_path
        )

        try:
            row = connection.execute(
                """
                SELECT *
                FROM paper_job_runs
                WHERE job_run_id = ?
                """,
                (job_run_id,),
            ).fetchone()
        finally:
            connection.close()

        if row is None:
            raise ValueError(
                f"Unknown job run: "
                f"{job_run_id}."
            )

        return self._from_row(row)

    def complete_job(
        self,
        job_run_id: str,
        *,
        status: JobStatus,
        completed_at: datetime,
        scan_id: str | None = None,
        execution_run_id: str | None = None,
        queued_notifications: int = 0,
        sent_notifications: int = 0,
        failed_notifications: int = 0,
        metadata: Mapping[
            str,
            object,
        ] | None = None,
        error_message: str | None = None,
    ) -> JobRun:
        if status is JobStatus.RUNNING:
            raise ValueError(
                "A completed job cannot "
                "remain RUNNING."
            )

        for name, value in (
            (
                "queued_notifications",
                queued_notifications,
            ),
            (
                "sent_notifications",
                sent_notifications,
            ),
            (
                "failed_notifications",
                failed_notifications,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be "
                    "non-negative."
                )

        with transaction(
            self.database_path
        ) as connection:
            existing = connection.execute(
                """
                SELECT job_run_id
                FROM paper_job_runs
                WHERE job_run_id = ?
                """,
                (job_run_id,),
            ).fetchone()

            if existing is None:
                raise ValueError(
                    f"Unknown job run: "
                    f"{job_run_id}."
                )

            connection.execute(
                """
                UPDATE paper_job_runs
                SET status = ?,
                    completed_at = ?,
                    scan_id = ?,
                    execution_run_id = ?,
                    queued_notifications = ?,
                    sent_notifications = ?,
                    failed_notifications = ?,
                    metadata_json = ?,
                    error_message = ?
                WHERE job_run_id = ?
                """,
                (
                    status.value,
                    _timestamp(
                        completed_at
                    ),
                    scan_id,
                    execution_run_id,
                    queued_notifications,
                    sent_notifications,
                    failed_notifications,
                    _json_dump(
                        dict(metadata or {})
                    ),
                    (
                        str(error_message)
                        .strip()
                        if error_message
                        else None
                    ),
                    job_run_id,
                ),
            )

        return self.get_job(job_run_id)

    def list_jobs(
        self,
        account_id: str,
    ) -> tuple[JobRun, ...]:
        connection = connect_database(
            self.database_path
        )

        try:
            rows = connection.execute(
                """
                SELECT *
                FROM paper_job_runs
                WHERE account_id = ?
                ORDER BY
                    scheduled_for,
                    job_run_id
                """,
                (account_id,),
            ).fetchall()
        finally:
            connection.close()

        return tuple(
            self._from_row(row)
            for row in rows
        )
