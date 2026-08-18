"""Persistent evidence for autonomous orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.paper import (
    DEFAULT_DATABASE_PATH,
    initialize_database,
    transaction,
)

from .orchestrator import (
    OrchestrationCycleReport,
)
from .schedule import ScheduledInvocation


def _timestamp(value: datetime) -> str:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            "Timestamp must be timezone-aware."
        )

    return value.astimezone(
        timezone.utc
    ).isoformat()


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


@dataclass(frozen=True, slots=True)
class PersistedInvocation:
    idempotency_key: str
    account_id: str
    policy_version: str
    job_kind: str
    scheduled_for: datetime
    strategy_horizon: str | None
    status: str
    first_seen_at: datetime
    completed_at: datetime | None
    attempt_count: int
    error_message: str | None
    updated_at: datetime


class OrchestrationRepository:
    """SQLite checkpoint and invocation evidence store."""

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
    def _from_row(row) -> PersistedInvocation:
        return PersistedInvocation(
            idempotency_key=(
                row["idempotency_key"]
            ),
            account_id=row["account_id"],
            policy_version=(
                row["policy_version"]
            ),
            job_kind=row["job_kind"],
            scheduled_for=_datetime(
                row["scheduled_for"]
            ),
            strategy_horizon=(
                row["strategy_horizon"]
            ),
            status=row["status"],
            first_seen_at=_datetime(
                row["first_seen_at"]
            ),
            completed_at=(
                _datetime(row["completed_at"])
                if row["completed_at"]
                else None
            ),
            attempt_count=int(
                row["attempt_count"]
            ),
            error_message=(
                row["error_message"]
            ),
            updated_at=_datetime(
                row["updated_at"]
            ),
        )

    def get_checkpoint(
        self,
        *,
        account_id: str,
        policy_version: str,
    ) -> datetime | None:
        with transaction(
            self.database_path
        ) as connection:
            row = connection.execute(
                """
                SELECT last_window_ended_at
                FROM paper_orchestration_checkpoints
                WHERE account_id = ?
                  AND policy_version = ?
                """,
                (
                    account_id,
                    policy_version,
                ),
            ).fetchone()

        return (
            _datetime(
                row["last_window_ended_at"]
            )
            if row is not None
            else None
        )

    def completed_keys(
        self,
        *,
        account_id: str,
        policy_version: str,
    ) -> frozenset[str]:
        with transaction(
            self.database_path
        ) as connection:
            rows = connection.execute(
                """
                SELECT idempotency_key
                FROM paper_orchestration_invocations
                WHERE account_id = ?
                  AND policy_version = ?
                  AND status IN (
                      'EXECUTED',
                      'DUPLICATE'
                  )
                """,
                (
                    account_id,
                    policy_version,
                ),
            ).fetchall()

        return frozenset(
            row["idempotency_key"]
            for row in rows
        )

    def record_missed(
        self,
        *,
        account_id: str,
        policy_version: str,
        invocations: tuple[
            ScheduledInvocation,
            ...,
        ],
        detected_at: datetime,
    ) -> int:
        recorded_at = _timestamp(
            detected_at
        )
        inserted = 0

        with transaction(
            self.database_path
        ) as connection:
            for item in invocations:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO
                    paper_orchestration_invocations(
                        idempotency_key,
                        account_id,
                        policy_version,
                        job_kind,
                        scheduled_for,
                        strategy_horizon,
                        status,
                        first_seen_at,
                        completed_at,
                        attempt_count,
                        error_message,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'MISSED',
                            ?, NULL, 0, ?, ?)
                    """,
                    (
                        item.idempotency_key,
                        account_id,
                        policy_version,
                        item.job_kind.value,
                        _timestamp(
                            item.scheduled_for
                        ),
                        (
                            item
                            .strategy_horizon
                            .value
                            if item
                            .strategy_horizon
                            is not None
                            else None
                        ),
                        recorded_at,
                        (
                            "Invocation fell outside "
                            "the bounded recovery "
                            "window."
                        ),
                        recorded_at,
                    ),
                )
                inserted += max(
                    cursor.rowcount,
                    0,
                )

        return inserted

    def record_cycle(
        self,
        *,
        account_id: str,
        policy_version: str,
        report: OrchestrationCycleReport,
        recorded_at: datetime,
    ) -> None:
        at = _timestamp(recorded_at)

        with transaction(
            self.database_path
        ) as connection:
            for result in report.results:
                item = result.invocation
                connection.execute(
                    """
                    INSERT INTO
                    paper_orchestration_invocations(
                        idempotency_key,
                        account_id,
                        policy_version,
                        job_kind,
                        scheduled_for,
                        strategy_horizon,
                        status,
                        first_seen_at,
                        completed_at,
                        attempt_count,
                        error_message,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                            ?, 1, ?, ?)
                    ON CONFLICT(idempotency_key)
                    DO UPDATE SET
                        status = excluded.status,
                        completed_at =
                            excluded.completed_at,
                        attempt_count =
                            paper_orchestration_invocations
                            .attempt_count + 1,
                        error_message =
                            excluded.error_message,
                        updated_at =
                            excluded.updated_at
                    """,
                    (
                        item.idempotency_key,
                        account_id,
                        policy_version,
                        item.job_kind.value,
                        _timestamp(
                            item.scheduled_for
                        ),
                        (
                            item
                            .strategy_horizon
                            .value
                            if item
                            .strategy_horizon
                            is not None
                            else None
                        ),
                        result.status.value,
                        at,
                        at,
                        result.error_message,
                        at,
                    ),
                )

            connection.execute(
                """
                INSERT INTO
                paper_orchestration_checkpoints(
                    account_id,
                    policy_version,
                    last_window_ended_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(
                    account_id,
                    policy_version
                )
                DO UPDATE SET
                    last_window_ended_at =
                        excluded.last_window_ended_at,
                    updated_at =
                        excluded.updated_at
                """,
                (
                    account_id,
                    policy_version,
                    _timestamp(
                        report.window_ended_at
                    ),
                    at,
                ),
            )

    def list_invocations(
        self,
        *,
        account_id: str,
        policy_version: str,
        status: str | None = None,
    ) -> tuple[PersistedInvocation, ...]:
        query = """
            SELECT *
            FROM paper_orchestration_invocations
            WHERE account_id = ?
              AND policy_version = ?
        """
        parameters: list[object] = [
            account_id,
            policy_version,
        ]

        if status is not None:
            query += " AND status = ?"
            parameters.append(status)

        query += (
            " ORDER BY scheduled_for, "
            "idempotency_key"
        )

        with transaction(
            self.database_path
        ) as connection:
            rows = connection.execute(
                query,
                tuple(parameters),
            ).fetchall()

        return tuple(
            self._from_row(row)
            for row in rows
        )
