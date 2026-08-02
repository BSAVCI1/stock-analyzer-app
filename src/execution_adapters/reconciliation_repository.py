"""Persistence for broker-paper reconciliation."""

from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from decimal import Decimal
from enum import Enum
import json
from pathlib import Path
from typing import (
    Mapping,
    Sequence,
)
from uuid import uuid4

from src.paper import (
    DEFAULT_DATABASE_PATH,
    connect_database,
    initialize_database,
    transaction,
)

from .reconciliation_models import (
    BrokerReconciliationCategory,
    BrokerReconciliationItem,
    BrokerReconciliationItemStatus,
    BrokerReconciliationRun,
    BrokerReconciliationRunStatus,
)


def _utc(
    value: datetime,
    *,
    field_name: str,
) -> datetime:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} must be "
            "timezone-aware."
        )

    return value.astimezone(
        timezone.utc
    )


def _timestamp(
    value: datetime,
) -> str:
    return _utc(
        value,
        field_name="datetime",
    ).isoformat()


def _datetime(
    value: str | None,
) -> datetime | None:
    if value is None:
        return None

    return datetime.fromisoformat(
        value
    ).astimezone(
        timezone.utc
    )


def _json_default(
    value: object,
) -> object:
    if isinstance(value, datetime):
        return _timestamp(value)

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    raise TypeError(
        "Unsupported reconciliation JSON "
        f"value: {type(value).__name__}."
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


def _json_load(
    value: str | None,
    *,
    fallback,
):
    if not value:
        return fallback

    return json.loads(value)


class BrokerReconciliationRepository:
    """Store immutable reconciliation evidence."""

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
    def _run_from_row(
        row,
    ) -> BrokerReconciliationRun:
        return BrokerReconciliationRun(
            reconciliation_run_id=(
                row[
                    "reconciliation_run_id"
                ]
            ),
            account_id=row["account_id"],
            reconciliation_key=(
                row["reconciliation_key"]
            ),
            provider=row["provider"],
            broker_account_id=(
                row["broker_account_id"]
            ),
            status=(
                BrokerReconciliationRunStatus(
                    row["status"]
                )
            ),
            started_at=_datetime(
                row["started_at"]
            ),
            completed_at=_datetime(
                row["completed_at"]
            ),
            account_item_count=(
                row["account_item_count"]
            ),
            order_item_count=(
                row["order_item_count"]
            ),
            position_item_count=(
                row["position_item_count"]
            ),
            matched_item_count=(
                row["matched_item_count"]
            ),
            mismatched_item_count=(
                row["mismatched_item_count"]
            ),
            missing_internal_item_count=(
                row[
                    "missing_internal_item_count"
                ]
            ),
            missing_broker_item_count=(
                row[
                    "missing_broker_item_count"
                ]
            ),
            metadata=_json_load(
                row["metadata_json"],
                fallback={},
            ),
            error_message=(
                row["error_message"]
            ),
        )

    @staticmethod
    def _item_from_row(
        row,
    ) -> BrokerReconciliationItem:
        return BrokerReconciliationItem(
            reconciliation_item_id=(
                row[
                    "reconciliation_item_id"
                ]
            ),
            reconciliation_run_id=(
                row[
                    "reconciliation_run_id"
                ]
            ),
            account_id=row["account_id"],
            category=(
                BrokerReconciliationCategory(
                    row["category"]
                )
            ),
            comparison_key=(
                row["comparison_key"]
            ),
            status=(
                BrokerReconciliationItemStatus(
                    row["status"]
                )
            ),
            internal_reference_ids=tuple(
                _json_load(
                    row[
                        "internal_reference_ids_json"
                    ],
                    fallback=[],
                )
            ),
            broker_reference_ids=tuple(
                _json_load(
                    row[
                        "broker_reference_ids_json"
                    ],
                    fallback=[],
                )
            ),
            differences=_json_load(
                row["differences_json"],
                fallback={},
            ),
            message=row["message"],
            created_at=_datetime(
                row["created_at"]
            ),
            metadata=_json_load(
                row["metadata_json"],
                fallback={},
            ),
        )

    def start_run(
        self,
        *,
        account_id: str,
        reconciliation_key: str,
        provider: str,
        broker_account_id: str,
        started_at: datetime,
        metadata: Mapping[
            str,
            object,
        ] | None = None,
    ) -> tuple[
        BrokerReconciliationRun,
        bool,
    ]:
        key = str(
            reconciliation_key
        ).strip()

        provider_name = str(
            provider
        ).strip()

        broker_id = str(
            broker_account_id
        ).strip()

        if not key:
            raise ValueError(
                "reconciliation_key is required."
            )

        if not provider_name:
            raise ValueError(
                "provider is required."
            )

        if not broker_id:
            raise ValueError(
                "broker_account_id is required."
            )

        at = _utc(
            started_at,
            field_name="started_at",
        )

        with transaction(
            self.database_path
        ) as connection:
            existing = connection.execute(
                """
                SELECT *
                FROM
                    paper_broker_reconciliation_runs
                WHERE account_id = ?
                  AND provider = ?
                  AND reconciliation_key = ?
                """,
                (
                    account_id,
                    provider_name,
                    key,
                ),
            ).fetchone()

            if existing is not None:
                return (
                    self._run_from_row(
                        existing
                    ),
                    False,
                )

            run_id = (
                "BRR-"
                + uuid4().hex
            )

            connection.execute(
                """
                INSERT INTO
                paper_broker_reconciliation_runs(
                    reconciliation_run_id,
                    account_id,
                    reconciliation_key,
                    provider,
                    broker_account_id,
                    status,
                    started_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    account_id,
                    key,
                    provider_name,
                    broker_id,
                    BrokerReconciliationRunStatus
                    .RUNNING.value,
                    _timestamp(at),
                    _json_dump(
                        dict(metadata or {})
                    ),
                ),
            )

            row = connection.execute(
                """
                SELECT *
                FROM
                    paper_broker_reconciliation_runs
                WHERE reconciliation_run_id = ?
                """,
                (run_id,),
            ).fetchone()

        return (
            self._run_from_row(row),
            True,
        )

    def get_run(
        self,
        reconciliation_run_id: str,
    ) -> BrokerReconciliationRun:
        connection = connect_database(
            self.database_path
        )

        try:
            row = connection.execute(
                """
                SELECT *
                FROM
                    paper_broker_reconciliation_runs
                WHERE reconciliation_run_id = ?
                """,
                (reconciliation_run_id,),
            ).fetchone()
        finally:
            connection.close()

        if row is None:
            raise ValueError(
                "Unknown broker reconciliation "
                f"run: {reconciliation_run_id}."
            )

        return self._run_from_row(row)

    def list_items(
        self,
        reconciliation_run_id: str,
    ) -> tuple[
        BrokerReconciliationItem,
        ...,
    ]:
        connection = connect_database(
            self.database_path
        )

        try:
            rows = connection.execute(
                """
                SELECT *
                FROM
                    paper_broker_reconciliation_items
                WHERE reconciliation_run_id = ?
                ORDER BY
                    category,
                    comparison_key,
                    reconciliation_item_id
                """,
                (reconciliation_run_id,),
            ).fetchall()
        finally:
            connection.close()

        return tuple(
            self._item_from_row(row)
            for row in rows
        )

    def list_runs(
        self,
        account_id: str,
    ) -> tuple[
        BrokerReconciliationRun,
        ...,
    ]:
        connection = connect_database(
            self.database_path
        )

        try:
            rows = connection.execute(
                """
                SELECT *
                FROM
                    paper_broker_reconciliation_runs
                WHERE account_id = ?
                ORDER BY
                    started_at,
                    reconciliation_run_id
                """,
                (account_id,),
            ).fetchall()
        finally:
            connection.close()

        return tuple(
            self._run_from_row(row)
            for row in rows
        )

    def latest_run(
        self,
        account_id: str,
    ) -> BrokerReconciliationRun | None:
        connection = connect_database(
            self.database_path
        )

        try:
            row = connection.execute(
                """
                SELECT *
                FROM
                    paper_broker_reconciliation_runs
                WHERE account_id = ?
                ORDER BY
                    started_at DESC,
                    reconciliation_run_id DESC
                LIMIT 1
                """,
                (account_id,),
            ).fetchone()
        finally:
            connection.close()

        return (
            self._run_from_row(row)
            if row is not None
            else None
        )

    def complete_run(
        self,
        reconciliation_run_id: str,
        *,
        items: Sequence[
            BrokerReconciliationItem
        ],
        completed_at: datetime,
    ) -> BrokerReconciliationRun:
        at = _utc(
            completed_at,
            field_name="completed_at",
        )

        item_tuple = tuple(items)

        for item in item_tuple:
            if (
                item.reconciliation_run_id
                != reconciliation_run_id
            ):
                raise ValueError(
                    "All reconciliation items "
                    "must belong to the run."
                )

        account_count = sum(
            item.category
            is BrokerReconciliationCategory
            .ACCOUNT
            for item in item_tuple
        )

        order_count = sum(
            item.category
            is BrokerReconciliationCategory
            .ORDER
            for item in item_tuple
        )

        position_count = sum(
            item.category
            is BrokerReconciliationCategory
            .POSITION
            for item in item_tuple
        )

        matched_count = sum(
            item.status
            is BrokerReconciliationItemStatus
            .MATCH
            for item in item_tuple
        )

        mismatched_count = sum(
            item.status
            is BrokerReconciliationItemStatus
            .MISMATCH
            for item in item_tuple
        )

        missing_internal_count = sum(
            item.status
            is BrokerReconciliationItemStatus
            .MISSING_INTERNAL
            for item in item_tuple
        )

        missing_broker_count = sum(
            item.status
            is BrokerReconciliationItemStatus
            .MISSING_BROKER
            for item in item_tuple
        )

        unresolved = (
            mismatched_count
            + missing_internal_count
            + missing_broker_count
        )

        status = (
            BrokerReconciliationRunStatus
            .MATCHED
            if unresolved == 0
            else
            BrokerReconciliationRunStatus
            .DIFFERENCES
        )

        with transaction(
            self.database_path
        ) as connection:
            existing = connection.execute(
                """
                SELECT status
                FROM
                    paper_broker_reconciliation_runs
                WHERE reconciliation_run_id = ?
                """,
                (reconciliation_run_id,),
            ).fetchone()

            if existing is None:
                raise ValueError(
                    "Unknown broker "
                    "reconciliation run: "
                    f"{reconciliation_run_id}."
                )

            if (
                existing["status"]
                != BrokerReconciliationRunStatus
                .RUNNING.value
            ):
                raise ValueError(
                    "Only RUNNING reconciliation "
                    "runs can be completed."
                )

            for item in item_tuple:
                connection.execute(
                    """
                    INSERT INTO
                    paper_broker_reconciliation_items(
                        reconciliation_item_id,
                        reconciliation_run_id,
                        account_id,
                        category,
                        comparison_key,
                        status,
                        internal_reference_ids_json,
                        broker_reference_ids_json,
                        differences_json,
                        message,
                        metadata_json,
                        created_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        item
                        .reconciliation_item_id,
                        item
                        .reconciliation_run_id,
                        item.account_id,
                        item.category.value,
                        item.comparison_key,
                        item.status.value,
                        _json_dump(
                            item
                            .internal_reference_ids
                        ),
                        _json_dump(
                            item
                            .broker_reference_ids
                        ),
                        _json_dump(
                            dict(
                                item.differences
                            )
                        ),
                        item.message,
                        _json_dump(
                            dict(item.metadata)
                        ),
                        _timestamp(
                            item.created_at
                        ),
                    ),
                )

            connection.execute(
                """
                UPDATE
                    paper_broker_reconciliation_runs
                SET status = ?,
                    completed_at = ?,
                    account_item_count = ?,
                    order_item_count = ?,
                    position_item_count = ?,
                    matched_item_count = ?,
                    mismatched_item_count = ?,
                    missing_internal_item_count = ?,
                    missing_broker_item_count = ?,
                    error_message = NULL
                WHERE reconciliation_run_id = ?
                """,
                (
                    status.value,
                    _timestamp(at),
                    account_count,
                    order_count,
                    position_count,
                    matched_count,
                    mismatched_count,
                    missing_internal_count,
                    missing_broker_count,
                    reconciliation_run_id,
                ),
            )

        return self.get_run(
            reconciliation_run_id
        )

    def fail_run(
        self,
        reconciliation_run_id: str,
        *,
        completed_at: datetime,
        error_message: str,
    ) -> BrokerReconciliationRun:
        at = _utc(
            completed_at,
            field_name="completed_at",
        )

        message = str(
            error_message
        ).strip()

        if not message:
            raise ValueError(
                "error_message cannot be empty."
            )

        with transaction(
            self.database_path
        ) as connection:
            cursor = connection.execute(
                """
                UPDATE
                    paper_broker_reconciliation_runs
                SET status = ?,
                    completed_at = ?,
                    error_message = ?
                WHERE reconciliation_run_id = ?
                  AND status = ?
                """,
                (
                    BrokerReconciliationRunStatus
                    .FAILED.value,
                    _timestamp(at),
                    message,
                    reconciliation_run_id,
                    BrokerReconciliationRunStatus
                    .RUNNING.value,
                ),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    "Unknown or non-running "
                    "broker reconciliation run."
                )

        return self.get_run(
            reconciliation_run_id
        )
