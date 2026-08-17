"""SQLite persistence for automatic market scans."""

from __future__ import annotations

from src.strategy import (
    coerce_strategy_horizon,
    normalise_strategy_version,
    strategy_horizon_value,
)

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
import json
from pathlib import Path
import sqlite3
from typing import Mapping
from uuid import uuid4

from src.paper.database import (
    DEFAULT_DATABASE_PATH,
    connect_database,
    transaction,
)
from src.paper.migrations import (
    initialize_database,
)

from .models import (
    MarketScan,
    MarketScanReport,
    ScanResult,
    ScanResultStatus,
    ScanStatus,
    StockUniverse,
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
            "Timestamp must be timezone-aware."
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
    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    raise TypeError(
        f"Unsupported JSON value: "
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


def _optional_float(
    value: object,
) -> float | None:
    if value is None:
        return None

    return float(value)


class ScannerRepository:
    """Persistent repository for scans and per-symbol outcomes."""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
    ) -> None:
        self.database_path = Path(database_path)
        initialize_database(self.database_path)

    @staticmethod
    def _scan_from_row(
        row: sqlite3.Row,
    ) -> MarketScan:
        return MarketScan(
            scan_id=row["scan_id"],
            account_id=row["account_id"],
            scan_key=row["scan_key"],
            universe=row["universe"],
            status=ScanStatus(row["status"]),
            started_at=_datetime(
                row["started_at"]
            ),
            completed_at=_datetime(
                row["completed_at"]
            ),
            requested_count=int(
                row["requested_count"]
            ),
            processed_count=int(
                row["processed_count"]
            ),
            rejected_count=int(
                row["rejected_count"]
            ),
            signal_count=int(
                row["signal_count"]
            ),
            order_count=int(
                row["order_count"]
            ),
            configuration=json.loads(
                row["configuration_json"]
            ),
            app_version=row["app_version"],
            error_message=(
                row["error_message"]
            ),
        )

    @staticmethod
    def _result_from_row(
        row: sqlite3.Row,
    ) -> ScanResult:
        return ScanResult(
            result_id=row["result_id"],
            scan_id=row["scan_id"],
            account_id=row["account_id"],
            symbol=row["symbol"],
            status=ScanResultStatus(
                row["status"]
            ),
            processed_at=_datetime(
                row["processed_at"]
            ),
            data_as_of=_datetime(
                row["data_as_of"]
            ),
            history_rows=int(
                row["history_rows"]
            ),
            latest_price=_optional_float(
                row["latest_price"]
            ),
            average_volume=_optional_float(
                row["average_volume"]
            ),
            average_dollar_volume=(
                _optional_float(
                    row[
                        "average_dollar_volume"
                    ]
                )
            ),
            recommendation=(
                row["recommendation"]
            ),
            strategy=row["strategy"],
            strategy_horizon=(
                coerce_strategy_horizon(
                    row["strategy_horizon"]
                )
            ),
            strategy_version=(
                normalise_strategy_version(
                    row["strategy_version"]
                )
            ),
            score=_optional_float(
                row["score"]
            ),
            confidence=_optional_float(
                row["confidence"]
            ),
            market_regime=(
                row["market_regime"]
            ),
            reward_to_risk=(
                _optional_float(
                    row["reward_to_risk"]
                )
            ),
            release_eligible=bool(
                row["release_eligible"]
            ),
            rank_score=_optional_float(
                row["rank_score"]
            ),
            rank_position=(
                int(row["rank_position"])
                if row["rank_position"]
                is not None
                else None
            ),
            signal_id=row["signal_id"],
            reasons=tuple(
                json.loads(
                    row["reasons_json"]
                )
            ),
            evidence=tuple(
                json.loads(
                    row["evidence_json"]
                )
            ),
            metadata=json.loads(
                row["metadata_json"]
            ),
        )

    def start_scan(
        self,
        *,
        account_id: str,
        universe: StockUniverse,
        configuration: Mapping[str, object],
        app_version: str,
        started_at: datetime,
        scan_key: str = "",
    ) -> tuple[MarketScan, bool]:
        key = str(scan_key or "").strip()

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
                    f"Unknown paper account: "
                    f"{account_id}."
                )

            if key:
                existing = connection.execute(
                    """
                    SELECT *
                    FROM paper_scans
                    WHERE account_id = ?
                      AND scan_key = ?
                    """,
                    (
                        account_id,
                        key,
                    ),
                ).fetchone()

                if existing is not None:
                    return (
                        self._scan_from_row(
                            existing
                        ),
                        False,
                    )

            scan_id = _new_id("SCAN")

            connection.execute(
                """
                INSERT INTO paper_scans(
                    scan_id,
                    account_id,
                    universe,
                    status,
                    started_at,
                    completed_at,
                    processed_count,
                    rejected_count,
                    signal_count,
                    order_count,
                    error_message,
                    scan_key,
                    requested_count,
                    configuration_json,
                    app_version
                )
                VALUES (
                    ?, ?, ?, ?, ?, NULL,
                    0, 0, 0, 0, NULL,
                    ?, ?, ?, ?
                )
                """,
                (
                    scan_id,
                    account_id,
                    universe.name,
                    ScanStatus.RUNNING.value,
                    _timestamp(started_at),
                    key,
                    len(universe.symbols),
                    _json_dump(
                        dict(configuration)
                    ),
                    app_version,
                ),
            )

        return self.get_scan(scan_id), True

    def get_scan(
        self,
        scan_id: str,
    ) -> MarketScan:
        connection = connect_database(
            self.database_path
        )

        try:
            row = connection.execute(
                """
                SELECT *
                FROM paper_scans
                WHERE scan_id = ?
                """,
                (scan_id,),
            ).fetchone()
        finally:
            connection.close()

        if row is None:
            raise ValueError(
                f"Unknown scan: {scan_id}."
            )

        return self._scan_from_row(row)

    def save_result(
        self,
        result: ScanResult,
    ) -> None:
        with transaction(
            self.database_path
        ) as connection:
            connection.execute(
                """
                INSERT INTO paper_scan_results(
                    result_id,
                    scan_id,
                    account_id,
                    symbol,
                    status,
                    processed_at,
                    data_as_of,
                    history_rows,
                    latest_price,
                    average_volume,
                    average_dollar_volume,
                    recommendation,
                    strategy,
                    strategy_horizon,
                    strategy_version,
                    score,
                    confidence,
                    market_regime,
                    reward_to_risk,
                    release_eligible,
                    rank_score,
                    rank_position,
                    signal_id,
                    reasons_json,
                    evidence_json,
                    metadata_json
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
                ON CONFLICT(scan_id, symbol)
                DO UPDATE SET
                    status = excluded.status,
                    processed_at = excluded.processed_at,
                    data_as_of = excluded.data_as_of,
                    history_rows = excluded.history_rows,
                    latest_price = excluded.latest_price,
                    average_volume = excluded.average_volume,
                    average_dollar_volume =
                        excluded.average_dollar_volume,
                    recommendation =
                        excluded.recommendation,
                    strategy = excluded.strategy,
                    strategy_horizon =
                        excluded.strategy_horizon,
                    strategy_version =
                        excluded.strategy_version,
                    score = excluded.score,
                    confidence = excluded.confidence,
                    market_regime =
                        excluded.market_regime,
                    reward_to_risk =
                        excluded.reward_to_risk,
                    release_eligible =
                        excluded.release_eligible,
                    rank_score = excluded.rank_score,
                    rank_position =
                        excluded.rank_position,
                    signal_id = excluded.signal_id,
                    reasons_json =
                        excluded.reasons_json,
                    evidence_json =
                        excluded.evidence_json,
                    metadata_json =
                        excluded.metadata_json
                """,
                (
                    result.result_id,
                    result.scan_id,
                    result.account_id,
                    result.symbol,
                    result.status.value,
                    _timestamp(
                        result.processed_at
                    ),
                    (
                        _timestamp(
                            result.data_as_of
                        )
                        if result.data_as_of
                        is not None
                        else None
                    ),
                    result.history_rows,
                    result.latest_price,
                    result.average_volume,
                    (
                        result
                        .average_dollar_volume
                    ),
                    result.recommendation,
                    result.strategy,
                    strategy_horizon_value(
                        result.strategy_horizon
                    ),
                    normalise_strategy_version(
                        result.strategy_version
                    ),
                    result.score,
                    result.confidence,
                    result.market_regime,
                    result.reward_to_risk,
                    int(
                        result.release_eligible
                    ),
                    result.rank_score,
                    result.rank_position,
                    result.signal_id,
                    _json_dump(
                        list(result.reasons)
                    ),
                    _json_dump(
                        list(result.evidence)
                    ),
                    _json_dump(
                        dict(result.metadata)
                    ),
                ),
            )

    def complete_scan(
        self,
        scan_id: str,
        *,
        completed_at: datetime,
    ) -> MarketScan:
        with transaction(
            self.database_path
        ) as connection:
            rows = connection.execute(
                """
                SELECT status, signal_id
                FROM paper_scan_results
                WHERE scan_id = ?
                """,
                (scan_id,),
            ).fetchall()

            processed_count = len(rows)

            rejected_statuses = {
                ScanResultStatus.DATA_REJECTED.value,
                ScanResultStatus
                .ANALYSIS_REJECTED.value,
                ScanResultStatus
                .RELEASE_INELIGIBLE.value,
                ScanResultStatus.SCAN_ERROR.value,
            }

            rejected_count = sum(
                row["status"]
                in rejected_statuses
                for row in rows
            )

            signal_count = sum(
                row["signal_id"] is not None
                for row in rows
            )

            has_errors = any(
                row["status"]
                == ScanResultStatus
                .SCAN_ERROR.value
                for row in rows
            )

            status = (
                ScanStatus
                .COMPLETED_WITH_ERRORS
                if has_errors
                else ScanStatus.COMPLETED
            )

            connection.execute(
                """
                UPDATE paper_scans
                SET status = ?,
                    completed_at = ?,
                    processed_count = ?,
                    rejected_count = ?,
                    signal_count = ?,
                    order_count = 0
                WHERE scan_id = ?
                """,
                (
                    status.value,
                    _timestamp(completed_at),
                    processed_count,
                    rejected_count,
                    signal_count,
                    scan_id,
                ),
            )

        return self.get_scan(scan_id)

    def list_results(
        self,
        scan_id: str,
    ) -> tuple[ScanResult, ...]:
        connection = connect_database(
            self.database_path
        )

        try:
            rows = connection.execute(
                """
                SELECT *
                FROM paper_scan_results
                WHERE scan_id = ?
                ORDER BY
                    CASE
                        WHEN rank_position IS NULL
                        THEN 1
                        ELSE 0
                    END,
                    rank_position,
                    symbol
                """,
                (scan_id,),
            ).fetchall()
        finally:
            connection.close()

        return tuple(
            self._result_from_row(row)
            for row in rows
        )

    def get_report(
        self,
        scan_id: str,
    ) -> MarketScanReport:
        return MarketScanReport(
            scan=self.get_scan(scan_id),
            results=self.list_results(
                scan_id
            ),
        )
