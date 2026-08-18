"""Persistence for execution runs and portfolio controls."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
import json
from pathlib import Path
import sqlite3
from typing import Mapping
from uuid import uuid4

from src.paper import (
    DEFAULT_DATABASE_PATH,
    FixedNotionalSizingPolicy,
    PaperExitReason,
    PositionSizingMode,
    connect_database,
    initialize_database,
    money,
    transaction,
)

from .models import (
    EquitySnapshot,
    ExecutionRun,
    ExecutionRunStatus,
    ExitRequest,
    ExitRequestStatus,
    PortfolioControl,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


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


def _datetime(
    value: str | None,
) -> datetime | None:
    if value is None:
        return None

    return datetime.fromisoformat(value)


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Enum):
        return value.value

    raise TypeError(
        f"Unsupported JSON value: "
        f"{type(value).__name__}."
    )


def _json_dump(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


class AutomationRepository:
    """SQLite execution-run and control repository."""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
    ) -> None:
        self.database_path = Path(database_path)
        initialize_database(self.database_path)

    @staticmethod
    def _run_from_row(
        row: sqlite3.Row,
    ) -> ExecutionRun:
        return ExecutionRun(
            run_id=row["run_id"],
            account_id=row["account_id"],
            run_key=row["run_key"],
            scan_id=row["scan_id"],
            status=ExecutionRunStatus(
                row["status"]
            ),
            started_at=_datetime(
                row["started_at"]
            ),
            completed_at=_datetime(
                row["completed_at"]
            ),
            created_orders=int(
                row["created_orders"]
            ),
            filled_orders=int(
                row["filled_orders"]
            ),
            expired_orders=int(
                row["expired_orders"]
            ),
            cancelled_orders=int(
                row["cancelled_orders"]
            ),
            closed_positions=int(
                row["closed_positions"]
            ),
            rejected_entries=int(
                row["rejected_entries"]
            ),
            error_count=int(
                row["error_count"]
            ),
            entry_block_reasons=tuple(
                json.loads(
                    row[
                        "entry_block_reasons_json"
                    ]
                )
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
    def _request_from_row(
        row: sqlite3.Row,
    ) -> ExitRequest:
        return ExitRequest(
            request_id=row["request_id"],
            account_id=row["account_id"],
            position_id=row["position_id"],
            reason=PaperExitReason(
                row["reason"]
            ),
            triggered_at=_datetime(
                row["triggered_at"]
            ),
            status=ExitRequestStatus(
                row["status"]
            ),
            created_at=_datetime(
                row["created_at"]
            ),
            executed_at=_datetime(
                row["executed_at"]
            ),
            error_message=(
                row["error_message"]
            ),
        )

    @staticmethod
    def _equity_from_row(
        row: sqlite3.Row,
    ) -> EquitySnapshot:
        return EquitySnapshot(
            snapshot_id=row["snapshot_id"],
            run_id=row["run_id"],
            account_id=row["account_id"],
            captured_at=_datetime(
                row["captured_at"]
            ),
            cash_balance=money(
                row["cash_balance"]
            ),
            reserved_cash=money(
                row["reserved_cash"]
            ),
            market_value=money(
                row["market_value"]
            ),
            equity=money(row["equity"]),
        )

    def get_control(
        self,
        account_id: str,
        *,
        at: datetime,
    ) -> PortfolioControl:
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
                    f"Unknown account: {account_id}."
                )

            connection.execute(
                """
                INSERT OR IGNORE INTO
                    paper_account_controls(
                        account_id,
                        updated_at
                    )
                VALUES (?, ?)
                """,
                (
                    account_id,
                    _timestamp(at),
                ),
            )

            row = connection.execute(
                """
                SELECT *
                FROM paper_account_controls
                WHERE account_id = ?
                """,
                (account_id,),
            ).fetchone()

        return PortfolioControl(
            account_id=row["account_id"],
            kill_switch_active=bool(
                row["kill_switch_active"]
            ),
            kill_switch_reason=(
                row["kill_switch_reason"]
            ),
            maximum_daily_loss_fraction=money(
                row[
                    "maximum_daily_loss_fraction"
                ]
            ),
            maximum_drawdown_fraction=money(
                row[
                    "maximum_drawdown_fraction"
                ]
            ),
            maximum_new_orders_per_run=int(
                row[
                    "maximum_new_orders_per_run"
                ]
            ),
            maximum_stale_market_days=int(
                row[
                    "maximum_stale_market_days"
                ]
            ),
            sizing_mode=(
                PositionSizingMode(
                    row["sizing_mode"]
                )
                if row["sizing_mode"]
                is not None
                else None
            ),
            portfolio_currency=(
                row["portfolio_currency"]
            ),
            target_order_value=(
                money(
                    row[
                        "target_order_value"
                    ]
                )
                if row[
                    "target_order_value"
                ]
                is not None
                else None
            ),
            maximum_order_value=(
                money(
                    row[
                        "maximum_order_value"
                    ]
                )
                if row[
                    "maximum_order_value"
                ]
                is not None
                else None
            ),
            maximum_planned_loss=(
                money(
                    row[
                        "maximum_planned_loss"
                    ]
                )
                if row[
                    "maximum_planned_loss"
                ]
                is not None
                else None
            ),
            maximum_open_positions=(
                int(
                    row[
                        "maximum_open_positions"
                    ]
                )
                if row[
                    "maximum_open_positions"
                ]
                is not None
                else None
            ),
            maximum_invested_exposure=(
                money(
                    row[
                        "maximum_invested_exposure"
                    ]
                )
                if row[
                    "maximum_invested_exposure"
                ]
                is not None
                else None
            ),
            updated_at=_datetime(
                row["updated_at"]
            ),
        )


    def set_fixed_notional_sizing(
        self,
        account_id: str,
        *,
        policy: FixedNotionalSizingPolicy,
        updated_at: datetime,
    ) -> PortfolioControl:
        """Persist the approved P4.1 sizing controls."""

        if not isinstance(
            policy,
            FixedNotionalSizingPolicy,
        ):
            raise ValueError(
                "policy must be a "
                "FixedNotionalSizingPolicy."
            )

        self.get_control(
            account_id,
            at=updated_at,
        )

        with transaction(
            self.database_path
        ) as connection:
            account = connection.execute(
                """
                SELECT base_currency
                FROM paper_accounts
                WHERE account_id = ?
                """,
                (account_id,),
            ).fetchone()

            if account is None:
                raise ValueError(
                    f"Unknown account: {account_id}."
                )

            account_currency = str(
                account["base_currency"]
            ).strip().upper()

            if (
                account_currency
                != policy.portfolio_currency
            ):
                raise ValueError(
                    "Sizing-policy currency must "
                    "match account base currency."
                )

            connection.execute(
                """
                UPDATE paper_account_controls
                SET sizing_mode = ?,
                    portfolio_currency = ?,
                    target_order_value = ?,
                    maximum_order_value = ?,
                    maximum_planned_loss = ?,
                    maximum_open_positions = ?,
                    maximum_invested_exposure = ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (
                    policy.mode.value,
                    policy.portfolio_currency,
                    str(
                        policy.target_order_value
                    ),
                    str(
                        policy.maximum_order_value
                    ),
                    str(
                        policy.maximum_planned_loss
                    ),
                    policy.maximum_open_positions,
                    str(
                        policy
                        .maximum_invested_exposure
                    ),
                    _timestamp(updated_at),
                    account_id,
                ),
            )

        return self.get_control(
            account_id,
            at=updated_at,
        )

    def set_kill_switch(
        self,
        account_id: str,
        *,
        active: bool,
        reason: str | None,
        updated_at: datetime,
        changed_by: str | None = None,
    ) -> PortfolioControl:
        current = self.get_control(
            account_id,
            at=updated_at,
        )

        clean_reason = str(reason).strip() if reason else None
        clean_operator = (
            str(changed_by).strip() if changed_by else None
        )

        if clean_operator is not None and not clean_reason:
            raise ValueError(
                "An operator kill-switch change requires a reason."
            )

        if current.kill_switch_active is bool(active):
            return current

        with transaction(
            self.database_path
        ) as connection:
            connection.execute(
                """
                UPDATE paper_account_controls
                SET kill_switch_active = ?,
                    kill_switch_reason = ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (
                    int(bool(active)),
                    clean_reason,
                    _timestamp(updated_at),
                    account_id,
                ),
            )

            if clean_operator is not None:
                connection.execute(
                    """
                    INSERT INTO paper_system_events(
                        event_id, account_id, event_type, severity,
                        reference_type, reference_id, message,
                        metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _new_id("EVT"),
                        account_id,
                        (
                            "GLOBAL_KILL_SWITCH_ACTIVATED"
                            if active
                            else "GLOBAL_KILL_SWITCH_DEACTIVATED"
                        ),
                        "CRITICAL" if active else "INFO",
                        "ACCOUNT",
                        account_id,
                        clean_reason,
                        _json_dump(
                            {
                                "active": bool(active),
                                "changed_by": clean_operator,
                            }
                        ),
                        _timestamp(updated_at),
                    ),
                )

        return self.get_control(
            account_id,
            at=updated_at,
        )

    def start_run(
        self,
        *,
        account_id: str,
        run_key: str,
        scan_id: str | None,
        configuration: Mapping[str, object],
        app_version: str,
        started_at: datetime,
    ) -> tuple[ExecutionRun, bool]:
        key = str(run_key).strip()

        if not key:
            raise ValueError(
                "run_key cannot be empty."
            )

        with transaction(
            self.database_path
        ) as connection:
            existing = connection.execute(
                """
                SELECT *
                FROM paper_execution_runs
                WHERE account_id = ?
                  AND run_key = ?
                """,
                (
                    account_id,
                    key,
                ),
            ).fetchone()

            if existing is not None:
                return (
                    self._run_from_row(existing),
                    False,
                )

            run_id = _new_id("RUN")

            connection.execute(
                """
                INSERT INTO paper_execution_runs(
                    run_id,
                    account_id,
                    run_key,
                    scan_id,
                    status,
                    started_at,
                    completed_at,
                    entry_block_reasons_json,
                    configuration_json,
                    app_version
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, NULL,
                    '[]', ?, ?
                )
                """,
                (
                    run_id,
                    account_id,
                    key,
                    scan_id,
                    ExecutionRunStatus
                    .RUNNING.value,
                    _timestamp(started_at),
                    _json_dump(
                        dict(configuration)
                    ),
                    app_version,
                ),
            )

        return self.get_run(run_id), True

    def get_run(
        self,
        run_id: str,
    ) -> ExecutionRun:
        connection = connect_database(
            self.database_path
        )

        try:
            row = connection.execute(
                """
                SELECT *
                FROM paper_execution_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        finally:
            connection.close()

        if row is None:
            raise ValueError(
                f"Unknown execution run: {run_id}."
            )

        return self._run_from_row(row)

    def complete_run(
        self,
        run_id: str,
        *,
        status: ExecutionRunStatus,
        completed_at: datetime,
        created_orders: int,
        filled_orders: int,
        expired_orders: int,
        cancelled_orders: int,
        closed_positions: int,
        rejected_entries: int,
        error_count: int,
        entry_block_reasons: tuple[str, ...],
        error_message: str | None = None,
    ) -> ExecutionRun:
        with transaction(
            self.database_path
        ) as connection:
            connection.execute(
                """
                UPDATE paper_execution_runs
                SET status = ?,
                    completed_at = ?,
                    created_orders = ?,
                    filled_orders = ?,
                    expired_orders = ?,
                    cancelled_orders = ?,
                    closed_positions = ?,
                    rejected_entries = ?,
                    error_count = ?,
                    entry_block_reasons_json = ?,
                    error_message = ?
                WHERE run_id = ?
                """,
                (
                    status.value,
                    _timestamp(completed_at),
                    created_orders,
                    filled_orders,
                    expired_orders,
                    cancelled_orders,
                    closed_positions,
                    rejected_entries,
                    error_count,
                    _json_dump(
                        list(
                            entry_block_reasons
                        )
                    ),
                    error_message,
                    run_id,
                ),
            )

        return self.get_run(run_id)

    def create_exit_request(
        self,
        *,
        account_id: str,
        position_id: str,
        reason: PaperExitReason,
        triggered_at: datetime,
        created_at: datetime,
    ) -> tuple[ExitRequest, bool]:
        with transaction(
            self.database_path
        ) as connection:
            existing = connection.execute(
                """
                SELECT *
                FROM paper_exit_requests
                WHERE position_id = ?
                  AND reason = ?
                  AND triggered_at = ?
                """,
                (
                    position_id,
                    reason.value,
                    _timestamp(triggered_at),
                ),
            ).fetchone()

            if existing is not None:
                return (
                    self._request_from_row(
                        existing
                    ),
                    False,
                )

            request_id = _new_id("EXIT")

            connection.execute(
                """
                INSERT INTO paper_exit_requests(
                    request_id,
                    account_id,
                    position_id,
                    reason,
                    triggered_at,
                    status,
                    created_at,
                    executed_at,
                    error_message
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    NULL, NULL
                )
                """,
                (
                    request_id,
                    account_id,
                    position_id,
                    reason.value,
                    _timestamp(triggered_at),
                    ExitRequestStatus
                    .PENDING.value,
                    _timestamp(created_at),
                ),
            )

            row = connection.execute(
                """
                SELECT *
                FROM paper_exit_requests
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()

        return self._request_from_row(row), True

    def list_pending_exit_requests(
        self,
        account_id: str,
    ) -> tuple[ExitRequest, ...]:
        connection = connect_database(
            self.database_path
        )

        try:
            rows = connection.execute(
                """
                SELECT *
                FROM paper_exit_requests
                WHERE account_id = ?
                  AND status = ?
                ORDER BY triggered_at, request_id
                """,
                (
                    account_id,
                    ExitRequestStatus
                    .PENDING.value,
                ),
            ).fetchall()
        finally:
            connection.close()

        return tuple(
            self._request_from_row(row)
            for row in rows
        )

    def update_exit_request(
        self,
        request_id: str,
        *,
        status: ExitRequestStatus,
        executed_at: datetime | None = None,
        error_message: str | None = None,
    ) -> ExitRequest:
        with transaction(
            self.database_path
        ) as connection:
            connection.execute(
                """
                UPDATE paper_exit_requests
                SET status = ?,
                    executed_at = ?,
                    error_message = ?
                WHERE request_id = ?
                """,
                (
                    status.value,
                    (
                        _timestamp(executed_at)
                        if executed_at
                        is not None
                        else None
                    ),
                    error_message,
                    request_id,
                ),
            )

            row = connection.execute(
                """
                SELECT *
                FROM paper_exit_requests
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()

        if row is None:
            raise ValueError(
                f"Unknown exit request: "
                f"{request_id}."
            )

        return self._request_from_row(row)

    def save_equity_snapshot(
        self,
        *,
        run_id: str,
        account_id: str,
        captured_at: datetime,
        cash_balance: object,
        reserved_cash: object,
        market_value: object,
    ) -> EquitySnapshot:
        cash = money(cash_balance)
        reserved = money(reserved_cash)
        market = money(market_value)
        equity = money(cash + market)

        with transaction(
            self.database_path
        ) as connection:
            existing = connection.execute(
                """
                SELECT snapshot_id
                FROM paper_equity_snapshots
                WHERE run_id = ?
                  AND account_id = ?
                """,
                (
                    run_id,
                    account_id,
                ),
            ).fetchone()

            snapshot_id = (
                existing["snapshot_id"]
                if existing is not None
                else _new_id("EQUITY")
            )

            connection.execute(
                """
                INSERT INTO paper_equity_snapshots(
                    snapshot_id,
                    run_id,
                    account_id,
                    captured_at,
                    cash_balance,
                    reserved_cash,
                    market_value,
                    equity
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, account_id)
                DO UPDATE SET
                    captured_at =
                        excluded.captured_at,
                    cash_balance =
                        excluded.cash_balance,
                    reserved_cash =
                        excluded.reserved_cash,
                    market_value =
                        excluded.market_value,
                    equity = excluded.equity
                """,
                (
                    snapshot_id,
                    run_id,
                    account_id,
                    _timestamp(captured_at),
                    str(cash),
                    str(reserved),
                    str(market),
                    str(equity),
                ),
            )

            row = connection.execute(
                """
                SELECT *
                FROM paper_equity_snapshots
                WHERE run_id = ?
                  AND account_id = ?
                """,
                (
                    run_id,
                    account_id,
                ),
            ).fetchone()

        return self._equity_from_row(row)

    def get_equity_for_run(
        self,
        run_id: str,
        account_id: str,
    ) -> EquitySnapshot | None:
        connection = connect_database(
            self.database_path
        )

        try:
            row = connection.execute(
                """
                SELECT *
                FROM paper_equity_snapshots
                WHERE run_id = ?
                  AND account_id = ?
                """,
                (
                    run_id,
                    account_id,
                ),
            ).fetchone()
        finally:
            connection.close()

        return (
            self._equity_from_row(row)
            if row is not None
            else None
        )

    def peak_equity(
        self,
        account_id: str,
    ) -> Decimal | None:
        connection = connect_database(
            self.database_path
        )

        try:
            rows = connection.execute(
                """
                SELECT equity
                FROM paper_equity_snapshots
                WHERE account_id = ?
                """,
                (account_id,),
            ).fetchall()
        finally:
            connection.close()

        if not rows:
            return None

        return max(
            money(row["equity"])
            for row in rows
        )

    def refresh_scan_order_count(
        self,
        scan_id: str,
    ) -> None:
        with transaction(
            self.database_path
        ) as connection:
            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM paper_orders AS orders
                    INNER JOIN paper_signals AS signals
                        ON signals.signal_id =
                           orders.signal_id
                    WHERE signals.scan_id = ?
                    """,
                    (scan_id,),
                ).fetchone()[0]
            )

            connection.execute(
                """
                UPDATE paper_scans
                SET order_count = ?
                WHERE scan_id = ?
                """,
                (
                    count,
                    scan_id,
                ),
            )
