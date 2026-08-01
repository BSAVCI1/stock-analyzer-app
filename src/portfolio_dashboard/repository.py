"""Read-only persisted queries for dashboard reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.automation import (
    AutomationRepository,
    EquitySnapshot,
    ExecutionRun,
)
from src.jobs import (
    JobRepository,
    JobRun,
)
from src.paper import (
    DEFAULT_DATABASE_PATH,
    ClosedPaperTrade,
    NotificationRecord,
    PaperAccount,
    PaperOrderRecord,
    PaperPositionRecord,
    PaperRepository,
    PersistedSignal,
    SystemEventRecord,
    connect_database,
)
from src.scanner import (
    MarketScanReport,
    ScannerRepository,
)


class PortfolioDashboardRepository:
    """Compose existing stores without mutating portfolio data."""

    def __init__(
        self,
        database_path: str | Path = (
            DEFAULT_DATABASE_PATH
        ),
    ) -> None:
        self.database_path = Path(
            database_path
        )

        self.paper = PaperRepository(
            self.database_path
        )

        self.scanner = ScannerRepository(
            self.database_path
        )

        self.automation = (
            AutomationRepository(
                self.database_path
            )
        )

        self.jobs = JobRepository(
            self.database_path
        )

    def _list_ids(
        self,
        *,
        query: str,
        parameters: Sequence[object],
        column: str,
    ) -> tuple[str, ...]:
        connection = connect_database(
            self.database_path
        )

        try:
            rows = connection.execute(
                query,
                tuple(parameters),
            ).fetchall()
        finally:
            connection.close()

        return tuple(
            str(row[column])
            for row in rows
        )

    def get_account(
        self,
        account_id: str,
    ) -> PaperAccount:
        return self.paper.get_account(
            account_id
        )

    def list_signals(
        self,
        account_id: str,
    ) -> tuple[PersistedSignal, ...]:
        signal_ids = self._list_ids(
            query="""
                SELECT signal_id
                FROM paper_signals
                WHERE account_id = ?
                ORDER BY
                    generated_at,
                    signal_id
            """,
            parameters=(account_id,),
            column="signal_id",
        )

        return tuple(
            self.paper.get_signal(
                signal_id
            )
            for signal_id in signal_ids
        )

    def list_pending_orders(
        self,
        account_id: str,
    ) -> tuple[PaperOrderRecord, ...]:
        return self.paper.list_pending_orders(
            account_id
        )

    def list_open_positions(
        self,
        account_id: str,
    ) -> tuple[PaperPositionRecord, ...]:
        return self.paper.list_open_positions(
            account_id
        )

    def list_closed_trades(
        self,
        account_id: str,
    ) -> tuple[ClosedPaperTrade, ...]:
        return self.paper.list_closed_trades(
            account_id
        )

    def list_scan_reports(
        self,
        account_id: str,
    ) -> tuple[MarketScanReport, ...]:
        scan_ids = self._list_ids(
            query="""
                SELECT scan_id
                FROM paper_scans
                WHERE account_id = ?
                ORDER BY
                    started_at,
                    scan_id
            """,
            parameters=(account_id,),
            column="scan_id",
        )

        return tuple(
            self.scanner.get_report(scan_id)
            for scan_id in scan_ids
        )

    def list_execution_runs(
        self,
        account_id: str,
    ) -> tuple[ExecutionRun, ...]:
        run_ids = self._list_ids(
            query="""
                SELECT run_id
                FROM paper_execution_runs
                WHERE account_id = ?
                ORDER BY
                    started_at,
                    run_id
            """,
            parameters=(account_id,),
            column="run_id",
        )

        return tuple(
            self.automation.get_run(run_id)
            for run_id in run_ids
        )

    def list_equity_snapshots(
        self,
        account_id: str,
    ) -> tuple[EquitySnapshot, ...]:
        run_ids = self._list_ids(
            query="""
                SELECT run_id
                FROM paper_equity_snapshots
                WHERE account_id = ?
                ORDER BY
                    captured_at,
                    snapshot_id
            """,
            parameters=(account_id,),
            column="run_id",
        )

        snapshots = []

        for run_id in run_ids:
            snapshot = (
                self.automation
                .get_equity_for_run(
                    run_id,
                    account_id,
                )
            )

            if snapshot is not None:
                snapshots.append(snapshot)

        return tuple(snapshots)

    def list_jobs(
        self,
        account_id: str,
    ) -> tuple[JobRun, ...]:
        return self.jobs.list_jobs(
            account_id
        )

    def list_notifications(
        self,
        account_id: str,
    ) -> tuple[NotificationRecord, ...]:
        return self.paper.list_notifications(
            account_id
        )

    def list_system_events(
        self,
        account_id: str,
    ) -> tuple[SystemEventRecord, ...]:
        return self.paper.list_system_events(
            account_id
        )

    def reconcile_account(
        self,
        account_id: str,
    ):
        return self.paper.reconcile_account(
            account_id
        )
