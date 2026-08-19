"""Exchange-aware scheduled paper-trading orchestration."""

from __future__ import annotations

from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal
from typing import Callable

from src.automation import (
    AutomatedPaperExecutionEngine,
    ExecutionRunReport,
    ExecutionRunStatus,
)
from src.notifications import (
    DispatchReport,
    NotificationService,
)
from src.paper import (
    NotificationChannel,
    NotificationStatus,
    PaperRepository,
)
from src.scanner import (
    AutomaticMarketScanner,
    MarketScanReport,
    StockUniverse,
    load_stock_universe,
)

from .calendar import ExchangeCalendar
from .models import (
    ExchangeSession,
    JobRun,
    JobStatus,
    JobType,
    ScheduledJobReport,
)
from .repository import JobRepository


UniverseLoader = Callable[[], StockUniverse]


class ScheduledJobService:
    """Run idempotent scans, execution, reports and delivery."""

    def __init__(
        self,
        *,
        job_repository: JobRepository,
        paper_repository: PaperRepository,
        scanner: AutomaticMarketScanner,
        execution_engine: AutomatedPaperExecutionEngine,
        notification_service: NotificationService,
        calendar: ExchangeCalendar | None = None,
        universe_loader: UniverseLoader = (
            load_stock_universe
        ),
        notification_channels: tuple[
            NotificationChannel,
            ...,
        ] = (),
    ) -> None:
        self.job_repository = job_repository
        self.paper_repository = (
            paper_repository
        )
        self.scanner = scanner
        self.execution_engine = (
            execution_engine
        )
        self.notification_service = (
            notification_service
        )
        self.calendar = (
            calendar
            or ExchangeCalendar()
        )
        self.universe_loader = universe_loader

        channels = tuple(
            dict.fromkeys(
                channel
                for channel
                in notification_channels
                if channel
                is not NotificationChannel.INTERNAL
            )
        )

        if not all(
            isinstance(
                channel,
                NotificationChannel,
            )
            for channel in channels
        ):
            raise ValueError(
                "notification_channels must "
                "contain NotificationChannel values."
            )

        self.notification_channels = channels

    @staticmethod
    def _aware(
        value: datetime,
        *,
        name: str,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                f"{name} must be timezone-aware."
            )

        return value.astimezone(
            timezone.utc
        )

    def _market_job_key(
        self,
        *,
        scheduled_for: datetime,
        session: ExchangeSession | None,
        after_close: bool,
    ) -> str:
        exchange = (
            self.calendar.exchange_code
        )

        if session is None:
            local_date = (
                scheduled_for
                .astimezone(
                    self.calendar.timezone
                )
                .date()
            )

            return (
                f"MARKET_CYCLE:{exchange}:"
                f"{local_date.isoformat()}:"
                "CLOSED"
            )

        if after_close:
            return (
                f"MARKET_CYCLE:{exchange}:"
                f"{session.session_date.isoformat()}"
            )

        minute = (
            scheduled_for
            .astimezone(timezone.utc)
            .replace(
                second=0,
                microsecond=0,
            )
            .isoformat()
        )

        return (
            f"MARKET_CYCLE_PRECHECK:"
            f"{exchange}:{minute}"
        )

    def _weekly_job_key(
        self,
        session_date: date,
    ) -> str:
        iso_year, iso_week, _ = (
            session_date.isocalendar()
        )

        return (
            f"WEEKLY_REPORT:"
            f"{self.calendar.exchange_code}:"
            f"{iso_year}-W{iso_week:02d}"
        )

    def _queue_external(
        self,
        *,
        account_id: str,
        event_type: str,
        reference_type: str,
        reference_id: str,
        subject: str,
        text: str,
        created_at: datetime,
    ) -> int:
        queued = 0

        for channel in (
            self.notification_channels
        ):
            notification = (
                self.paper_repository
                .queue_notification(
                    account_id=account_id,
                    event_type=event_type,
                    reference_type=(
                        reference_type
                    ),
                    reference_id=reference_id,
                    channel=channel,
                    payload={
                        "subject": subject,
                        "text": text,
                    },
                    created_at=created_at,
                )
            )

            if (
                notification.status
                is NotificationStatus.PENDING
            ):
                queued += 1

        return queued

    def _resume_if_interrupted(
        self,
        job: JobRun,
        *,
        created: bool,
        resumed_at: datetime,
    ) -> tuple[JobRun, bool]:
        """Resume RUNNING jobs; completed jobs remain duplicates."""

        if created:
            return job, False

        if job.status is not JobStatus.RUNNING:
            return job, True

        return (
            self.job_repository.resume_running_job(
                job.job_run_id,
                resumed_at=resumed_at,
            ),
            False,
        )

    def _dispatch(
        self,
        *,
        account_id: str,
        attempted_at: datetime,
    ) -> tuple[int, DispatchReport]:
        fanned_out = (
            self.notification_service
            .fan_out_internal(
                account_id,
                channels=(
                    self.notification_channels
                ),
                created_at=attempted_at,
            )
        )

        report = (
            self.notification_service
            .dispatch_pending(
                account_id,
                attempted_at=attempted_at,
            )
        )

        return fanned_out, report

    def _daily_summary(
        self,
        *,
        account_id: str,
        session: ExchangeSession,
        scan_report: MarketScanReport,
        execution_report: ExecutionRunReport,
    ) -> tuple[str, str]:
        account = (
            self.paper_repository
            .get_account(account_id)
        )

        positions = (
            self.paper_repository
            .list_open_positions(
                account_id
            )
        )

        pending_orders = (
            self.paper_repository
            .list_pending_orders(
                account_id
            )
        )

        trades = (
            self.paper_repository
            .list_closed_trades(
                account_id
            )
        )

        local_timezone = (
            session.closes_at.tzinfo
        )

        daily_trades = tuple(
            trade
            for trade in trades
            if (
                trade.exit_time
                .astimezone(local_timezone)
                .date()
                == session.session_date
            )
        )

        daily_net_pnl = sum(
            (
                trade.net_pnl
                for trade in daily_trades
            ),
            Decimal("0"),
        )

        reconciliation = (
            execution_report
            .reconciliation
        )

        candidates = len(
            scan_report.candidates
        )

        subject = (
            "Paper portfolio summary — "
            f"{session.session_date.isoformat()}"
        )

        text = "\n".join(
            (
                "Daily paper portfolio summary",
                (
                    "Session: "
                    f"{session.session_date.isoformat()}"
                ),
                (
                    "Scan status: "
                    f"{scan_report.scan.status.value}"
                ),
                (
                    "Symbols processed: "
                    f"{scan_report.scan.processed_count}"
                ),
                (
                    "Order candidates: "
                    f"{candidates}"
                ),
                (
                    "Orders created: "
                    f"{execution_report.run.created_orders}"
                ),
                (
                    "Orders filled: "
                    f"{execution_report.run.filled_orders}"
                ),
                (
                    "Positions closed: "
                    f"{execution_report.run.closed_positions}"
                ),
                (
                    "Rejected entries: "
                    f"{execution_report.run.rejected_entries}"
                ),
                (
                    "Open positions: "
                    f"{len(positions)}"
                ),
                (
                    "Pending orders: "
                    f"{len(pending_orders)}"
                ),
                (
                    "Cash balance: "
                    f"{account.cash_balance} "
                    f"{account.base_currency}"
                ),
                (
                    "Reserved cash: "
                    f"{account.reserved_cash} "
                    f"{account.base_currency}"
                ),
                (
                    "Daily realised P&L: "
                    f"{daily_net_pnl} "
                    f"{account.base_currency}"
                ),
                (
                    "Account reconciled: "
                    f"{reconciliation.reconciled}"
                ),
            )
        )

        return subject, text

    def _weekly_summary(
        self,
        *,
        account_id: str,
        session: ExchangeSession,
    ) -> tuple[str, str]:
        local_timezone = (
            session.closes_at.tzinfo
        )

        weekday = (
            session.session_date.weekday()
        )

        week_start = (
            session.session_date
            - timedelta(days=weekday)
        )

        week_end = (
            week_start
            + timedelta(days=6)
        )

        account = (
            self.paper_repository
            .get_account(account_id)
        )

        all_trades = (
            self.paper_repository
            .list_closed_trades(
                account_id
            )
        )

        trades = tuple(
            trade
            for trade in all_trades
            if (
                week_start
                <= trade.exit_time
                .astimezone(local_timezone)
                .date()
                <= week_end
            )
        )

        net_pnl = sum(
            (
                trade.net_pnl
                for trade in trades
            ),
            Decimal("0"),
        )

        wins = sum(
            trade.net_pnl > 0
            for trade in trades
        )

        losses = sum(
            trade.net_pnl < 0
            for trade in trades
        )

        jobs = tuple(
            job
            for job
            in self.job_repository
            .list_jobs(account_id)
            if (
                week_start
                <= job.scheduled_for
                .astimezone(local_timezone)
                .date()
                <= week_end
            )
        )

        completed_jobs = sum(
            job.status
            in {
                JobStatus.COMPLETED,
                JobStatus
                .COMPLETED_WITH_ERRORS,
            }
            for job in jobs
        )

        failed_jobs = sum(
            job.status is JobStatus.FAILED
            for job in jobs
        )

        events = tuple(
            event
            for event
            in self.paper_repository
            .list_system_events(account_id)
            if (
                week_start
                <= event.created_at
                .astimezone(local_timezone)
                .date()
                <= week_end
            )
        )

        error_events = sum(
            event.severity.upper()
            == "ERROR"
            for event in events
        )

        notifications = tuple(
            notification
            for notification
            in self.paper_repository
            .list_notifications(account_id)
            if (
                week_start
                <= notification.created_at
                .astimezone(local_timezone)
                .date()
                <= week_end
            )
        )

        sent_notifications = sum(
            notification.status
            is NotificationStatus.SENT
            for notification
            in notifications
        )

        failed_notifications = sum(
            notification.status
            is NotificationStatus.FAILED
            for notification
            in notifications
        )

        reconciliation = (
            self.paper_repository
            .reconcile_account(account_id)
        )

        iso_year, iso_week, _ = (
            session.session_date
            .isocalendar()
        )

        subject = (
            "Weekly paper performance — "
            f"{iso_year}-W{iso_week:02d}"
        )

        text = "\n".join(
            (
                "Weekly paper performance "
                "and reliability report",
                (
                    "Period: "
                    f"{week_start.isoformat()} "
                    f"to {week_end.isoformat()}"
                ),
                (
                    "Closed trades: "
                    f"{len(trades)}"
                ),
                f"Wins: {wins}",
                f"Losses: {losses}",
                (
                    "Net realised P&L: "
                    f"{net_pnl} "
                    f"{account.base_currency}"
                ),
                (
                    "Completed jobs: "
                    f"{completed_jobs}"
                ),
                (
                    "Failed jobs: "
                    f"{failed_jobs}"
                ),
                (
                    "System error events: "
                    f"{error_events}"
                ),
                (
                    "Notifications sent: "
                    f"{sent_notifications}"
                ),
                (
                    "Notification failures: "
                    f"{failed_notifications}"
                ),
                (
                    "Account reconciled: "
                    f"{reconciliation.reconciled}"
                ),
            )
        )

        return subject, text

    def _skip_market_cycle(
        self,
        *,
        account_id: str,
        scheduled_for: datetime,
        session: ExchangeSession | None,
        reason: str,
        after_close: bool,
    ) -> ScheduledJobReport:
        key = self._market_job_key(
            scheduled_for=scheduled_for,
            session=session,
            after_close=after_close,
        )

        job, created = (
            self.job_repository
            .start_job(
                account_id=account_id,
                job_key=key,
                job_type=(
                    JobType.MARKET_CYCLE
                ),
                scheduled_for=(
                    scheduled_for
                ),
                exchange_code=(
                    self.calendar
                    .exchange_code
                ),
                metadata={
                    "skip_reason": reason,
                },
                started_at=(
                    scheduled_for
                ),
            )
        )

        job, duplicate = self._resume_if_interrupted(
            job,
            created=created,
            resumed_at=scheduled_for,
        )

        if duplicate:
            return ScheduledJobReport(
                job=job,
                session=session,
                duplicate=True,
                skipped_reason=reason,
            )

        job = (
            self.job_repository
            .complete_job(
                job.job_run_id,
                status=JobStatus.SKIPPED,
                completed_at=(
                    scheduled_for
                ),
                metadata={
                    "skip_reason": reason,
                },
            )
        )

        return ScheduledJobReport(
            job=job,
            session=session,
            duplicate=False,
            skipped_reason=reason,
        )

    def run_market_cycle(
        self,
        *,
        account_id: str,
        scheduled_for: datetime,
    ) -> ScheduledJobReport:
        at = self._aware(
            scheduled_for,
            name="scheduled_for",
        )

        session = (
            self.calendar
            .session_for_run(at)
        )

        if session is None:
            return self._skip_market_cycle(
                account_id=account_id,
                scheduled_for=at,
                session=None,
                reason=(
                    "The exchange is closed "
                    "on this date."
                ),
                after_close=False,
            )

        after_close = (
            self.calendar.is_after_close(at)
        )

        if not after_close:
            return self._skip_market_cycle(
                account_id=account_id,
                scheduled_for=at,
                session=session,
                reason=(
                    "The scheduled run occurred "
                    "before the regular market "
                    "close."
                ),
                after_close=False,
            )

        job_key = self._market_job_key(
            scheduled_for=at,
            session=session,
            after_close=True,
        )

        job, created = (
            self.job_repository
            .start_job(
                account_id=account_id,
                job_key=job_key,
                job_type=(
                    JobType.MARKET_CYCLE
                ),
                scheduled_for=at,
                exchange_code=(
                    self.calendar
                    .exchange_code
                ),
                metadata={
                    "session_date":
                    session
                    .session_date
                    .isoformat(),
                },
                started_at=at,
            )
        )

        job, duplicate = self._resume_if_interrupted(
            job,
            created=created,
            resumed_at=at,
        )

        if duplicate:
            return ScheduledJobReport(
                job=job,
                session=session,
                duplicate=True,
            )

        scan_id: str | None = None
        execution_run_id: str | None = None
        directly_queued = 0

        try:
            universe = (
                self.universe_loader()
            )

            scan_report = (
                self.scanner.run_scan(
                    account_id=account_id,
                    universe=universe,
                    started_at=at,
                    scan_key=(
                        f"{job_key}:SCAN"
                    ),
                )
            )

            scan_id = (
                scan_report.scan.scan_id
            )

            execution_report = (
                self.execution_engine.run(
                    account_id=account_id,
                    run_key=(
                        f"{job_key}:EXECUTION"
                    ),
                    scan_id=scan_id,
                    run_at=at,
                )
            )

            execution_run_id = (
                execution_report
                .run
                .run_id
            )

            subject, text = (
                self._daily_summary(
                    account_id=account_id,
                    session=session,
                    scan_report=(
                        scan_report
                    ),
                    execution_report=(
                        execution_report
                    ),
                )
            )

            directly_queued += (
                self._queue_external(
                    account_id=account_id,
                    event_type=(
                        "DAILY_PORTFOLIO_SUMMARY"
                    ),
                    reference_type=(
                        "JOB_RUN"
                    ),
                    reference_id=(
                        job.job_run_id
                    ),
                    subject=subject,
                    text=text,
                    created_at=at,
                )
            )

            if (
                execution_report
                .entry_block_reasons
            ):
                block_text = "\n".join(
                    (
                        "New paper entries were "
                        "blocked.",
                        "",
                        *execution_report
                        .entry_block_reasons,
                    )
                )

                directly_queued += (
                    self._queue_external(
                        account_id=(
                            account_id
                        ),
                        event_type=(
                            "RISK_REJECTION"
                        ),
                        reference_type=(
                            "EXECUTION_RUN"
                        ),
                        reference_id=(
                            execution_run_id
                        ),
                        subject=(
                            "Paper-entry risk "
                            "rejection"
                        ),
                        text=block_text,
                        created_at=at,
                    )
                )

            scan_status = (
                scan_report.scan.status.value
            )

            execution_has_errors = (
                execution_report
                .run
                .status
                is ExecutionRunStatus
                .COMPLETED_WITH_ERRORS
                or execution_report
                .run
                .error_count
                > 0
            )

            scan_has_errors = (
                "ERROR" in scan_status
                or "FAILED" in scan_status
            )

            if (
                execution_has_errors
                or scan_has_errors
            ):
                directly_queued += (
                    self._queue_external(
                        account_id=(
                            account_id
                        ),
                        event_type=(
                            "SYSTEM_WARNING"
                        ),
                        reference_type=(
                            "JOB_RUN"
                        ),
                        reference_id=(
                            job.job_run_id
                        ),
                        subject=(
                            "Paper job completed "
                            "with errors"
                        ),
                        text=(
                            "The scheduled paper "
                            "cycle completed with "
                            "one or more processing "
                            "errors.\n"
                            f"Scan status: "
                            f"{scan_status}\n"
                            "Execution errors: "
                            f"{execution_report.run.error_count}"
                        ),
                        created_at=at,
                    )
                )

            fanned_out, dispatch = (
                self._dispatch(
                    account_id=(
                        account_id
                    ),
                    attempted_at=at,
                )
            )

            completed_with_errors = (
                dispatch.failed > 0
                or execution_has_errors
                or scan_has_errors
            )

            job = (
                self.job_repository
                .complete_job(
                    job.job_run_id,
                    status=(
                        JobStatus
                        .COMPLETED_WITH_ERRORS
                        if completed_with_errors
                        else JobStatus.COMPLETED
                    ),
                    completed_at=at,
                    scan_id=scan_id,
                    execution_run_id=(
                        execution_run_id
                    ),
                    queued_notifications=(
                        directly_queued
                        + fanned_out
                    ),
                    sent_notifications=(
                        dispatch.sent
                    ),
                    failed_notifications=(
                        dispatch.failed
                    ),
                    metadata={
                        "session_date":
                        session
                        .session_date
                        .isoformat(),
                        "scan_status":
                        scan_status,
                        "execution_status":
                        execution_report
                        .run
                        .status
                        .value,
                        "entries_enabled":
                        execution_report
                        .entries_enabled,
                        "candidate_count":
                        len(
                            scan_report
                            .candidates
                        ),
                    },
                )
            )

            return ScheduledJobReport(
                job=job,
                session=session,
                duplicate=False,
            )

        except Exception as exc:
            failure_text = (
                "The scheduled paper-trading "
                "cycle failed.\n"
                f"Error type: "
                f"{type(exc).__name__}\n"
                f"Error: {exc}"
            )

            directly_queued += (
                self._queue_external(
                    account_id=account_id,
                    event_type=(
                        "SYSTEM_FAILURE"
                    ),
                    reference_type=(
                        "JOB_RUN"
                    ),
                    reference_id=(
                        job.job_run_id
                    ),
                    subject=(
                        "Scheduled paper job "
                        "failed"
                    ),
                    text=failure_text,
                    created_at=at,
                )
            )

            try:
                fanned_out, dispatch = (
                    self._dispatch(
                        account_id=(
                            account_id
                        ),
                        attempted_at=at,
                    )
                )
            except Exception:
                fanned_out = 0
                dispatch = DispatchReport(
                    processed=0,
                    sent=0,
                    failed=0,
                    skipped=0,
                    sent_notification_ids=(),
                    failed_notification_ids=(),
                )

            job = (
                self.job_repository
                .complete_job(
                    job.job_run_id,
                    status=JobStatus.FAILED,
                    completed_at=at,
                    scan_id=scan_id,
                    execution_run_id=(
                        execution_run_id
                    ),
                    queued_notifications=(
                        directly_queued
                        + fanned_out
                    ),
                    sent_notifications=(
                        dispatch.sent
                    ),
                    failed_notifications=(
                        dispatch.failed
                    ),
                    metadata={
                        "session_date":
                        session
                        .session_date
                        .isoformat(),
                    },
                    error_message=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )

            return ScheduledJobReport(
                job=job,
                session=session,
                duplicate=False,
            )

    def run_weekly_report(
        self,
        *,
        account_id: str,
        scheduled_for: datetime,
    ) -> ScheduledJobReport:
        at = self._aware(
            scheduled_for,
            name="scheduled_for",
        )

        session = (
            self.calendar
            .session_for_run(at)
        )

        if (
            session is None
            or not self.calendar
            .is_after_close(at)
            or not self.calendar
            .is_last_session_of_week(
                session.session_date
            )
        ):
            local_date = (
                at.astimezone(
                    self.calendar.timezone
                )
                .date()
            )

            key = (
                "WEEKLY_REPORT_PRECHECK:"
                f"{self.calendar.exchange_code}:"
                f"{at.replace(second=0, microsecond=0).isoformat()}"
            )

            job, created = (
                self.job_repository
                .start_job(
                    account_id=account_id,
                    job_key=key,
                    job_type=(
                        JobType.WEEKLY_REPORT
                    ),
                    scheduled_for=at,
                    exchange_code=(
                        self.calendar
                        .exchange_code
                    ),
                    metadata={
                        "local_date":
                        local_date
                        .isoformat(),
                    },
                    started_at=at,
                )
            )

            reason = (
                "The run is not after the "
                "final exchange session of "
                "the week."
            )

            job, duplicate = self._resume_if_interrupted(
                job,
                created=created,
                resumed_at=at,
            )

            if not duplicate:
                job = (
                    self.job_repository
                    .complete_job(
                        job.job_run_id,
                        status=(
                            JobStatus.SKIPPED
                        ),
                        completed_at=at,
                        metadata={
                            "skip_reason":
                            reason,
                        },
                    )
                )

            return ScheduledJobReport(
                job=job,
                session=session,
                duplicate=duplicate,
                skipped_reason=reason,
            )

        job_key = self._weekly_job_key(
            session.session_date
        )

        job, created = (
            self.job_repository
            .start_job(
                account_id=account_id,
                job_key=job_key,
                job_type=(
                    JobType.WEEKLY_REPORT
                ),
                scheduled_for=at,
                exchange_code=(
                    self.calendar
                    .exchange_code
                ),
                metadata={
                    "session_date":
                    session
                    .session_date
                    .isoformat(),
                },
                started_at=at,
            )
        )

        job, duplicate = self._resume_if_interrupted(
            job,
            created=created,
            resumed_at=at,
        )

        if duplicate:
            return ScheduledJobReport(
                job=job,
                session=session,
                duplicate=True,
            )

        directly_queued = 0

        try:
            subject, text = (
                self._weekly_summary(
                    account_id=account_id,
                    session=session,
                )
            )

            directly_queued += (
                self._queue_external(
                    account_id=account_id,
                    event_type=(
                        "WEEKLY_PERFORMANCE_REPORT"
                    ),
                    reference_type=(
                        "JOB_RUN"
                    ),
                    reference_id=(
                        job.job_run_id
                    ),
                    subject=subject,
                    text=text,
                    created_at=at,
                )
            )

            fanned_out, dispatch = (
                self._dispatch(
                    account_id=(
                        account_id
                    ),
                    attempted_at=at,
                )
            )

            job = (
                self.job_repository
                .complete_job(
                    job.job_run_id,
                    status=(
                        JobStatus
                        .COMPLETED_WITH_ERRORS
                        if dispatch.failed
                        else JobStatus.COMPLETED
                    ),
                    completed_at=at,
                    queued_notifications=(
                        directly_queued
                        + fanned_out
                    ),
                    sent_notifications=(
                        dispatch.sent
                    ),
                    failed_notifications=(
                        dispatch.failed
                    ),
                    metadata={
                        "session_date":
                        session
                        .session_date
                        .isoformat(),
                    },
                )
            )

            return ScheduledJobReport(
                job=job,
                session=session,
                duplicate=False,
            )

        except Exception as exc:
            directly_queued += (
                self._queue_external(
                    account_id=account_id,
                    event_type=(
                        "SYSTEM_FAILURE"
                    ),
                    reference_type=(
                        "JOB_RUN"
                    ),
                    reference_id=(
                        job.job_run_id
                    ),
                    subject=(
                        "Weekly paper report "
                        "failed"
                    ),
                    text=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                    created_at=at,
                )
            )

            try:
                fanned_out, dispatch = (
                    self._dispatch(
                        account_id=(
                            account_id
                        ),
                        attempted_at=at,
                    )
                )
            except Exception:
                fanned_out = 0
                dispatch = DispatchReport(
                    processed=0,
                    sent=0,
                    failed=0,
                    skipped=0,
                    sent_notification_ids=(),
                    failed_notification_ids=(),
                )

            job = (
                self.job_repository
                .complete_job(
                    job.job_run_id,
                    status=JobStatus.FAILED,
                    completed_at=at,
                    queued_notifications=(
                        directly_queued
                        + fanned_out
                    ),
                    sent_notifications=(
                        dispatch.sent
                    ),
                    failed_notifications=(
                        dispatch.failed
                    ),
                    error_message=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )
            )

            return ScheduledJobReport(
                job=job,
                session=session,
                duplicate=False,
            )
