from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from types import SimpleNamespace

from src.automation import (
    AutomationRepository,
    ExecutionRunReport,
    ExecutionRunStatus,
)
from src.jobs import (
    JobRepository,
    JobStatus,
    JobType,
    ScheduledJobService,
)
from src.notifications import (
    DeliveryResult,
    NotificationService,
)
from src.paper import (
    NotificationChannel,
    PaperRepository,
)
from src.scanner import (
    ScannerRepository,
    StockUniverse,
)


AFTER_CLOSE = datetime(
    2026,
    8,
    3,
    20,
    30,
    tzinfo=timezone.utc,
)

BEFORE_CLOSE = datetime(
    2026,
    8,
    3,
    19,
    30,
    tzinfo=timezone.utc,
)

FRIDAY_AFTER_CLOSE = datetime(
    2026,
    8,
    7,
    20,
    30,
    tzinfo=timezone.utc,
)

SATURDAY = datetime(
    2026,
    8,
    1,
    20,
    30,
    tzinfo=timezone.utc,
)


class SuccessfulSender:
    def __init__(self) -> None:
        self.messages = []

    def send(self, notification):
        self.messages.append(notification)

        return DeliveryResult(
            provider_message_id=(
                f"MSG-{len(self.messages)}"
            ),
            metadata={
                "provider": "test",
            },
        )


class FakeScanner:
    def __init__(
        self,
        repository: ScannerRepository,
    ) -> None:
        self.repository = repository
        self.calls = 0
        self.fail = False

    def run_scan(
        self,
        *,
        account_id,
        universe,
        started_at,
        scan_key,
    ):
        self.calls += 1

        if self.fail:
            raise RuntimeError(
                "Scanner unavailable."
            )

        scan, _ = (
            self.repository.start_scan(
                account_id=account_id,
                universe=universe,
                configuration={},
                app_version="test",
                started_at=started_at,
                scan_key=scan_key,
            )
        )

        self.repository.complete_scan(
            scan.scan_id,
            completed_at=started_at,
        )

        return self.repository.get_report(
            scan.scan_id
        )


class FakeExecutionEngine:
    def __init__(
        self,
        *,
        automation_repository:
        AutomationRepository,
        paper_repository:
        PaperRepository,
    ) -> None:
        self.automation_repository = (
            automation_repository
        )
        self.paper_repository = (
            paper_repository
        )
        self.calls = 0
        self.entry_block_reasons = ()
        self.error_count = 0

    def run(
        self,
        *,
        account_id,
        run_key,
        scan_id=None,
        run_at=None,
    ):
        self.calls += 1

        run, created = (
            self.automation_repository
            .start_run(
                account_id=account_id,
                run_key=run_key,
                scan_id=scan_id,
                configuration={},
                app_version="test",
                started_at=run_at,
            )
        )

        if created:
            run = (
                self.automation_repository
                .complete_run(
                    run.run_id,
                    status=(
                        ExecutionRunStatus
                        .COMPLETED_WITH_ERRORS
                        if self.error_count
                        else ExecutionRunStatus
                        .COMPLETED
                    ),
                    completed_at=run_at,
                    created_orders=0,
                    filled_orders=0,
                    expired_orders=0,
                    cancelled_orders=0,
                    closed_positions=0,
                    rejected_entries=0,
                    error_count=(
                        self.error_count
                    ),
                    entry_block_reasons=(
                        self.entry_block_reasons
                    ),
                )
            )

        return ExecutionRunReport(
            run=run,
            entries_enabled=(
                not bool(
                    self.entry_block_reasons
                )
            ),
            entry_block_reasons=(
                self.entry_block_reasons
            ),
            reconciliation=(
                self.paper_repository
                .reconcile_account(
                    account_id
                )
            ),
            equity_snapshot=None,
        )


def make_environment(tmp_path):
    database_path = (
        tmp_path / "scheduled.db"
    )

    paper_repository = PaperRepository(
        database_path
    )

    account = (
        paper_repository
        .create_account(
            name="Scheduled Test",
            base_currency="USD",
            starting_balance="10000",
            created_at=AFTER_CLOSE,
        )
    )

    scanner_repository = ScannerRepository(
        database_path
    )

    automation_repository = (
        AutomationRepository(
            database_path
        )
    )

    job_repository = JobRepository(
        database_path
    )

    scanner = FakeScanner(
        scanner_repository
    )

    execution = FakeExecutionEngine(
        automation_repository=(
            automation_repository
        ),
        paper_repository=(
            paper_repository
        ),
    )

    sender = SuccessfulSender()

    notification_service = (
        NotificationService(
            paper_repository,
            senders={
                NotificationChannel.EMAIL:
                sender,
            },
        )
    )

    service = ScheduledJobService(
        job_repository=job_repository,
        paper_repository=(
            paper_repository
        ),
        scanner=scanner,
        execution_engine=execution,
        notification_service=(
            notification_service
        ),
        universe_loader=lambda: (
            StockUniverse(
                name="test",
                symbols=("AAPL",),
            )
        ),
        notification_channels=(
            NotificationChannel.EMAIL,
        ),
    )

    return SimpleNamespace(
        database_path=database_path,
        paper_repository=(
            paper_repository
        ),
        job_repository=job_repository,
        scanner=scanner,
        execution=execution,
        sender=sender,
        service=service,
        account=account,
    )


def test_closed_day_is_skipped(
    tmp_path,
) -> None:
    env = make_environment(tmp_path)

    report = (
        env.service.run_market_cycle(
            account_id=(
                env.account.account_id
            ),
            scheduled_for=SATURDAY,
        )
    )

    assert (
        report.job.status
        is JobStatus.SKIPPED
    )

    assert env.scanner.calls == 0
    assert env.execution.calls == 0


def test_before_close_does_not_block_after_close(
    tmp_path,
) -> None:
    env = make_environment(tmp_path)

    early = (
        env.service.run_market_cycle(
            account_id=(
                env.account.account_id
            ),
            scheduled_for=(
                BEFORE_CLOSE
            ),
        )
    )

    later = (
        env.service.run_market_cycle(
            account_id=(
                env.account.account_id
            ),
            scheduled_for=(
                AFTER_CLOSE
            ),
        )
    )

    assert (
        early.job.status
        is JobStatus.SKIPPED
    )

    assert (
        later.job.status
        is JobStatus.COMPLETED
    )

    assert env.scanner.calls == 1
    assert env.execution.calls == 1


def test_market_cycle_persists_links_and_summary(
    tmp_path,
) -> None:
    env = make_environment(tmp_path)

    report = (
        env.service.run_market_cycle(
            account_id=(
                env.account.account_id
            ),
            scheduled_for=(
                AFTER_CLOSE
            ),
        )
    )

    assert (
        report.job.status
        is JobStatus.COMPLETED
    )

    assert report.job.scan_id
    assert report.job.execution_run_id

    assert (
        report.job.sent_notifications
        == 1
    )

    assert len(env.sender.messages) == 1

    assert (
        "Daily paper portfolio summary"
        in env.sender.messages[0].text
    )


def test_repeated_market_cycle_is_idempotent(
    tmp_path,
) -> None:
    env = make_environment(tmp_path)

    first = (
        env.service.run_market_cycle(
            account_id=(
                env.account.account_id
            ),
            scheduled_for=(
                AFTER_CLOSE
            ),
        )
    )

    second = (
        env.service.run_market_cycle(
            account_id=(
                env.account.account_id
            ),
            scheduled_for=(
                AFTER_CLOSE
            ),
        )
    )

    assert second.duplicate is True

    assert (
        first.job.job_run_id
        == second.job.job_run_id
    )

    assert env.scanner.calls == 1
    assert env.execution.calls == 1
    assert len(env.sender.messages) == 1


def test_interrupted_market_cycle_resumes_once_after_restart(
    tmp_path,
) -> None:
    env = make_environment(tmp_path)
    session = env.service.calendar.session_for_run(
        AFTER_CLOSE
    )
    assert session is not None

    job_key = env.service._market_job_key(
        scheduled_for=AFTER_CLOSE,
        session=session,
        after_close=True,
    )
    interrupted, created = (
        env.job_repository.start_job(
            account_id=env.account.account_id,
            job_key=job_key,
            job_type=JobType.MARKET_CYCLE,
            scheduled_for=AFTER_CLOSE,
            exchange_code="XNYS",
            metadata={
                "session_date": (
                    session.session_date.isoformat()
                ),
            },
            started_at=AFTER_CLOSE,
        )
    )
    assert created is True
    assert interrupted.status is JobStatus.RUNNING

    restarted_paper = PaperRepository(
        env.database_path
    )
    restarted_automation = AutomationRepository(
        env.database_path
    )
    restarted_scanner = FakeScanner(
        ScannerRepository(env.database_path)
    )
    restarted_execution = FakeExecutionEngine(
        automation_repository=restarted_automation,
        paper_repository=restarted_paper,
    )
    restarted_sender = SuccessfulSender()
    restarted_service = ScheduledJobService(
        job_repository=JobRepository(env.database_path),
        paper_repository=restarted_paper,
        scanner=restarted_scanner,
        execution_engine=restarted_execution,
        notification_service=NotificationService(
            restarted_paper,
            senders={
                NotificationChannel.EMAIL:
                restarted_sender,
            },
        ),
        universe_loader=lambda: StockUniverse(
            name="test",
            symbols=("AAPL",),
        ),
        notification_channels=(
            NotificationChannel.EMAIL,
        ),
    )

    recovered = restarted_service.run_market_cycle(
        account_id=env.account.account_id,
        scheduled_for=AFTER_CLOSE,
    )
    replay = restarted_service.run_market_cycle(
        account_id=env.account.account_id,
        scheduled_for=AFTER_CLOSE,
    )

    assert recovered.duplicate is False
    assert recovered.job.job_run_id == interrupted.job_run_id
    assert recovered.job.status is JobStatus.COMPLETED
    assert recovered.job.metadata["recovery_count"] == 1
    assert recovered.job.metadata["last_recovered_at"] == (
        AFTER_CLOSE.isoformat()
    )

    assert replay.duplicate is True
    assert replay.job.job_run_id == interrupted.job_run_id
    assert restarted_scanner.calls == 1
    assert restarted_execution.calls == 1
    assert len(restarted_sender.messages) == 1

    events = env.paper_repository.list_system_events(
        env.account.account_id
    )
    recovery_events = tuple(
        event
        for event in events
        if event.event_type == "JOB_RUN_RECOVERED"
    )
    assert len(recovery_events) == 1
    assert recovery_events[0].reference_id == (
        interrupted.job_run_id
    )
    assert recovery_events[0].metadata["recovery_count"] == 1


def test_risk_block_queues_alert(
    tmp_path,
) -> None:
    env = make_environment(tmp_path)

    env.execution.entry_block_reasons = (
        "Daily realised-loss circuit "
        "breaker is active.",
    )

    report = (
        env.service.run_market_cycle(
            account_id=(
                env.account.account_id
            ),
            scheduled_for=(
                AFTER_CLOSE
            ),
        )
    )

    assert (
        report.job.status
        is JobStatus.COMPLETED
    )

    assert (
        report.job.sent_notifications
        == 2
    )

    texts = [
        message.text
        for message in env.sender.messages
    ]

    assert any(
        "Daily paper portfolio summary"
        in text
        for text in texts
    )

    assert any(
        "New paper entries were blocked"
        in text
        for text in texts
    )


def test_scanner_failure_is_persisted_and_alerted(
    tmp_path,
) -> None:
    env = make_environment(tmp_path)

    env.scanner.fail = True

    report = (
        env.service.run_market_cycle(
            account_id=(
                env.account.account_id
            ),
            scheduled_for=(
                AFTER_CLOSE
            ),
        )
    )

    assert (
        report.job.status
        is JobStatus.FAILED
    )

    assert "Scanner unavailable" in (
        report.job.error_message
    )

    assert (
        report.job.sent_notifications
        == 1
    )

    assert (
        "scheduled paper-trading "
        "cycle failed"
        in env.sender.messages[0]
        .text.lower()
    )


def test_weekly_report_runs_on_final_session(
    tmp_path,
) -> None:
    env = make_environment(tmp_path)

    report = (
        env.service.run_weekly_report(
            account_id=(
                env.account.account_id
            ),
            scheduled_for=(
                FRIDAY_AFTER_CLOSE
            ),
        )
    )

    assert (
        report.job.status
        is JobStatus.COMPLETED
    )

    assert (
        report.job.sent_notifications
        == 1
    )

    assert (
        "Weekly paper performance"
        in env.sender.messages[0].text
    )


def test_weekly_report_is_idempotent(
    tmp_path,
) -> None:
    env = make_environment(tmp_path)

    first = (
        env.service.run_weekly_report(
            account_id=(
                env.account.account_id
            ),
            scheduled_for=(
                FRIDAY_AFTER_CLOSE
            ),
        )
    )

    second = (
        env.service.run_weekly_report(
            account_id=(
                env.account.account_id
            ),
            scheduled_for=(
                FRIDAY_AFTER_CLOSE
            ),
        )
    )

    assert second.duplicate is True

    assert (
        first.job.job_run_id
        == second.job.job_run_id
    )

    assert len(env.sender.messages) == 1
