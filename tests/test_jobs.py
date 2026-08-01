from __future__ import annotations

from datetime import (
    date,
    datetime,
    timezone,
)

from src.jobs import (
    ExchangeCalendar,
    JobRepository,
    JobStatus,
    JobType,
)
from src.paper import PaperRepository


T0 = datetime(
    2026,
    8,
    3,
    20,
    30,
    tzinfo=timezone.utc,
)


def create_account(
    tmp_path,
):
    database_path = (
        tmp_path / "jobs.db"
    )

    paper_repository = PaperRepository(
        database_path
    )

    account = (
        paper_repository
        .create_account(
            name="Job Test",
            base_currency="USD",
            starting_balance="10000",
            created_at=T0,
        )
    )

    return (
        database_path,
        account,
    )


def test_weekend_is_not_exchange_session() -> None:
    calendar = ExchangeCalendar()

    assert calendar.is_session(
        date(2026, 8, 1)
    ) is False

    assert calendar.is_session(
        date(2026, 8, 2)
    ) is False


def test_independence_day_observed_is_closed() -> None:
    calendar = ExchangeCalendar()

    # 4 July 2026 is Saturday, so
    # Friday 3 July is observed.
    assert calendar.is_session(
        date(2026, 7, 3)
    ) is False


def test_regular_monday_is_exchange_session() -> None:
    calendar = ExchangeCalendar()

    assert calendar.is_session(
        date(2026, 8, 3)
    ) is True


def test_run_is_due_only_after_market_close() -> None:
    calendar = ExchangeCalendar()

    before_close = datetime(
        2026,
        8,
        3,
        19,
        59,
        tzinfo=timezone.utc,
    )

    after_close = datetime(
        2026,
        8,
        3,
        20,
        1,
        tzinfo=timezone.utc,
    )

    assert calendar.is_after_close(
        before_close
    ) is False

    assert calendar.is_after_close(
        after_close
    ) is True


def test_friday_is_last_session_of_week() -> None:
    calendar = ExchangeCalendar()

    assert (
        calendar
        .is_last_session_of_week(
            date(2026, 8, 7)
        )
        is True
    )

    assert (
        calendar
        .is_last_session_of_week(
            date(2026, 8, 6)
        )
        is False
    )


def test_holiday_thursday_makes_wednesday_last_session() -> None:
    calendar = ExchangeCalendar()

    # Thanksgiving is Thursday,
    # but Friday remains a session.
    assert (
        calendar
        .is_last_session_of_week(
            date(2026, 11, 25)
        )
        is False
    )


def test_job_creation_is_idempotent(
    tmp_path,
) -> None:
    database_path, account = (
        create_account(tmp_path)
    )

    repository = JobRepository(
        database_path
    )

    first, first_created = (
        repository.start_job(
            account_id=(
                account.account_id
            ),
            job_key=(
                "MARKET_CYCLE:"
                "2026-08-03"
            ),
            job_type=(
                JobType.MARKET_CYCLE
            ),
            scheduled_for=T0,
            exchange_code="XNYS",
            metadata={
                "session_date":
                "2026-08-03",
            },
        )
    )

    second, second_created = (
        repository.start_job(
            account_id=(
                account.account_id
            ),
            job_key=(
                "MARKET_CYCLE:"
                "2026-08-03"
            ),
            job_type=(
                JobType.MARKET_CYCLE
            ),
            scheduled_for=T0,
            exchange_code="XNYS",
            metadata={},
        )
    )

    assert first_created is True
    assert second_created is False

    assert (
        first.job_run_id
        == second.job_run_id
    )


def test_completed_job_is_persisted(
    tmp_path,
) -> None:
    database_path, account = (
        create_account(tmp_path)
    )

    repository = JobRepository(
        database_path
    )

    job, _ = repository.start_job(
        account_id=account.account_id,
        job_key="WEEKLY:2026-W32",
        job_type=JobType.WEEKLY_REPORT,
        scheduled_for=T0,
        exchange_code="XNYS",
    )

    completed = repository.complete_job(
        job.job_run_id,
        status=JobStatus.COMPLETED,
        completed_at=T0,
        queued_notifications=2,
        sent_notifications=2,
        failed_notifications=0,
        metadata={
            "trade_count": 3,
        },
    )

    assert (
        completed.status
        is JobStatus.COMPLETED
    )

    assert (
        completed.queued_notifications
        == 2
    )

    assert (
        completed.sent_notifications
        == 2
    )

    assert (
        completed.metadata[
            "trade_count"
        ]
        == 3
    )
