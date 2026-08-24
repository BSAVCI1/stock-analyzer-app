from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
import sqlite3

from src.jobs import (
    OrchestrationRepository,
    PersistentOrchestrationService,
)
from src.paper import PaperRepository


NOW = datetime(
    2026,
    8,
    3,
    14,
    1,
    tzinfo=timezone.utc,
)


def make_service(
    tmp_path,
    *,
    executor,
    initial_lookback=timedelta(
        minutes=62
    ),
    maximum_recovery=timedelta(
        hours=6
    ),
):
    path = tmp_path / "orchestration.db"
    paper = PaperRepository(path)
    account = paper.create_account(
        name="Orchestration Test",
        base_currency="EUR",
        starting_balance="2000",
        created_at=(
            NOW - timedelta(days=1)
        ),
    )
    repository = (
        OrchestrationRepository(path)
    )
    service = (
        PersistentOrchestrationService(
            account_id=account.account_id,
            repository=repository,
            executor=executor,
            initial_lookback=(
                initial_lookback
            ),
            maximum_recovery=(
                maximum_recovery
            ),
        )
    )

    return (
        path,
        account,
        repository,
        service,
    )


def test_cycle_persists_outcomes_and_checkpoint(
    tmp_path,
) -> None:
    calls = []
    (
        path,
        account,
        repository,
        service,
    ) = make_service(
        tmp_path,
        executor=lambda item:
        calls.append(item),
    )

    report = service.run(now=NOW)
    records = repository.list_invocations(
        account_id=account.account_id,
        policy_version=(
            service.policy_version
        ),
    )

    assert calls
    assert report.cycle.executed_count == (
        len(calls)
    )
    assert records
    assert all(
        item.status == "EXECUTED"
        for item in records
    )
    assert repository.get_checkpoint(
        account_id=account.account_id,
        policy_version=(
            service.policy_version
        ),
    ) == NOW

    connection = sqlite3.connect(path)

    try:
        version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }
    finally:
        connection.close()

    assert version == 17
    assert {
        "paper_orchestration_invocations",
        "paper_orchestration_checkpoints",
    }.issubset(tables)


def test_failed_job_is_retried_without_repeating_success(
    tmp_path,
) -> None:
    first_calls = []

    def first_executor(item):
        first_calls.append(item)

        if len(first_calls) == 1:
            raise RuntimeError(
                "temporary failure"
            )

        return True

    (
        _,
        account,
        repository,
        service,
    ) = make_service(
        tmp_path,
        executor=first_executor,
    )
    first = service.run(now=NOW)

    assert first.cycle.failed_count == 1
    assert (
        first.stored_checkpoint
        < first.cycle.results[0]
        .invocation.scheduled_for
    )

    second_calls = []
    second_service = (
        PersistentOrchestrationService(
            account_id=account.account_id,
            repository=repository,
            executor=lambda item:
            second_calls.append(item),
        )
    )
    second = second_service.run(
        now=NOW + timedelta(minutes=1)
    )

    assert second.cycle.failed_count == 0
    assert second_calls
    assert (
        second.cycle.duplicate_count
        >= first.cycle.executed_count
    )

    records = repository.list_invocations(
        account_id=account.account_id,
        policy_version=(
            service.policy_version
        ),
    )
    retried = next(
        item
        for item in records
        if item.idempotency_key
        == first.cycle.results[0]
        .invocation.idempotency_key
    )

    assert retried.status == "EXECUTED"
    assert retried.attempt_count == 2


def test_downtime_beyond_recovery_is_visible_as_missed(
    tmp_path,
) -> None:
    (
        _,
        account,
        repository,
        service,
    ) = make_service(
        tmp_path,
        executor=lambda item: True,
        initial_lookback=timedelta(
            hours=8
        ),
        maximum_recovery=timedelta(
            hours=1
        ),
    )

    report = service.run(now=NOW)
    missed = repository.list_invocations(
        account_id=account.account_id,
        policy_version=(
            service.policy_version
        ),
        status="MISSED",
    )

    assert report.missed_count > 0
    assert (
        len(missed)
        == report.missed_count
    )
    assert all(
        item.attempt_count == 0
        for item in missed
    )
    assert all(
        item.error_message
        == (
            "Invocation fell outside the "
            "bounded recovery window."
        )
        for item in missed
    )
