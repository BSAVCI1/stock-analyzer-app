from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.jobs import (
    InvocationStatus,
    due_invocations,
    run_orchestration_cycle,
)


def at(hour: int, minute: int) -> datetime:
    return datetime(
        2026,
        8,
        3,
        hour,
        minute,
        tzinfo=timezone.utc,
    )


def test_due_window_is_chronological_and_bounded() -> None:
    due = due_invocations(
        window_started_at=at(12, 59),
        window_ended_at=at(14, 1),
    )

    assert due
    assert all(
        at(12, 59)
        < item.scheduled_for
        <= at(14, 1)
        for item in due
    )
    assert [
        item.scheduled_for
        for item in due
    ] == sorted(
        item.scheduled_for
        for item in due
    )


def test_due_window_uses_open_start_boundary() -> None:
    due = due_invocations(
        window_started_at=at(13, 0),
        window_ended_at=at(13, 30),
    )

    assert due
    assert all(
        item.scheduled_for > at(13, 0)
        for item in due
    )
    assert any(
        item.scheduled_for == at(13, 30)
        for item in due
    )


def test_replayed_completed_keys_are_duplicates() -> None:
    due = due_invocations(
        window_started_at=at(12, 59),
        window_ended_at=at(13, 31),
    )
    calls = []

    report = run_orchestration_cycle(
        window_started_at=at(12, 59),
        window_ended_at=at(13, 31),
        completed_keys={
            item.idempotency_key
            for item in due
        },
        executor=lambda invocation:
        calls.append(invocation),
    )

    assert calls == []
    assert report.executed_count == 0
    assert report.failed_count == 0
    assert (
        report.duplicate_count
        == len(due)
    )


def test_failure_does_not_suppress_later_jobs() -> None:
    calls = []

    def executor(invocation):
        calls.append(invocation)

        if len(calls) == 1:
            raise RuntimeError(
                "simulated job failure"
            )

        return True

    report = run_orchestration_cycle(
        window_started_at=at(12, 59),
        window_ended_at=at(14, 1),
        executor=executor,
    )

    assert report.failed_count == 1
    assert report.executed_count >= 1
    assert len(calls) == len(
        report.due_invocations
    )
    assert (
        "simulated job failure"
        in report.results[0].error_message
    )
    assert all(
        item.status
        is InvocationStatus.EXECUTED
        for item in report.results[1:]
    )


def test_executor_can_report_downstream_duplicate() -> None:
    report = run_orchestration_cycle(
        window_started_at=at(12, 59),
        window_ended_at=at(13, 1),
        executor=lambda invocation: False,
    )

    assert report.due_invocations
    assert report.executed_count == 0
    assert (
        report.duplicate_count
        == len(report.due_invocations)
    )


def test_invalid_recovery_window_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be before",
    ):
        due_invocations(
            window_started_at=at(14, 0),
            window_ended_at=at(13, 0),
        )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        due_invocations(
            window_started_at=datetime(
                2026,
                8,
                3,
                13,
                0,
            ),
            window_ended_at=at(14, 0),
        )
