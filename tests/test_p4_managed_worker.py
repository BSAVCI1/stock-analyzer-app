from datetime import (
    datetime,
    timedelta,
    timezone,
)
from threading import Event, Thread

import pytest

from src.deployment import (
    HeartbeatStore,
    ScheduledWorker,
    ScheduledWorkerConfig,
    scheduled_run_key,
)


NOW = datetime(
    2026,
    8,
    18,
    12,
    0,
    tzinfo=timezone.utc,
)


def test_scheduled_run_key_is_deterministic():
    assert scheduled_run_key(
        prefix="paper",
        run_at=NOW,
    ) == "paper:20260818T120000Z"


def test_scheduled_run_key_requires_timezone():
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        scheduled_run_key(
            prefix="paper",
            run_at=NOW.replace(
                tzinfo=None
            ),
        )


def test_run_once_records_running_and_idle(
    tmp_path,
):
    heartbeat = HeartbeatStore(
        tmp_path / "heartbeat.json"
    )
    calls = []

    def cycle(*, run_at, run_key):
        calls.append((run_at, run_key))
        assert (
            heartbeat.read().status
            == "running"
        )

    worker = ScheduledWorker(
        cycle,
        heartbeat_store=heartbeat,
        config=ScheduledWorkerConfig(
            run_key_prefix="paper",
        ),
        now=lambda: NOW,
    )

    assert worker.run_once(
        run_at=NOW
    ) is True
    assert calls == [
        (
            NOW,
            "paper:20260818T120000Z",
        )
    ]
    final = heartbeat.read()
    assert final.status == "idle"
    assert (
        final.run_id
        == "paper:20260818T120000Z"
    )


def test_run_once_records_error_without_secret(
    tmp_path,
):
    heartbeat = HeartbeatStore(
        tmp_path / "heartbeat.json"
    )

    def cycle(*, run_at, run_key):
        raise RuntimeError(
            "password=super-secret"
        )

    worker = ScheduledWorker(
        cycle,
        heartbeat_store=heartbeat,
        now=lambda: NOW,
    )

    with pytest.raises(
        RuntimeError,
        match="super-secret",
    ):
        worker.run_once(run_at=NOW)

    payload = (
        tmp_path / "heartbeat.json"
    ).read_text(encoding="utf-8")
    assert "super-secret" not in payload
    assert heartbeat.read().status == "error"


def test_overlapping_cycle_is_skipped(
    tmp_path,
):
    started = Event()
    release = Event()
    heartbeat = HeartbeatStore(
        tmp_path / "heartbeat.json"
    )

    def cycle(*, run_at, run_key):
        started.set()
        release.wait(2)

    worker = ScheduledWorker(
        cycle,
        heartbeat_store=heartbeat,
        now=lambda: NOW,
    )
    thread = Thread(
        target=worker.run_once,
        kwargs={"run_at": NOW},
    )
    thread.start()
    assert started.wait(1)

    assert worker.run_once(
        run_at=NOW + timedelta(seconds=1)
    ) is False

    release.set()
    thread.join(timeout=2)
    assert thread.is_alive() is False


def test_worker_config_rejects_bad_interval():
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        ScheduledWorkerConfig(
            interval_seconds=0
        )


def test_stop_sets_worker_event(tmp_path):
    stop_event = Event()
    worker = ScheduledWorker(
        lambda **kwargs: None,
        heartbeat_store=HeartbeatStore(
            tmp_path / "heartbeat.json"
        ),
        stop_event=stop_event,
    )

    worker.stop()

    assert stop_event.is_set() is True
