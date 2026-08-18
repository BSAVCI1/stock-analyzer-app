from datetime import (
    datetime,
    timedelta,
    timezone,
)
import json

import pytest

from src.deployment import (
    HeartbeatRecord,
    HeartbeatStore,
    HealthEvaluator,
)


NOW = datetime(
    2026,
    8,
    18,
    10,
    0,
    tzinfo=timezone.utc,
)


def test_liveness_is_process_only(tmp_path):
    evaluator = HealthEvaluator(
        database_path=(
            tmp_path / "missing" / "paper.db"
        ),
        heartbeat_store=HeartbeatStore(
            tmp_path / "missing-heartbeat.json"
        ),
        now=lambda: NOW,
    )

    result = evaluator.liveness()

    assert result.healthy is True
    assert result.checks == {
        "process": "running"
    }


def test_readiness_creates_and_checks_database(
    tmp_path,
):
    database_path = (
        tmp_path / "persistent" / "paper.db"
    )
    evaluator = HealthEvaluator(
        database_path=database_path,
        now=lambda: NOW,
    )

    result = evaluator.readiness()

    assert result.healthy is True
    assert result.checks == {
        "database": "available"
    }
    assert database_path.exists()


def test_heartbeat_round_trip_is_persisted(
    tmp_path,
):
    path = (
        tmp_path / "state" / "heartbeat.json"
    )
    store = HeartbeatStore(path)
    expected = HeartbeatRecord(
        observed_at=NOW,
        status="idle",
        run_id="RUN-1",
    )

    store.write(expected)

    assert store.read() == expected
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    assert payload["run_id"] == "RUN-1"


def test_worker_health_requires_heartbeat(
    tmp_path,
):
    evaluator = HealthEvaluator(
        database_path=tmp_path / "paper.db",
        heartbeat_store=HeartbeatStore(
            tmp_path / "heartbeat.json"
        ),
        now=lambda: NOW,
    )

    result = evaluator.worker_health()

    assert result.healthy is False
    assert result.checks == {
        "heartbeat": "missing"
    }


def test_worker_health_rejects_stale_heartbeat(
    tmp_path,
):
    store = HeartbeatStore(
        tmp_path / "heartbeat.json"
    )
    store.write(
        HeartbeatRecord(
            observed_at=(
                NOW - timedelta(minutes=16)
            ),
            status="idle",
        )
    )
    evaluator = HealthEvaluator(
        database_path=tmp_path / "paper.db",
        heartbeat_store=store,
        heartbeat_max_age_seconds=900,
        now=lambda: NOW,
    )

    result = evaluator.worker_health()

    assert result.healthy is False
    assert result.checks == {
        "heartbeat": "stale"
    }


def test_worker_health_accepts_current_heartbeat(
    tmp_path,
):
    store = HeartbeatStore(
        tmp_path / "heartbeat.json"
    )
    store.write(
        HeartbeatRecord(
            observed_at=(
                NOW - timedelta(minutes=2)
            ),
            status="idle",
        )
    )
    evaluator = HealthEvaluator(
        database_path=tmp_path / "paper.db",
        heartbeat_store=store,
        heartbeat_max_age_seconds=900,
        now=lambda: NOW,
    )

    result = evaluator.worker_health()

    assert result.healthy is True
    assert result.checks == {
        "heartbeat": "current",
        "worker_status": "idle",
    }


def test_heartbeat_requires_timezone():
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        HeartbeatRecord(
            observed_at=datetime(
                2026,
                8,
                18,
                10,
                0,
            ),
            status="idle",
        )
