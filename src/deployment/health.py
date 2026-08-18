"""Dependency-free deployment health and heartbeat service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from http.server import (
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
)
import json
import os
from pathlib import Path
import sqlite3
from typing import Callable


DEFAULT_DATABASE_PATH = Path("data/paper_trading.db")
DEFAULT_HEARTBEAT_PATH = Path("data/worker-heartbeat.json")
DEFAULT_HEARTBEAT_MAX_AGE_SECONDS = 900


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            "Heartbeat time must be timezone-aware."
        )

    return value.astimezone(
        timezone.utc
    ).isoformat()


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)

    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
    ):
        raise ValueError(
            "Heartbeat time must be timezone-aware."
        )

    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class HeartbeatRecord:
    observed_at: datetime
    status: str
    run_id: str | None = None

    def __post_init__(self) -> None:
        normalized_status = self.status.strip()

        if not normalized_status:
            raise ValueError(
                "Heartbeat status is required."
            )

        object.__setattr__(
            self,
            "observed_at",
            _parse_timestamp(
                _timestamp(self.observed_at)
            ),
        )
        object.__setattr__(
            self,
            "status",
            normalized_status,
        )


class HeartbeatStore:
    """Persist the worker heartbeat outside process memory."""

    def __init__(
        self,
        path: str | Path = DEFAULT_HEARTBEAT_PATH,
    ) -> None:
        self.path = Path(path)

    def write(
        self,
        record: HeartbeatRecord,
    ) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        temporary_path = self.path.with_suffix(
            self.path.suffix + ".tmp"
        )
        payload = {
            "observed_at": _timestamp(
                record.observed_at
            ),
            "status": record.status,
            "run_id": record.run_id,
        }
        temporary_path.write_text(
            json.dumps(
                payload,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)

    def read(self) -> HeartbeatRecord | None:
        if not self.path.exists():
            return None

        payload = json.loads(
            self.path.read_text(
                encoding="utf-8"
            )
        )

        return HeartbeatRecord(
            observed_at=_parse_timestamp(
                payload["observed_at"]
            ),
            status=payload["status"],
            run_id=payload.get("run_id"),
        )


@dataclass(frozen=True, slots=True)
class HealthResult:
    status: str
    checks: dict[str, str]

    @property
    def healthy(self) -> bool:
        return self.status == "ok"

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


class HealthEvaluator:
    """Evaluate process, database and worker health separately."""

    def __init__(
        self,
        *,
        database_path: str | Path = (
            DEFAULT_DATABASE_PATH
        ),
        heartbeat_store: HeartbeatStore | None = None,
        heartbeat_max_age_seconds: int = (
            DEFAULT_HEARTBEAT_MAX_AGE_SECONDS
        ),
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        if heartbeat_max_age_seconds < 1:
            raise ValueError(
                "heartbeat_max_age_seconds "
                "must be positive."
            )

        self.database_path = Path(database_path)
        self.heartbeat_store = (
            heartbeat_store or HeartbeatStore()
        )
        self.heartbeat_max_age = timedelta(
            seconds=heartbeat_max_age_seconds
        )
        self.now = now

    def liveness(self) -> HealthResult:
        return HealthResult(
            status="ok",
            checks={"process": "running"},
        )

    def readiness(self) -> HealthResult:
        try:
            self.database_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            connection = sqlite3.connect(
                str(self.database_path),
                timeout=5,
            )

            try:
                connection.execute(
                    "SELECT 1"
                ).fetchone()
            finally:
                connection.close()
        except Exception:
            return HealthResult(
                status="error",
                checks={
                    "database": "unavailable"
                },
            )

        return HealthResult(
            status="ok",
            checks={"database": "available"},
        )

    def worker_health(self) -> HealthResult:
        try:
            heartbeat = (
                self.heartbeat_store.read()
            )
        except Exception:
            return HealthResult(
                status="error",
                checks={
                    "heartbeat": "invalid"
                },
            )

        if heartbeat is None:
            return HealthResult(
                status="error",
                checks={
                    "heartbeat": "missing"
                },
            )

        age = self.now() - heartbeat.observed_at

        if (
            age < timedelta(0)
            or age > self.heartbeat_max_age
        ):
            return HealthResult(
                status="error",
                checks={"heartbeat": "stale"},
            )

        return HealthResult(
            status="ok",
            checks={
                "heartbeat": "current",
                "worker_status": heartbeat.status,
            },
        )


def _handler_factory(
    evaluator: HealthEvaluator,
) -> type[BaseHTTPRequestHandler]:
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            routes = {
                "/health/live":
                    evaluator.liveness,
                "/health/ready":
                    evaluator.readiness,
                "/health/worker":
                    evaluator.worker_health,
            }
            check = routes.get(self.path)

            if check is None:
                self.send_error(404)
                return

            result = check()
            body = json.dumps(
                result.as_payload(),
                sort_keys=True,
            ).encode("utf-8")
            self.send_response(
                200 if result.healthy else 503
            )
            self.send_header(
                "Content-Type",
                "application/json",
            )
            self.send_header(
                "Content-Length",
                str(len(body)),
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(
            self,
            format: str,
            *args: object,
        ) -> None:
            return None

    return HealthHandler


def serve() -> None:
    host = os.getenv(
        "BSAVCI_HEALTH_HOST",
        "0.0.0.0",
    )
    port = int(
        os.getenv(
            "BSAVCI_HEALTH_PORT",
            "8080",
        )
    )
    database_path = os.getenv(
        "BSAVCI_DATABASE_PATH",
        str(DEFAULT_DATABASE_PATH),
    )
    heartbeat_path = os.getenv(
        "BSAVCI_HEARTBEAT_PATH",
        str(DEFAULT_HEARTBEAT_PATH),
    )
    maximum_age = int(
        os.getenv(
            "BSAVCI_HEARTBEAT_MAX_AGE_SECONDS",
            str(
                DEFAULT_HEARTBEAT_MAX_AGE_SECONDS
            ),
        )
    )
    evaluator = HealthEvaluator(
        database_path=database_path,
        heartbeat_store=HeartbeatStore(
            heartbeat_path
        ),
        heartbeat_max_age_seconds=maximum_age,
    )
    server = ThreadingHTTPServer(
        (host, port),
        _handler_factory(evaluator),
    )
    server.serve_forever()


if __name__ == "__main__":
    serve()
