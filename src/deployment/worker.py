"""Managed, non-overlapping scheduler for paper cycles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import importlib
import os
import signal
from threading import Event, Lock
from typing import Callable, Protocol

from .health import (
    HeartbeatRecord,
    HeartbeatStore,
)


class ScheduledCycle(Protocol):
    def __call__(
        self,
        *,
        run_at: datetime,
        run_key: str,
    ) -> object:
        ...


@dataclass(frozen=True, slots=True)
class ScheduledWorkerConfig:
    interval_seconds: int = 900
    run_immediately: bool = True
    run_key_prefix: str = "managed"

    def __post_init__(self) -> None:
        if (
            isinstance(self.interval_seconds, bool)
            or not isinstance(
                self.interval_seconds,
                int,
            )
            or self.interval_seconds < 1
        ):
            raise ValueError(
                "interval_seconds must be positive."
            )

        prefix = self.run_key_prefix.strip()

        if not prefix:
            raise ValueError(
                "run_key_prefix is required."
            )

        object.__setattr__(
            self,
            "run_key_prefix",
            prefix,
        )


def scheduled_run_key(
    *,
    prefix: str,
    run_at: datetime,
) -> str:
    if (
        run_at.tzinfo is None
        or run_at.utcoffset() is None
    ):
        raise ValueError(
            "run_at must be timezone-aware."
        )

    normalized = run_at.astimezone(
        timezone.utc
    )

    return (
        f"{prefix}:"
        f"{normalized.strftime('%Y%m%dT%H%M%SZ')}"
    )


class ScheduledWorker:
    """Run one idempotent paper cycle per schedule slot."""

    def __init__(
        self,
        cycle: ScheduledCycle,
        *,
        heartbeat_store: HeartbeatStore,
        config: ScheduledWorkerConfig | None = None,
        now: Callable[[], datetime] = (
            lambda: datetime.now(timezone.utc)
        ),
        stop_event: Event | None = None,
    ) -> None:
        if not callable(cycle):
            raise ValueError(
                "cycle must be callable."
            )

        self.cycle = cycle
        self.heartbeat_store = heartbeat_store
        self.config = (
            config or ScheduledWorkerConfig()
        )
        self.now = now
        self.stop_event = stop_event or Event()
        self._run_lock = Lock()

    def _heartbeat(
        self,
        *,
        status: str,
        observed_at: datetime,
        run_id: str | None = None,
    ) -> None:
        self.heartbeat_store.write(
            HeartbeatRecord(
                observed_at=observed_at,
                status=status,
                run_id=run_id,
            )
        )

    def run_once(
        self,
        *,
        run_at: datetime | None = None,
    ) -> bool:
        at = run_at or self.now()

        if not self._run_lock.acquire(
            blocking=False
        ):
            return False

        run_key = scheduled_run_key(
            prefix=self.config.run_key_prefix,
            run_at=at,
        )

        try:
            self._heartbeat(
                status="running",
                observed_at=self.now(),
                run_id=run_key,
            )
            self.cycle(
                run_at=at,
                run_key=run_key,
            )
            self._heartbeat(
                status="idle",
                observed_at=self.now(),
                run_id=run_key,
            )
            return True
        except Exception:
            self._heartbeat(
                status="error",
                observed_at=self.now(),
                run_id=run_key,
            )
            raise
        finally:
            self._run_lock.release()

    def stop(self) -> None:
        self.stop_event.set()

    def run_forever(self) -> None:
        interval = timedelta(
            seconds=self.config.interval_seconds
        )
        current = self.now()
        next_run = (
            current
            if self.config.run_immediately
            else current + interval
        )
        self._heartbeat(
            status="starting",
            observed_at=current,
        )

        try:
            while not self.stop_event.is_set():
                delay = max(
                    0.0,
                    (
                        next_run - self.now()
                    ).total_seconds(),
                )

                if self.stop_event.wait(delay):
                    break

                try:
                    self.run_once(
                        run_at=next_run
                    )
                except Exception:
                    pass

                next_run += interval
                current = self.now()

                while next_run <= current:
                    next_run += interval
        finally:
            self._heartbeat(
                status="stopped",
                observed_at=self.now(),
            )


def load_scheduled_cycle(
    import_path: str,
) -> ScheduledCycle:
    module_name, separator, attribute = (
        import_path.partition(":")
    )

    if (
        not separator
        or not module_name
        or not attribute
    ):
        raise ValueError(
            "Cycle import path must use "
            "'module:function' format."
        )

    module = importlib.import_module(
        module_name
    )
    cycle = getattr(module, attribute)

    if not callable(cycle):
        raise ValueError(
            "Configured cycle is not callable."
        )

    return cycle


def serve() -> None:
    import_path = os.environ.get(
        "BSAVCI_WORKER_CYCLE",
        "",
    ).strip()

    if not import_path:
        raise RuntimeError(
            "BSAVCI_WORKER_CYCLE is required."
        )

    interval_seconds = int(
        os.environ.get(
            "BSAVCI_WORKER_INTERVAL_SECONDS",
            "900",
        )
    )
    run_immediately = (
        os.environ.get(
            "BSAVCI_WORKER_RUN_IMMEDIATELY",
            "true",
        ).strip().lower()
        in {"1", "true", "yes"}
    )
    heartbeat_path = os.environ.get(
        "BSAVCI_HEARTBEAT_PATH",
        "data/worker-heartbeat.json",
    )
    worker = ScheduledWorker(
        load_scheduled_cycle(import_path),
        heartbeat_store=HeartbeatStore(
            heartbeat_path
        ),
        config=ScheduledWorkerConfig(
            interval_seconds=interval_seconds,
            run_immediately=run_immediately,
            run_key_prefix=os.environ.get(
                "BSAVCI_RUN_KEY_PREFIX",
                "managed",
            ),
        ),
    )

    def request_stop(
        signum: int,
        frame: object,
    ) -> None:
        worker.stop()

    signal.signal(
        signal.SIGTERM,
        request_stop,
    )
    signal.signal(
        signal.SIGINT,
        request_stop,
    )
    worker.run_forever()


if __name__ == "__main__":
    serve()
