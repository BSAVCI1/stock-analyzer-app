"""Portable process hosting health and scheduled work together."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import signal
from threading import Event, Thread
from typing import Mapping

from .health import (
    HeartbeatStore,
    HealthEvaluator,
    create_health_server,
)
from .worker import (
    ScheduledWorker,
    ScheduledWorkerConfig,
    load_scheduled_cycle,
)


class PortableRuntime:
    """Coordinate one health server and one worker process."""

    def __init__(
        self,
        *,
        worker: ScheduledWorker,
        health_server: object,
        stop_event: Event | None = None,
    ) -> None:
        self.worker = worker
        self.health_server = health_server
        self.stop_event = (
            stop_event or Event()
        )
        self._health_thread: Thread | None = None
        self._worker_thread: Thread | None = None

    def start(self) -> None:
        if (
            self._health_thread is not None
            or self._worker_thread is not None
        ):
            raise RuntimeError(
                "Portable runtime already started."
            )

        self._health_thread = Thread(
            target=(
                self.health_server
                .serve_forever
            ),
            name="health-server",
            daemon=True,
        )
        self._worker_thread = Thread(
            target=self.worker.run_forever,
            name="scheduled-worker",
            daemon=True,
        )
        self._health_thread.start()
        self._worker_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.worker.stop()
        self.health_server.shutdown()

    def wait(self) -> None:
        if (
            self._health_thread is None
            or self._worker_thread is None
        ):
            raise RuntimeError(
                "Portable runtime has not started."
            )

        try:
            while not self.stop_event.wait(1):
                if not self._health_thread.is_alive():
                    raise RuntimeError(
                        "Health server stopped "
                        "unexpectedly."
                    )

                if not self._worker_thread.is_alive():
                    raise RuntimeError(
                        "Scheduled worker stopped "
                        "unexpectedly."
                    )
        finally:
            self.worker.stop()
            self.health_server.shutdown()
            self._worker_thread.join(
                timeout=10
            )
            self._health_thread.join(
                timeout=10
            )
            self.health_server.server_close()


def build_portable_runtime(
    environ: Mapping[str, str] | None = None,
) -> PortableRuntime:
    values = (
        environ
        if environ is not None
        else os.environ
    )
    cycle_path = values.get(
        "BSAVCI_WORKER_CYCLE",
        "",
    ).strip()

    if not cycle_path:
        raise RuntimeError(
            "BSAVCI_WORKER_CYCLE is required."
        )

    heartbeat_path = values.get(
        "BSAVCI_HEARTBEAT_PATH",
        "data/worker-heartbeat.json",
    )
    heartbeat_store = HeartbeatStore(
        heartbeat_path
    )
    worker = ScheduledWorker(
        load_scheduled_cycle(cycle_path),
        heartbeat_store=heartbeat_store,
        config=ScheduledWorkerConfig(
            interval_seconds=int(
                values.get(
                    "BSAVCI_WORKER_INTERVAL_SECONDS",
                    "900",
                )
            ),
            run_immediately=(
                values.get(
                    "BSAVCI_WORKER_RUN_IMMEDIATELY",
                    "true",
                ).strip().lower()
                in {"1", "true", "yes"}
            ),
            run_key_prefix=values.get(
                "BSAVCI_RUN_KEY_PREFIX",
                "managed",
            ),
        ),
    )
    evaluator = HealthEvaluator(
        database_path=values.get(
            "BSAVCI_DATABASE_PATH",
            "data/paper_trading.db",
        ),
        heartbeat_store=heartbeat_store,
        heartbeat_max_age_seconds=int(
            values.get(
                "BSAVCI_HEARTBEAT_MAX_AGE_SECONDS",
                "900",
            )
        ),
    )
    server = create_health_server(
        evaluator,
        host=values.get(
            "BSAVCI_HEALTH_HOST",
            "0.0.0.0",
        ),
        port=int(
            values.get(
                "BSAVCI_HEALTH_PORT",
                "8080",
            )
        ),
    )

    return PortableRuntime(
        worker=worker,
        health_server=server,
    )


def serve() -> None:
    runtime = build_portable_runtime()

    def request_stop(
        signum: int,
        frame: object,
    ) -> None:
        runtime.stop()

    signal.signal(
        signal.SIGTERM,
        request_stop,
    )
    signal.signal(
        signal.SIGINT,
        request_stop,
    )
    runtime.start()
    runtime.wait()


if __name__ == "__main__":
    serve()
