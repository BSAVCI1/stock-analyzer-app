from datetime import (
    datetime,
    timezone,
)
from threading import Event
import sqlite3
import subprocess
import sys

import pytest

from src.deployment.combined import (
    PortableRuntime,
)
from src.deployment.validation import (
    validation_cycle,
)


NOW = datetime(
    2026,
    8,
    18,
    18,
    0,
    tzinfo=timezone.utc,
)


class Worker:
    def __init__(self):
        self.started = Event()
        self.stopped = Event()

    def run_forever(self):
        self.started.set()
        self.stopped.wait(2)

    def stop(self):
        self.stopped.set()


class Server:
    def __init__(self):
        self.started = Event()
        self.stopped = Event()
        self.closed = False

    def serve_forever(self):
        self.started.set()
        self.stopped.wait(2)

    def shutdown(self):
        self.stopped.set()

    def server_close(self):
        self.closed = True


def test_portable_runtime_starts_and_stops_both():
    worker = Worker()
    server = Server()
    runtime = PortableRuntime(
        worker=worker,
        health_server=server,
    )

    runtime.start()

    assert worker.started.wait(1)
    assert server.started.wait(1)

    runtime.stop()
    runtime.wait()

    assert worker.stopped.is_set()
    assert server.stopped.is_set()
    assert server.closed is True


def test_portable_runtime_rejects_double_start():
    worker = Worker()
    server = Server()
    runtime = PortableRuntime(
        worker=worker,
        health_server=server,
    )
    runtime.start()

    try:
        with pytest.raises(
            RuntimeError,
            match="already started",
        ):
            runtime.start()
    finally:
        runtime.stop()
        runtime.wait()


def test_combined_module_can_run_without_preload_warning():
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            "-c",
            (
                "import runpy; "
                "runpy.run_module("
                "'src.deployment.combined', "
                "run_name='deployment_test'"
                ")"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "found in sys.modules" not in result.stderr


def test_validation_cycle_is_idempotent(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "validation.db"
    monkeypatch.setenv(
        "BSAVCI_VALIDATION_DATABASE_PATH",
        str(path),
    )

    validation_cycle(
        run_at=NOW,
        run_key="local:1",
    )
    validation_cycle(
        run_at=NOW,
        run_key="local:1",
    )

    connection = sqlite3.connect(path)

    try:
        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM deployment_validation_cycles
            """
        ).fetchone()[0]
    finally:
        connection.close()

    assert count == 1


def test_validation_cycle_requires_timezone(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "BSAVCI_VALIDATION_DATABASE_PATH",
        str(tmp_path / "validation.db"),
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        validation_cycle(
            run_at=NOW.replace(
                tzinfo=None
            ),
            run_key="local:1",
        )
