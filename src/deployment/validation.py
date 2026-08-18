"""Harmless local cycle for deployment validation."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3


def validation_cycle(
    *,
    run_at: datetime,
    run_key: str,
) -> None:
    if (
        run_at.tzinfo is None
        or run_at.utcoffset() is None
    ):
        raise ValueError(
            "run_at must be timezone-aware."
        )

    path = Path(
        os.environ.get(
            "BSAVCI_VALIDATION_DATABASE_PATH",
            "data/deployment_validation.db",
        )
    )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    connection = sqlite3.connect(
        str(path),
        timeout=30,
    )

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS
                deployment_validation_cycles(
                    run_key TEXT PRIMARY KEY,
                    run_at TEXT NOT NULL
                )
            """
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO
                deployment_validation_cycles(
                    run_key,
                    run_at
                )
            VALUES (?, ?)
            """,
            (
                run_key,
                run_at.astimezone(
                    timezone.utc
                ).isoformat(),
            ),
        )
        connection.commit()
    finally:
        connection.close()
