"""Managed adapter for the real internal paper-trading cycle."""

from __future__ import annotations

from datetime import datetime
import os
from typing import Mapping

from src.jobs.runtime import (
    build_runtime,
    load_runtime_settings,
)


_TRUE_VALUES = {"1", "true", "yes"}


def _enabled(
    values: Mapping[str, str],
    name: str,
) -> bool:
    return (
        values.get(name, "")
        .strip()
        .lower()
        in _TRUE_VALUES
    )


def paper_cycle(
    *,
    run_at: datetime,
    run_key: str,
    environ: Mapping[str, str] | None = None,
):
    """Run one persistent internal-paper orchestration cycle."""
    if run_at.tzinfo is None or run_at.utcoffset() is None:
        raise ValueError("run_at must be timezone-aware.")

    if not str(run_key).strip():
        raise ValueError("run_key is required.")

    values = os.environ if environ is None else environ

    if not _enabled(values, "BSAVCI_PAPER_CYCLE_ENABLED"):
        raise RuntimeError(
            "Managed paper cycle is disabled. "
            "Set BSAVCI_PAPER_CYCLE_ENABLED=true "
            "only in an approved paper profile."
        )

    if (
        _enabled(values, "PAPER_BROKER_ENABLED")
        or _enabled(values, "PAPER_BROKER_LIVE_TRADING")
    ):
        raise RuntimeError(
            "Managed deployment supports internal "
            "paper execution only; broker and live "
            "trading are prohibited."
        )

    database_path = (
        values.get("BSAVCI_DATABASE_PATH")
        or values.get("PAPER_DATABASE_PATH")
    )
    settings = load_runtime_settings(
        values,
        database_path=database_path,
    )
    runtime = build_runtime(
        settings,
        environ=values,
    )
    report = runtime.orchestration_service.run(
        now=run_at,
    )

    if report.cycle.failed_count:
        raise RuntimeError(
            "Managed paper cycle persisted "
            f"{report.cycle.failed_count} failed "
            "invocation(s)."
        )

    return report
