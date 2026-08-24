"""P4.10.4 watchlist conversion and stale-signal analytics tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.portfolio_dashboard import actionability_rows, calculate_actionability
from src.scanner import WatchlistState
from src.strategy import StrategyHorizon


T0 = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


def _result(result_id, symbol, state, minutes, *, version="swing-v1"):
    return SimpleNamespace(
        result_id=result_id,
        symbol=symbol,
        watchlist_state=state,
        processed_at=T0 + timedelta(minutes=minutes),
        strategy_horizon=StrategyHorizon.SWING,
        strategy_version=version,
    )


def _signal(signal_id, expires_minutes):
    return SimpleNamespace(
        signal_id=signal_id,
        generated_at=T0,
        expires_at=T0 + timedelta(minutes=expires_minutes),
    )


def test_watchlist_conversion_is_episode_and_version_safe() -> None:
    scans = (
        SimpleNamespace(results=(
            _result("R1", "AAPL", WatchlistState.WATCH, 1),
            _result("R2", "MSFT", WatchlistState.PREPARE, 1),
        )),
        SimpleNamespace(results=(
            _result("R3", "AAPL", WatchlistState.PREPARE, 2),
            _result("R4", "MSFT", WatchlistState.ACTIONABLE, 2),
            _result("R5", "SAP", WatchlistState.WATCH, 2, version="swing-v2"),
        )),
        SimpleNamespace(results=(
            _result("R6", "AAPL", WatchlistState.ACTIONABLE, 3),
            _result("R7", "SAP", WatchlistState.REJECT, 3, version="swing-v2"),
        )),
    )
    summary = calculate_actionability(
        signals=(), orders=(), scans=scans, at=T0 + timedelta(hours=1),
    )
    assert summary.watchlist_entries == 3
    assert summary.converted_entries == 2
    assert summary.open_entries == 0
    assert summary.abandoned_entries == 1
    assert summary.conversion_rate_pct == pytest.approx(66.6666666667)
    assert {row.key for row in summary.cohorts} == {
        "SWING|swing-v1", "SWING|swing-v2",
    }
    rows = actionability_rows(SimpleNamespace(actionability=summary))
    assert rows[0]["watchlist_entries"] == 2


def test_stale_signal_rate_uses_only_matured_persisted_signals() -> None:
    signals = (
        _signal("S-ORDERED", 30),
        _signal("S-STALE", 30),
        _signal("S-FUTURE", 180),
    )
    orders = (SimpleNamespace(
        order_id="O1", signal_id="S-ORDERED", created_at=T0 + timedelta(minutes=10)
    ),)
    summary = calculate_actionability(
        signals=signals,
        orders=orders,
        scans=(),
        at=T0 + timedelta(minutes=60),
    )
    assert summary.signal_count == 3
    assert summary.matured_signal_count == 2
    assert summary.ordered_signal_count == 1
    assert summary.stale_signal_count == 1
    assert summary.stale_signal_rate_pct == 50.0
    assert "S-FUTURE" not in summary.provenance.record_ids


def test_actionability_empty_evidence_is_explicit() -> None:
    summary = calculate_actionability(
        signals=(), orders=(), scans=(), at=T0,
    )
    assert summary.conversion_rate_pct is None
    assert summary.stale_signal_rate_pct is None
    assert summary.cohorts == ()
