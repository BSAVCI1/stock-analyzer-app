"""Immutable market-data snapshots for independent horizon validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
import hashlib
import json
from typing import Any

import pandas as pd

from src.data.market_data import MarketSnapshot


SnapshotLoader = Callable[..., MarketSnapshot]
_VERSIONS = {
    "swing": "p4.3-swing-v1",
    "medium_term": "p4.3-medium-term-v1",
}
_SETTINGS = {
    "swing": ("2y", "1d", 252),
    "medium_term": ("5y", "1wk", 156),
}
_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _timestamp(value: object, label: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware datetime.")
    return value.isoformat()


def _symbols(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("symbols must be a sequence of ticker symbols.")
    result = tuple(sorted({str(item).strip().upper() for item in value}))
    if not result or any(not item for item in result):
        raise ValueError("symbols must contain non-empty ticker symbols.")
    return result


def _rows(frame: pd.DataFrame, symbol: str) -> list[dict[str, object]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{symbol} history cannot be empty.")
    missing = [column for column in _COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{symbol} history is missing: {', '.join(missing)}.")
    clean = frame.loc[:, list(_COLUMNS)].copy()
    clean.index = pd.to_datetime(clean.index, utc=True, errors="coerce")
    if clean.index.isna().any() or clean.index.duplicated().any():
        raise ValueError(f"{symbol} history has invalid or duplicate timestamps.")
    clean = clean.sort_index()
    for column in _COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean.isna().any().any():
        raise ValueError(f"{symbol} history contains non-numeric values.")
    return [
        {
            "at": at.isoformat(),
            "open": float(open_price),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
        }
        for at, open_price, high, low, close, volume
        in clean.itertuples(index=True, name=None)
    ]


def capture_horizon_dataset(
    *,
    horizon: str,
    symbols: Sequence[str],
    captured_at: datetime,
    loader: SnapshotLoader,
    policy_version: str,
    universe_policy_version: str,
) -> dict[str, Any]:
    """Download and fingerprint one horizon without mixing its observations."""

    name = str(horizon).strip().lower()
    if name not in _VERSIONS:
        raise ValueError("horizon must be swing or medium_term.")
    if not callable(loader):
        raise ValueError("loader must be callable.")
    capture_time = _timestamp(captured_at, "captured_at")
    period, interval, minimum_rows = _SETTINGS[name]
    instruments = []
    for symbol in _symbols(symbols):
        snapshot = loader(
            symbol, period=period, interval=interval, min_rows=minimum_rows
        )
        if snapshot.symbol != symbol:
            raise ValueError(f"Provider returned {snapshot.symbol} for {symbol}.")
        rows = _rows(snapshot.history, symbol)
        instruments.append({
            "symbol": symbol,
            "provider_fetched_at": _timestamp(
                snapshot.fetched_at_utc, f"{symbol}.provider_fetched_at"
            ),
            "row_count": len(rows),
            "first_observation": rows[0]["at"],
            "last_observation": rows[-1]["at"],
            "rows": rows,
        })
    body: dict[str, Any] = {
        "schema_version": 1,
        "horizon": name,
        "strategy_version": _VERSIONS[name],
        "captured_at": capture_time,
        "provider": "YAHOO_FINANCE",
        "period": period,
        "interval": interval,
        "policy_version": str(policy_version).strip(),
        "universe_policy_version": str(universe_policy_version).strip(),
        "instruments": instruments,
    }
    if not body["policy_version"] or not body["universe_policy_version"]:
        raise ValueError("policy versions must be non-empty.")
    body["dataset_id"] = "sha256:" + hashlib.sha256(_canonical(body)).hexdigest()
    return body


__all__ = ["capture_horizon_dataset"]
