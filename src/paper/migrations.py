"""SQLite schema migrations for the paper portfolio."""

from __future__ import annotations

from pathlib import Path
import sqlite3

from .database import (
    DEFAULT_DATABASE_PATH,
    connect_database,
)


SCHEMA_VERSION = 1


_SCHEMA_V1 = """
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_accounts (
    account_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    base_currency TEXT NOT NULL,
    starting_balance TEXT NOT NULL,
    cash_balance TEXT NOT NULL,
    reserved_cash TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_scans (
    scan_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    universe TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    processed_count INTEGER NOT NULL DEFAULT 0,
    rejected_count INTEGER NOT NULL DEFAULT 0,
    signal_count INTEGER NOT NULL DEFAULT 0,
    order_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    FOREIGN KEY(account_id)
        REFERENCES paper_accounts(account_id)
);

CREATE TABLE IF NOT EXISTS paper_signals (
    signal_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    scan_id TEXT,
    symbol TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    strategy TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    market_regime TEXT NOT NULL,
    score REAL NOT NULL,
    confidence REAL NOT NULL,
    reward_to_risk REAL NOT NULL,
    entry_low TEXT NOT NULL,
    entry_high TEXT NOT NULL,
    stop_price TEXT NOT NULL,
    targets_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    conflicts_json TEXT NOT NULL,
    threshold_version TEXT NOT NULL,
    app_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(account_id)
        REFERENCES paper_accounts(account_id),
    FOREIGN KEY(scan_id)
        REFERENCES paper_scans(scan_id)
);

CREATE TABLE IF NOT EXISTS paper_orders (
    order_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity TEXT NOT NULL,
    entry_low TEXT NOT NULL,
    entry_high TEXT NOT NULL,
    stop_price TEXT NOT NULL,
    targets_json TEXT NOT NULL,
    estimated_cash_required TEXT NOT NULL,
    reserved_cash TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    filled_at TEXT,
    closed_at TEXT,
    UNIQUE(account_id, idempotency_key),
    FOREIGN KEY(account_id)
        REFERENCES paper_accounts(account_id),
    FOREIGN KEY(signal_id)
        REFERENCES paper_signals(signal_id)
);

CREATE TABLE IF NOT EXISTS paper_fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL UNIQUE,
    price TEXT NOT NULL,
    quantity TEXT NOT NULL,
    fees TEXT NOT NULL,
    slippage TEXT NOT NULL,
    filled_at TEXT NOT NULL,
    FOREIGN KEY(order_id)
        REFERENCES paper_orders(order_id)
);

CREATE TABLE IF NOT EXISTS paper_positions (
    position_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    order_id TEXT NOT NULL UNIQUE,
    fill_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity TEXT NOT NULL,
    entry_price TEXT NOT NULL,
    stop_price TEXT NOT NULL,
    targets_json TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    status TEXT NOT NULL,
    closed_at TEXT,
    FOREIGN KEY(account_id)
        REFERENCES paper_accounts(account_id),
    FOREIGN KEY(order_id)
        REFERENCES paper_orders(order_id),
    FOREIGN KEY(fill_id)
        REFERENCES paper_fills(fill_id)
);

CREATE TABLE IF NOT EXISTS paper_closed_trades (
    trade_id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL UNIQUE,
    account_id TEXT NOT NULL,
    order_id TEXT NOT NULL,
    fill_id TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    market_regime TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price TEXT NOT NULL,
    exit_time TEXT NOT NULL,
    exit_price TEXT NOT NULL,
    exit_reason TEXT NOT NULL,
    quantity TEXT NOT NULL,
    gross_pnl TEXT NOT NULL,
    fees TEXT NOT NULL,
    slippage TEXT NOT NULL,
    net_pnl TEXT NOT NULL,
    return_pct REAL NOT NULL,
    holding_seconds INTEGER NOT NULL,
    FOREIGN KEY(position_id)
        REFERENCES paper_positions(position_id),
    FOREIGN KEY(account_id)
        REFERENCES paper_accounts(account_id),
    FOREIGN KEY(order_id)
        REFERENCES paper_orders(order_id),
    FOREIGN KEY(fill_id)
        REFERENCES paper_fills(fill_id),
    FOREIGN KEY(signal_id)
        REFERENCES paper_signals(signal_id)
);

CREATE TABLE IF NOT EXISTS paper_ledger_entries (
    ledger_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    amount TEXT NOT NULL,
    balance_after TEXT NOT NULL,
    reference_type TEXT NOT NULL,
    reference_id TEXT NOT NULL,
    description TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    FOREIGN KEY(account_id)
        REFERENCES paper_accounts(account_id)
);

CREATE TABLE IF NOT EXISTS paper_notifications (
    notification_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    reference_type TEXT NOT NULL,
    reference_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    sent_at TEXT,
    error_message TEXT,
    UNIQUE(
        account_id,
        event_type,
        reference_type,
        reference_id,
        channel
    ),
    FOREIGN KEY(account_id)
        REFERENCES paper_accounts(account_id)
);

CREATE TABLE IF NOT EXISTS paper_system_events (
    event_id TEXT PRIMARY KEY,
    account_id TEXT,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    reference_type TEXT,
    reference_id TEXT,
    message TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(account_id)
        REFERENCES paper_accounts(account_id)
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_time
    ON paper_signals(symbol, generated_at);

CREATE INDEX IF NOT EXISTS idx_orders_account_status
    ON paper_orders(account_id, status);

CREATE INDEX IF NOT EXISTS idx_positions_account_status
    ON paper_positions(account_id, status);

CREATE INDEX IF NOT EXISTS idx_trades_account_exit
    ON paper_closed_trades(account_id, exit_time);

CREATE INDEX IF NOT EXISTS idx_events_account_time
    ON paper_system_events(account_id, created_at);

CREATE INDEX IF NOT EXISTS idx_notifications_status
    ON paper_notifications(status, created_at);

PRAGMA user_version = 1;

COMMIT;
"""


def apply_migrations(
    connection: sqlite3.Connection,
) -> None:
    current_version = int(
        connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]
    )

    if current_version > SCHEMA_VERSION:
        raise RuntimeError(
            "Database schema is newer than this application."
        )

    if current_version == 0:
        connection.executescript(_SCHEMA_V1)
        connection.execute(
            """
            INSERT OR REPLACE INTO schema_migrations(
                version,
                applied_at
            )
            VALUES (
                1,
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            )
            """
        )


def initialize_database(
    path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    connection = connect_database(path)

    try:
        apply_migrations(connection)
    finally:
        connection.close()
