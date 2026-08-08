"""SQLite schema migrations for the paper portfolio."""

from __future__ import annotations

from pathlib import Path
import sqlite3

from .database import (
    DEFAULT_DATABASE_PATH,
    connect_database,
)


SCHEMA_VERSION = 8


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


_SCHEMA_V2 = """
BEGIN IMMEDIATE;

ALTER TABLE paper_scans
    ADD COLUMN scan_key TEXT NOT NULL DEFAULT '';

ALTER TABLE paper_scans
    ADD COLUMN requested_count INTEGER NOT NULL DEFAULT 0;

ALTER TABLE paper_scans
    ADD COLUMN configuration_json TEXT NOT NULL DEFAULT '{}';

ALTER TABLE paper_scans
    ADD COLUMN app_version TEXT NOT NULL DEFAULT 'unknown';

CREATE UNIQUE INDEX IF NOT EXISTS
    idx_scans_account_key
    ON paper_scans(account_id, scan_key)
    WHERE scan_key <> '';

CREATE TABLE IF NOT EXISTS paper_scan_results (
    result_id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    status TEXT NOT NULL,
    processed_at TEXT NOT NULL,

    data_as_of TEXT,
    history_rows INTEGER NOT NULL,

    latest_price REAL,
    average_volume REAL,
    average_dollar_volume REAL,

    recommendation TEXT,
    strategy TEXT,
    score REAL,
    confidence REAL,
    market_regime TEXT,
    reward_to_risk REAL,

    release_eligible INTEGER NOT NULL,

    rank_score REAL,
    rank_position INTEGER,

    signal_id TEXT,

    reasons_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,

    UNIQUE(scan_id, symbol),

    FOREIGN KEY(scan_id)
        REFERENCES paper_scans(scan_id),

    FOREIGN KEY(account_id)
        REFERENCES paper_accounts(account_id),

    FOREIGN KEY(signal_id)
        REFERENCES paper_signals(signal_id)
);

CREATE INDEX IF NOT EXISTS
    idx_scan_results_status
    ON paper_scan_results(
        scan_id,
        status
    );

CREATE INDEX IF NOT EXISTS
    idx_scan_results_rank
    ON paper_scan_results(
        scan_id,
        rank_position
    );

PRAGMA user_version = 2;

COMMIT;
"""


_SCHEMA_V3 = """
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS paper_execution_runs (
    run_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    run_key TEXT NOT NULL,
    scan_id TEXT,

    status TEXT NOT NULL,

    started_at TEXT NOT NULL,
    completed_at TEXT,

    created_orders INTEGER NOT NULL DEFAULT 0,
    filled_orders INTEGER NOT NULL DEFAULT 0,
    expired_orders INTEGER NOT NULL DEFAULT 0,
    cancelled_orders INTEGER NOT NULL DEFAULT 0,
    closed_positions INTEGER NOT NULL DEFAULT 0,
    rejected_entries INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,

    entry_block_reasons_json TEXT NOT NULL DEFAULT '[]',
    configuration_json TEXT NOT NULL,
    app_version TEXT NOT NULL,
    error_message TEXT,

    UNIQUE(account_id, run_key),

    FOREIGN KEY(account_id)
        REFERENCES paper_accounts(account_id),

    FOREIGN KEY(scan_id)
        REFERENCES paper_scans(scan_id)
);

CREATE TABLE IF NOT EXISTS paper_account_controls (
    account_id TEXT PRIMARY KEY,

    kill_switch_active INTEGER NOT NULL DEFAULT 0,
    kill_switch_reason TEXT,

    maximum_daily_loss_fraction TEXT NOT NULL DEFAULT '0.03',
    maximum_drawdown_fraction TEXT NOT NULL DEFAULT '0.10',
    maximum_new_orders_per_run INTEGER NOT NULL DEFAULT 3,
    maximum_stale_market_days INTEGER NOT NULL DEFAULT 7,

    updated_at TEXT NOT NULL,

    FOREIGN KEY(account_id)
        REFERENCES paper_accounts(account_id)
);

CREATE TABLE IF NOT EXISTS paper_exit_requests (
    request_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    position_id TEXT NOT NULL,

    reason TEXT NOT NULL,
    triggered_at TEXT NOT NULL,

    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    executed_at TEXT,
    error_message TEXT,

    UNIQUE(
        position_id,
        reason,
        triggered_at
    ),

    FOREIGN KEY(account_id)
        REFERENCES paper_accounts(account_id),

    FOREIGN KEY(position_id)
        REFERENCES paper_positions(position_id)
);

CREATE TABLE IF NOT EXISTS paper_equity_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    account_id TEXT NOT NULL,

    captured_at TEXT NOT NULL,

    cash_balance TEXT NOT NULL,
    reserved_cash TEXT NOT NULL,
    market_value TEXT NOT NULL,
    equity TEXT NOT NULL,

    UNIQUE(run_id, account_id),

    FOREIGN KEY(run_id)
        REFERENCES paper_execution_runs(run_id),

    FOREIGN KEY(account_id)
        REFERENCES paper_accounts(account_id)
);

CREATE INDEX IF NOT EXISTS
    idx_execution_runs_account_time
    ON paper_execution_runs(
        account_id,
        started_at
    );

CREATE INDEX IF NOT EXISTS
    idx_exit_requests_status
    ON paper_exit_requests(
        account_id,
        status,
        triggered_at
    );

CREATE INDEX IF NOT EXISTS
    idx_equity_snapshots_account_time
    ON paper_equity_snapshots(
        account_id,
        captured_at
    );

PRAGMA user_version = 3;

COMMIT;
"""


_SCHEMA_V4 = """
BEGIN IMMEDIATE;

ALTER TABLE paper_notifications
    ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0;

ALTER TABLE paper_notifications
    ADD COLUMN last_attempt_at TEXT;

ALTER TABLE paper_notifications
    ADD COLUMN provider_message_id TEXT;

ALTER TABLE paper_notifications
    ADD COLUMN delivery_metadata_json TEXT NOT NULL DEFAULT '{}';

CREATE TABLE IF NOT EXISTS paper_job_runs (
    job_run_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,

    job_key TEXT NOT NULL,
    job_type TEXT NOT NULL,

    scheduled_for TEXT NOT NULL,
    exchange_code TEXT NOT NULL,

    status TEXT NOT NULL,

    started_at TEXT NOT NULL,
    completed_at TEXT,

    scan_id TEXT,
    execution_run_id TEXT,

    queued_notifications INTEGER NOT NULL DEFAULT 0,
    sent_notifications INTEGER NOT NULL DEFAULT 0,
    failed_notifications INTEGER NOT NULL DEFAULT 0,

    metadata_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,

    UNIQUE(account_id, job_key),

    FOREIGN KEY(account_id)
        REFERENCES paper_accounts(account_id),

    FOREIGN KEY(scan_id)
        REFERENCES paper_scans(scan_id),

    FOREIGN KEY(execution_run_id)
        REFERENCES paper_execution_runs(run_id)
);

CREATE INDEX IF NOT EXISTS
    idx_notifications_delivery
    ON paper_notifications(
        account_id,
        status,
        channel,
        created_at
    );

CREATE INDEX IF NOT EXISTS
    idx_job_runs_account_time
    ON paper_job_runs(
        account_id,
        scheduled_for
    );

PRAGMA user_version = 4;

COMMIT;
"""


_SCHEMA_V5 = """
CREATE TABLE IF NOT EXISTS
paper_broker_reconciliation_runs (
    reconciliation_run_id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    reconciliation_key TEXT NOT NULL,
    provider TEXT NOT NULL,
    broker_account_id TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (
            status IN (
                'RUNNING',
                'MATCHED',
                'DIFFERENCES',
                'FAILED'
            )
        ),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    account_item_count INTEGER
        NOT NULL DEFAULT 0,
    order_item_count INTEGER
        NOT NULL DEFAULT 0,
    position_item_count INTEGER
        NOT NULL DEFAULT 0,
    matched_item_count INTEGER
        NOT NULL DEFAULT 0,
    mismatched_item_count INTEGER
        NOT NULL DEFAULT 0,
    missing_internal_item_count INTEGER
        NOT NULL DEFAULT 0,
    missing_broker_item_count INTEGER
        NOT NULL DEFAULT 0,
    metadata_json TEXT
        NOT NULL DEFAULT '{}',
    error_message TEXT,
    FOREIGN KEY (account_id)
        REFERENCES paper_accounts(account_id),
    UNIQUE (
        account_id,
        provider,
        reconciliation_key
    )
);

CREATE TABLE IF NOT EXISTS
paper_broker_reconciliation_items (
    reconciliation_item_id TEXT PRIMARY KEY,
    reconciliation_run_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    category TEXT NOT NULL
        CHECK (
            category IN (
                'ACCOUNT',
                'ORDER',
                'POSITION'
            )
        ),
    comparison_key TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (
            status IN (
                'MATCH',
                'MISMATCH',
                'MISSING_INTERNAL',
                'MISSING_BROKER'
            )
        ),
    internal_reference_ids_json TEXT
        NOT NULL DEFAULT '[]',
    broker_reference_ids_json TEXT
        NOT NULL DEFAULT '[]',
    differences_json TEXT
        NOT NULL DEFAULT '{}',
    message TEXT NOT NULL,
    metadata_json TEXT
        NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (reconciliation_run_id)
        REFERENCES
        paper_broker_reconciliation_runs(
            reconciliation_run_id
        )
        ON DELETE CASCADE,
    FOREIGN KEY (account_id)
        REFERENCES paper_accounts(account_id),
    UNIQUE (
        reconciliation_run_id,
        category,
        comparison_key
    )
);

CREATE INDEX IF NOT EXISTS
idx_broker_reconciliation_runs_account
ON paper_broker_reconciliation_runs(
    account_id,
    started_at,
    reconciliation_run_id
);

CREATE INDEX IF NOT EXISTS
idx_broker_reconciliation_runs_status
ON paper_broker_reconciliation_runs(
    account_id,
    status,
    started_at
);

CREATE INDEX IF NOT EXISTS
idx_broker_reconciliation_items_run
ON paper_broker_reconciliation_items(
    reconciliation_run_id,
    category,
    status
);

CREATE INDEX IF NOT EXISTS
idx_broker_reconciliation_items_unresolved
ON paper_broker_reconciliation_items(
    account_id,
    status,
    created_at
)
WHERE status <> 'MATCH';

PRAGMA user_version = 5;
"""


_SCHEMA_V6 = """
ALTER TABLE paper_account_controls
    ADD COLUMN sizing_mode TEXT;

ALTER TABLE paper_account_controls
    ADD COLUMN portfolio_currency TEXT;

ALTER TABLE paper_account_controls
    ADD COLUMN target_order_value TEXT;

ALTER TABLE paper_account_controls
    ADD COLUMN maximum_order_value TEXT;

ALTER TABLE paper_account_controls
    ADD COLUMN maximum_planned_loss TEXT;

ALTER TABLE paper_account_controls
    ADD COLUMN maximum_open_positions INTEGER;

ALTER TABLE paper_account_controls
    ADD COLUMN maximum_invested_exposure TEXT;

PRAGMA user_version = 6;
"""


_SCHEMA_V7 = """
ALTER TABLE paper_signals
    ADD COLUMN quote_currency TEXT;

PRAGMA user_version = 7;
"""


_SCHEMA_V8 = """
ALTER TABLE paper_orders
    ADD COLUMN quote_currency TEXT;

ALTER TABLE paper_orders
    ADD COLUMN portfolio_currency TEXT;

ALTER TABLE paper_orders
    ADD COLUMN reservation_fx_rate TEXT;

ALTER TABLE paper_orders
    ADD COLUMN reservation_fx_as_of TEXT;

ALTER TABLE paper_orders
    ADD COLUMN reservation_fx_source TEXT;


ALTER TABLE paper_fills
    ADD COLUMN quote_currency TEXT;

ALTER TABLE paper_fills
    ADD COLUMN portfolio_currency TEXT;

ALTER TABLE paper_fills
    ADD COLUMN entry_fx_rate TEXT;

ALTER TABLE paper_fills
    ADD COLUMN entry_fx_as_of TEXT;

ALTER TABLE paper_fills
    ADD COLUMN entry_fx_source TEXT;

ALTER TABLE paper_fills
    ADD COLUMN cash_required_portfolio TEXT;


ALTER TABLE paper_positions
    ADD COLUMN quote_currency TEXT;

ALTER TABLE paper_positions
    ADD COLUMN portfolio_currency TEXT;

ALTER TABLE paper_positions
    ADD COLUMN entry_fx_rate TEXT;

ALTER TABLE paper_positions
    ADD COLUMN entry_fx_as_of TEXT;

ALTER TABLE paper_positions
    ADD COLUMN entry_fx_source TEXT;

ALTER TABLE paper_positions
    ADD COLUMN entry_cash_portfolio TEXT;


ALTER TABLE paper_closed_trades
    ADD COLUMN quote_currency TEXT;

ALTER TABLE paper_closed_trades
    ADD COLUMN portfolio_currency TEXT;

ALTER TABLE paper_closed_trades
    ADD COLUMN entry_fx_rate TEXT;

ALTER TABLE paper_closed_trades
    ADD COLUMN entry_fx_as_of TEXT;

ALTER TABLE paper_closed_trades
    ADD COLUMN entry_fx_source TEXT;

ALTER TABLE paper_closed_trades
    ADD COLUMN exit_fx_rate TEXT;

ALTER TABLE paper_closed_trades
    ADD COLUMN exit_fx_as_of TEXT;

ALTER TABLE paper_closed_trades
    ADD COLUMN exit_fx_source TEXT;

PRAGMA user_version = 8;
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
                strftime(
                    '%Y-%m-%dT%H:%M:%fZ',
                    'now'
                )
            )
            """
        )

        current_version = 1

    if current_version == 1:
        connection.executescript(_SCHEMA_V2)

        connection.execute(
            """
            INSERT OR REPLACE INTO schema_migrations(
                version,
                applied_at
            )
            VALUES (
                2,
                strftime(
                    '%Y-%m-%dT%H:%M:%fZ',
                    'now'
                )
            )
            """
        )

        current_version = 2

    if current_version == 2:
        connection.executescript(_SCHEMA_V3)

        connection.execute(
            """
            INSERT OR REPLACE INTO schema_migrations(
                version,
                applied_at
            )
            VALUES (
                3,
                strftime(
                    '%Y-%m-%dT%H:%M:%fZ',
                    'now'
                )
            )
            """
        )

        current_version = 3

    if current_version == 3:
        connection.executescript(_SCHEMA_V4)

        connection.execute(
            """
            INSERT OR REPLACE INTO schema_migrations(
                version,
                applied_at
            )
            VALUES (
                4,
                strftime(
                    '%Y-%m-%dT%H:%M:%fZ',
                    'now'
                )
            )
            """
        )

        current_version = 4

    if current_version < 5:
        connection.executescript(
            _SCHEMA_V5
        )
        current_version = 5

    if current_version < 6:
        connection.executescript(
            _SCHEMA_V6
        )
        current_version = 6

    if current_version < 7:
        connection.executescript(
            _SCHEMA_V7
        )
        current_version = 7

    if current_version < 8:
        connection.executescript(
            _SCHEMA_V8
        )
        current_version = 8

    if current_version != SCHEMA_VERSION:
        raise RuntimeError(
            "Database migration did not reach "
            f"schema version {SCHEMA_VERSION}; "
            f"stopped at {current_version}."
        )


def initialize_database(
    path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    connection = connect_database(path)

    try:
        apply_migrations(connection)
    finally:
        connection.close()
