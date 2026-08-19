"""P4.1 lifecycle FX provenance persistence tests."""

from __future__ import annotations

import sqlite3
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

from src.paper import PaperRepository
from src.paper import migrations
from src.paper.migrations import (
    initialize_database,
)


T0 = datetime(
    2026,
    8,
    1,
    20,
    0,
    tzinfo=timezone.utc,
)

T1 = T0 + timedelta(days=1)
T3 = T0 + timedelta(days=3)
T5 = T0 + timedelta(days=5)


EXPECTED_COLUMNS = {
    "paper_orders": {
        "quote_currency",
        "portfolio_currency",
        "reservation_fx_rate",
        "reservation_fx_as_of",
        "reservation_fx_source",
    },
    "paper_fills": {
        "quote_currency",
        "portfolio_currency",
        "entry_fx_rate",
        "entry_fx_as_of",
        "entry_fx_source",
        "cash_required_portfolio",
    },
    "paper_positions": {
        "quote_currency",
        "portfolio_currency",
        "entry_fx_rate",
        "entry_fx_as_of",
        "entry_fx_source",
        "entry_cash_portfolio",
    },
    "paper_closed_trades": {
        "quote_currency",
        "portfolio_currency",
        "entry_fx_rate",
        "entry_fx_as_of",
        "entry_fx_source",
        "exit_fx_rate",
        "exit_fx_as_of",
        "exit_fx_source",
    },
}


def table_columns(
    connection: sqlite3.Connection,
    table: str,
) -> set[str]:
    return {
        row[1]
        for row in connection.execute(
            f"PRAGMA table_info({table})"
        )
    }


def make_v7_database(
    database,
) -> None:
    connection = sqlite3.connect(database)

    try:
        for script in (
            migrations._SCHEMA_V1,
            migrations._SCHEMA_V2,
            migrations._SCHEMA_V3,
            migrations._SCHEMA_V4,
            migrations._SCHEMA_V5,
            migrations._SCHEMA_V6,
            migrations._SCHEMA_V7,
        ):
            connection.executescript(script)
    finally:
        connection.close()


def insert_legacy_rows(
    database,
) -> None:
    connection = sqlite3.connect(database)

    try:
        connection.execute(
            """
            INSERT INTO paper_orders(
                order_id,
                account_id,
                signal_id,
                idempotency_key,
                symbol,
                side,
                quantity,
                entry_low,
                entry_high,
                stop_price,
                targets_json,
                estimated_cash_required,
                reserved_cash,
                status,
                created_at,
                expires_at,
                filled_at,
                closed_at
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                "ORD-LEGACY",
                "ACC-LEGACY",
                "SIG-LEGACY",
                "legacy-order",
                "AAPL",
                "LONG",
                "10.00000000",
                "99.00000000",
                "101.00000000",
                "95.00000000",
                '["110.00000000"]',
                "1011.00000000",
                "0.00000000",
                "CLOSED",
                T0.isoformat(),
                T5.isoformat(),
                T1.isoformat(),
                T3.isoformat(),
            ),
        )

        connection.execute(
            """
            INSERT INTO paper_fills(
                fill_id,
                order_id,
                price,
                quantity,
                fees,
                slippage,
                filled_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "FILL-LEGACY",
                "ORD-LEGACY",
                "100.00000000",
                "10.00000000",
                "1.00000000",
                "0.50000000",
                T1.isoformat(),
            ),
        )

        connection.execute(
            """
            INSERT INTO paper_positions(
                position_id,
                account_id,
                order_id,
                fill_id,
                symbol,
                side,
                quantity,
                entry_price,
                stop_price,
                targets_json,
                opened_at,
                expires_at,
                status,
                closed_at
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?
            )
            """,
            (
                "POS-LEGACY",
                "ACC-LEGACY",
                "ORD-LEGACY",
                "FILL-LEGACY",
                "AAPL",
                "LONG",
                "10.00000000",
                "100.00000000",
                "95.00000000",
                '["110.00000000"]',
                T1.isoformat(),
                T5.isoformat(),
                "CLOSED",
                T3.isoformat(),
            ),
        )

        connection.execute(
            """
            INSERT INTO paper_closed_trades(
                trade_id,
                position_id,
                account_id,
                order_id,
                fill_id,
                signal_id,
                symbol,
                strategy,
                market_regime,
                entry_time,
                entry_price,
                exit_time,
                exit_price,
                exit_reason,
                quantity,
                gross_pnl,
                fees,
                slippage,
                net_pnl,
                return_pct,
                holding_seconds
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                "TRD-LEGACY",
                "POS-LEGACY",
                "ACC-LEGACY",
                "ORD-LEGACY",
                "FILL-LEGACY",
                "SIG-LEGACY",
                "AAPL",
                "trend_pullback",
                "BULLISH",
                T1.isoformat(),
                "100.00000000",
                T3.isoformat(),
                "110.00000000",
                "TARGET",
                "10.00000000",
                "100.00000000",
                "2.00000000",
                "1.00000000",
                "98.00000000",
                0.098,
                int(
                    (
                        T3 - T1
                    ).total_seconds()
                ),
            ),
        )

        connection.commit()
    finally:
        connection.close()


def test_latest_schema_has_lifecycle_fx_columns(
    tmp_path,
) -> None:
    database = tmp_path / "latest-v8.db"

    initialize_database(database)

    connection = sqlite3.connect(database)

    try:
        version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        actual = {
            table: table_columns(
                connection,
                table,
            )
            for table in EXPECTED_COLUMNS
        }
    finally:
        connection.close()

    assert version == 15

    for table, expected in (
        EXPECTED_COLUMNS.items()
    ):
        assert expected.issubset(
            actual[table]
        )


def test_genuine_v7_rows_upgrade_with_null_fx(
    tmp_path,
) -> None:
    database = tmp_path / "legacy-v7.db"

    make_v7_database(database)
    insert_legacy_rows(database)

    connection = sqlite3.connect(database)

    try:
        before = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        for table, expected in (
            EXPECTED_COLUMNS.items()
        ):
            assert expected.isdisjoint(
                table_columns(
                    connection,
                    table,
                )
            )
    finally:
        connection.close()

    assert before == 7

    initialize_database(database)

    repository = PaperRepository(database)

    order = repository.get_order(
        "ORD-LEGACY"
    )

    fill = repository.get_fill_for_order(
        "ORD-LEGACY"
    )

    position = (
        repository.get_position_by_order(
            "ORD-LEGACY"
        )
    )

    trade = (
        repository
        .get_closed_trade_by_position(
            "POS-LEGACY"
        )
    )

    connection = sqlite3.connect(database)

    try:
        after = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]
    finally:
        connection.close()

    assert after == 15

    assert order.quote_currency is None
    assert order.portfolio_currency is None
    assert order.reservation_fx_rate is None
    assert order.reservation_fx_as_of is None
    assert order.reservation_fx_source is None

    assert fill.quote_currency is None
    assert fill.portfolio_currency is None
    assert fill.entry_fx_rate is None
    assert fill.entry_fx_as_of is None
    assert fill.entry_fx_source is None
    assert fill.cash_required_portfolio is None

    assert position.quote_currency is None
    assert position.portfolio_currency is None
    assert position.entry_fx_rate is None
    assert position.entry_fx_as_of is None
    assert position.entry_fx_source is None
    assert position.entry_cash_portfolio is None

    assert trade.quote_currency is None
    assert trade.portfolio_currency is None
    assert trade.entry_fx_rate is None
    assert trade.entry_fx_as_of is None
    assert trade.entry_fx_source is None
    assert trade.exit_fx_rate is None
    assert trade.exit_fx_as_of is None
    assert trade.exit_fx_source is None


def test_fx_provenance_round_trips_from_v8_rows(
    tmp_path,
) -> None:
    database = tmp_path / "roundtrip-v8.db"

    make_v7_database(database)
    insert_legacy_rows(database)
    initialize_database(database)

    connection = sqlite3.connect(database)

    try:
        connection.execute(
            """
            UPDATE paper_orders
            SET quote_currency = 'USD',
                portfolio_currency = 'EUR',
                reservation_fx_rate =
                    '0.90000000',
                reservation_fx_as_of = ?,
                reservation_fx_source =
                    'TEST_RESERVATION'
            WHERE order_id = 'ORD-LEGACY'
            """,
            (T0.isoformat(),),
        )

        connection.execute(
            """
            UPDATE paper_fills
            SET quote_currency = 'USD',
                portfolio_currency = 'EUR',
                entry_fx_rate = '0.91000000',
                entry_fx_as_of = ?,
                entry_fx_source = 'TEST_ENTRY',
                cash_required_portfolio =
                    '910.91000000'
            WHERE fill_id = 'FILL-LEGACY'
            """,
            (T1.isoformat(),),
        )

        connection.execute(
            """
            UPDATE paper_positions
            SET quote_currency = 'USD',
                portfolio_currency = 'EUR',
                entry_fx_rate = '0.91000000',
                entry_fx_as_of = ?,
                entry_fx_source = 'TEST_ENTRY',
                entry_cash_portfolio =
                    '910.91000000'
            WHERE position_id = 'POS-LEGACY'
            """,
            (T1.isoformat(),),
        )

        connection.execute(
            """
            UPDATE paper_closed_trades
            SET quote_currency = 'USD',
                portfolio_currency = 'EUR',
                entry_fx_rate = '0.91000000',
                entry_fx_as_of = ?,
                entry_fx_source = 'TEST_ENTRY',
                exit_fx_rate = '0.93000000',
                exit_fx_as_of = ?,
                exit_fx_source = 'TEST_EXIT'
            WHERE trade_id = 'TRD-LEGACY'
            """,
            (
                T1.isoformat(),
                T3.isoformat(),
            ),
        )

        connection.commit()
    finally:
        connection.close()

    repository = PaperRepository(database)

    order = repository.get_order(
        "ORD-LEGACY"
    )

    fill = repository.get_fill_for_order(
        "ORD-LEGACY"
    )

    position = (
        repository.get_position_by_order(
            "ORD-LEGACY"
        )
    )

    trade = (
        repository
        .get_closed_trade_by_position(
            "POS-LEGACY"
        )
    )

    assert order.quote_currency == "USD"
    assert order.portfolio_currency == "EUR"
    assert (
        order.reservation_fx_rate
        == Decimal("0.90000000")
    )
    assert order.reservation_fx_as_of == T0
    assert (
        order.reservation_fx_source
        == "TEST_RESERVATION"
    )

    assert fill.quote_currency == "USD"
    assert fill.portfolio_currency == "EUR"
    assert (
        fill.entry_fx_rate
        == Decimal("0.91000000")
    )
    assert fill.entry_fx_as_of == T1
    assert fill.entry_fx_source == "TEST_ENTRY"
    assert (
        fill.cash_required_portfolio
        == Decimal("910.91000000")
    )

    assert position.quote_currency == "USD"
    assert position.portfolio_currency == "EUR"
    assert (
        position.entry_fx_rate
        == Decimal("0.91000000")
    )
    assert position.entry_fx_as_of == T1
    assert (
        position.entry_fx_source
        == "TEST_ENTRY"
    )
    assert (
        position.entry_cash_portfolio
        == Decimal("910.91000000")
    )

    assert trade.quote_currency == "USD"
    assert trade.portfolio_currency == "EUR"
    assert (
        trade.entry_fx_rate
        == Decimal("0.91000000")
    )
    assert trade.entry_fx_as_of == T1
    assert trade.entry_fx_source == "TEST_ENTRY"
    assert (
        trade.exit_fx_rate
        == Decimal("0.93000000")
    )
    assert trade.exit_fx_as_of == T3
    assert trade.exit_fx_source == "TEST_EXIT"
