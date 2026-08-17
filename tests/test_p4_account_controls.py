from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import sqlite3

import pytest

from src.automation import (
    AutomationRepository,
)
from src.paper import (
    FixedNotionalSizingPolicy,
    PaperRepository,
    PositionSizingMode,
    initialize_database,
)
import src.paper.migrations as migrations


T0 = datetime(
    2026,
    8,
    8,
    10,
    0,
    tzinfo=timezone.utc,
)


EXPECTED_COLUMNS = {
    "sizing_mode",
    "portfolio_currency",
    "target_order_value",
    "maximum_order_value",
    "maximum_planned_loss",
    "maximum_open_positions",
    "maximum_invested_exposure",
}


def make_version_five_database(
    database_path,
) -> None:
    connection = sqlite3.connect(
        database_path
    )

    try:
        for script in (
            migrations._SCHEMA_V1,
            migrations._SCHEMA_V2,
            migrations._SCHEMA_V3,
            migrations._SCHEMA_V4,
            migrations._SCHEMA_V5,
        ):
            connection.executescript(script)
    finally:
        connection.close()


def test_schema_version_six_adds_sizing_controls(
    tmp_path,
) -> None:
    database_path = (
        tmp_path / "schema-v6.db"
    )

    initialize_database(
        database_path
    )

    connection = sqlite3.connect(
        database_path
    )

    try:
        version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        columns = {
            row[1]
            for row in connection.execute(
                """
                PRAGMA table_info(
                    paper_account_controls
                )
                """
            )
        }
    finally:
        connection.close()

    assert version == 10

    assert EXPECTED_COLUMNS.issubset(
        columns
    )


def test_version_five_control_upgrades_as_legacy(
    tmp_path,
) -> None:
    database_path = (
        tmp_path / "legacy-v5.db"
    )

    make_version_five_database(
        database_path
    )

    connection = sqlite3.connect(
        database_path
    )

    try:
        connection.execute(
            """
            INSERT INTO paper_accounts(
                account_id,
                name,
                base_currency,
                starting_balance,
                cash_balance,
                reserved_cash,
                status,
                created_at,
                updated_at
            )
            VALUES (
                'ACC-LEGACY',
                'Legacy P3',
                'USD',
                '10000.00000000',
                '10000.00000000',
                '0E-8',
                'ACTIVE',
                ?,
                ?
            )
            """,
            (
                T0.isoformat(),
                T0.isoformat(),
            ),
        )

        connection.execute(
            """
            INSERT INTO paper_account_controls(
                account_id,
                kill_switch_active,
                maximum_daily_loss_fraction,
                maximum_drawdown_fraction,
                maximum_new_orders_per_run,
                maximum_stale_market_days,
                updated_at
            )
            VALUES (
                'ACC-LEGACY',
                0,
                '0.03',
                '0.10',
                3,
                7,
                ?
            )
            """,
            (T0.isoformat(),),
        )

        connection.commit()
    finally:
        connection.close()

    initialize_database(
        database_path
    )

    automation = AutomationRepository(
        database_path
    )

    control = automation.get_control(
        "ACC-LEGACY",
        at=T0,
    )

    assert control.sizing_mode is None
    assert control.portfolio_currency is None
    assert control.target_order_value is None
    assert control.maximum_order_value is None
    assert control.maximum_planned_loss is None
    assert control.maximum_open_positions is None
    assert (
        control.maximum_invested_exposure
        is None
    )

    assert (
        control.maximum_daily_loss_fraction
        == Decimal("0.03000000")
    )

    assert (
        control.maximum_drawdown_fraction
        == Decimal("0.10000000")
    )

    connection = sqlite3.connect(
        database_path
    )

    try:
        version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]
    finally:
        connection.close()

    assert version == 10


def test_fixed_notional_control_round_trip(
    tmp_path,
) -> None:
    database_path = (
        tmp_path / "fixed-control.db"
    )

    paper = PaperRepository(
        database_path
    )

    account = paper.create_account(
        account_id="ACC-P4-EUR",
        name="P4 EUR Portfolio",
        base_currency="EUR",
        starting_balance="2000",
        created_at=T0,
    )

    automation = AutomationRepository(
        database_path
    )

    legacy = automation.get_control(
        account.account_id,
        at=T0,
    )

    assert legacy.sizing_mode is None

    policy = FixedNotionalSizingPolicy()

    control = (
        automation
        .set_fixed_notional_sizing(
            account.account_id,
            policy=policy,
            updated_at=T0,
        )
    )

    assert control.sizing_mode is (
        PositionSizingMode
        .FIXED_NOTIONAL_WITH_RISK_CAP
    )

    assert control.portfolio_currency == "EUR"

    assert (
        control.target_order_value
        == Decimal("100.00000000")
    )

    assert (
        control.maximum_order_value
        == Decimal("100.00000000")
    )

    assert (
        control.maximum_planned_loss
        == Decimal("10.00000000")
    )

    assert control.maximum_open_positions == 5

    assert (
        control.maximum_invested_exposure
        == Decimal("500.00000000")
    )


def test_fixed_notional_control_requires_account_currency(
    tmp_path,
) -> None:
    database_path = (
        tmp_path / "currency-mismatch.db"
    )

    paper = PaperRepository(
        database_path
    )

    account = paper.create_account(
        account_id="ACC-P3-USD",
        name="Historical USD Portfolio",
        base_currency="USD",
        starting_balance="10000",
        created_at=T0,
    )

    automation = AutomationRepository(
        database_path
    )

    with pytest.raises(
        ValueError,
        match="must match account base currency",
    ):
        automation.set_fixed_notional_sizing(
            account.account_id,
            policy=FixedNotionalSizingPolicy(),
            updated_at=T0,
        )

    control = automation.get_control(
        account.account_id,
        at=T0,
    )

    assert control.sizing_mode is None
