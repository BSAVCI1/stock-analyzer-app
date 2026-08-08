from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from decimal import Decimal
from enum import Enum
import sqlite3
import src.paper.migrations as migrations
from types import SimpleNamespace

import pytest

from src.execution_adapters import (
    BrokerAccountSnapshot,
    BrokerOrderSide,
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerPaperConnectionConfig,
    BrokerPositionSide,
    BrokerPositionSnapshot,
    BrokerReconciliationCategory,
    BrokerReconciliationItemStatus,
    BrokerReconciliationRepository,
    BrokerReconciliationRunStatus,
    BrokerReconciliationService,
    InMemoryBrokerPaperTransport,
)
from src.paper import (
    initialize_database,
)


T0 = datetime(
    2026,
    8,
    3,
    20,
    0,
    tzinfo=timezone.utc,
)


class InternalSide(str, Enum):
    LONG = "LONG"


class FakePaperRepository:
    def __init__(
        self,
        *,
        account,
        orders=(),
        positions=(),
    ) -> None:
        self.account = account
        self.orders = tuple(orders)
        self.positions = tuple(
            positions
        )
        self.system_events = []

    def get_account(
        self,
        account_id,
    ):
        assert (
            account_id
            == self.account.account_id
        )

        return self.account

    def list_pending_orders(
        self,
        account_id,
    ):
        assert (
            account_id
            == self.account.account_id
        )

        return self.orders

    def list_open_positions(
        self,
        account_id,
    ):
        assert (
            account_id
            == self.account.account_id
        )

        return self.positions

    def record_system_event(
        self,
        **values,
    ) -> None:
        self.system_events.append(
            values
        )


def internal_account(
    *,
    cash="10000",
):
    return SimpleNamespace(
        account_id="ACC-1",
        base_currency="USD",
        cash_balance=Decimal(cash),
        reserved_cash=Decimal("0"),
    )


def broker_account(
    *,
    cash="10000",
):
    return BrokerAccountSnapshot(
        provider_account_id=(
            "BROKER-PAPER-1"
        ),
        currency="USD",
        cash=Decimal(cash),
        buying_power=Decimal(cash),
        equity=Decimal(cash),
        captured_at=T0,
    )


def internal_order(
    key="CLIENT-1",
):
    return SimpleNamespace(
        order_id="ORDER-1",
        idempotency_key=key,
        symbol="AAPL",
        side=InternalSide.LONG,
        quantity=Decimal("10"),
    )


def broker_order(
    key="CLIENT-1",
):
    return BrokerOrderSnapshot(
        broker_order_id="BROKER-ORDER-1",
        client_order_id=key,
        symbol="AAPL",
        side=BrokerOrderSide.BUY,
        status=BrokerOrderStatus.NEW,
        quantity=Decimal("10"),
        filled_quantity=Decimal("0"),
        submitted_at=T0,
        updated_at=T0,
    )


def internal_position(
    *,
    position_id,
    quantity,
    entry_price,
):
    return SimpleNamespace(
        position_id=position_id,
        symbol="AAPL",
        side=InternalSide.LONG,
        quantity=Decimal(quantity),
        entry_price=Decimal(
            entry_price
        ),
    )


def broker_position():
    return BrokerPositionSnapshot(
        broker_position_id=(
            "BROKER-POSITION-1"
        ),
        symbol="AAPL",
        side=BrokerPositionSide.LONG,
        quantity=Decimal("10"),
        average_entry_price=(
            Decimal("106")
        ),
        market_value=Decimal("1100"),
        unrealized_pnl=Decimal("40"),
        captured_at=T0,
    )


def make_transport(
    *,
    account=None,
    orders=(),
    positions=(),
):
    return InMemoryBrokerPaperTransport(
        config=BrokerPaperConnectionConfig(
            provider="Example",
            base_url=(
                "https://paper-api."
                "example.com"
            ),
            account_id=(
                "BROKER-PAPER-1"
            ),
            api_key="test-key",
        ),
        account=(
            account or broker_account()
        ),
        orders=tuple(orders),
        positions=tuple(positions),
    )


def make_service(
    tmp_path,
    *,
    paper,
    transport,
):
    database_path = (
        tmp_path / "reconciliation.db"
    )

    repository = (
        BrokerReconciliationRepository(
            database_path
        )
    )

    connection = sqlite3.connect(
        database_path
    )

    try:
        connection.execute(
            """
            INSERT OR IGNORE INTO
            paper_accounts(
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper.account.account_id,
                "Broker reconciliation test",
                paper.account.base_currency,
                str(
                    paper.account.cash_balance
                ),
                str(
                    paper.account.cash_balance
                ),
                str(
                    paper.account.reserved_cash
                ),
                "ACTIVE",
                T0.isoformat(),
                T0.isoformat(),
            ),
        )

        connection.commit()
    finally:
        connection.close()

    service = BrokerReconciliationService(
        paper_repository=paper,
        reconciliation_repository=(
            repository
        ),
        transport=transport,
    )

    return repository, service


def test_schema_version_five_and_tables(
    tmp_path,
) -> None:
    database_path = (
        tmp_path / "schema.db"
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

        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }
    finally:
        connection.close()

    assert version == 6

    assert (
        "paper_broker_reconciliation_runs"
        in tables
    )

    assert (
        "paper_broker_reconciliation_items"
        in tables
    )


def test_version_four_database_upgrades(
    tmp_path,
) -> None:
    database_path = (
        tmp_path / "upgrade.db"
    )

    connection = sqlite3.connect(
        database_path
    )

    try:
        for script in (
            migrations._SCHEMA_V1,
            migrations._SCHEMA_V2,
            migrations._SCHEMA_V3,
            migrations._SCHEMA_V4,
        ):
            connection.executescript(
                script
            )
    finally:
        connection.close()

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

        run_table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE name =
              'paper_broker_reconciliation_runs'
            """
        ).fetchone()
    finally:
        connection.close()

    assert version == 6
    assert run_table is not None


def test_matching_account_is_persisted(
    tmp_path,
) -> None:
    paper = FakePaperRepository(
        account=internal_account()
    )

    repository, service = make_service(
        tmp_path,
        paper=paper,
        transport=make_transport(),
    )

    report = service.reconcile(
        "ACC-1",
        reconciliation_key="RUN-1",
        reconciled_at=T0,
    )

    assert report.reconciled is True

    assert (
        report.run.status
        is BrokerReconciliationRunStatus
        .MATCHED
    )

    assert len(report.items) == 1

    assert (
        report.items[0].category
        is BrokerReconciliationCategory
        .ACCOUNT
    )

    assert (
        repository.latest_run(
            "ACC-1"
        )
        == report.run
    )


def test_account_difference_is_persisted(
    tmp_path,
) -> None:
    paper = FakePaperRepository(
        account=internal_account(
            cash="10000"
        )
    )

    _, service = make_service(
        tmp_path,
        paper=paper,
        transport=make_transport(
            account=broker_account(
                cash="9990"
            )
        ),
    )

    report = service.reconcile(
        "ACC-1",
        reconciliation_key="RUN-2",
        reconciled_at=T0,
    )

    assert report.reconciled is False

    item = report.items[0]

    assert (
        item.status
        is BrokerReconciliationItemStatus
        .MISMATCH
    )

    assert "cash" in item.differences


def test_matching_active_order(
    tmp_path,
) -> None:
    paper = FakePaperRepository(
        account=internal_account(),
        orders=(internal_order(),),
    )

    _, service = make_service(
        tmp_path,
        paper=paper,
        transport=make_transport(
            orders=(broker_order(),)
        ),
    )

    report = service.reconcile(
        "ACC-1",
        reconciliation_key="RUN-3",
        reconciled_at=T0,
    )

    order_item = next(
        item
        for item in report.items
        if item.category
        is BrokerReconciliationCategory
        .ORDER
    )

    assert (
        order_item.status
        is BrokerReconciliationItemStatus
        .MATCH
    )


def test_missing_orders_are_reported(
    tmp_path,
) -> None:
    paper = FakePaperRepository(
        account=internal_account(),
        orders=(
            internal_order(
                "INTERNAL-ONLY"
            ),
        ),
    )

    _, service = make_service(
        tmp_path,
        paper=paper,
        transport=make_transport(
            orders=(
                broker_order(
                    "BROKER-ONLY"
                ),
            )
        ),
    )

    report = service.reconcile(
        "ACC-1",
        reconciliation_key="RUN-4",
        reconciled_at=T0,
    )

    statuses = {
        item.status
        for item in report.items
        if item.category
        is BrokerReconciliationCategory
        .ORDER
    }

    assert statuses == {
        BrokerReconciliationItemStatus
        .MISSING_INTERNAL,
        BrokerReconciliationItemStatus
        .MISSING_BROKER,
    }


def test_positions_are_aggregated(
    tmp_path,
) -> None:
    paper = FakePaperRepository(
        account=internal_account(),
        positions=(
            internal_position(
                position_id="POSITION-1",
                quantity="4",
                entry_price="100",
            ),
            internal_position(
                position_id="POSITION-2",
                quantity="6",
                entry_price="110",
            ),
        ),
    )

    _, service = make_service(
        tmp_path,
        paper=paper,
        transport=make_transport(
            positions=(
                broker_position(),
            )
        ),
    )

    report = service.reconcile(
        "ACC-1",
        reconciliation_key="RUN-5",
        reconciled_at=T0,
    )

    position_item = next(
        item
        for item in report.items
        if item.category
        is BrokerReconciliationCategory
        .POSITION
    )

    assert (
        position_item.status
        is BrokerReconciliationItemStatus
        .MATCH
    )

    assert (
        position_item
        .internal_reference_ids
        == (
            "POSITION-1",
            "POSITION-2",
        )
    )


def test_reconciliation_key_is_idempotent(
    tmp_path,
) -> None:
    paper = FakePaperRepository(
        account=internal_account()
    )

    repository, service = make_service(
        tmp_path,
        paper=paper,
        transport=make_transport(),
    )

    first = service.reconcile(
        "ACC-1",
        reconciliation_key=(
            "IDEMPOTENT-RUN"
        ),
        reconciled_at=T0,
    )

    second = service.reconcile(
        "ACC-1",
        reconciliation_key=(
            "IDEMPOTENT-RUN"
        ),
        reconciled_at=T0,
    )

    assert second.duplicate is True

    assert (
        second.run.reconciliation_run_id
        == first.run.reconciliation_run_id
    )

    assert len(
        repository.list_runs("ACC-1")
    ) == 1
