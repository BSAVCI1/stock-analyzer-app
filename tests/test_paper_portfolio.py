from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

import pytest

from src.paper import (
    NotificationStatus,
    OrderStatus,
    PaperExitReason,
    PaperPortfolioConfig,
    PaperRepository,
    PaperTradingService,
    PositionStatus,
)


T0 = datetime(
    2026,
    8,
    1,
    20,
    0,
    tzinfo=timezone.utc,
)


@pytest.fixture
def repository(
    tmp_path,
) -> PaperRepository:
    return PaperRepository(
        tmp_path / "paper.db"
    )


@pytest.fixture
def service(
    repository,
) -> PaperTradingService:
    return PaperTradingService(
        repository,
        config=PaperPortfolioConfig(
            starting_balance=Decimal("10000"),
            base_currency="USD",
            maximum_open_positions=5,
            maximum_allocation_fraction=Decimal(
                "0.20"
            ),
            risk_fraction_per_trade=Decimal(
                "0.01"
            ),
            maximum_open_risk_fraction=Decimal(
                "0.04"
            ),
            minimum_reward_to_risk=Decimal(
                "2"
            ),
        ),
        app_version="test-version",
        threshold_version="test-thresholds",
    )


def create_account_and_signal(
    service: PaperTradingService,
):
    account = service.create_account(
        created_at=T0
    )

    signal = service.persist_signal(
        account_id=account.account_id,
        signal_id="SIG-001",
        symbol="AAPL",
        generated_at=T0,
        expires_at=T0 + timedelta(days=5),
        strategy="trend_pullback",
        recommendation="BUY",
        market_regime="BULLISH",
        score=80,
        confidence=0.90,
        reward_to_risk=2.5,
        entry_low=99,
        entry_high=101,
        stop_price=95,
        targets=(110, 120, 130),
        evidence=(
            "Price is above MA200.",
            "Pullback is near MA20.",
        ),
        conflicts=(),
    )

    return account, signal


def test_account_creation_and_initial_ledger_reconcile(
    service,
    repository,
) -> None:
    account = service.create_account(
        created_at=T0
    )

    assert account.cash_balance == Decimal(
        "10000.00000000"
    )

    assert account.reserved_cash == Decimal(
        "0.00000000"
    )

    reconciliation = (
        repository.reconcile_account(
            account.account_id
        )
    )

    assert reconciliation.reconciled is True
    assert reconciliation.difference == Decimal(
        "0.00000000"
    )


def test_signal_is_persisted_with_versions(
    service,
    repository,
) -> None:
    _, signal = create_account_and_signal(
        service
    )

    stored = repository.get_signal(
        signal.signal_id
    )

    assert stored.symbol == "AAPL"
    assert stored.strategy == "trend_pullback"
    assert stored.recommendation == "BUY"
    assert stored.threshold_version == (
        "test-thresholds"
    )
    assert stored.app_version == (
        "test-version"
    )
    assert len(stored.evidence) == 2


def test_order_creation_is_idempotent(
    service,
    repository,
) -> None:
    account, signal = (
        create_account_and_signal(service)
    )

    first, first_created = (
        service.create_automatic_buy(
            account_id=account.account_id,
            signal_id=signal.signal_id,
            quantity=10,
            idempotency_key=(
                "AAPL-2026-08-01-BUY"
            ),
            estimated_fees=1,
            created_at=T0
            + timedelta(minutes=1),
        )
    )

    second, second_created = (
        service.create_automatic_buy(
            account_id=account.account_id,
            signal_id=signal.signal_id,
            quantity=10,
            idempotency_key=(
                "AAPL-2026-08-01-BUY"
            ),
            estimated_fees=1,
            created_at=T0
            + timedelta(minutes=2),
        )
    )

    assert first_created is True
    assert second_created is False
    assert first.order_id == second.order_id

    updated_account = repository.get_account(
        account.account_id
    )

    assert updated_account.reserved_cash == Decimal(
        "1011.00000000"
    )


def test_fill_creates_position_and_updates_cash(
    service,
    repository,
) -> None:
    account, signal = (
        create_account_and_signal(service)
    )

    order, _ = service.create_automatic_buy(
        account_id=account.account_id,
        signal_id=signal.signal_id,
        quantity=10,
        idempotency_key="AAPL-BUY-1",
        estimated_fees=1,
        created_at=T0
        + timedelta(minutes=1),
    )

    fill, position = (
        service.record_automatic_buy_fill(
            order_id=order.order_id,
            fill_price=100,
            fees=1,
            slippage=0.50,
            filled_at=T0
            + timedelta(days=1),
        )
    )

    assert fill.price == Decimal(
        "100.00000000"
    )

    assert position.status is PositionStatus.OPEN
    assert position.entry_price == Decimal(
        "100.00000000"
    )

    stored_order = repository.get_order(
        order.order_id
    )

    assert stored_order.status is OrderStatus.FILLED

    updated_account = repository.get_account(
        account.account_id
    )

    assert updated_account.cash_balance == Decimal(
        "8999.00000000"
    )

    assert updated_account.reserved_cash == Decimal(
        "0.00000000"
    )

    assert (
        repository
        .reconcile_account(
            account.account_id
        )
        .reconciled
        is True
    )


def test_repeated_fill_does_not_duplicate_position(
    service,
    repository,
) -> None:
    account, signal = (
        create_account_and_signal(service)
    )

    order, _ = service.create_automatic_buy(
        account_id=account.account_id,
        signal_id=signal.signal_id,
        quantity=10,
        idempotency_key="AAPL-BUY-2",
        estimated_fees=1,
        created_at=T0
        + timedelta(minutes=1),
    )

    first_fill, first_position = (
        service.record_automatic_buy_fill(
            order_id=order.order_id,
            fill_price=100,
            fees=1,
            slippage=0.50,
            filled_at=T0
            + timedelta(days=1),
        )
    )

    second_fill, second_position = (
        service.record_automatic_buy_fill(
            order_id=order.order_id,
            fill_price=100,
            fees=1,
            slippage=0.50,
            filled_at=T0
            + timedelta(days=1),
        )
    )

    assert first_fill.fill_id == (
        second_fill.fill_id
    )

    assert first_position.position_id == (
        second_position.position_id
    )

    assert len(
        repository.list_open_positions(
            account.account_id
        )
    ) == 1


def test_position_closure_records_realised_profit(
    service,
    repository,
) -> None:
    account, signal = (
        create_account_and_signal(service)
    )

    order, _ = service.create_automatic_buy(
        account_id=account.account_id,
        signal_id=signal.signal_id,
        quantity=10,
        idempotency_key="AAPL-BUY-3",
        estimated_fees=1,
        created_at=T0
        + timedelta(minutes=1),
    )

    _, position = (
        service.record_automatic_buy_fill(
            order_id=order.order_id,
            fill_price=100,
            fees=1,
            slippage=0.50,
            filled_at=T0
            + timedelta(days=1),
        )
    )

    trade = (
        service.close_automatic_position(
            position_id=position.position_id,
            exit_price=110,
            exit_reason=(
                PaperExitReason.TARGET
            ),
            exit_fees=1,
            exit_slippage=0.50,
            closed_at=T0
            + timedelta(days=3),
        )
    )

    assert trade.gross_pnl == Decimal(
        "100.00000000"
    )

    assert trade.fees == Decimal(
        "2.00000000"
    )

    assert trade.slippage == Decimal(
        "1.00000000"
    )

    assert trade.net_pnl == Decimal(
        "98.00000000"
    )

    assert trade.return_pct == pytest.approx(
        0.098
    )

    updated_account = repository.get_account(
        account.account_id
    )

    assert updated_account.cash_balance == Decimal(
        "10098.00000000"
    )

    assert (
        repository
        .reconcile_account(
            account.account_id
        )
        .reconciled
        is True
    )

    assert len(
        repository.list_closed_trades(
            account.account_id
        )
    ) == 1

    assert repository.list_open_positions(
        account.account_id
    ) == ()


def test_buy_and_sell_notifications_are_queued_once(
    service,
    repository,
) -> None:
    account, signal = (
        create_account_and_signal(service)
    )

    order, _ = service.create_automatic_buy(
        account_id=account.account_id,
        signal_id=signal.signal_id,
        quantity=10,
        idempotency_key="AAPL-BUY-4",
        estimated_fees=1,
        created_at=T0
        + timedelta(minutes=1),
    )

    _, position = (
        service.record_automatic_buy_fill(
            order_id=order.order_id,
            fill_price=100,
            fees=1,
            filled_at=T0
            + timedelta(days=1),
        )
    )

    service.record_automatic_buy_fill(
        order_id=order.order_id,
        fill_price=100,
        fees=1,
        filled_at=T0
        + timedelta(days=1),
    )

    service.close_automatic_position(
        position_id=position.position_id,
        exit_price=110,
        exit_reason=PaperExitReason.TARGET,
        exit_fees=1,
        closed_at=T0
        + timedelta(days=2),
    )

    notifications = (
        repository.list_notifications(
            account.account_id
        )
    )

    assert len(notifications) == 2

    assert {
        notification.event_type
        for notification in notifications
    } == {
        "PAPER_BUY_EXECUTED",
        "PAPER_SELL_EXECUTED",
    }

    assert all(
        notification.status
        is NotificationStatus.PENDING
        for notification in notifications
    )


def test_risk_limit_rejects_excessive_quantity(
    service,
) -> None:
    account, signal = (
        create_account_and_signal(service)
    )

    with pytest.raises(
        ValueError,
        match="maximum risk per trade",
    ):
        service.create_automatic_buy(
            account_id=account.account_id,
            signal_id=signal.signal_id,
            quantity=17,
            idempotency_key="TOO-MUCH-RISK",
            created_at=T0
            + timedelta(minutes=1),
        )


def test_cancel_order_releases_reserved_cash(
    service,
    repository,
) -> None:
    account, signal = (
        create_account_and_signal(service)
    )

    order, _ = service.create_automatic_buy(
        account_id=account.account_id,
        signal_id=signal.signal_id,
        quantity=10,
        idempotency_key="AAPL-CANCEL",
        estimated_fees=1,
        created_at=T0
        + timedelta(minutes=1),
    )

    cancelled = service.cancel_pending_order(
        order_id=order.order_id,
        reason="Signal invalidated.",
        cancelled_at=T0
        + timedelta(hours=1),
    )

    assert cancelled.status is (
        OrderStatus.CANCELLED
    )

    updated_account = repository.get_account(
        account.account_id
    )

    assert updated_account.reserved_cash == Decimal(
        "0.00000000"
    )


def test_records_persist_across_repository_instances(
    service,
    repository,
) -> None:
    account, signal = (
        create_account_and_signal(service)
    )

    reopened = PaperRepository(
        repository.database_path
    )

    assert (
        reopened
        .get_account(account.account_id)
        .cash_balance
        == Decimal("10000.00000000")
    )

    assert reopened.get_signal(
        signal.signal_id
    ).symbol == "AAPL"


def test_material_changes_have_audit_events(
    service,
    repository,
) -> None:
    account, signal = (
        create_account_and_signal(service)
    )

    order, _ = service.create_automatic_buy(
        account_id=account.account_id,
        signal_id=signal.signal_id,
        quantity=10,
        idempotency_key="AAPL-AUDIT",
        estimated_fees=1,
        created_at=T0
        + timedelta(minutes=1),
    )

    _, position = (
        service.record_automatic_buy_fill(
            order_id=order.order_id,
            fill_price=100,
            fees=1,
            filled_at=T0
            + timedelta(days=1),
        )
    )

    service.close_automatic_position(
        position_id=position.position_id,
        exit_price=110,
        exit_reason=PaperExitReason.TARGET,
        exit_fees=1,
        closed_at=T0
        + timedelta(days=2),
    )

    event_types = {
        event.event_type
        for event
        in repository.list_system_events(
            account.account_id
        )
    }

    assert {
        "ACCOUNT_CREATED",
        "SIGNAL_PERSISTED",
        "ORDER_CREATED",
        "ORDER_FILLED",
        "POSITION_OPENED",
        "POSITION_CLOSED",
    }.issubset(event_types)
