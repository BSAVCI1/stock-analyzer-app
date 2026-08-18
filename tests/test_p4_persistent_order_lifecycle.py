from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

import pytest

from src.paper import (
    CostAwareOrderProposalRequest,
    FixedNotionalSizingPolicy,
    OrderStatus,
    PaperPortfolioConfig,
    PaperRepository,
    PaperTradingService,
    propose_cost_aware_us_long_order,
)


NOW = datetime(
    2026,
    8,
    18,
    10,
    0,
    tzinfo=timezone.utc,
)
EXPIRY = NOW + timedelta(days=2)


@pytest.fixture
def lifecycle(tmp_path):
    repository = PaperRepository(
        tmp_path / "paper.db"
    )
    service = PaperTradingService(
        repository,
        config=PaperPortfolioConfig(
            starting_balance=Decimal("10000"),
            base_currency="USD",
            minimum_reward_to_risk=(
                Decimal("2")
            ),
        ),
        app_version="p4.6.2-test",
        threshold_version="p4.6.2-test",
    )
    account = service.create_account(
        created_at=NOW - timedelta(hours=1)
    )
    signal = service.persist_signal(
        account_id=account.account_id,
        signal_id="SIG-P4-6-2",
        symbol="AAPL",
        quote_currency="USD",
        generated_at=(
            NOW - timedelta(minutes=5)
        ),
        expires_at=EXPIRY,
        strategy="trend_pullback",
        recommendation="BUY",
        market_regime="BULLISH",
        score=85,
        confidence=0.90,
        reward_to_risk=Decimal("3.16"),
        entry_low=Decimal("99"),
        entry_high=Decimal("101"),
        stop_price=Decimal("95"),
        targets=(
            Decimal("120"),
            Decimal("130"),
        ),
        evidence=("qualified",),
        conflicts=(),
    )
    proposal = propose_cost_aware_us_long_order(
        CostAwareOrderProposalRequest(
            symbol="AAPL",
            quote_currency="USD",
            entry_price=Decimal("101"),
            stop_price=Decimal("95"),
            target_price=Decimal("120"),
            proposed_at=NOW,
            expires_at=EXPIRY,
            quote_to_portfolio_rate=(
                Decimal("1")
            ),
            available_cash_portfolio=(
                account.available_cash
            ),
            invested_exposure_portfolio=(
                Decimal("0")
            ),
            current_position_count=0,
            minimum_net_reward_to_risk=(
                Decimal("2")
            ),
        ),
        policy=FixedNotionalSizingPolicy(
            portfolio_currency="USD"
        ),
    )

    return repository, service, account, signal, proposal


def test_approved_proposal_is_persisted_and_reserves_cash(
    lifecycle,
):
    (
        repository,
        service,
        account,
        signal,
        proposal,
    ) = lifecycle

    order, created = service.create_cost_aware_buy(
        account_id=account.account_id,
        signal_id=signal.signal_id,
        proposal=proposal,
        idempotency_key="P4-6-2:AAPL",
    )
    stored_account = repository.get_account(
        account.account_id
    )

    assert created is True
    assert order.status is OrderStatus.PENDING
    assert order.quantity == proposal.quantity
    assert order.expires_at == proposal.expires_at
    assert order.reserved_cash > 0
    assert (
        stored_account.reserved_cash
        == order.reserved_cash
    )


def test_proposal_persistence_is_idempotent(
    lifecycle,
):
    _, service, account, signal, proposal = lifecycle

    first, first_created = (
        service.create_cost_aware_buy(
            account_id=account.account_id,
            signal_id=signal.signal_id,
            proposal=proposal,
            idempotency_key="P4-6-2:REPLAY",
        )
    )
    second, second_created = (
        service.create_cost_aware_buy(
            account_id=account.account_id,
            signal_id=signal.signal_id,
            proposal=proposal,
            idempotency_key="P4-6-2:REPLAY",
        )
    )

    assert first_created is True
    assert second_created is False
    assert first.order_id == second.order_id


def test_cancellation_releases_proposal_reservation(
    lifecycle,
):
    (
        repository,
        service,
        account,
        signal,
        proposal,
    ) = lifecycle
    order, _ = service.create_cost_aware_buy(
        account_id=account.account_id,
        signal_id=signal.signal_id,
        proposal=proposal,
        idempotency_key="P4-6-2:CANCEL",
    )

    cancelled = service.cancel_pending_order(
        order_id=order.order_id,
        reason="Thesis invalidated.",
        cancelled_at=NOW + timedelta(hours=1),
    )

    assert cancelled.status is OrderStatus.CANCELLED
    assert (
        repository.get_account(
            account.account_id
        ).reserved_cash
        == Decimal("0")
    )


def test_expiry_releases_proposal_reservation(
    lifecycle,
):
    (
        repository,
        service,
        account,
        signal,
        proposal,
    ) = lifecycle
    order, _ = service.create_cost_aware_buy(
        account_id=account.account_id,
        signal_id=signal.signal_id,
        proposal=proposal,
        idempotency_key="P4-6-2:EXPIRE",
    )

    expired = repository.expire_order(
        order.order_id,
        expired_at=EXPIRY,
        reason="Entry window expired.",
    )

    assert expired.status is OrderStatus.EXPIRED
    assert (
        repository.get_account(
            account.account_id
        ).reserved_cash
        == Decimal("0")
    )


def test_proposal_must_share_signal_expiry(
    lifecycle,
):
    _, service, account, signal, proposal = lifecycle
    changed = propose_cost_aware_us_long_order(
        CostAwareOrderProposalRequest(
            symbol=proposal.symbol,
            quote_currency=(
                proposal.quote_currency
            ),
            entry_price=proposal.entry_price,
            stop_price=proposal.stop_price,
            target_price=proposal.target_price,
            proposed_at=proposal.proposed_at,
            expires_at=(
                proposal.expires_at
                - timedelta(hours=1)
            ),
            quote_to_portfolio_rate=(
                Decimal("1")
            ),
            available_cash_portfolio=(
                Decimal("10000")
            ),
            invested_exposure_portfolio=(
                Decimal("0")
            ),
            current_position_count=0,
            minimum_net_reward_to_risk=(
                Decimal("2")
            ),
        ),
        policy=FixedNotionalSizingPolicy(
            portfolio_currency="USD"
        ),
    )

    with pytest.raises(
        ValueError,
        match="same clock",
    ):
        service.create_cost_aware_buy(
            account_id=account.account_id,
            signal_id=signal.signal_id,
            proposal=changed,
            idempotency_key="P4-6-2:CLOCK",
        )
