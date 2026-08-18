from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

import pytest

from src.costs import IBKREconomicDecision
from src.paper.proposal import (
    CostAwareOrderProposalRequest,
    OrderProposalRejected,
    propose_cost_aware_us_long_order,
)


NOW = datetime(
    2026,
    8,
    18,
    8,
    0,
    tzinfo=timezone.utc,
)


def request_for(
    *,
    target_price: str = "130",
    expires_at=None,
):
    return CostAwareOrderProposalRequest(
        symbol="aapl",
        quote_currency="usd",
        entry_price=Decimal("100"),
        stop_price=Decimal("90"),
        target_price=Decimal(
            target_price
        ),
        proposed_at=NOW,
        expires_at=(
            expires_at
            or NOW + timedelta(hours=2)
        ),
        quote_to_portfolio_rate=(
            Decimal("0.90")
        ),
        available_cash_portfolio=(
            Decimal("500")
        ),
        invested_exposure_portfolio=(
            Decimal("0")
        ),
        current_position_count=0,
        minimum_net_reward_to_risk=(
            Decimal("1.50")
        ),
    )


def test_proposal_is_complete_cost_aware_and_fractional():
    proposal = (
        propose_cost_aware_us_long_order(
            request_for()
        )
    )

    assert proposal.symbol == "AAPL"
    assert proposal.quantity > 0
    assert (
        proposal.quantity
        % Decimal("1")
        != 0
    )
    assert (
        proposal.economics.decision
        is IBKREconomicDecision.ACCEPT
    )
    assert (
        proposal.net_reward_to_risk
        >= Decimal("1.50")
    )
    assert (
        proposal.capital_required_portfolio
        <= Decimal("100")
    )
    assert (
        proposal.planned_loss_portfolio
        <= Decimal("10")
    )
    assert proposal.expires_at > proposal.proposed_at


def test_costs_can_reject_grossly_positive_trade():
    with pytest.raises(
        OrderProposalRejected,
        match="not economical after modeled costs",
    ):
        propose_cost_aware_us_long_order(
            request_for(
                target_price="105"
            )
        )


def test_proposal_requires_future_expiry():
    with pytest.raises(
        ValueError,
        match="expires_at must be after",
    ):
        request_for(expires_at=NOW)


def test_proposal_requires_protective_stop():
    with pytest.raises(
        ValueError,
        match="stop_price must be below",
    ):
        CostAwareOrderProposalRequest(
            symbol="AAPL",
            quote_currency="USD",
            entry_price=Decimal("100"),
            stop_price=Decimal("100"),
            target_price=Decimal("130"),
            proposed_at=NOW,
            expires_at=(
                NOW + timedelta(hours=1)
            ),
            quote_to_portfolio_rate=(
                Decimal("0.90")
            ),
            available_cash_portfolio=(
                Decimal("500")
            ),
            invested_exposure_portfolio=(
                Decimal("0")
            ),
            current_position_count=0,
            minimum_net_reward_to_risk=(
                Decimal("1.5")
            ),
        )


def test_non_us_quote_currency_is_not_guessed():
    with pytest.raises(
        ValueError,
        match="USD-quoted instruments only",
    ):
        CostAwareOrderProposalRequest(
            symbol="VWCE",
            quote_currency="EUR",
            entry_price=Decimal("100"),
            stop_price=Decimal("90"),
            target_price=Decimal("130"),
            proposed_at=NOW,
            expires_at=(
                NOW + timedelta(hours=1)
            ),
            quote_to_portfolio_rate=(
                Decimal("1")
            ),
            available_cash_portfolio=(
                Decimal("500")
            ),
            invested_exposure_portfolio=(
                Decimal("0")
            ),
            current_position_count=0,
            minimum_net_reward_to_risk=(
                Decimal("1.5")
            ),
        )
