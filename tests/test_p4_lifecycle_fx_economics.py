"""P4.1 operational multicurrency lifecycle tests."""

from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

import pytest

from src.paper import (
    FXRateError,
    PaperExitReason,
    PaperRepository,
    PaperTradingService,
    QuoteToPortfolioFXRate,
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


class TimedFXProvider:
    def get_rate(
        self,
        *,
        quote_currency: str,
        portfolio_currency: str,
        as_of: datetime,
    ) -> QuoteToPortfolioFXRate:
        if (
            quote_currency != "USD"
            or portfolio_currency != "EUR"
        ):
            raise FXRateError(
                "Unexpected test currency pair."
            )

        if as_of <= T0:
            rate = Decimal("0.90")
        elif as_of <= T1:
            rate = Decimal("0.91")
        else:
            rate = Decimal("0.93")

        return QuoteToPortfolioFXRate(
            quote_currency="USD",
            portfolio_currency="EUR",
            rate=rate,
            as_of=as_of,
            source="TEST_TIMED_FX",
        )


def make_environment(
    tmp_path,
    *,
    quote_currency: str,
    account_currency: str,
    provider=None,
):
    database = tmp_path / "paper.db"

    repository = PaperRepository(
        database
    )

    account = repository.create_account(
        account_id="ACC-P4-FX",
        name="P4 FX Portfolio",
        base_currency=account_currency,
        starting_balance="2000",
        created_at=T0 - timedelta(days=1),
    )

    service = PaperTradingService(
        repository,
        fx_rate_provider=provider,
    )

    signal = service.persist_signal(
        account_id=account.account_id,
        signal_id="SIG-P4-FX",
        symbol="AAPL",
        quote_currency=quote_currency,
        generated_at=T0 - timedelta(hours=1),
        expires_at=T5,
        recommendation="BUY",
        strategy="trend_pullback",
        market_regime="BULLISH",
        score=85,
        confidence=0.9,
        reward_to_risk=2.5,
        entry_low=99,
        entry_high=100,
        stop_price=95,
        targets=(110,),
        evidence=("P4 FX test",),
    )

    return (
        repository,
        service,
        account,
        signal,
    )


def test_usd_security_eur_portfolio_lifecycle_reconciles(
    tmp_path,
) -> None:
    (
        repository,
        service,
        account,
        signal,
    ) = make_environment(
        tmp_path,
        quote_currency="USD",
        account_currency="EUR",
        provider=TimedFXProvider(),
    )

    assert signal.scan_id is None

    order, created = (
        service.create_automatic_buy(
            account_id=account.account_id,
            signal_id=signal.signal_id,
            quantity=1,
            idempotency_key="P4-USD-EUR",
            estimated_fees=1,
            created_at=T0,
        )
    )

    assert created is True
    assert order.quote_currency == "USD"
    assert order.portfolio_currency == "EUR"
    assert (
        order.reservation_fx_rate
        == Decimal("0.90000000")
    )
    assert (
        order.estimated_cash_required
        == Decimal("90.90000000")
    )
    assert (
        order.reserved_cash
        == Decimal("90.90000000")
    )

    fill, position = (
        service.record_automatic_buy_fill(
            order_id=order.order_id,
            fill_price=100,
            fees=1,
            slippage=0,
            filled_at=T1,
        )
    )

    assert fill.quote_currency == "USD"
    assert fill.portfolio_currency == "EUR"
    assert (
        fill.entry_fx_rate
        == Decimal("0.91000000")
    )
    assert (
        fill.cash_required_portfolio
        == Decimal("91.91000000")
    )

    assert position.quote_currency == "USD"
    assert (
        position.portfolio_currency
        == "EUR"
    )
    assert (
        position.entry_fx_rate
        == Decimal("0.91000000")
    )
    assert (
        position.entry_cash_portfolio
        == Decimal("91.91000000")
    )

    after_fill = repository.get_account(
        account.account_id
    )

    assert (
        after_fill.cash_balance
        == Decimal("1908.09000000")
    )
    assert (
        after_fill.reserved_cash
        == Decimal("0.00000000")
    )
    assert (
        repository.reconcile_account(
            account.account_id
        ).reconciled
        is True
    )

    trade = service.close_automatic_position(
        position_id=position.position_id,
        exit_price=110,
        exit_reason=PaperExitReason.TARGET,
        exit_fees=1,
        exit_slippage=0,
        closed_at=T3,
    )

    assert trade.quote_currency == "USD"
    assert trade.portfolio_currency == "EUR"
    assert (
        trade.entry_fx_rate
        == Decimal("0.91000000")
    )
    assert (
        trade.exit_fx_rate
        == Decimal("0.93000000")
    )
    assert (
        trade.gross_pnl
        == Decimal("11.30000000")
    )
    assert (
        trade.fees
        == Decimal("1.84000000")
    )
    assert (
        trade.net_pnl
        == Decimal("9.46000000")
    )

    final_account = repository.get_account(
        account.account_id
    )

    assert (
        final_account.cash_balance
        == Decimal("2009.46000000")
    )
    assert (
        repository.reconcile_account(
            account.account_id
        ).reconciled
        is True
    )


def test_cross_currency_order_requires_fx_provider(
    tmp_path,
) -> None:
    (
        _,
        service,
        account,
        signal,
    ) = make_environment(
        tmp_path,
        quote_currency="USD",
        account_currency="EUR",
        provider=None,
    )

    with pytest.raises(
        FXRateError,
        match="No FX rate provider",
    ):
        service.create_automatic_buy(
            account_id=account.account_id,
            signal_id=signal.signal_id,
            quantity=1,
            idempotency_key="P4-NO-FX",
            estimated_fees=1,
            created_at=T0,
        )


def test_same_currency_uses_identity_fx(
    tmp_path,
) -> None:
    (
        repository,
        service,
        account,
        signal,
    ) = make_environment(
        tmp_path,
        quote_currency="EUR",
        account_currency="EUR",
        provider=None,
    )

    order, _ = service.create_automatic_buy(
        account_id=account.account_id,
        signal_id=signal.signal_id,
        quantity=1,
        idempotency_key="P4-EUR-EUR",
        estimated_fees=1,
        created_at=T0,
    )

    assert (
        order.reservation_fx_rate
        == Decimal("1.00000000")
    )
    assert (
        order.reservation_fx_source
        == "IDENTITY"
    )

    fill, position = (
        service.record_automatic_buy_fill(
            order_id=order.order_id,
            fill_price=100,
            fees=1,
            slippage=0,
            filled_at=T1,
        )
    )

    assert (
        fill.entry_fx_rate
        == Decimal("1.00000000")
    )

    trade = service.close_automatic_position(
        position_id=position.position_id,
        exit_price=110,
        exit_reason=PaperExitReason.TARGET,
        exit_fees=1,
        exit_slippage=0,
        closed_at=T3,
    )

    assert (
        trade.entry_fx_rate
        == Decimal("1.00000000")
    )
    assert (
        trade.exit_fx_rate
        == Decimal("1.00000000")
    )
    assert (
        trade.gross_pnl
        == Decimal("10.00000000")
    )
    assert (
        trade.fees
        == Decimal("2.00000000")
    )
    assert (
        trade.net_pnl
        == Decimal("8.00000000")
    )

    assert (
        repository.reconcile_account(
            account.account_id
        ).reconciled
        is True
    )
