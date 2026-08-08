"""P4.1 engine fixed-notional and FX valuation tests."""

from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

import pandas as pd
import pytest

from src.automation.engine import (
    AutomatedPaperExecutionEngine,
)
from src.automation.repository import (
    AutomationRepository,
)
from src.paper import (
    PaperExitReason,
    PaperPortfolioConfig,
    PaperRepository,
    PaperTradingService,
    StaticFXRateProvider,
)
from src.paper.sizing import (
    FixedNotionalSizingPolicy,
    PositionSizingRejected,
)
from src.scanner import ScannerRepository


T0 = datetime(
    2026,
    8,
    8,
    12,
    0,
    tzinfo=timezone.utc,
)


def make_environment(tmp_path):
    database = tmp_path / "engine.db"

    paper_repository = PaperRepository(
        database
    )

    fx_provider = StaticFXRateProvider(
        {
            ("USD", "EUR"):
            Decimal("0.90"),
        },
        source="TEST_ENGINE_FX",
    )

    paper_service = PaperTradingService(
        paper_repository,
        config=PaperPortfolioConfig(
            starting_balance=Decimal("2000"),
            base_currency="EUR",
        ),
        fx_rate_provider=fx_provider,
        app_version="test",
        threshold_version="test",
    )

    account = paper_service.create_account(
        created_at=T0
    )

    scanner_repository = ScannerRepository(
        database
    )

    automation_repository = (
        AutomationRepository(
            database
        )
    )

    control = (
        automation_repository
        .set_fixed_notional_sizing(
            account.account_id,
            policy=(
                FixedNotionalSizingPolicy()
            ),
            updated_at=T0,
        )
    )

    engine = (
        AutomatedPaperExecutionEngine(
            paper_repository=(
                paper_repository
            ),
            paper_service=paper_service,
            scanner_repository=(
                scanner_repository
            ),
            automation_repository=(
                automation_repository
            ),
            snapshot_loader=(
                lambda symbol: None
            ),
            app_version="test",
        )
    )

    return (
        paper_repository,
        paper_service,
        automation_repository,
        engine,
        account,
        control,
    )


def persist_signal(
    service,
    account_id,
    *,
    signal_id,
    entry,
    stop,
):
    return service.persist_signal(
        account_id=account_id,
        signal_id=signal_id,
        symbol="AAPL",
        quote_currency="USD",
        generated_at=(
            T0 - timedelta(hours=1)
        ),
        expires_at=(
            T0 + timedelta(days=5)
        ),
        strategy="trend_pullback",
        recommendation="BUY",
        market_regime="BULLISH",
        score=85,
        confidence=0.90,
        reward_to_risk=2.5,
        entry_low=entry,
        entry_high=entry,
        stop_price=stop,
        targets=(
            Decimal(str(entry))
            + Decimal("15"),
        ),
        evidence=(
            "P4 engine sizing test",
        ),
    )


def test_fixed_notional_engine_sizes_usd_signal_in_eur(
    tmp_path,
) -> None:
    (
        _,
        service,
        _,
        engine,
        account,
        control,
    ) = make_environment(tmp_path)

    signal = persist_signal(
        service,
        account.account_id,
        signal_id="SIG-P4-SIZE",
        entry=50,
        stop=47,
    )

    quantity = engine._calculate_quantity(
        account_id=account.account_id,
        signal_id=signal.signal_id,
        control=control,
        run_at=T0,
    )

    assert quantity == Decimal(
        "2.00000000"
    )

    # USD 100 * 0.90 = EUR 90,
    # safely below the EUR 100 hard ceiling.
    assert (
        Decimal("50")
        * quantity
        * Decimal("0.90")
        == Decimal("90.0000000000")
    )


def test_fixed_notional_engine_rejects_one_share_above_loss_cap(
    tmp_path,
) -> None:
    (
        _,
        service,
        _,
        engine,
        account,
        control,
    ) = make_environment(tmp_path)

    signal = persist_signal(
        service,
        account.account_id,
        signal_id="SIG-P4-RISK",
        entry=100,
        stop=80,
    )

    with pytest.raises(
        PositionSizingRejected,
        match="No positive quantity step",
    ):
        engine._calculate_quantity(
            account_id=account.account_id,
            signal_id=signal.signal_id,
            control=control,
            run_at=T0,
        )


def test_market_value_converts_current_usd_value_to_eur(
    tmp_path,
    monkeypatch,
) -> None:
    (
        repository,
        service,
        _,
        engine,
        account,
        control,
    ) = make_environment(tmp_path)

    signal = persist_signal(
        service,
        account.account_id,
        signal_id="SIG-P4-MV",
        entry=100,
        stop=95,
    )

    order, created = (
        service.create_automatic_buy(
            account_id=account.account_id,
            signal_id=signal.signal_id,
            quantity=1,
            idempotency_key="P4-MV-ORDER",
            estimated_fees=0,
            created_at=T0,
        )
    )

    assert created is True

    _, position = (
        service.record_automatic_buy_fill(
            order_id=order.order_id,
            fill_price=100,
            fees=0,
            slippage=0,
            filled_at=(
                T0 + timedelta(hours=1)
            ),
        )
    )

    assert position.quote_currency == "USD"
    assert (
        position.portfolio_currency
        == "EUR"
    )

    history = pd.DataFrame(
        {
            "Close": [
                Decimal("110"),
            ],
        }
    )

    monkeypatch.setattr(
        engine,
        "_load_market",
        lambda *args, **kwargs: (
            None,
            history,
        ),
    )

    market_value = (
        engine._portfolio_market_value(
            account_id=account.account_id,
            run_at=(
                T0 + timedelta(days=1)
            ),
            control=control,
            cache={},
        )
    )

    assert market_value == Decimal(
        "99.00000000"
    )

    assert (
        repository.reconcile_account(
            account.account_id
        ).reconciled
        is True
    )


def test_pending_order_counts_as_committed_position(
    tmp_path,
) -> None:
    (
        _,
        service,
        automation,
        engine,
        account,
        _,
    ) = make_environment(tmp_path)

    restrictive_policy = (
        FixedNotionalSizingPolicy(
            maximum_open_positions=1,
        )
    )

    control = (
        automation
        .set_fixed_notional_sizing(
            account.account_id,
            policy=restrictive_policy,
            updated_at=(
                T0 + timedelta(minutes=1)
            ),
        )
    )

    first_signal = persist_signal(
        service,
        account.account_id,
        signal_id="SIG-P4-FIRST",
        entry=50,
        stop=47,
    )

    first_quantity = (
        engine._calculate_quantity(
            account_id=account.account_id,
            signal_id=(
                first_signal.signal_id
            ),
            control=control,
            run_at=T0,
        )
    )

    first_fee = Decimal("0")

    service.create_automatic_buy(
        account_id=account.account_id,
        signal_id=first_signal.signal_id,
        quantity=first_quantity,
        idempotency_key="P4-FIRST",
        estimated_fees=first_fee,
        created_at=T0,
    )

    second_signal = persist_signal(
        service,
        account.account_id,
        signal_id="SIG-P4-SECOND",
        entry=50,
        stop=47,
    )

    with pytest.raises(
        PositionSizingRejected,
        match="open-position limit",
    ):
        engine._calculate_quantity(
            account_id=account.account_id,
            signal_id=(
                second_signal.signal_id
            ),
            control=control,
            run_at=(
                T0 + timedelta(minutes=2)
            ),
        )
