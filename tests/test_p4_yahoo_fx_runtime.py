"""P4.1 Yahoo FX provider and runtime wiring tests."""

from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

import pandas as pd
import pytest

from src.jobs.runtime import (
    build_runtime,
    load_runtime_settings,
)
from src.paper import (
    FXRateError,
    PaperRepository,
    YahooFXRateProvider,
)


T0 = datetime(
    2026,
    8,
    8,
    20,
    0,
    tzinfo=timezone.utc,
)


class FakeTicker:
    def __init__(
        self,
        *,
        history=None,
        error: Exception | None = None,
    ):
        self._history = history
        self._error = error
        self.calls = []

    def history(
        self,
        **kwargs,
    ):
        self.calls.append(kwargs)

        if self._error is not None:
            raise self._error

        return self._history.copy()


class FakeFactory:
    def __init__(self, tickers):
        self.tickers = tickers
        self.symbols = []

    def __call__(self, symbol):
        self.symbols.append(symbol)
        return self.tickers[symbol]


def history(*rows):
    return pd.DataFrame(
        {
            "Close": [
                value
                for _, value in rows
            ],
        },
        index=pd.DatetimeIndex(
            [
                timestamp
                for timestamp, _ in rows
            ]
        ),
    )


def test_yahoo_fx_uses_latest_direct_observation() -> None:
    direct = FakeTicker(
        history=history(
            (
                T0 - timedelta(days=1),
                0.90,
            ),
            (
                T0,
                0.91,
            ),
            (
                T0 + timedelta(days=1),
                0.92,
            ),
        )
    )

    factory = FakeFactory(
        {
            "USDEUR=X": direct,
        }
    )

    provider = YahooFXRateProvider(
        factory
    )

    rate = provider.get_rate(
        quote_currency="usd",
        portfolio_currency="eur",
        as_of=T0,
    )

    assert factory.symbols == [
        "USDEUR=X",
    ]

    assert rate.quote_currency == "USD"
    assert (
        rate.portfolio_currency
        == "EUR"
    )

    assert rate.rate == Decimal(
        "0.91000000"
    )

    assert (
        rate.source
        == "YAHOO_FINANCE:USDEUR=X"
    )

    assert rate.as_of.date() == T0.date()

    assert len(direct.calls) == 1

    assert (
        direct.calls[0]["interval"]
        == "1d"
    )


def test_yahoo_fx_falls_back_to_inverse_pair() -> None:
    direct = FakeTicker(
        error=RuntimeError(
            "direct unavailable"
        )
    )

    inverse = FakeTicker(
        history=history(
            (
                T0,
                1.10,
            ),
        )
    )

    factory = FakeFactory(
        {
            "USDEUR=X": direct,
            "EURUSD=X": inverse,
        }
    )

    provider = YahooFXRateProvider(
        factory
    )

    rate = provider.get_rate(
        quote_currency="USD",
        portfolio_currency="EUR",
        as_of=T0,
    )

    assert factory.symbols == [
        "USDEUR=X",
        "EURUSD=X",
    ]

    assert rate.rate == Decimal(
        "0.90909091"
    )

    assert (
        rate.source
        == (
            "YAHOO_FINANCE:"
            "EURUSD=X:INVERTED"
        )
    )


def test_yahoo_fx_identity_does_not_call_yahoo() -> None:
    def forbidden_factory(symbol):
        raise AssertionError(
            f"Yahoo must not be called: {symbol}"
        )

    provider = YahooFXRateProvider(
        forbidden_factory
    )

    rate = provider.get_rate(
        quote_currency="EUR",
        portfolio_currency="EUR",
        as_of=T0,
    )

    assert rate.rate == Decimal(
        "1.00000000"
    )

    assert rate.source == "IDENTITY"


def test_yahoo_fx_rejects_when_both_pairs_fail() -> None:
    factory = FakeFactory(
        {
            "USDEUR=X": FakeTicker(
                error=RuntimeError("direct")
            ),
            "EURUSD=X": FakeTicker(
                history=pd.DataFrame()
            ),
        }
    )

    provider = YahooFXRateProvider(
        factory
    )

    with pytest.raises(
        FXRateError,
        match=(
            "both direct and inverse "
            "pairs failed"
        ),
    ):
        provider.get_rate(
            quote_currency="USD",
            portfolio_currency="EUR",
            as_of=T0,
        )


def test_runtime_wires_yahoo_fx_without_network(
    tmp_path,
) -> None:
    database_path = (
        tmp_path / "runtime.db"
    )

    repository = PaperRepository(
        database_path
    )

    account = repository.create_account(
        name="Runtime FX Test",
        base_currency="USD",
        starting_balance="10000",
        created_at=T0,
    )

    settings = load_runtime_settings(
        {
            "PAPER_DATABASE_PATH":
            str(database_path),
            "PAPER_ACCOUNT_ID":
            account.account_id,
        }
    )

    runtime = build_runtime(
        settings,
        environ={
            "PAPER_DATABASE_PATH":
            str(database_path),
            "PAPER_ACCOUNT_ID":
            account.account_id,
        },
    )

    assert isinstance(
        runtime.paper_service
        .fx_rate_provider,
        YahooFXRateProvider,
    )

    # Construction must not make a
    # provider/network request.
    assert (
        runtime.paper_service
        .fx_rate_provider
        .source
        == "YAHOO_FINANCE"
    )
