from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.paper.fx import (
    FXRateError,
    QuoteToPortfolioFXRate,
    StaticFXRateProvider,
    identity_fx_rate,
)


AS_OF = datetime(
    2026,
    8,
    4,
    20,
    0,
    tzinfo=timezone.utc,
)


def test_identity_rate_is_exact() -> None:
    rate = identity_fx_rate(
        "eur",
        as_of=AS_OF,
    )

    assert rate.quote_currency == "EUR"
    assert rate.portfolio_currency == "EUR"

    assert rate.rate == Decimal(
        "1.00000000"
    )

    assert (
        rate.convert_quote_to_portfolio(
            "100"
        )
        == Decimal("100.00000000")
    )


def test_usd_amount_converts_to_eur() -> None:
    rate = QuoteToPortfolioFXRate(
        quote_currency="USD",
        portfolio_currency="EUR",
        rate="0.90",
        as_of=AS_OF,
        source="TEST_RATE",
    )

    assert (
        rate.convert_quote_to_portfolio(
            "100"
        )
        == Decimal("90.00000000")
    )

    assert (
        rate.convert_quote_to_portfolio(
            "-10"
        )
        == Decimal("-9.00000000")
    )


def test_inverse_conversion_is_deterministic() -> None:
    rate = QuoteToPortfolioFXRate(
        quote_currency="USD",
        portfolio_currency="EUR",
        rate="0.80",
        as_of=AS_OF,
        source="TEST_RATE",
    )

    assert (
        rate.convert_portfolio_to_quote(
            "80"
        )
        == Decimal("100.00000000")
    )


def test_same_currency_requires_rate_one() -> None:
    with pytest.raises(
        FXRateError,
        match="must equal 1",
    ):
        QuoteToPortfolioFXRate(
            quote_currency="EUR",
            portfolio_currency="EUR",
            rate="0.99",
            as_of=AS_OF,
            source="INVALID",
        )


@pytest.mark.parametrize(
    "value",
    (
        "0",
        "-1",
        "NaN",
        "Infinity",
    ),
)
def test_invalid_rate_is_rejected(
    value,
) -> None:
    with pytest.raises(
        FXRateError,
    ):
        QuoteToPortfolioFXRate(
            quote_currency="USD",
            portfolio_currency="EUR",
            rate=value,
            as_of=AS_OF,
            source="INVALID",
        )


def test_timestamp_must_be_timezone_aware() -> None:
    with pytest.raises(
        FXRateError,
        match="timezone-aware",
    ):
        QuoteToPortfolioFXRate(
            quote_currency="USD",
            portfolio_currency="EUR",
            rate="0.90",
            as_of=datetime(
                2026,
                8,
                4,
                20,
                0,
            ),
            source="INVALID",
        )


def test_static_provider_returns_explicit_rate() -> None:
    provider = StaticFXRateProvider(
        {
            (
                "USD",
                "EUR",
            ): "0.90",
        },
        source="TEST_CONFIGURATION",
    )

    rate = provider.get_rate(
        quote_currency="usd",
        portfolio_currency="eur",
        as_of=AS_OF,
    )

    assert rate.rate == Decimal(
        "0.90000000"
    )

    assert rate.source == (
        "TEST_CONFIGURATION"
    )


def test_static_provider_returns_identity_rate() -> None:
    provider = StaticFXRateProvider(
        {},
        source="TEST_CONFIGURATION",
    )

    rate = provider.get_rate(
        quote_currency="EUR",
        portfolio_currency="EUR",
        as_of=AS_OF,
    )

    assert rate.rate == Decimal(
        "1.00000000"
    )

    assert rate.source == "IDENTITY"


def test_missing_non_base_rate_is_rejected() -> None:
    provider = StaticFXRateProvider(
        {},
        source="TEST_CONFIGURATION",
    )

    with pytest.raises(
        FXRateError,
        match="No explicit FX rate",
    ):
        provider.get_rate(
            quote_currency="USD",
            portfolio_currency="EUR",
            as_of=AS_OF,
        )


def test_identity_pair_cannot_be_configured() -> None:
    with pytest.raises(
        FXRateError,
        match="must not be configured",
    ):
        StaticFXRateProvider(
            {
                (
                    "EUR",
                    "EUR",
                ): "1",
            },
            source="INVALID",
        )
