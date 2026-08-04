"""Deterministic foreign-exchange conversion for paper trading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Protocol, runtime_checkable

from .models import money


class FXRateError(ValueError):
    """Raised when an FX rate is missing or invalid."""


def _decimal(
    name: str,
    value: object,
) -> Decimal:
    if isinstance(value, bool):
        raise FXRateError(
            f"{name} must be numeric."
        )

    try:
        result = Decimal(str(value))
    except (
        InvalidOperation,
        TypeError,
        ValueError,
    ) as exc:
        raise FXRateError(
            f"{name} must be numeric."
        ) from exc

    if not result.is_finite():
        raise FXRateError(
            f"{name} must be finite."
        )

    return result


def _positive_decimal(
    name: str,
    value: object,
) -> Decimal:
    result = _decimal(name, value)

    if result <= 0:
        raise FXRateError(
            f"{name} must be positive."
        )

    return money(result)


def _currency(
    name: str,
    value: object,
) -> str:
    result = str(value).strip().upper()

    if (
        len(result) != 3
        or not result.isalpha()
    ):
        raise FXRateError(
            f"{name} must be a three-letter "
            "currency code."
        )

    return result


def _aware_datetime(
    name: str,
    value: object,
) -> datetime:
    if not isinstance(value, datetime):
        raise FXRateError(
            f"{name} must be a datetime."
        )

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise FXRateError(
            f"{name} must be timezone-aware."
        )

    return value


@dataclass(frozen=True, slots=True)
class QuoteToPortfolioFXRate:
    """Portfolio-currency units for one quote-currency unit."""

    quote_currency: str
    portfolio_currency: str
    rate: Decimal

    as_of: datetime
    source: str

    def __post_init__(self) -> None:
        quote_currency = _currency(
            "quote_currency",
            self.quote_currency,
        )

        portfolio_currency = _currency(
            "portfolio_currency",
            self.portfolio_currency,
        )

        rate = _positive_decimal(
            "rate",
            self.rate,
        )

        as_of = _aware_datetime(
            "as_of",
            self.as_of,
        )

        source = str(
            self.source
        ).strip()

        if not source:
            raise FXRateError(
                "source must be a non-empty string."
            )

        if (
            quote_currency
            == portfolio_currency
            and rate != Decimal("1.00000000")
        ):
            raise FXRateError(
                "Same-currency FX rate must "
                "equal 1."
            )

        object.__setattr__(
            self,
            "quote_currency",
            quote_currency,
        )

        object.__setattr__(
            self,
            "portfolio_currency",
            portfolio_currency,
        )

        object.__setattr__(
            self,
            "rate",
            rate,
        )

        object.__setattr__(
            self,
            "as_of",
            as_of,
        )

        object.__setattr__(
            self,
            "source",
            source,
        )

    def convert_quote_to_portfolio(
        self,
        amount: object,
    ) -> Decimal:
        """Convert a signed quote-currency amount."""

        value = _decimal(
            "amount",
            amount,
        )

        return money(
            value * self.rate
        )

    def convert_portfolio_to_quote(
        self,
        amount: object,
    ) -> Decimal:
        """Convert a signed portfolio-currency amount."""

        value = _decimal(
            "amount",
            amount,
        )

        return money(
            value / self.rate
        )


def identity_fx_rate(
    currency: str,
    *,
    as_of: datetime,
    source: str = "IDENTITY",
) -> QuoteToPortfolioFXRate:
    """Return the exact identity rate for one currency."""

    normalised = _currency(
        "currency",
        currency,
    )

    return QuoteToPortfolioFXRate(
        quote_currency=normalised,
        portfolio_currency=normalised,
        rate=Decimal("1"),
        as_of=as_of,
        source=source,
    )


@runtime_checkable
class FXRateProvider(Protocol):
    """Source of explicit quote-to-portfolio FX rates."""

    def get_rate(
        self,
        *,
        quote_currency: str,
        portfolio_currency: str,
        as_of: datetime,
    ) -> QuoteToPortfolioFXRate:
        """Return the applicable explicit conversion rate."""


class StaticFXRateProvider:
    """Deterministic provider backed by configured rates."""

    def __init__(
        self,
        rates: Mapping[
            tuple[str, str],
            object,
        ],
        *,
        source: str,
    ) -> None:
        if not isinstance(
            rates,
            Mapping,
        ):
            raise FXRateError(
                "rates must be a mapping."
            )

        source_value = str(source).strip()

        if not source_value:
            raise FXRateError(
                "source must be a non-empty string."
            )

        normalised: dict[
            tuple[str, str],
            Decimal,
        ] = {}

        for raw_pair, raw_rate in rates.items():
            if (
                not isinstance(raw_pair, tuple)
                or len(raw_pair) != 2
            ):
                raise FXRateError(
                    "FX rate keys must be "
                    "(quote_currency, "
                    "portfolio_currency) pairs."
                )

            quote_currency = _currency(
                "quote_currency",
                raw_pair[0],
            )

            portfolio_currency = _currency(
                "portfolio_currency",
                raw_pair[1],
            )

            if (
                quote_currency
                == portfolio_currency
            ):
                raise FXRateError(
                    "Identity rates must not be "
                    "configured explicitly."
                )

            pair = (
                quote_currency,
                portfolio_currency,
            )

            if pair in normalised:
                raise FXRateError(
                    "Duplicate normalised FX pair: "
                    f"{quote_currency}/"
                    f"{portfolio_currency}."
                )

            normalised[pair] = (
                _positive_decimal(
                    "rate",
                    raw_rate,
                )
            )

        self._rates = normalised
        self._source = source_value

    def get_rate(
        self,
        *,
        quote_currency: str,
        portfolio_currency: str,
        as_of: datetime,
    ) -> QuoteToPortfolioFXRate:
        quote = _currency(
            "quote_currency",
            quote_currency,
        )

        portfolio = _currency(
            "portfolio_currency",
            portfolio_currency,
        )

        timestamp = _aware_datetime(
            "as_of",
            as_of,
        )

        if quote == portfolio:
            return identity_fx_rate(
                quote,
                as_of=timestamp,
            )

        pair = (
            quote,
            portfolio,
        )

        try:
            rate = self._rates[pair]
        except KeyError as exc:
            raise FXRateError(
                "No explicit FX rate configured "
                f"for {quote}/{portfolio}."
            ) from exc

        return QuoteToPortfolioFXRate(
            quote_currency=quote,
            portfolio_currency=portfolio,
            rate=rate,
            as_of=timestamp,
            source=self._source,
        )
