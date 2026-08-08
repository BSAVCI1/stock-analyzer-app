"""Yahoo Finance FX-rate provider for paper trading."""

from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal
from typing import (
    Any,
    Callable,
    Protocol,
)

import pandas as pd

from .fx import (
    FXRateError,
    QuoteToPortfolioFXRate,
    identity_fx_rate,
)


class YahooFXTickerLike(Protocol):
    """Minimal yfinance ticker contract used for FX."""

    def history(
        self,
        **kwargs: Any,
    ) -> pd.DataFrame:
        ...


YahooFXTickerFactory = Callable[
    [str],
    YahooFXTickerLike,
]


def _currency(
    name: str,
    value: object,
) -> str:
    normalised = str(value).strip().upper()

    if (
        len(normalised) != 3
        or not normalised.isalpha()
    ):
        raise FXRateError(
            f"{name} must be a valid "
            "three-letter currency code."
        )

    return normalised


def _aware_utc(
    value: datetime,
) -> datetime:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise FXRateError(
            "FX as_of must be timezone-aware."
        )

    return value.astimezone(
        timezone.utc
    )


def _default_ticker_factory(
    symbol: str,
) -> YahooFXTickerLike:
    """Create yfinance ticker lazily."""

    try:
        import yfinance as yf
    except ImportError as exc:
        raise FXRateError(
            "yfinance is not installed."
        ) from exc

    try:
        yf.config.network.retries = 2
    except (AttributeError, TypeError):
        pass

    return yf.Ticker(symbol)


class YahooFXRateProvider:
    """Resolve quote-to-portfolio FX from Yahoo Finance.

    Yahoo FX symbols use the form ``USDEUR=X`` for
    portfolio-currency units per one quote-currency unit.

    If Yahoo has no usable direct pair, the provider tries
    the inverse pair and reciprocates it.
    """

    def __init__(
        self,
        ticker_factory: (
            YahooFXTickerFactory | None
        ) = None,
        *,
        source: str = "YAHOO_FINANCE",
        lookback_days: int = 10,
    ) -> None:
        if (
            isinstance(
                lookback_days,
                bool,
            )
            or not isinstance(
                lookback_days,
                int,
            )
            or lookback_days < 2
        ):
            raise ValueError(
                "lookback_days must be "
                "an integer of at least 2."
            )

        source_value = str(
            source
        ).strip()

        if not source_value:
            raise ValueError(
                "source must not be empty."
            )

        self.ticker_factory = (
            ticker_factory
            or _default_ticker_factory
        )

        self.source = source_value
        self.lookback_days = (
            lookback_days
        )

    def _load_pair(
        self,
        *,
        symbol: str,
        as_of: datetime,
    ) -> tuple[
        Decimal,
        datetime,
    ]:
        start = (
            as_of
            - timedelta(
                days=self.lookback_days
            )
        )

        end = (
            as_of
            + timedelta(days=1)
        )

        try:
            ticker = self.ticker_factory(
                symbol
            )

            history = ticker.history(
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                actions=False,
                repair=False,
                timeout=10,
                raise_errors=True,
            )

        except Exception as exc:
            raise FXRateError(
                f"Yahoo FX data could not be "
                f"downloaded for {symbol}: "
                f"{type(exc).__name__}."
            ) from exc

        if (
            history is None
            or history.empty
            or "Close" not in history.columns
        ):
            raise FXRateError(
                f"Yahoo FX returned no usable "
                f"close data for {symbol}."
            )

        observations: list[
            tuple[
                datetime,
                Decimal,
            ]
        ] = []

        for index, raw_value in (
            history["Close"].items()
        ):
            if pd.isna(raw_value):
                continue

            timestamp = pd.Timestamp(
                index
            )

            if timestamp.tzinfo is None:
                timestamp = (
                    timestamp.tz_localize(
                        "UTC"
                    )
                )
            else:
                timestamp = (
                    timestamp.tz_convert(
                        "UTC"
                    )
                )

            observed_at = (
                timestamp
                .to_pydatetime()
            )

            # Daily FX rows represent that
            # observation date. Never select
            # a row from a future calendar day.
            if (
                observed_at.date()
                > as_of.date()
            ):
                continue

            try:
                close = Decimal(
                    str(raw_value)
                )
            except Exception:
                continue

            if (
                not close.is_finite()
                or close <= 0
            ):
                continue

            observations.append(
                (
                    observed_at,
                    close,
                )
            )

        if not observations:
            raise FXRateError(
                f"Yahoo FX returned no usable "
                f"observation for {symbol} "
                "at or before the requested "
                "date."
            )

        return max(
            observations,
            key=lambda item: item[0],
        )

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

        requested_at = _aware_utc(
            as_of
        )

        if quote == portfolio:
            return identity_fx_rate(
                quote,
                as_of=requested_at,
            )

        direct_symbol = (
            f"{quote}{portfolio}=X"
        )

        inverse_symbol = (
            f"{portfolio}{quote}=X"
        )

        direct_error: (
            FXRateError | None
        ) = None

        try:
            observed_at, close = (
                self._load_pair(
                    symbol=direct_symbol,
                    as_of=requested_at,
                )
            )

            return QuoteToPortfolioFXRate(
                quote_currency=quote,
                portfolio_currency=(
                    portfolio
                ),
                rate=close,
                as_of=observed_at,
                source=(
                    f"{self.source}:"
                    f"{direct_symbol}"
                ),
            )

        except FXRateError as exc:
            direct_error = exc

        try:
            observed_at, inverse_close = (
                self._load_pair(
                    symbol=inverse_symbol,
                    as_of=requested_at,
                )
            )

            reciprocal = (
                Decimal("1")
                / inverse_close
            )

            return QuoteToPortfolioFXRate(
                quote_currency=quote,
                portfolio_currency=(
                    portfolio
                ),
                rate=reciprocal,
                as_of=observed_at,
                source=(
                    f"{self.source}:"
                    f"{inverse_symbol}:"
                    "INVERTED"
                ),
            )

        except FXRateError as exc:
            raise FXRateError(
                "Yahoo FX could not resolve "
                f"{quote}/{portfolio}; "
                "both direct and inverse "
                "pairs failed. "
                f"Direct={type(direct_error).__name__}; "
                f"Inverse={type(exc).__name__}."
            ) from exc
