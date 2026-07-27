"""Reliable, testable market-data loading for the stock analyser.

The module deliberately contains no Streamlit code. This keeps network access,
validation, and UI concerns separate and makes the data layer easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Callable, Mapping, Protocol

import pandas as pd


_ALLOWED_SYMBOL = re.compile(r"^[A-Z0-9.\-^=]{1,32}$")
_REQUIRED_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


class MarketDataError(RuntimeError):
    """Raised when usable market data cannot be produced."""


class InvalidSymbolError(ValueError):
    """Raised when a ticker symbol is empty or contains unsupported characters."""


class TickerLike(Protocol):
    """Small protocol describing the yfinance methods used by this module."""

    fast_info: Any

    def history(self, **kwargs: Any) -> pd.DataFrame: ...

    def get_info(self) -> Mapping[str, Any]: ...

    def get_history_metadata(self) -> Mapping[str, Any]: ...


TickerFactory = Callable[[str], TickerLike]


@dataclass(frozen=True)
class MarketSnapshot:
    """Validated data returned to the application layer."""

    symbol: str
    history: pd.DataFrame
    metadata: dict[str, Any]
    fetched_at_utc: datetime
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def latest_close(self) -> float:
        """Return the latest validated closing price."""
        return float(self.history["Close"].iloc[-1])

    @property
    def first_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.history.index[0])

    @property
    def last_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.history.index[-1])



def normalise_symbol(raw_symbol: str) -> str:
    """Normalise and validate a Yahoo Finance ticker symbol."""
    symbol = (raw_symbol or "").strip().upper()
    if not symbol:
        raise InvalidSymbolError("Enter a ticker symbol, for example AAPL or VWCE.DE.")
    if not _ALLOWED_SYMBOL.fullmatch(symbol):
        raise InvalidSymbolError(
            "Ticker contains unsupported characters. Allowed characters are "
            "letters, numbers, dot, hyphen, caret and equals sign."
        )
    return symbol



def _default_ticker_factory(symbol: str) -> TickerLike:
    """Create a yfinance Ticker lazily so unit tests do not need network access."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - deployment/configuration issue
        raise MarketDataError(
            "yfinance is not installed. Install the packages from requirements.txt."
        ) from exc

    # yfinance 1.x supports configurable retries. Keep this optional so the
    # application remains compatible if the setting changes in a later release.
    try:
        yf.config.network.retries = 2
    except (AttributeError, TypeError):
        pass

    return yf.Ticker(symbol)



def _safe_mapping(value: Any) -> dict[str, Any]:
    """Best-effort conversion of provider metadata into a normal dictionary."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)

    # yfinance FastInfo behaves like a mapping but can raise for individual keys.
    result: dict[str, Any] = {}
    try:
        keys = list(value.keys())
    except (AttributeError, TypeError, KeyError):
        return result

    for key in keys:
        try:
            result[str(key)] = value[key]
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return result



def _clean_history(raw_history: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Validate and clean provider price history."""
    if not isinstance(raw_history, pd.DataFrame) or raw_history.empty:
        raise MarketDataError(f"No price history was returned for {symbol}.")

    missing = [column for column in _REQUIRED_PRICE_COLUMNS if column not in raw_history.columns]
    if missing:
        raise MarketDataError(
            f"Price history for {symbol} is missing required columns: {', '.join(missing)}."
        )

    history = raw_history.loc[:, list(_REQUIRED_PRICE_COLUMNS)].copy()
    history.index = pd.to_datetime(history.index, errors="coerce")
    history = history.loc[~history.index.isna()]
    history = history.loc[~history.index.duplicated(keep="last")].sort_index()

    for column in _REQUIRED_PRICE_COLUMNS:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    # A row without a closing price cannot be used by indicators or strategies.
    history = history.dropna(subset=["Close"])
    if history.empty:
        raise MarketDataError(f"No valid closing prices were returned for {symbol}.")

    return history



def _read_metadata(ticker: TickerLike) -> tuple[dict[str, Any], list[str]]:
    """Load metadata without allowing a metadata failure to break price analysis."""
    metadata: dict[str, Any] = {}
    warnings: list[str] = []

    try:
        metadata.update(_safe_mapping(ticker.fast_info))
    except Exception as exc:  # provider exceptions vary between yfinance versions
        warnings.append(f"Fast metadata unavailable: {type(exc).__name__}.")

    try:
        metadata.update(_safe_mapping(ticker.get_history_metadata()))
    except Exception as exc:
        warnings.append(f"History metadata unavailable: {type(exc).__name__}.")

    try:
        # Full info is slower, but it supplies sector, industry and quote type.
        # It is deliberately optional: price analysis should still work without it.
        metadata.update(_safe_mapping(ticker.get_info()))
    except Exception as exc:
        warnings.append(f"Full company metadata unavailable: {type(exc).__name__}.")

    return metadata, warnings



def load_market_snapshot(
    raw_symbol: str,
    *,
    period: str = "2y",
    interval: str = "1d",
    min_rows: int = 2,
    ticker_factory: TickerFactory | None = None,
) -> MarketSnapshot:
    """Fetch and validate a single instrument snapshot.

    Parameters
    ----------
    raw_symbol:
        Yahoo Finance ticker, such as ``AAPL``, ``VWCE.DE`` or ``^GSPC``.
    period:
        yfinance history period. Two years is the default because the production
        app will require enough observations for a 200-session moving average.
    interval:
        Price interval. P0-P2 are designed around end-of-day data, so ``1d`` is
        the default.
    min_rows:
        Minimum usable price rows required by the caller.
    ticker_factory:
        Optional dependency injection hook used by unit tests.
    """
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1.")

    symbol = normalise_symbol(raw_symbol)
    factory = ticker_factory or _default_ticker_factory

    try:
        ticker = factory(symbol)
        raw_history = ticker.history(
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            repair=True,
            timeout=10,
            raise_errors=True,
        )
    except InvalidSymbolError:
        raise
    except Exception as exc:
        raise MarketDataError(
            f"Market data could not be downloaded for {symbol}: {type(exc).__name__}."
        ) from exc

    history = _clean_history(raw_history, symbol)
    if len(history) < min_rows:
        raise MarketDataError(
            f"Only {len(history)} usable rows were returned for {symbol}; "
            f"at least {min_rows} are required."
        )

    metadata, warnings = _read_metadata(ticker)
    metadata.setdefault("symbol", symbol)
    metadata.setdefault("regularMarketPrice", float(history["Close"].iloc[-1]))

    return MarketSnapshot(
        symbol=symbol,
        history=history,
        metadata=metadata,
        fetched_at_utc=datetime.now(timezone.utc),
        warnings=tuple(warnings),
    )
