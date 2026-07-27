from __future__ import annotations

import pandas as pd
import pytest

from src.data.market_data import (
    InvalidSymbolError,
    MarketDataError,
    load_market_snapshot,
    normalise_symbol,
)


class FakeTicker:
    def __init__(
        self,
        history: pd.DataFrame,
        *,
        fast_info=None,
        history_metadata=None,
        info=None,
        metadata_error: bool = False,
    ):
        self._history = history
        self.fast_info = fast_info if fast_info is not None else {}
        self._history_metadata = history_metadata if history_metadata is not None else {}
        self._info = info if info is not None else {}
        self._metadata_error = metadata_error
        self.history_kwargs = None

    def history(self, **kwargs):
        self.history_kwargs = kwargs
        return self._history.copy()

    def get_history_metadata(self):
        if self._metadata_error:
            raise RuntimeError("metadata unavailable")
        return self._history_metadata

    def get_info(self):
        if self._metadata_error:
            raise RuntimeError("info unavailable")
        return self._info


def sample_history(rows: int = 5) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "Open": range(100, 100 + rows),
            "High": range(101, 101 + rows),
            "Low": range(99, 99 + rows),
            "Close": range(100, 100 + rows),
            "Volume": [1_000] * rows,
        },
        index=index,
    )


def test_normalise_symbol_accepts_common_yahoo_formats():
    assert normalise_symbol(" aapl ") == "AAPL"
    assert normalise_symbol("vwce.de") == "VWCE.DE"
    assert normalise_symbol("^gspc") == "^GSPC"
    assert normalise_symbol("eurusd=x") == "EURUSD=X"


@pytest.mark.parametrize("symbol", ["", "   ", "AAPL;DROP", "AAPL/USD", "AAPL 🤖"])
def test_normalise_symbol_rejects_invalid_input(symbol):
    with pytest.raises(InvalidSymbolError):
        normalise_symbol(symbol)


def test_load_snapshot_returns_clean_validated_data():
    raw = sample_history()
    raw = pd.concat([raw.iloc[::-1], raw.iloc[[-1]]])  # unsorted + duplicate date
    fake = FakeTicker(
        raw,
        fast_info={"currency": "USD", "exchange": "NMS"},
        history_metadata={"exchangeTimezoneName": "America/New_York"},
        info={"shortName": "Example Corp", "quoteType": "EQUITY"},
    )

    snapshot = load_market_snapshot(
        "test",
        min_rows=5,
        ticker_factory=lambda _: fake,
    )

    assert snapshot.symbol == "TEST"
    assert len(snapshot.history) == 5
    assert snapshot.history.index.is_monotonic_increasing
    assert snapshot.history.index.is_unique
    assert snapshot.latest_close == 104.0
    assert snapshot.metadata["currency"] == "USD"
    assert snapshot.metadata["shortName"] == "Example Corp"
    assert fake.history_kwargs["period"] == "2y"
    assert fake.history_kwargs["interval"] == "1d"
    assert fake.history_kwargs["raise_errors"] is True


def test_metadata_failure_does_not_block_price_analysis():
    fake = FakeTicker(sample_history(), metadata_error=True)

    snapshot = load_market_snapshot(
        "AAPL",
        ticker_factory=lambda _: fake,
    )

    assert snapshot.latest_close == 104.0
    assert snapshot.metadata["symbol"] == "AAPL"
    assert len(snapshot.warnings) == 2


def test_empty_history_returns_controlled_error():
    fake = FakeTicker(pd.DataFrame())

    with pytest.raises(MarketDataError, match="No price history"):
        load_market_snapshot("AAPL", ticker_factory=lambda _: fake)


def test_missing_required_column_returns_controlled_error():
    history = sample_history().drop(columns=["Volume"])
    fake = FakeTicker(history)

    with pytest.raises(MarketDataError, match="missing required columns: Volume"):
        load_market_snapshot("AAPL", ticker_factory=lambda _: fake)


def test_minimum_row_requirement_is_enforced():
    fake = FakeTicker(sample_history(rows=3))

    with pytest.raises(MarketDataError, match="at least 4 are required"):
        load_market_snapshot("AAPL", min_rows=4, ticker_factory=lambda _: fake)
