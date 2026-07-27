"""Isolated P0.1 diagnostic app.

Run this before integrating the new data layer into the existing 503-line app:
    streamlit run step_01_data_check.py
"""

from __future__ import annotations

import streamlit as st

from src.data.market_data import InvalidSymbolError, MarketDataError, load_market_snapshot


st.set_page_config(page_title="BSAVCI Data Check", layout="wide")
st.title("BSAVCI Trading Expert — Step P0.1")
st.caption("Safe market-data loading and validation. No trading signal is generated in this step.")


@st.cache_data(ttl=900, show_spinner=False)
def get_snapshot(symbol: str, period: str):
    """Cache external data for 15 minutes between Streamlit reruns."""
    return load_market_snapshot(symbol, period=period, interval="1d", min_rows=2)


with st.sidebar:
    st.header("Data test")
    symbol = st.text_input("Ticker", value="AAPL").strip()
    period = st.selectbox("History period", options=["2y", "5y", "10y"], index=0)
    run_check = st.button("Load and validate", type="primary", use_container_width=True)
    st.markdown(
        "**Acceptance tickers**\n\n"
        "`AAPL` · `SPCE` · `SXR8.DE` · `VWCE.DE` · one invalid ticker"
    )

if not run_check:
    st.info("Enter a ticker and select **Load and validate**.")
    st.stop()

try:
    with st.spinner(f"Loading {symbol.upper()}..."):
        snapshot = get_snapshot(symbol, period)
except InvalidSymbolError as exc:
    st.error(str(exc))
    st.stop()
except MarketDataError as exc:
    st.error(str(exc))
    st.caption("This is a controlled data error; the app has not crashed.")
    st.stop()
except Exception as exc:
    st.exception(exc)
    st.stop()

history = snapshot.history
metadata = snapshot.metadata

st.success(f"{snapshot.symbol} loaded and validated successfully.")

name = metadata.get("shortName") or metadata.get("longName") or snapshot.symbol
currency = metadata.get("currency") or metadata.get("financialCurrency") or "Unknown"
exchange = metadata.get("exchange") or metadata.get("fullExchangeName") or "Unknown"
quote_type = metadata.get("quoteType") or "Unknown"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Instrument", str(name))
c2.metric("Latest close", f"{snapshot.latest_close:,.2f} {currency}")
c3.metric("Validated rows", f"{len(history):,}")
c4.metric("Quote type", str(quote_type))

st.write(
    {
        "symbol": snapshot.symbol,
        "exchange": exchange,
        "currency": currency,
        "first_price_date": snapshot.first_date.isoformat(),
        "last_price_date": snapshot.last_date.isoformat(),
        "fetched_at_utc": snapshot.fetched_at_utc.isoformat(),
    }
)

if len(history) >= 200:
    st.success("There are enough observations to calculate a 200-session moving average.")
else:
    st.warning(
        f"Only {len(history)} observations are available. The later indicator layer "
        "must not display MA200 until at least 200 observations exist."
    )

if snapshot.warnings:
    with st.expander("Provider warnings"):
        for warning in snapshot.warnings:
            st.warning(warning)

st.subheader("Closing-price history")
st.line_chart(history["Close"])

st.subheader("Latest validated rows")
st.dataframe(history.tail(10).iloc[::-1], use_container_width=True)
