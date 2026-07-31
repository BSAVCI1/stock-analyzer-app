"""BSAVCI Stock Analyser — P0.2 integrated production baseline.

This checkpoint integrates the validated P0.1 market-data layer into the
production Streamlit app. It intentionally does not implement the later P1
trading-decision engine or any broker/AI-platform connection.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
import time

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from src.data.market_data import (
    InvalidSymbolError,
    MarketDataError,
    MarketSnapshot,
    load_market_snapshot,
)


# -----------------------------------------------------------------------------
# Page configuration and styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="BSAVCI Stock Analyser",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.10);
        padding: 18px;
        margin-bottom: 18px;
        border-radius: 12px;
    }
    .card-dark {
        background: rgba(43, 43, 43, 0.90);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 18px;
        margin-bottom: 18px;
        border-radius: 12px;
    }
    .positive { color: #2ecc71; font-weight: 700; }
    .negative { color: #ff5c5c; font-weight: 700; }
    .neutral  { color: #aab2bd; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

POPULAR_TICKERS = [
    "BBAI",
    "ARCC",
    "SPCE",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "QS",
    "TSLA",
    "NVDA",
    "SXR8.DE",
    "VWCE.DE",
]

INDUSTRY_PEERS: dict[str, list[str]] = {
    "Information Technology Services": ["SOUN", "CRNC", "AI", "NVDA", "PLTR"],
    "Software—Infrastructure": ["NOW", "CRM", "ORCL", "ADBE", "SNOW"],
    "Software - Infrastructure": ["NOW", "CRM", "ORCL", "ADBE", "SNOW"],
}

SECTOR_PEERS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "Consumer Cyclical": ["AMZN", "TSLA", "BBWI"],
    "Communication Services": ["META", "NFLX", "DIS", "GOOGL"],
}

CORPORATE_QUOTE_TYPES = {"EQUITY"}
ETF_QUOTE_TYPES = {"ETF", "MUTUALFUND", "INDEX"}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def as_float(value: Any, default: float = np.nan) -> float:
    """Return a finite float when possible, otherwise ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def safe_mean(values: Iterable[Any]) -> float:
    """Return a mean without emitting warnings for an empty collection."""
    cleaned = [as_float(value) for value in values]
    finite = [value for value in cleaned if np.isfinite(value)]
    return float(np.mean(finite)) if finite else np.nan


def normalise_percentage(value: Any) -> float:
    """Normalise provider ratios to display percentages.

    Yahoo fields such as dividend yield are normally decimal ratios, but some
    provider versions may already expose percentage values.
    """
    number = as_float(value)
    if not np.isfinite(number):
        return np.nan
    return number * 100 if abs(number) <= 1 else number


def currency_text(value: Any, currency: str, decimals: int = 2) -> str:
    number = as_float(value)
    if not np.isfinite(number):
        return "N/A"
    return f"{number:,.{decimals}f} {currency}"


def compact_number(value: Any, currency: str | None = None) -> str:
    number = as_float(value)
    if not np.isfinite(number):
        return "N/A"

    absolute = abs(number)
    if absolute >= 1_000_000_000_000:
        rendered = f"{number / 1_000_000_000_000:.2f}T"
    elif absolute >= 1_000_000_000:
        rendered = f"{number / 1_000_000_000:.2f}B"
    elif absolute >= 1_000_000:
        rendered = f"{number / 1_000_000:.2f}M"
    elif absolute >= 1_000:
        rendered = f"{number / 1_000:.2f}K"
    else:
        rendered = f"{number:,.2f}"

    return f"{rendered} {currency}" if currency else rendered


def percentage_text(value: Any, decimals: int = 2) -> str:
    number = as_float(value)
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.{decimals}f}%"


def comparison_label(
    value: Any,
    peer_average: Any,
    *,
    preference: str,
) -> tuple[str, str]:
    """Return a factual peer-comparison label and display class.

    ``preference`` can be ``higher``, ``lower``, ``near_one`` or ``neutral``.
    Enterprise value, for example, is a size measure and is therefore neutral.
    """
    current = as_float(value)
    average = as_float(peer_average)

    if not np.isfinite(current) or not np.isfinite(average):
        return "Peer comparison unavailable", "neutral"

    if np.isclose(current, average, rtol=0.03, atol=0.01):
        return "Near peer average", "neutral"

    relation = "Above peer average" if current > average else "Below peer average"

    if preference == "neutral":
        return relation, "neutral"
    if preference == "higher":
        return relation, "positive" if current > average else "negative"
    if preference == "lower":
        return relation, "positive" if current < average else "negative"
    if preference == "near_one":
        current_distance = abs(current - 1.0)
        peer_distance = abs(average - 1.0)
        return relation, "positive" if current_distance < peer_distance else "negative"

    return relation, "neutral"


def format_metric_value(key: str, value: Any, currency: str) -> str:
    number = as_float(value)
    if not np.isfinite(number):
        return "N/A"
    if key in {"profitMargins", "returnOnEquity"}:
        return f"{number * 100:.2f}%"
    if key == "enterpriseValue":
        return compact_number(number, currency)
    return f"{number:.2f}"


def instrument_name(snapshot: MarketSnapshot) -> str:
    metadata = snapshot.metadata
    return str(
        metadata.get("shortName")
        or metadata.get("longName")
        or metadata.get("displayName")
        or snapshot.symbol
    )


def instrument_currency(snapshot: MarketSnapshot) -> str:
    metadata = snapshot.metadata
    return str(metadata.get("currency") or metadata.get("financialCurrency") or "N/A")


def instrument_exchange(snapshot: MarketSnapshot) -> str:
    metadata = snapshot.metadata
    return str(metadata.get("exchange") or metadata.get("fullExchangeName") or "N/A")


def instrument_quote_type(snapshot: MarketSnapshot) -> str:
    return str(snapshot.metadata.get("quoteType") or "UNKNOWN").upper()


# -----------------------------------------------------------------------------
# Cached provider services
# -----------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def get_snapshot(symbol: str, period: str = "2y", min_rows: int = 2) -> MarketSnapshot:
    """Return one validated, cached provider snapshot per instrument."""
    return load_market_snapshot(
        symbol,
        period=period,
        interval="1d",
        min_rows=min_rows,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def get_quarterly_financials(symbol: str) -> pd.DataFrame:
    """Load corporate quarterly income-statement data with a safe fallback."""
    try:
        frame = yf.Ticker(symbol).quarterly_financials
    except Exception:
        return pd.DataFrame()

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()

    return frame.T.copy()


@st.cache_data(ttl=1800, show_spinner=False)
def get_yfinance_news(symbol: str) -> list[dict[str, Any]]:
    """Return normalised Yahoo Finance headlines."""
    try:
        raw_items = getattr(yf.Ticker(symbol), "news", []) or []
    except Exception:
        return []

    normalised: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = content.get("title") or item.get("title")
        if not title:
            continue

        published_timestamp: float | None = None
        raw_timestamp = item.get("providerPublishTime") or content.get("providerPublishTime")
        if raw_timestamp is not None:
            candidate = as_float(raw_timestamp)
            if np.isfinite(candidate):
                published_timestamp = candidate

        if published_timestamp is None:
            raw_date = content.get("pubDate") or content.get("displayTime")
            if raw_date:
                try:
                    parsed = pd.to_datetime(raw_date, utc=True)
                    published_timestamp = float(parsed.timestamp())
                except (TypeError, ValueError):
                    published_timestamp = None

        if published_timestamp is None:
            continue

        normalised.append(
            {
                "title": str(title),
                "providerPublishTime": published_timestamp,
                "source": "Yahoo Finance",
            }
        )

    return normalised


@st.cache_data(ttl=1800, show_spinner=False)
def get_rss_news(symbol: str) -> list[dict[str, Any]]:
    """Return Yahoo RSS headlines when the built-in endpoint is unavailable."""
    url = (
        "https://feeds.finance.yahoo.com/rss/2.0/headline"
        f"?s={symbol}&region=US&lang=en-US"
    )

    try:
        feed = feedparser.parse(url)
    except Exception:
        return []

    normalised: list[dict[str, Any]] = []
    for entry in getattr(feed, "entries", []):
        title = entry.get("title")
        published_parsed = entry.get("published_parsed")
        if not title or not published_parsed:
            continue

        normalised.append(
            {
                "title": str(title),
                "providerPublishTime": float(time.mktime(published_parsed)),
                "source": "Yahoo RSS",
            }
        )

    return normalised


@st.cache_data(ttl=1800, show_spinner=False)
def get_news(symbol: str) -> list[dict[str, Any]]:
    """Use one coherent news service with a single fallback."""
    news = get_yfinance_news(symbol)
    return news if news else get_rss_news(symbol)


# -----------------------------------------------------------------------------
# Technical calculations
# -----------------------------------------------------------------------------

def calculate_indicators(
    history: pd.DataFrame,
    *,
    rsi_period: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    bb_window: int,
    bb_multiplier: float,
    atr_period: int,
) -> pd.DataFrame:
    """Calculate the current dashboard indicators on validated OHLCV data."""
    result = history.copy()

    result["MA20"] = result["Close"].rolling(20, min_periods=20).mean()
    result["MA50"] = result["Close"].rolling(50, min_periods=50).mean()
    result["MA200"] = result["Close"].rolling(200, min_periods=200).mean()

    delta = result["Close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(
        alpha=1 / rsi_period,
        adjust=False,
        min_periods=rsi_period,
    ).mean()
    average_loss = losses.ewm(
        alpha=1 / rsi_period,
        adjust=False,
        min_periods=rsi_period,
    ).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    result["RSI"] = 100 - (100 / (1 + relative_strength))

    result["EMA_FAST"] = result["Close"].ewm(span=macd_fast, adjust=False).mean()
    result["EMA_SLOW"] = result["Close"].ewm(span=macd_slow, adjust=False).mean()
    result["MACD"] = result["EMA_FAST"] - result["EMA_SLOW"]
    result["MACD_SIGNAL"] = result["MACD"].ewm(span=macd_signal, adjust=False).mean()
    result["MACD_HIST"] = result["MACD"] - result["MACD_SIGNAL"]

    result["BB_MIDDLE"] = result["Close"].rolling(
        bb_window,
        min_periods=bb_window,
    ).mean()
    result["BB_STD"] = result["Close"].rolling(
        bb_window,
        min_periods=bb_window,
    ).std()
    result["BB_UPPER"] = result["BB_MIDDLE"] + bb_multiplier * result["BB_STD"]
    result["BB_LOWER"] = result["BB_MIDDLE"] - bb_multiplier * result["BB_STD"]
    band_width = (result["BB_UPPER"] - result["BB_LOWER"]).replace(0, np.nan)
    result["BB_PERCENT_B"] = (result["Close"] - result["BB_LOWER"]) / band_width

    true_range = pd.concat(
        [
            result["High"] - result["Low"],
            (result["High"] - result["Close"].shift(1)).abs(),
            (result["Low"] - result["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["ATR"] = true_range.ewm(
        alpha=1 / atr_period,
        adjust=False,
        min_periods=atr_period,
    ).mean()

    direction = np.sign(result["Close"].diff()).fillna(0)
    result["OBV"] = (direction * result["Volume"].fillna(0)).cumsum()

    return result


def build_indicator_events(history: pd.DataFrame) -> pd.Series:
    """Describe indicator events without claiming that they are final orders."""
    events: list[str] = []

    for index in range(len(history)):
        if index == 0:
            events.append("No prior session")
            continue

        current = history.iloc[index]
        previous = history.iloc[index - 1]
        day_events: list[str] = []

        if np.isfinite(as_float(current.get("RSI"))):
            if current["RSI"] < 30:
                day_events.append("RSI oversold")
            elif current["RSI"] > 70:
                day_events.append("RSI overbought")

        macd_values = [
            current.get("MACD"),
            current.get("MACD_SIGNAL"),
            previous.get("MACD"),
            previous.get("MACD_SIGNAL"),
        ]
        if all(np.isfinite(as_float(value)) for value in macd_values):
            if (
                current["MACD"] > current["MACD_SIGNAL"]
                and previous["MACD"] <= previous["MACD_SIGNAL"]
            ):
                day_events.append("MACD bullish crossover")
            elif (
                current["MACD"] < current["MACD_SIGNAL"]
                and previous["MACD"] >= previous["MACD_SIGNAL"]
            ):
                day_events.append("MACD bearish crossover")

        bollinger_values = [
            current.get("Close"),
            current.get("BB_LOWER"),
            current.get("BB_UPPER"),
        ]
        if all(np.isfinite(as_float(value)) for value in bollinger_values):
            if current["Close"] < current["BB_LOWER"]:
                day_events.append("Below lower Bollinger Band")
            elif current["Close"] > current["BB_UPPER"]:
                day_events.append("Above upper Bollinger Band")

        moving_average_values = [
            current.get("MA20"),
            current.get("MA50"),
            previous.get("MA20"),
            previous.get("MA50"),
        ]
        if all(np.isfinite(as_float(value)) for value in moving_average_values):
            if current["MA20"] > current["MA50"] and previous["MA20"] <= previous["MA50"]:
                day_events.append("MA20 crossed above MA50")
            elif current["MA20"] < current["MA50"] and previous["MA20"] >= previous["MA50"]:
                day_events.append("MA20 crossed below MA50")

        events.append(" | ".join(day_events) if day_events else "No new event")

    return pd.Series(events, index=history.index, name="Indicator Event")


# -----------------------------------------------------------------------------
# Header and input controls
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div class="card" style="text-align:center;">
        <h1 style="margin-bottom:5px;">📊 BSAVCI Stock Analyser</h1>
        <p style="font-size:16px; margin-bottom:0;">
            Validated market data, fundamentals and technical research.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Select instrument")
    selected_ticker = st.selectbox(
        "Popular instruments",
        POPULAR_TICKERS,
        index=POPULAR_TICKERS.index("SPCE"),
    )
    entered_ticker = st.text_input("Or enter a Yahoo ticker", "").strip().upper()
    ticker = entered_ticker or selected_ticker

    auto_select_peers = st.checkbox("Auto-select peers", value=True)
    manual_peer_text = ""
    if not auto_select_peers:
        manual_peer_text = st.text_input(
            "Peers, comma separated",
            ",".join(POPULAR_TICKERS[:6]),
        )

    st.divider()
    st.header("Technical settings")
    rsi_period = st.slider("RSI period", 5, 30, 14)
    macd_fast = st.slider("MACD fast EMA", 5, 30, 12)
    macd_slow = st.slider("MACD slow EMA", 10, 60, 26)
    macd_signal_period = st.slider("MACD signal EMA", 5, 20, 9)
    bb_window = st.slider("Bollinger window", 10, 60, 20)
    bb_multiplier = st.slider("Bollinger standard deviations", 1.0, 3.0, 2.0)
    atr_period = st.slider("ATR period", 5, 30, 14)


# -----------------------------------------------------------------------------
# Validated primary snapshot
# -----------------------------------------------------------------------------

try:
    with st.spinner(f"Loading and validating {ticker}..."):
        snapshot = get_snapshot(ticker, period="2y", min_rows=2)
except InvalidSymbolError as exc:
    st.error(str(exc))
    st.stop()
except MarketDataError as exc:
    st.error(str(exc))
    st.caption("This is a controlled provider error; the dashboard has not crashed.")
    st.stop()
except Exception as exc:
    st.error(f"Unexpected application error: {type(exc).__name__}.")
    st.exception(exc)
    st.stop()

history = snapshot.history.copy()
metadata = snapshot.metadata
name = instrument_name(snapshot)
currency = instrument_currency(snapshot)
exchange = instrument_exchange(snapshot)
quote_type = instrument_quote_type(snapshot)

if snapshot.warnings:
    with st.expander("Provider warnings"):
        for warning in snapshot.warnings:
            st.warning(warning)

if auto_select_peers:
    industry = metadata.get("industry")
    sector = metadata.get("sector")
    candidate_peers = INDUSTRY_PEERS.get(str(industry), []) or SECTOR_PEERS.get(str(sector), [])
    if not candidate_peers:
        candidate_peers = POPULAR_TICKERS
else:
    candidate_peers = [
        value.strip().upper()
        for value in manual_peer_text.split(",")
        if value.strip()
    ]

peer_list: list[str] = []
for peer in candidate_peers:
    if peer != snapshot.symbol and peer not in peer_list:
        peer_list.append(peer)
    if len(peer_list) == 5:
        break


# -----------------------------------------------------------------------------
# Market overview
# -----------------------------------------------------------------------------

st.subheader(f"{name} ({snapshot.symbol})")
st.caption(
    f"{quote_type} · {exchange} · {currency} · "
    f"data through {snapshot.last_date.date().isoformat()} · "
    f"fetched {snapshot.fetched_at_utc.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
)

latest_row = history.iloc[-1]
previous_close = as_float(history["Close"].iloc[-2]) if len(history) > 1 else snapshot.latest_close
latest_close = snapshot.latest_close
absolute_change = latest_close - previous_close
percentage_change = (absolute_change / previous_close * 100) if previous_close else np.nan
latest_volume = as_float(latest_row.get("Volume"), default=0.0)
average_volume_30 = as_float(history["Volume"].tail(30).mean(), default=0.0)

market_cap = as_float(metadata.get("marketCap"))
total_revenue = as_float(metadata.get("totalRevenue"))
dividend_yield = normalise_percentage(metadata.get("dividendYield"))
beta = as_float(metadata.get("beta"))

row_one = st.columns(4)
row_one[0].metric(
    "Latest close",
    currency_text(latest_close, currency),
    percentage_text(percentage_change),
)
row_one[1].metric(
    "Latest volume",
    compact_number(latest_volume),
    f"{(latest_volume / average_volume_30 - 1) * 100:.1f}% vs 30-day avg"
    if average_volume_30 > 0
    else None,
)
row_one[2].metric("Validated sessions", f"{len(history):,}")
row_one[3].metric("Quote type", quote_type)

row_two = st.columns(4)
row_two[0].metric("Market capitalisation", compact_number(market_cap, currency))
row_two[1].metric("Revenue, trailing 12 months", compact_number(total_revenue, currency))
row_two[2].metric("Dividend yield", percentage_text(dividend_yield))
row_two[3].metric("Beta", f"{beta:.2f}" if np.isfinite(beta) else "N/A")

interest_text = "above" if latest_volume > average_volume_30 else "below"
volatility_text = (
    "higher than the wider market"
    if np.isfinite(beta) and beta > 1
    else "lower than or similar to the wider market"
    if np.isfinite(beta)
    else "not available"
)

st.markdown(
    "<div class='card-dark'>"
    f"Latest volume was <b>{interest_text}</b> its 30-session average. "
    f"Historical beta is <b>{volatility_text}</b>. "
    "These observations describe the instrument; they are not a final trade order."
    "</div>",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached peer data
# -----------------------------------------------------------------------------

peer_snapshots: list[MarketSnapshot] = []
peer_failures: list[str] = []

for peer_symbol in peer_list:
    try:
        peer_snapshots.append(get_snapshot(peer_symbol, period="1y", min_rows=2))
    except (InvalidSymbolError, MarketDataError):
        peer_failures.append(peer_symbol)
    except Exception:
        peer_failures.append(peer_symbol)

if peer_failures:
    st.caption(f"Peer data unavailable for: {', '.join(peer_failures)}")

peer_metadata = [peer.metadata for peer in peer_snapshots]


# -----------------------------------------------------------------------------
# Fundamental comparison
# -----------------------------------------------------------------------------

st.markdown("<div class='card'><h2>📑 Fundamental comparison</h2></div>", unsafe_allow_html=True)

if quote_type in ETF_QUOTE_TYPES:
    st.info(
        "Corporate P/E, margins, return on equity and leverage are not scored for "
        "this ETF/index instrument. ETF-specific holdings and exposure analysis will "
        "be added in a later checkpoint."
    )
else:
    metric_definitions = [
        ("P/E ratio", "trailingPE", "lower"),
        ("PEG ratio", "pegRatio", "near_one"),
        ("Net margin", "profitMargins", "higher"),
        ("Return on equity", "returnOnEquity", "higher"),
        ("Debt/equity", "debtToEquity", "lower"),
        ("Enterprise value", "enterpriseValue", "neutral"),
    ]

    fundamental_rows: list[dict[str, str]] = []
    for label, key, preference in metric_definitions:
        current_value = metadata.get(key)
        peer_average = safe_mean(peer.get(key) for peer in peer_metadata)
        comparison, css_class = comparison_label(
            current_value,
            peer_average,
            preference=preference,
        )
        fundamental_rows.append(
            {
                "Metric": label,
                "Instrument": format_metric_value(key, current_value, currency),
                "Peer average": format_metric_value(key, peer_average, currency),
                "Interpretation": comparison,
                "Status": css_class,
            }
        )

    fundamental_frame = pd.DataFrame(fundamental_rows)
    st.dataframe(
        fundamental_frame.drop(columns=["Status"]),
        use_container_width=True,
        hide_index=True,
    )

    notes: list[str] = []
    for row in fundamental_rows:
        if row["Status"] == "positive":
            notes.append(f"✅ {row['Metric']}: {row['Interpretation'].lower()}.")
        elif row["Status"] == "negative":
            notes.append(f"⚠️ {row['Metric']}: {row['Interpretation'].lower()}.")

    st.markdown(
        "<div class='card-dark'><b>Fundamental observations</b><br>"
        + ("<br>".join(notes) if notes else "No reliable peer advantage could be calculated.")
        + "</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Quarterly financial review
# -----------------------------------------------------------------------------

st.markdown("<div class='card'><h2>📊 Quarterly financial review</h2></div>", unsafe_allow_html=True)

if quote_type not in CORPORATE_QUOTE_TYPES:
    st.info("Quarterly corporate income-statement analysis is not applicable to this instrument type.")
else:
    quarterly = get_quarterly_financials(snapshot.symbol)
    desired_metrics = [
        "Total Revenue",
        "Revenue",
        "Gross Profit",
        "Operating Income",
        "EBIT",
        "Net Income",
    ]
    available_metrics = [metric for metric in desired_metrics if metric in quarterly.columns]

    if quarterly.empty or not available_metrics:
        st.info("Quarterly financial data is currently unavailable from the provider.")
    else:
        quarterly = quarterly.loc[:, available_metrics].copy()
        quarterly.index = pd.to_datetime(quarterly.index, errors="coerce")
        quarterly = quarterly.loc[~quarterly.index.isna()].sort_index()
        quarterly = quarterly.tail(5)

        qoq_changes = quarterly.pct_change(fill_method=None) * 100
        latest_four = quarterly.tail(4)
        latest_changes = qoq_changes.reindex(latest_four.index)

        display_frame = pd.DataFrame(index=latest_four.index)
        for metric in available_metrics:
            display_frame[metric] = latest_four[metric].map(compact_number)
            display_frame[f"{metric} QoQ"] = latest_changes[metric].map(
                lambda value: percentage_text(value, 1)
            )

        display_frame.index = display_frame.index.to_period("Q").astype(str)
        display_frame = display_frame.iloc[::-1]
        st.dataframe(display_frame, use_container_width=True)

        latest_period = latest_four.index[-1]
        financial_notes: list[str] = []
        for metric in available_metrics:
            change = as_float(qoq_changes.loc[latest_period, metric])
            if not np.isfinite(change):
                continue
            direction = "increased" if change > 0 else "decreased" if change < 0 else "was unchanged"
            financial_notes.append(
                f"• {metric} {direction} by {abs(change):.1f}% quarter over quarter."
            )

        st.markdown(
            "<div class='card-dark'><b>Latest-quarter observations</b><br>"
            + ("<br>".join(financial_notes) if financial_notes else "No comparable prior quarter was available.")
            + "</div>",
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# Technical indicators and events
# -----------------------------------------------------------------------------

indicator_history = calculate_indicators(
    history,
    rsi_period=rsi_period,
    macd_fast=macd_fast,
    macd_slow=macd_slow,
    macd_signal=macd_signal_period,
    bb_window=bb_window,
    bb_multiplier=bb_multiplier,
    atr_period=atr_period,
)
indicator_history["Indicator Event"] = build_indicator_events(indicator_history)

st.markdown("<div class='card'><h2>📈 Technical overview</h2></div>", unsafe_allow_html=True)

latest_indicator = indicator_history.iloc[-1]
cutoff_90_days = indicator_history.index.max() - pd.Timedelta(days=90)
recent_90 = indicator_history.loc[indicator_history.index >= cutoff_90_days]
lower_boundary = as_float(recent_90["Low"].quantile(0.10))
upper_boundary = as_float(recent_90["High"].quantile(0.90))

ma50 = as_float(latest_indicator.get("MA50"))
ma200 = as_float(latest_indicator.get("MA200"))
if not np.isfinite(ma50) or not np.isfinite(ma200):
    cross_status = "Insufficient history"
elif ma50 > ma200:
    cross_status = "MA50 above MA200"
elif ma50 < ma200:
    cross_status = "MA50 below MA200"
else:
    cross_status = "MA50 equals MA200"

rsi_value = as_float(latest_indicator.get("RSI"))
if not np.isfinite(rsi_value):
    rsi_status = "Unavailable"
elif rsi_value > 70:
    rsi_status = "Overbought zone"
elif rsi_value < 30:
    rsi_status = "Oversold zone"
else:
    rsi_status = "Neutral zone"

macd_value = as_float(latest_indicator.get("MACD"))
macd_signal_value = as_float(latest_indicator.get("MACD_SIGNAL"))
if not np.isfinite(macd_value) or not np.isfinite(macd_signal_value):
    macd_status = "Unavailable"
elif macd_value > macd_signal_value:
    macd_status = "MACD above signal line"
elif macd_value < macd_signal_value:
    macd_status = "MACD below signal line"
else:
    macd_status = "MACD on signal line"

obv_status = "Unavailable"
if len(indicator_history) >= 10:
    current_obv = as_float(indicator_history["OBV"].iloc[-1])
    previous_obv = as_float(indicator_history["OBV"].iloc[-10])
    if np.isfinite(current_obv) and np.isfinite(previous_obv):
        obv_status = "Rising" if current_obv > previous_obv else "Falling"

technical_frame = pd.DataFrame(
    [
        ["RSI", f"{rsi_value:.1f}" if np.isfinite(rsi_value) else "N/A", rsi_status],
        ["MACD", f"{macd_value:.3f}" if np.isfinite(macd_value) else "N/A", macd_status],
        [
            "MACD histogram",
            f"{as_float(latest_indicator.get('MACD_HIST')):.3f}"
            if np.isfinite(as_float(latest_indicator.get("MACD_HIST")))
            else "N/A",
            "Momentum spread",
        ],
        [
            "MA20 / MA50 / MA200",
            " / ".join(
                f"{as_float(latest_indicator.get(column)):.2f}"
                if np.isfinite(as_float(latest_indicator.get(column)))
                else "N/A"
                for column in ["MA20", "MA50", "MA200"]
            ),
            cross_status,
        ],
        [
            "Bollinger %B",
            f"{as_float(latest_indicator.get('BB_PERCENT_B')):.2f}"
            if np.isfinite(as_float(latest_indicator.get("BB_PERCENT_B")))
            else "N/A",
            "Position within volatility bands",
        ],
        [
            "ATR",
            currency_text(latest_indicator.get("ATR"), currency),
            "Average price movement",
        ],
        ["OBV", compact_number(latest_indicator.get("OBV")), obv_status],
        ["90-day lower statistical boundary", currency_text(lower_boundary, currency), "10th percentile of lows"],
        ["90-day upper statistical boundary", currency_text(upper_boundary, currency), "90th percentile of highs"],
    ],
    columns=["Indicator", "Value", "Interpretation"],
)

st.dataframe(technical_frame, use_container_width=True, hide_index=True)

st.warning(
    "Indicator events below are descriptive flags only. The P1 strategy engine will "
    "later resolve conflicts and generate one evidence-based BUY, WATCH, HOLD, REDUCE "
    "or SELL decision."
)

signal_columns = [
    "Close",
    "RSI",
    "MACD",
    "MACD_SIGNAL",
    "BB_PERCENT_B",
    "MA20",
    "MA50",
    "MA200",
    "Indicator Event",
]
st.dataframe(
    indicator_history.loc[:, signal_columns].tail(50).iloc[::-1],
    use_container_width=True,
)


# -----------------------------------------------------------------------------
# Price, volume and news chart
# -----------------------------------------------------------------------------

raw_news = get_news(snapshot.symbol)
six_month_cutoff = datetime.now() - timedelta(days=180)
filtered_news = [
    item
    for item in raw_news
    if datetime.fromtimestamp(item["providerPublishTime"]) >= six_month_cutoff
]

big_move_days = set(
    indicator_history.index[indicator_history["Close"].pct_change().abs() > 0.05].date
)
last_60 = indicator_history.tail(60)
event_news = [
    item
    for item in filtered_news
    if datetime.fromtimestamp(item["providerPublishTime"]).date() in big_move_days
]

figure = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.72, 0.28],
    vertical_spacing=0.05,
)
figure.add_trace(
    go.Candlestick(
        x=last_60.index,
        open=last_60["Open"],
        high=last_60["High"],
        low=last_60["Low"],
        close=last_60["Close"],
        name="Price",
    ),
    row=1,
    col=1,
)
figure.add_trace(
    go.Bar(
        x=last_60.index,
        y=last_60["Volume"],
        name="Volume",
    ),
    row=2,
    col=1,
)

for item in event_news:
    event_date = datetime.fromtimestamp(item["providerPublishTime"]).date()
    if event_date in last_60.index.date:
        figure.add_vline(
            x=pd.Timestamp(event_date),
            line_dash="dot",
            row=1,
            col=1,
        )

figure.update_layout(
    title=f"{snapshot.symbol}: latest 60 sessions",
    xaxis_rangeslider_visible=False,
    height=700,
    legend_orientation="h",
)
st.plotly_chart(figure, use_container_width=True)

st.markdown("<div class='card'><h2>📰 News around large price moves</h2></div>", unsafe_allow_html=True)
if event_news:
    for item in sorted(
        event_news,
        key=lambda value: value["providerPublishTime"],
        reverse=True,
    )[:15]:
        published = datetime.fromtimestamp(item["providerPublishTime"]).date().isoformat()
        st.markdown(f"- **{published}** — {item['title']} *({item['source']})*")
else:
    st.info("No matching headlines were found for >5% daily moves during the selected period.")


# -----------------------------------------------------------------------------
# Peer comparison
# -----------------------------------------------------------------------------

st.markdown("<div class='card'><h2>🤝 Peer comparison</h2></div>", unsafe_allow_html=True)

peer_rows: list[dict[str, Any]] = []
for peer_snapshot in peer_snapshots:
    peer_currency = instrument_currency(peer_snapshot)
    peer_rows.append(
        {
            "Ticker": peer_snapshot.symbol,
            "Instrument": instrument_name(peer_snapshot),
            "Latest close": peer_snapshot.latest_close,
            "Currency": peer_currency,
            "P/E": as_float(peer_snapshot.metadata.get("trailingPE")),
            "Quote type": instrument_quote_type(peer_snapshot),
        }
    )

if not peer_rows:
    st.info("No peer data is available for this selection.")
else:
    peer_frame = pd.DataFrame(peer_rows).set_index("Ticker")
    st.dataframe(peer_frame, use_container_width=True)

    pe_values = peer_frame["P/E"].dropna()
    if not pe_values.empty:
        st.caption("P/E comparison for peers with available corporate valuation data")
        st.bar_chart(pe_values)


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.markdown(
    """
    <hr style="margin-top:2em;">
    <div style="text-align:center">
        <p style="color:#888;">
            Created by <b>BSAVCI1</b> · P0.2 validated-data integration ·
            Powered by Streamlit and Yahoo Finance
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
