# ai_stock_analyzer_app/main.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests
import datetime
import feedparser
from src.analysis import (
    AnalysisSnapshot,
    IndicatorSnapshot,
    build_trading_expert_report,
)
from src.data.market_data import (
    InvalidSymbolError,
    MarketDataError,
    load_market_snapshot,
)

# --- PAGE CONFIG ---
st.set_page_config(page_title="📈 AI Stock Analyzer", layout="wide")

# --- GLOBAL STYLES ---
st.markdown("""
<style>
.card {background:#ffffff; color:#222; padding:20px; margin-bottom:20px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.1);}
.card-dark {background:#2b2b2b; color:#fff; padding:20px; margin-bottom:20px; border-radius:10px;}
.metric-tooltip {text-decoration:underline; cursor:help;}
.arrow-up {color:green;}
.arrow-down {color:red;}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="card" style="text-align:center;">
    <h1 style="color:#4CAF50; margin-bottom:5px;">📊 AI Stock Analyzer</h1>
    <p style="font-size:16px; color:#555;">Interactive, non-finance friendly insights with action recommendations</p>
</div>
""", unsafe_allow_html=True)

# --- USER INPUT & PEERS ---
st.sidebar.header("Select Stock & Peers")
popular = ["BBAI","ARCC","SPCE","AAPL","MSFT","GOOGL","AMZN","QS","TSLA","NVDA"]
ticker_select = st.sidebar.selectbox("Choose from popular tickers", popular, index=popular.index("SPCE"))
ticker_input  = st.sidebar.text_input("Or enter any ticker symbol", "").upper().strip()
ticker = ticker_input or ticker_select

# --- VALIDATE AND FETCH PRIMARY DATA ---
try:
    snapshot = load_market_snapshot(
        ticker,
        period="2y",
        interval="1d",
        min_rows=200,
    )
except (InvalidSymbolError, MarketDataError) as exc:
    st.error(str(exc))
    st.stop()

# Use the validated and normalised symbol throughout the production app.
ticker = snapshot.symbol
hist = snapshot.history.copy()
info = dict(snapshot.metadata)

quote_type = str(
    info.get("quoteType")
    or info.get("instrumentType")
    or "UNKNOWN"
).upper()
is_etf = quote_type == "ETF"

# Additional yfinance endpoints are used later for dividends, financial
# statements and news. They are created only after the symbol is validated.
data = yf.Ticker(ticker)

# Auto-select peers only after validated metadata is available.
if st.sidebar.checkbox("Auto-select peers by sector/industry", True):
    sector = info.get("sector")
    industry = info.get("industry")

    industry_map = {
        'Information Technology Services': ['SOUN', 'CRNC', 'AI', 'NVDA', 'PLTR'],
        'Software—Infrastructure': ['NOW', 'CRM', 'ORCL', 'ADBE', 'SNOW'],
    }
    sector_map = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'Consumer Cyclical': ['AMZN', 'TSLA', 'BBWI'],
        'Communication Services': ['META', 'NFLX', 'DIS'],
    }

    peer_list = industry_map.get(industry) or sector_map.get(sector) or popular

    if is_etf:
        etf_peer_map = {
            "VWCE.DE": ["VWRL.AS", "IUSQ.DE", "EUNL.DE"],
            "SXR8.DE": ["CSPX.L", "VUAA.DE", "IUSA.L"],
        }
        peer_list = etf_peer_map.get(ticker, [])
else:
    peer_text = st.sidebar.text_input(
        "Or enter peers (comma separated)",
        ",".join(popular),
    )
    peer_list = [
        symbol.strip().upper()
        for symbol in peer_text.split(",")
        if symbol.strip()
    ]

# --- EXCLUDE SELECTED INSTRUMENT FROM PEERS ---
# Remove blank values, duplicates and the selected ticker itself.
cleaned_peer_list = []
seen_peer_symbols = set()

for peer_symbol in peer_list:
    candidate = str(peer_symbol or "").strip().upper()

    if not candidate:
        continue

    if candidate == ticker:
        continue

    if candidate in seen_peer_symbols:
        continue

    seen_peer_symbols.add(candidate)
    cleaned_peer_list.append(candidate)

# Keep a usable fallback if automatic or manual selection returns no peers.
peer_list = cleaned_peer_list or [
    symbol
    for symbol in popular
    if symbol != ticker
]

hist['MA20'] = hist['Close'].rolling(20).mean()
hist['MA50'] = hist['Close'].rolling(50).mean()

# --- DIVIDEND DATES FIX ---
try:
    div_dates = [dt.date() for dt in data.dividends.index]
except Exception as exc:
    div_dates = []
    st.warning(
        "Dividend dates could not be loaded "
        f"({type(exc).__name__}). Price analysis remains available."
    )

# Safe previous-close lookup
prev_close = hist['Close'].shift(1).iloc[-1]
if pd.isna(prev_close): prev_close = hist['Close'].iloc[-1]

# --- MARKET OVERVIEW & SUPPORT/RESISTANCE ---
instrument_currency = str(info.get("currency") or "N/A").upper()
financial_currency = str(
    info.get("financialCurrency") or instrument_currency
).upper()
exchange = str(
    info.get("fullExchangeName")
    or info.get("exchangeName")
    or info.get("exchange")
    or "N/A"
)

latest_market_date = snapshot.last_date.strftime("%Y-%m-%d")
fetched_at_utc = snapshot.fetched_at_utc.astimezone(
    datetime.timezone.utc
).strftime("%Y-%m-%d %H:%M UTC")

st.markdown(f"### {info.get('shortName', ticker)} ({ticker})")

financial_currency_note = (
    f" · Financial currency: **{financial_currency}**"
    if financial_currency != instrument_currency
    else ""
)

st.caption(
    f"Type: **{quote_type}**"
    f" · Currency: **{instrument_currency}**"
    f"{financial_currency_note}"
    f" · Exchange: **{exchange}**"
    f" · Latest market date: **{latest_market_date}**"
    f" · Retrieved: **{fetched_at_utc}**"
)

st.markdown(
    "<div class='card'><h2>📈 Market & Trading Overview</h2></div>",
    unsafe_allow_html=True,
)

CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CHF": "CHF ",
    "CAD": "C$",
    "AUD": "A$",
    "CNY": "CN¥",
    "HKD": "HK$",
    "SEK": "SEK ",
    "NOK": "NOK ",
    "DKK": "DKK ",
}


def numeric_or_nan(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def format_money(value, currency: str, decimals: int = 0) -> str:
    number = numeric_or_nan(value)
    if np.isnan(number):
        return "N/A"

    code = str(currency or "N/A").upper()
    symbol = CURRENCY_SYMBOLS.get(code)
    sign = "-" if number < 0 else ""
    absolute_value = abs(number)

    if symbol:
        return f"{sign}{symbol}{absolute_value:,.{decimals}f}"

    return f"{sign}{absolute_value:,.{decimals}f} {code}"


def format_ratio(value, decimals: int = 2) -> str:
    number = numeric_or_nan(value)
    if np.isnan(number):
        return "N/A"
    return f"{number:.{decimals}f}"


def format_count(value) -> str:
    number = numeric_or_nan(value)
    if np.isnan(number):
        return "N/A"
    return f"{number:,.0f}"


def dividend_yield_percent(metadata: dict) -> float:
    direct = numeric_or_nan(metadata.get("dividendYield"))
    trailing = numeric_or_nan(
        metadata.get("trailingAnnualDividendYield")
    )

    if not np.isnan(direct) and not np.isnan(trailing):
        trailing_percent = trailing * 100
        direct_as_percent = direct
        direct_as_fraction = direct * 100
        return (
            direct_as_percent
            if abs(direct_as_percent - trailing_percent)
            <= abs(direct_as_fraction - trailing_percent)
            else direct_as_fraction
        )

    if not np.isnan(direct):
        return direct * 100 if abs(direct) < 0.20 else direct

    if not np.isnan(trailing):
        return trailing * 100

    return np.nan


def comparison_arrow(left, right) -> str:
    lhs = numeric_or_nan(left)
    rhs = numeric_or_nan(right)

    if np.isnan(lhs) or np.isnan(rhs) or lhs == rhs:
        return ""
    if lhs > rhs:
        return '<span class="arrow-up">▲</span>'
    return '<span class="arrow-down">▼</span>'


vol = numeric_or_nan(info.get("volume"))
avg_vol = numeric_or_nan(info.get("averageVolume"))
mc = numeric_or_nan(info.get("marketCap"))
rev = numeric_or_nan(info.get("totalRevenue"))
beta = numeric_or_nan(info.get("beta"))
dy = dividend_yield_percent(info)

c1, c2, c3 = st.columns(3)

volume_arrow = comparison_arrow(vol, avg_vol)
average_volume_arrow = comparison_arrow(avg_vol, vol)

shares_outstanding = numeric_or_nan(info.get("sharesOutstanding"))
previous_market_cap = (
    prev_close * shares_outstanding
    if not np.isnan(shares_outstanding)
    else np.nan
)
market_cap_arrow = (
    ""
    if is_etf
    else comparison_arrow(mc, previous_market_cap)
)

peer_dividend_yields = []
peer_dividend_warnings = []

for peer_symbol in dict.fromkeys(peer_list):
    try:
        peer_metadata = yf.Ticker(peer_symbol).get_info() or {}
        peer_yield = dividend_yield_percent(peer_metadata)
        if not np.isnan(peer_yield):
            peer_dividend_yields.append(peer_yield)
    except Exception as exc:
        peer_dividend_warnings.append(
            f"{peer_symbol}: {type(exc).__name__}"
        )

if peer_dividend_warnings:
    st.warning(
        "Some peer dividend data could not be loaded: "
        + ", ".join(peer_dividend_warnings)
    )

average_peer_dividend_yield = (
    float(np.mean(peer_dividend_yields))
    if peer_dividend_yields
    else np.nan
)

dividend_arrow = comparison_arrow(
    dy,
    average_peer_dividend_yield,
)
beta_arrow = comparison_arrow(beta, 1)

market_cap_text = (
    "N/A"
    if is_etf
    else format_money(mc, instrument_currency, 0)
)
revenue_text = (
    "Not applicable to ETF"
    if is_etf
    else format_money(rev, financial_currency, 0)
)
dividend_text = "N/A" if np.isnan(dy) else f"{dy:.2f}%"
beta_text = format_ratio(beta)

c1.markdown(
    f"**Volume:** {format_count(vol)} {volume_arrow} "
    "<abbr title='Shares traded during the latest session.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)
c2.markdown(
    f"**Avg Volume:** {format_count(avg_vol)} {average_volume_arrow} "
    "<abbr title='Average recent trading volume.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)
c3.markdown(
    f"**Market Cap:** {market_cap_text} {market_cap_arrow} "
    "<abbr title='Corporate market value; unavailable for ETFs.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)

c4, c5, c6 = st.columns(3)
c4.markdown(
    f"**Revenue (TTM):** {revenue_text} "
    "<abbr title='Corporate trailing-twelve-month revenue.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)
c5.markdown(
    f"**Dividend Yield:** {dividend_text} {dividend_arrow} "
    "<abbr title='Annual dividend or distribution yield.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)
c6.markdown(
    f"**Beta:** {beta_text} {beta_arrow} "
    "<abbr title='Historical volatility relative to the wider market.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)

overview_notes = []

if not np.isnan(vol) and not np.isnan(avg_vol):
    overview_notes.append(
        "Volume was "
        f"{'above' if vol > avg_vol else 'below'} its recent average; "
        f"{'stronger interest' if vol > avg_vol else 'muted trading'}."
    )
else:
    overview_notes.append("Volume comparison is unavailable.")

if is_etf:
    overview_notes.append(
        "Corporate market-cap, revenue and profitability metrics "
        "are not applicable to this ETF."
    )
else:
    if not np.isnan(mc):
        size_label = "small" if mc < 1e9 else "mid/large"
        overview_notes.append(
            f"Market cap {format_money(mc, instrument_currency, 0)} "
            f"({size_label}-cap)."
        )
    if not np.isnan(rev):
        overview_notes.append(
            f"TTM revenue {format_money(rev, financial_currency, 0)}."
        )

if not np.isnan(dy):
    overview_notes.append(
        f"Dividend yield {dy:.2f}% "
        f"({'pays a distribution' if dy > 0 else 'no payout reported'})."
    )
else:
    overview_notes.append("Dividend yield is unavailable.")

if not np.isnan(beta):
    overview_notes.append(
        f"Beta {beta:.2f} "
        f"({'high' if beta > 1 else 'lower'} relative volatility)."
    )
else:
    overview_notes.append("Beta is unavailable.")

st.markdown(
    "<div class='card-dark'>🔍 "
    + " ".join(overview_notes)
    + "</div>",
    unsafe_allow_html=True,
)

# --- EXTENDED FUNDAMENTALS vs PEERS ---
if is_etf:
    st.markdown(
        "<div class='card'><h2>📑 Fund Structure</h2></div>",
        unsafe_allow_html=True,
    )
    st.info(
        "Corporate valuation, profitability, leverage and quarterly "
        "earnings metrics are not applicable to ETFs. "
        "ETF-specific holdings, fees, assets and tracking metrics "
        "will be added in a later implementation step."
    )
else:
    st.markdown("<div class='card'><h2>📑 Fundamental Breakdown vs Peers</h2></div>", unsafe_allow_html=True)

    # Gather peer metadata with controlled provider warnings.
    peer_info = []
    peer_info_warnings = []

    for peer_symbol in dict.fromkeys(peer_list):
        try:
            provider_info = yf.Ticker(peer_symbol).get_info() or {}
            if provider_info:
                peer_info.append(provider_info)
        except Exception as exc:
            peer_info_warnings.append(
                f"{peer_symbol}: {type(exc).__name__}"
            )

    if peer_info_warnings:
        st.warning(
            "Some peer fundamental data could not be loaded: "
            + ", ".join(peer_info_warnings)
        )

    keys = ['trailingPE','pegRatio','profitMargins','returnOnEquity','debtToEquity','enterpriseValue']
    avg_vals = {k: np.nanmean([pi.get(k) for pi in peer_info if isinstance(pi.get(k), (int,float))]) for k in keys}

    cols = st.columns(3)
    sections = {
        'Valuation'     : [('P/E Ratio','trailingPE','15–25 fair'), ('PEG Ratio','pegRatio','~1 fair')],
        'Profitability' : [('Net Margin','profitMargins','>5% profitable'), ('ROE','returnOnEquity','>15% strong')],
        'Leverage'      : [('Debt/Equity','debtToEquity','<1 manageable'), ('Enterprise Value','enterpriseValue','incl debt & cash')]
    }

    for idx, (sec, items) in enumerate(sections.items()):
        with cols[idx]:
            st.markdown(f"**{sec}**")
            for name,key,tip in items:
                val     = info.get(key)
                peer_av = avg_vals.get(key, np.nan)
                # display text
                if pd.isna(val) or pd.isna(peer_av):
                    disp, color = 'N/A','gray'
                else:
                    better = (val>=peer_av) if key!='debtToEquity' else (val<=peer_av)
                    color  = 'green' if better else 'red'
                    if name in ['Net Margin','ROE']:
                        disp = f"{val*100:.2f}%"
                    elif key=='enterpriseValue':
                        disp = format_money(val, instrument_currency, 0)
                    else:
                        disp = f"{val:.2f}"
                st.markdown(f"- {name}: <span style='color:{color};font-weight:bold'>{disp}</span> <abbr title='{tip}'>ℹ️</abbr>", unsafe_allow_html=True)

    # AI insight for fundamentals
    vd = info.get('trailingPE',np.nan) - avg_vals['trailingPE']
    pdiff = (info.get('returnOnEquity',0) - avg_vals['returnOnEquity'])*100
    ld = avg_vals['debtToEquity'] - info.get('debtToEquity',np.nan)
    notes=[]
    if not np.isnan(vd):
        notes.append("📈 Valuation attractive vs peers." if vd<0 else "⚠️ Valuation above peers.")
    if not np.isnan(pdiff):
        notes.append("👍 ROE outperforms peers." if pdiff>0 else "🔻 ROE lags peers.")
    if not np.isnan(ld):
        notes.append("🏦 Lower debt vs peers." if ld>0 else "⚠️ Higher leverage.")
    st.markdown(f"<div class='card-dark'>💡 {' '.join(notes)}</div>", unsafe_allow_html=True)

    # --- QUARTERLY EARNINGS REVIEW ---
    def render_fundamental_analysis(ticker):
        """Render correctly ordered quarterly financial results."""
        company = yf.Ticker(ticker)

        st.markdown(
            "<div class='card'><h2>📊 Quarterly Earnings Review</h2></div>",
            unsafe_allow_html=True,
        )

        try:
            financials = company.quarterly_financials.T.copy()
        except Exception as exc:
            st.warning(
                "Quarterly financial statements could not be loaded "
                f"({type(exc).__name__})."
            )
            return

        if not isinstance(financials, pd.DataFrame) or financials.empty:
            st.info(f"No quarterly financial statements are available for {ticker}.")
            return

        # Yahoo normally returns newest first, but enforce the ordering explicitly.
        financials.index = pd.to_datetime(
            financials.index,
            errors="coerce",
        )
        financials = (
            financials.loc[~financials.index.isna()]
            .sort_index(ascending=False)
        )

        metrics = [
            "Total Revenue",
            "Revenue",
            "Gross Profit",
            "Operating Income",
            "EBIT",
            "Net Income",
            "Operating Cash Flow",
        ]

        available_metrics = [
            metric
            for metric in metrics
            if metric in financials.columns
        ]

        # Avoid displaying Revenue twice when Total Revenue is also available.
        if (
            "Total Revenue" in available_metrics
            and "Revenue" in available_metrics
        ):
            available_metrics.remove("Revenue")

        if not available_metrics:
            st.info("No supported quarterly earnings metrics are available.")
            return

        quarterly_values = (
            financials.loc[:, available_metrics]
            .head(4)
            .apply(pd.to_numeric, errors="coerce")
        )

        if quarterly_values.empty:
            st.info("No usable quarterly earnings values are available.")
            return

        # Rows are newest to oldest.
        # shift(-1) places the immediately preceding quarter beside each row.
        previous_quarter = quarterly_values.shift(-1)

        # Absolute denominator handles negative income values correctly:
        # becoming more negative is deterioration, becoming less negative is improvement.
        quarterly_changes = (
            quarterly_values
            .subtract(previous_quarter)
            .divide(previous_quarter.abs())
            .multiply(100)
            .round(1)
        )

        percentage_frame = quarterly_changes.add_suffix(" % Change")

        display_frame = pd.concat(
            [quarterly_values, percentage_frame],
            axis=1,
        )

        def short_format(value):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return "-"

            if pd.isna(numeric_value):
                return "-"

            absolute_value = abs(numeric_value)

            if absolute_value >= 1_000_000_000:
                return f"{numeric_value / 1_000_000_000:.2f}B"

            if absolute_value >= 1_000_000:
                return f"{numeric_value / 1_000_000:.2f}M"

            if absolute_value >= 1_000:
                return f"{numeric_value / 1_000:.2f}K"

            return f"{numeric_value:.0f}"

        formatted_frame = display_frame.copy()

        for metric in available_metrics:
            formatted_frame[metric] = formatted_frame[metric].map(
                short_format
            )

        for percentage_column in percentage_frame.columns:
            formatted_frame[percentage_column] = (
                formatted_frame[percentage_column]
                .map(
                    lambda value: (
                        f"{value:.1f}%"
                        if pd.notna(value)
                        else "-"
                    )
                )
            )

        formatted_frame.index = (
            pd.DatetimeIndex(formatted_frame.index)
            .to_period("Q")
            .astype(str)
        )

        st.dataframe(
            formatted_frame,
            width="stretch",
        )

        # Row zero is now the latest quarter compared with the previous quarter.
        latest_changes = quarterly_changes.iloc[0]
        insights = []

        for metric in available_metrics:
            change = latest_changes.get(metric)

            if pd.isna(change):
                continue

            change = float(change)
            magnitude = abs(change)

            if magnitude < 0.05:
                insights.append(
                    f"• {metric} was broadly unchanged versus "
                    "the previous quarter."
                )
                continue

            if magnitude >= 10:
                strength = "significant"
            elif magnitude >= 5:
                strength = "notable"
            else:
                strength = "modest"

            direction = "increase" if change > 0 else "decrease"

            insights.append(
                f"• {metric} recorded a {strength} {direction} "
                f"of {magnitude:.1f}% versus the previous quarter."
            )

        summary = (
            "<br>".join(insights)
            if insights
            else "No valid latest-quarter comparisons are available."
        )

        st.markdown(
            "<div class='card-dark'>"
            "<b>💡 Earnings Insights:</b><br>"
            f"{summary}"
            "</div>",
            unsafe_allow_html=True,
        )

    render_fundamental_analysis(ticker)

# --- TECHNICAL PARAMETER CONTROLS ---
st.sidebar.header("🔧 Technical Settings")
rsi_p   = st.sidebar.slider("RSI Period", 5, 30, 14)
macd_f  = st.sidebar.slider("MACD Fast EMA", 5, 30, 12)
macd_s  = st.sidebar.slider("MACD Slow EMA", 10, 60, 26)
macd_sig= st.sidebar.slider("MACD Signal EMA", 5, 20, 9)
bb_w    = st.sidebar.slider("BB Window", 10, 60, 20)
bb_m    = st.sidebar.slider("BB Std Mult", 1.0, 3.0, 2.0)
atr_p   = st.sidebar.slider("ATR Period", 5, 30, 14)

# --- TECHNICAL INDICATORS COMPUTATION ---
hist['MA20']    = hist['Close'].rolling(20).mean()
hist['MA50']    = hist['Close'].rolling(50).mean()
hist['MA200']   = hist['Close'].rolling(200).mean()

delta          = hist['Close'].diff()
gain           = delta.clip(lower=0).rolling(rsi_p).mean()
loss           = -delta.clip(upper=0).rolling(rsi_p).mean()
hist['RSI']    = 100 - (100 / (1 + gain / loss))

hist['EMAf']   = hist['Close'].ewm(span=macd_f, adjust=False).mean()
hist['EMAs']   = hist['Close'].ewm(span=macd_s, adjust=False).mean()
hist['MACD']   = hist['EMAf'] - hist['EMAs']
hist['MACDs']  = hist['MACD'].ewm(span=macd_sig, adjust=False).mean()
hist['MACD_h'] = hist['MACD'] - hist['MACDs']

hist['BBm']    = hist['Close'].rolling(bb_w).mean()
hist['BBstd']  = hist['Close'].rolling(bb_w).std()
hist['BBu']    = hist['BBm'] + bb_m * hist['BBstd']
hist['BBl']    = hist['BBm'] - bb_m * hist['BBstd']
hist['BBpctB'] = (hist['Close'] - hist['BBl']) / (hist['BBu'] - hist['BBl'])

tr             = pd.concat([
    hist['High'] - hist['Low'],
    (hist['High'] - hist['Close'].shift()).abs(),
    (hist['Low'] - hist['Close'].shift()).abs()
], axis=1).max(axis=1)
hist['ATR']    = tr.rolling(atr_p).mean()

hist['OBV']    = (np.sign(hist['Close'].diff()) * hist['Volume']).fillna(0).cumsum()

# --- SIGNAL GENERATION ---
signals = []

for i in range(1, len(hist)):
    signal = ""
    # RSI Signal
    if hist['RSI'].iloc[i] < 30:
        signal += "RSI Buy | "
    elif hist['RSI'].iloc[i] > 70:
        signal += "RSI Sell | "

    # MACD Crossover
    if hist['MACD'].iloc[i] > hist['MACDs'].iloc[i] and hist['MACD'].iloc[i-1] <= hist['MACDs'].iloc[i-1]:
        signal += "MACD Buy | "
    elif hist['MACD'].iloc[i] < hist['MACDs'].iloc[i] and hist['MACD'].iloc[i-1] >= hist['MACDs'].iloc[i-1]:
        signal += "MACD Sell | "

    # Bollinger Band Signal
    if hist['Close'].iloc[i] < hist['BBl'].iloc[i]:
        signal += "BB Buy | "
    elif hist['Close'].iloc[i] > hist['BBu'].iloc[i]:
        signal += "BB Sell | "

    # Moving Average Crossover (MA20 vs MA50)
    if hist['MA20'].iloc[i] > hist['MA50'].iloc[i] and hist['MA20'].iloc[i-1] <= hist['MA50'].iloc[i-1]:
        signal += "MA Bullish | "
    elif hist['MA20'].iloc[i] < hist['MA50'].iloc[i] and hist['MA20'].iloc[i-1] >= hist['MA50'].iloc[i-1]:
        signal += "MA Bearish | "

    signals.append(signal.strip(" | ") if signal else "Hold")

hist['Signal'] = [""] + signals  # Add empty first signal

# --- DISPLAY SIGNAL TABLE ---
st.subheader("📊 Technical Signals Table")
st.dataframe(
    hist[['Close', 'RSI', 'MACD', 'MACDs', 'BBpctB', 'MA20', 'MA50', 'Signal']]
    .dropna()
    .tail(50)
    .iloc[::-1]  # newest data on top
)

# signals & overview
cutoff_date = hist.index.max() - pd.Timedelta(days=90)
recent = hist.loc[hist.index >= cutoff_date].copy()
sup = np.percentile(recent['Low'], 10)
res = np.percentile(recent['High'], 90)
latest = hist.iloc[-1]

# Golden / Death Cross logic
cross = ("Golden Cross ✅" if latest['MA50'] > latest['MA200']
         else "Death Cross ⚠️" if latest['MA50'] < latest['MA200']
         else "No Cross")

# RSI Signal
rsi_sig = "Overbought 🔺" if latest['RSI'] > 70 else "Oversold 🔻" if latest['RSI'] < 30 else "Neutral ⚪"

# MACD Signal
macd_sig = "Bullish 📈" if latest['MACD'] > latest['MACDs'] else "Bearish 📉" if latest['MACD'] < latest['MACDs'] else "Neutral ⚪"

# MACD Histogram
macd_hist_sig = "Increasing Momentum 🔼" if latest['MACD_h'] > 0 else "Decreasing Momentum 🔽"

# %B Signal
bb_sig = "Above Upper Band 🔺" if latest['BBpctB'] > 1 else "Below Lower Band 🔻" if latest['BBpctB'] < 0 else "Inside Bands ⚪"

# OBV Signal
obv_sig = "Rising 📊" if hist['OBV'].iloc[-1] > hist['OBV'].iloc[-10] else "Falling 📉"

# Create table
tech_df = pd.DataFrame([
    ["RSI",       f"{latest['RSI']:.1f}",              rsi_sig],
    ["MACD",      f"{latest['MACD']:.2f}",             macd_sig],
    ["MACD Hist", f"{latest['MACD_h']:.2f}",           macd_hist_sig],
    ["MA20/50/200", f"{latest['MA20']:.2f}/{latest['MA50']:.2f}/{latest['MA200']:.2f}", cross],
    ["%B",        f"{latest['BBpctB']:.2f}",           bb_sig],
    ["ATR",       f"{latest['ATR']:.2f}",              "Volatility"],
    ["OBV",       f"{int(latest['OBV'])}",             obv_sig],
    ["Support",   f"{sup:.2f}",                        "Local Support"],
    ["Resistance",f"{res:.2f}",                        "Local Resistance"],
], columns=["Indicator", "Value", "Signal"])

# Display
st.markdown("<div class='card'><h2>📈 Technical Overview</h2></div>", unsafe_allow_html=True)
st.dataframe(tech_df, width="stretch")

# Additional textual insights
ins = []
ins.append(f"RSI {latest['RSI']:.1f} ({rsi_sig}).")
ins.append(f"MACD {macd_sig} vs Signal Line.")
ins.append(f"MACD Histogram shows {macd_hist_sig}.")
ins.append(f"50/200MA: {cross}.")
ins.append(f"%B {latest['BBpctB']:.2f} of range ({bb_sig}).")
ins.append(f"ATR {latest['ATR']:.2f} indicates volatility.")
ins.append(f"OBV trend is {obv_sig.lower()}.")

st.markdown(f"<div class='card-dark'><b>📊 Technical Insights:</b><br>{'<br>'.join(ins)}</div>", unsafe_allow_html=True)


# --- TRADING EXPERT DASHBOARD ---
st.markdown(
    "<div class='card'>"
    "<h2 style='color:#222;'>🧭 Trading Expert</h2>"
    "<p style='color:#555;margin-bottom:0;'>"
    "Deterministic decision, conflict resolution and "
    "paper-only risk plan."
    "</p>"
    "</div>",
    unsafe_allow_html=True,
)

try:
    # Preserve the provider's local market-session date for display.
    expert_market_timestamp = pd.Timestamp(
        hist.index[-1]
    )
    expert_market_date = (
        expert_market_timestamp.date()
    )

    # The canonical analysis model still requires an aware timestamp.
    if expert_market_timestamp.tzinfo is None:
        expert_as_of = expert_market_timestamp.tz_localize(
            "UTC"
        )
    else:
        expert_as_of = expert_market_timestamp.tz_convert(
            "UTC"
        )

    expert_snapshot = AnalysisSnapshot(
        symbol=ticker,
        display_name=str(
            info.get("shortName")
            or info.get("longName")
            or ticker
        ),
        fetched_at_utc=snapshot.fetched_at_utc,
        history_rows=len(hist),
        indicators=IndicatorSnapshot(
            as_of=expert_as_of.to_pydatetime(),
            close=float(latest["Close"]),
            volume=float(latest["Volume"]),
            ma20=float(latest["MA20"]),
            ma50=float(latest["MA50"]),
            ma200=float(latest["MA200"]),
            rsi=float(latest["RSI"]),
            macd=float(latest["MACD"]),
            macd_signal=float(latest["MACDs"]),
            macd_histogram=float(
                latest["MACD_h"]
            ),
            bollinger_percent_b=float(
                latest["BBpctB"]
            ),
            atr=float(latest["ATR"]),
            obv=float(latest["OBV"]),
            support=float(sup),
            resistance=float(res),
        ),
        quote_type=quote_type,
        currency=instrument_currency,
        exchange=exchange,
        warnings=tuple(snapshot.warnings),
    )

    expert_report = build_trading_expert_report(
        expert_snapshot,
        hist,
        info,
    )

    expert_decision = (
        expert_report
        .risk_decision
        .recommendation
    )

    decision_styles = {
        "BUY": (
            "#123d2a",
            "#55e38f",
        ),
        "WATCH": (
            "#3d3512",
            "#ffd866",
        ),
        "HOLD": (
            "#263043",
            "#b8c7e0",
        ),
        "REDUCE": (
            "#472d16",
            "#ffad66",
        ),
        "SELL": (
            "#481f25",
            "#ff6b75",
        ),
    }

    card_background, card_text = (
        decision_styles.get(
            expert_decision.signal.value,
            ("#263043", "#ffffff"),
        )
    )

    st.markdown(
        (
            "<div style='"
            f"background:{card_background};"
            f"color:{card_text};"
            "padding:24px;"
            "border-radius:12px;"
            "margin-bottom:18px;"
            "border:1px solid rgba(255,255,255,0.12);"
            "'>"
            "<div style='font-size:14px;"
            "font-weight:700;letter-spacing:0.08em;'>"
            "FINAL DETERMINISTIC DECISION"
            "</div>"
            "<div style='font-size:42px;"
            "font-weight:800;margin:6px 0;'>"
            f"{expert_decision.signal.value}"
            "</div>"
            "<div style='font-size:16px;'>"
            f"Score {expert_decision.score:.1f}/100"
            " · "
            f"Confidence "
            f"{expert_decision.confidence * 100:.0f}%"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    decision_col1, decision_col2, decision_col3, decision_col4 = (
        st.columns(4)
    )

    decision_col1.metric(
        "Market regime",
        expert_report.regime.regime.value,
    )

    decision_col2.metric(
        "Regime confidence",
        (
            f"{expert_report.regime.confidence * 100:.0f}%"
        ),
    )

    decision_col3.metric(
        "Bullish / bearish votes",
        (
            f"{expert_report.regime.bullish_votes}"
            " / "
            f"{expert_report.regime.bearish_votes}"
        ),
    )

    decision_col4.metric(
        "Completed session",
        expert_market_date.isoformat(),
    )

    st.caption(
        "⚠️ Paper-only analytical output. "
        "No broker connection, live-order submission "
        "or automatic execution is available."
    )

    component_rows = [
        {
            "Component": trace.name,
            "Score": trace.score,
            "Deterministic calculation": (
                trace.explanation
            ),
        }
        for trace in expert_report.component_traces
    ]

    st.markdown("### Weighted score components")

    st.dataframe(
        pd.DataFrame(component_rows),
        width="stretch",
        hide_index=True,
    )

    strategy_rows = [
        {
            "Strategy": result.strategy,
            "Signal": result.signal.value,
            "Score": result.score,
            "Confidence": (
                f"{result.confidence * 100:.0f}%"
            ),
            "Vetoed": "Yes" if result.vetoed else "No",
            "Veto reason": (
                result.veto_reason or ""
            ),
        }
        for result in expert_report.strategy_results
    ]

    st.markdown(
        "### Strategy agreement and conflicts"
    )

    st.dataframe(
        pd.DataFrame(strategy_rows),
        width="stretch",
        hide_index=True,
    )

    with st.expander(
        "Evidence and conflict-resolution trace",
        expanded=False,
    ):
        evidence_rows = [
            {
                "Code": item.code,
                "Direction": item.direction.value,
                "Strength": item.strength,
                "Observed value": str(
                    item.observed_value
                ),
                "Explanation": item.message,
            }
            for item in expert_decision.evidence
        ]

        st.dataframe(
            pd.DataFrame(evidence_rows),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Market-regime reasons")

        for reason in expert_report.regime.reasons:
            st.markdown(f"- {reason}")

    risk_decision = expert_report.risk_decision

    if risk_decision.risk_vetoes:
        st.error(
            "Risk veto: "
            + " ".join(
                risk_decision.risk_vetoes
            )
        )

    paper_order = risk_decision.order

    if paper_order is not None:
        st.success(
            "✅ Risk gate passed. "
            "A paper-only setup has been generated."
        )

        order_col1, order_col2, order_col3, order_col4 = (
            st.columns(4)
        )

        order_col1.metric(
            "Entry zone",
            (
                f"{paper_order.entry_low:,.2f}"
                "–"
                f"{paper_order.entry_high:,.2f} "
                f"{instrument_currency}"
            ),
        )

        order_col2.metric(
            "Invalidation stop",
            (
                f"{paper_order.stop_price:,.2f} "
                f"{instrument_currency}"
            ),
        )

        order_col3.metric(
            "Reward / risk",
            f"{paper_order.reward_to_risk:.2f}",
        )

        order_col4.metric(
            "Expires",
            paper_order.expires_at.strftime(
                "%Y-%m-%d"
            ),
        )

        targets_frame = pd.DataFrame(
            {
                "Target": [
                    f"T{index}"
                    for index in range(
                        1,
                        len(paper_order.targets) + 1,
                    )
                ],
                "Price": [
                    (
                        f"{target:,.2f} "
                        f"{instrument_currency}"
                    )
                    for target in paper_order.targets
                ],
            }
        )

        st.dataframe(
            targets_frame,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "This setup is informational and paper-only. "
            "Its stop price is the defined invalidation point."
        )

    elif not risk_decision.risk_vetoes:
        st.info(
            "No paper order was generated because the "
            f"final decision is "
            f"{expert_decision.signal.value}."
        )

except (
    IndexError,
    KeyError,
    TypeError,
    ValueError,
) as exc:
    st.warning(
        "Trading Expert could not be calculated from "
        "the current completed-session data "
        f"({type(exc).__name__}: {exc})."
    )


# --- CONSOLIDATED MARKET CHART, NEWS AND PEER COMPARISON ---
def load_yahoo_rss_news(
    symbol: str,
) -> tuple[list[dict[str, object]], str | None]:
    """Load one normalized news feed from Yahoo Finance RSS."""
    endpoint = (
        "https://feeds.finance.yahoo.com/rss/2.0/headline"
    )
    params = {
        "s": symbol,
        "region": "US",
        "lang": "en-US",
    }

    try:
        response = requests.get(
            endpoint,
            params=params,
            timeout=8,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 BSAVCI-Stock-Analyzer/1.0"
                )
            },
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return (
            [],
            "Yahoo Finance RSS news could not be loaded "
            f"({type(exc).__name__}).",
        )

    try:
        feed = feedparser.parse(response.content)
        entries = list(getattr(feed, "entries", []) or [])
        articles: list[dict[str, object]] = []

        for entry in entries:
            published_parsed = entry.get("published_parsed")
            title = str(entry.get("title", "")).strip()

            if not published_parsed or not title:
                continue

            published_at = datetime.datetime(
                *published_parsed[:6],
                tzinfo=datetime.timezone.utc,
            )

            articles.append(
                {
                    "title": title,
                    "link": str(entry.get("link", "")).strip(),
                    "published_at": published_at,
                }
            )

        articles.sort(
            key=lambda item: item["published_at"],
            reverse=True,
        )

        if not articles:
            parse_error = getattr(feed, "bozo_exception", None)
            error_name = (
                type(parse_error).__name__
                if parse_error is not None
                else "EmptyFeed"
            )
            return (
                [],
                "Yahoo Finance RSS returned no usable headlines "
                f"({error_name}).",
            )

        return articles, None

    except (AttributeError, TypeError, ValueError) as exc:
        return (
            [],
            "Yahoo Finance RSS could not be interpreted "
            f"({type(exc).__name__}).",
        )


raw_news, news_warning = load_yahoo_rss_news(ticker)

news_cutoff = (
    datetime.datetime.now(datetime.timezone.utc)
    - datetime.timedelta(days=180)
)

recent_news = [
    article
    for article in raw_news
    if article["published_at"] >= news_cutoff
]

big_move_dates = set(
    hist.index[
        hist["Close"].pct_change().abs() > 0.05
    ].date
)

event_news = [
    article
    for article in recent_news
    if article["published_at"].date() in big_move_dates
]

last_sessions = hist.tail(60)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.72, 0.28],
    vertical_spacing=0.05,
)

fig.add_trace(
    go.Candlestick(
        x=last_sessions.index,
        open=last_sessions["Open"],
        high=last_sessions["High"],
        low=last_sessions["Low"],
        close=last_sessions["Close"],
        name="Price",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Bar(
        x=last_sessions.index,
        y=last_sessions["Volume"],
        name="Volume",
    ),
    row=2,
    col=1,
)

session_index_by_date = {
    timestamp.date(): timestamp
    for timestamp in last_sessions.index
}

annotated_dates = set()

for article in event_news:
    article_date = article["published_at"].date()
    chart_timestamp = session_index_by_date.get(article_date)

    if (
        chart_timestamp is None
        or article_date in annotated_dates
    ):
        continue

    annotated_dates.add(article_date)

    fig.add_vline(
        x=chart_timestamp,
        line_dash="dot",
        line_width=1,
        row=1,
        col=1,
    )

    fig.add_annotation(
        x=chart_timestamp,
        y=float(last_sessions["High"].max()),
        text="News",
        showarrow=True,
        arrowhead=2,
        row=1,
        col=1,
    )

fig.update_layout(
    title=f"{ticker} — latest {len(last_sessions)} sessions",
    height=650,
    xaxis_rangeslider_visible=False,
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
)

fig.update_yaxes(
    title_text=f"Price ({instrument_currency})",
    row=1,
    col=1,
)

fig.update_yaxes(
    title_text="Volume",
    row=2,
    col=1,
)

st.markdown(
    "<div class='card'>"
    "<h2 style='color:#222;'>📊 Market Chart</h2>"
    "</div>",
    unsafe_allow_html=True,
)

st.plotly_chart(
    fig,
    width="stretch",
    config={"displaylogo": False},
)


# One normalized news panel.
st.markdown(
    "<div class='card'>"
    "<h2 style='color:#222;'>📰 Market News</h2>"
    "</div>",
    unsafe_allow_html=True,
)

if news_warning:
    st.warning(news_warning)

headlines_to_display = (
    event_news
    if event_news
    else recent_news
)

if headlines_to_display:
    if event_news:
        st.caption(
            "Showing headlines published on sessions with "
            "an absolute price move above 5%."
        )
    else:
        st.caption(
            "No matching big-move headlines were found. "
            "Showing the latest available headlines."
        )

    for article in headlines_to_display[:8]:
        publication_date = article[
            "published_at"
        ].date().isoformat()
        article_title = str(article["title"])
        article_link = str(article.get("link", ""))

        safe_title = (
            article_title
            .replace("[", r"\[")
            .replace("]", r"\]")
        )

        if article_link:
            st.markdown(
                f"- **{publication_date}** "
                f"[{safe_title}]({article_link})"
            )
        else:
            st.markdown(
                f"- **{publication_date}** {safe_title}"
            )
else:
    st.info(
        "No recent Yahoo Finance RSS headlines are available "
        f"for {ticker}."
    )


# Peer comparison table. Prices retain each peer's own currency.
st.markdown(
    "<div class='card'>"
    "<h2 style='color:#222;'>🤝 Peer Comparison</h2>"
    "</div>",
    unsafe_allow_html=True,
)

peer_rows = []
peer_errors = []

for peer_symbol in dict.fromkeys(peer_list):
    try:
        peer_metadata = (
            yf.Ticker(peer_symbol).get_info() or {}
        )

        peer_price = peer_metadata.get("currentPrice")
        if peer_price is None:
            peer_price = peer_metadata.get(
                "regularMarketPrice"
            )

        peer_currency = str(
            peer_metadata.get("currency") or "N/A"
        ).upper()

        peer_pe = peer_metadata.get("trailingPE")

        if peer_price is None and peer_pe is None:
            peer_errors.append(
                f"{peer_symbol}: no usable values"
            )
            continue

        peer_rows.append(
            {
                "Ticker": peer_symbol,
                "Price": format_money(
                    peer_price,
                    peer_currency,
                    2,
                ),
                "Currency": peer_currency,
                "P/E": format_ratio(peer_pe),
            }
        )

    except Exception as exc:
        peer_errors.append(
            f"{peer_symbol}: {type(exc).__name__}"
        )

if peer_errors:
    st.warning(
        "Some peer data could not be loaded: "
        + ", ".join(peer_errors)
    )

if peer_rows:
    st.dataframe(
        pd.DataFrame(peer_rows),
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No peer comparison data are available.")

# --- FOOTER ---
st.markdown("""
<hr style="margin-top:2em;">
<div style="text-align:center"><p style="color:#888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p></div>
""", unsafe_allow_html=True)
