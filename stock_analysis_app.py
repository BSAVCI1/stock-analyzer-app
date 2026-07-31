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
    f"Currency: **{instrument_currency}**"
    f"{financial_currency_note}"
    f" · Exchange: **{exchange}**"
    f" · Latest market date: **{latest_market_date}**"
    f" · Retrieved: **{fetched_at_utc}**"
)

st.markdown(
    "<div class='card'><h2>📈 Market & Trading Overview</h2></div>",
    unsafe_allow_html=True,
)

vol = info.get("volume") or 0
avg_vol = info.get("averageVolume") or 0
mc = info.get("marketCap") or 0
rev = info.get("totalRevenue") or 0
dy = (info.get("dividendYield") or 0) * 100
beta = info.get("beta") or 0

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


def format_money(value, currency: str, decimals: int = 0) -> str:
    """Format a monetary value using its actual provider currency."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if not np.isfinite(number):
        return "N/A"

    code = str(currency or "N/A").upper()
    symbol = CURRENCY_SYMBOLS.get(code)
    sign = "-" if number < 0 else ""
    absolute_value = abs(number)

    if symbol:
        return f"{sign}{symbol}{absolute_value:,.{decimals}f}"

    return f"{sign}{absolute_value:,.{decimals}f} {code}"


def format_ratio(value, decimals: int = 2) -> str:
    """Safely format a numeric ratio."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if not np.isfinite(number):
        return "N/A"

    return f"{number:.{decimals}f}"


def arrow_markup(condition: bool) -> str:
    """Return safe HTML for a directional indicator."""
    if condition:
        return '<span class="arrow-up">▲</span>'
    return '<span class="arrow-down">▼</span>'


c1, c2, c3 = st.columns(3)

volume_arrow = arrow_markup(vol > avg_vol)
average_volume_arrow = arrow_markup(avg_vol > vol)

shares_outstanding = info.get("sharesOutstanding") or 0
previous_market_cap = prev_close * shares_outstanding
market_cap_arrow = arrow_markup(mc > previous_market_cap)

c1.markdown(
    f"**Volume:** {vol:,} {volume_arrow} "
    "<abbr title='Shares traded during the latest session.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)

c2.markdown(
    f"**Avg Volume:** {avg_vol:,} {average_volume_arrow} "
    "<abbr title='Average recent trading volume.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)

c3.markdown(
    f"**Market Cap:** {format_money(mc, instrument_currency, 0)} {market_cap_arrow} "
    "<abbr title='Total market value of company equity.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)


c4, c5, c6 = st.columns(3)

revenue_arrow = arrow_markup(rev > previous_market_cap)

peer_dividend_yields = []
peer_dividend_warnings = []

for peer_symbol in dict.fromkeys(peer_list):
    try:
        peer_metadata = yf.Ticker(peer_symbol).get_info() or {}
        peer_yield = peer_metadata.get("dividendYield")

        if isinstance(peer_yield, (int, float)):
            peer_dividend_yields.append(peer_yield * 100)
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
    float(np.nanmean(peer_dividend_yields))
    if peer_dividend_yields
    else np.nan
)

dividend_arrow = arrow_markup(
    not np.isnan(average_peer_dividend_yield)
    and dy > average_peer_dividend_yield
)

beta_arrow = arrow_markup(beta > 1)

c4.markdown(
    f"**Revenue (TTM):** {format_money(rev, financial_currency, 0)} {revenue_arrow} "
    "<abbr title='Revenue reported for the trailing twelve months.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)

c5.markdown(
    f"**Dividend Yield:** {dy:.2f}% {dividend_arrow} "
    "<abbr title='Annual dividend yield.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)

c6.markdown(
    f"**Beta:** {beta:.2f} {beta_arrow} "
    "<abbr title='Historical volatility relative to the wider market.'>ℹ️</abbr>",
    unsafe_allow_html=True,
)


ins = (
    f"Volume was {'above' if vol>avg_vol else 'below'} its 30-day avg; "
    f"{'strong interest' if vol>avg_vol else 'muted trading'}. "
    f"Market cap {format_money(mc, instrument_currency, 0)} "
    f"({'small' if mc < 1e9 else 'mid/large'}-cap). "
    f"TTM revenue {format_money(rev, financial_currency, 0)}; "
    f"Dividend yield {dy:.2f}% ({'pays' if dy>0 else 'no payout'}); "
    f"Beta {beta:.2f} ({'high' if beta>1 else 'low'} volatility)."
)
st.markdown(f"<div class='card-dark'>🔍 {ins}</div>", unsafe_allow_html=True)

# --- EXTENDED FUNDAMENTALS vs PEERS ---
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
    data = yf.Ticker(ticker)
    st.markdown("<div class='card'><h2>📊 Quarterly Earnings Review</h2></div>", unsafe_allow_html=True)

    try:
        df = data.quarterly_financials.T
    except Exception as exc:
        st.warning(
            "Quarterly financial statements could not be loaded "
            f"({type(exc).__name__})."
        )
        return

    if df.empty:
        st.warning(
            "No quarterly financial statements are available "
            f"for {ticker}."
        )
        return
    metrics = ['Total Revenue','Revenue','Gross Profit','Operating Income','EBIT','Net Income','Operating Cash Flow']
    avail   = [m for m in metrics if m in df.columns]
    df_q     = df[avail].iloc[:4]
    df_q.index = pd.to_datetime(df_q.index).to_period('Q').astype(str)

    # compute QoQ %
    df_pct = (df_q.pct_change()*100).round(1)
    df_pct.columns = [f"{c} % Change" for c in df_pct.columns]

    df_show = pd.concat([df_q, df_pct],axis=1)

    def short_fmt(x):
        try: x=float(x)
        except (TypeError, ValueError): return "-"
        if abs(x)>=1e9: return f"{x/1e9:.2f}B"
        if abs(x)>=1e6: return f"{x/1e6:.2f}M"
        if abs(x)>=1e3: return f"{x/1e3:.2f}K"
        return f"{x:.0f}"

    df_fmt = df_show.copy()
    for c in avail: df_fmt[c] = df_fmt[c].apply(short_fmt)
    for c in df_pct.columns: df_fmt[c] = df_fmt[c].apply(lambda v: f"{v:.1f}%" if pd.notna(v) else "-")

    st.dataframe(df_fmt, width="stretch")

    # insights
    latest = df_pct.index[-1]
    prev   = df_pct.index[-2] if len(df_pct)>1 else None
    ins    = []
    def senti(ch):
        if ch>5: return "strong growth"
        if ch>0: return "modest increase"
        if ch>-5: return "slight decline"
        return "notable decrease"

    if prev:
        for m in avail:
            key = f"{m} % Change"
            if key in df_pct.columns:
                ch = df_pct.loc[latest,key]
                ins.append(f"• {m} {senti(ch)} of {abs(ch):.1f}% this quarter.")
        # analyst style
        rc = df_pct.loc[latest,"Revenue % Change"] if "Revenue % Change" in df_pct else None
        if rc is not None:
            mood = "bullish" if rc>0 else "cautious"
            ins.append(f"🧐 Analysts are {mood} on rev after a {abs(rc):.1f}% {'rise' if rc>0 else 'drop'}.")

    summary = "<br>".join(ins) if ins else "No significant quarter-over-quarter changes."
    st.markdown(f"<div class='card-dark'><b>💡 Earnings Insights:</b><br>{summary}</div>", unsafe_allow_html=True)

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
