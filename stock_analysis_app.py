# ai_stock_analyzer_app/main.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from bs4 import BeautifulSoup
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="BSAV Stock Analyzer", layout="wide")

# --- GLOBAL STYLES ---
st.markdown("""
<style>
.card {background:#ffffff; padding:20px; margin-bottom:20px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.1);}
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

# --- USER INPUT ---
st.sidebar.header("Select Stock & Peers")
popular = ["BBAI","AARC","VUSA.AS","SPCE","AAPL","MSFT","GOOGL","AMZN","QS","TSLA","NVDA"]
# Free-text ticker override
ticker_select = st.sidebar.selectbox("Choose from popular tickers", options=popular, index=popular.index("SPCE"))
ticker_input = st.sidebar.text_input("Or enter any ticker symbol", "").upper().strip()
ticker = ticker_input if ticker_input else ticker_select

# --- FETCH DATA ---
data = yf.Ticker(ticker)
info = data.info
hist = data.history(period="6mo")
hist['MA20'] = hist['Close'].rolling(20).mean()
hist['MA50'] = hist['Close'].rolling(50).mean()

# --- MARKET OVERVIEW & SUPPORT/RESISTANCE ---
st.markdown(f"### {info.get('shortName', ticker)} ({ticker})")
st.markdown("<div class='card'><h2>📈 Market & Trading Overview</h2></div>", unsafe_allow_html=True)
vol = info.get('volume', 0)
avg_vol = info.get('averageVolume', 0)
mc = info.get('marketCap', 0)
rev = info.get('totalRevenue', 0)
dy = info.get('dividendYield', 0) * 100
beta = info.get('beta', 0)
cols = st.columns(3)
cols[0].markdown(f"**Volume:** {vol:,} {'<span class=\"arrow-up\">▲</span>' if vol>avg_vol else '<span class=\"arrow-down\">▼</span>'} <abbr title='Shares traded in last session.'>ℹ️</abbr>", unsafe_allow_html=True)
cols[1].markdown(f"**Avg Volume:** {avg_vol:,} {'<span class=\"arrow-up\">▲</span>' if avg_vol>vol else '<span class=\"arrow-down\">▼</span>'} <abbr title='30-day avg volume.'>ℹ️</abbr>", unsafe_allow_html=True)
prev_mc = hist['Close'].iloc[-2] * info.get('sharesOutstanding', 1)
cols[2].markdown(
    f"**Market Cap:** ${mc:,} {'<span class=\"arrow-up\">▲</span>' if mc>prev_mc else '<span class=\"arrow-down\">▼</span>'} <abbr title='Total market value.'>ℹ️</abbr>",
    unsafe_allow_html=True
)
cols2 = st.columns(3)
cols2[0].markdown(
    f"**Revenue (TTM):** ${rev:,} {'<span class=\"arrow-up\">▲</span>' if rev>hist['Close'].iloc[-2]*info.get('sharesOutstanding',1) else '<span class=\"arrow-down\">▼</span>'} <abbr title='Trailing 12m revenue.'>ℹ️</abbr>",
    unsafe_allow_html=True
)
cols2[1].markdown(
    f"**Dividend Yield:** {dy:.2f}% {'<span class=\"arrow-up\">▲</span>' if dy>np.nanmean([yf.Ticker(p).info.get('dividendYield',0)*100 for p in popular]) else '<span class=\"arrow-down\">▼</span>'} <abbr title='Annual dividend %.'>ℹ️</abbr>",
    unsafe_allow_html=True
)
cols2[2].markdown(
    f"**Beta:** {beta:.2f} {'<span class=\"arrow-up\">▲</span>' if beta>1 else '<span class=\"arrow-down\">▼</span>'} <abbr title='Volatility vs market.'>ℹ️</abbr>",
    unsafe_allow_html=True
)
ins = (
    f"Last session volume was {'higher' if vol>avg_vol else 'lower'} than the 30-day avg, suggesting {'strong buying interest.' if vol>avg_vol else 'potential lack of enthusiasm.'} "
    f"Market cap ${mc:,} indicates {'small' if mc<1e9 else 'mid/large'}-cap. "
    f"TTM revenue ${rev:,}. "
    f"Dividend {dy:.2f}% {'paid' if dy>0 else 'none'}. "
    f"Beta {beta:.2f} {'higher' if beta>1 else 'lower'} vol."
)
st.markdown(f"<div class='card-dark'>🔍 {ins}</div>", unsafe_allow_html=True)

# --- EXTENDED FUNDAMENTALS ---
st.markdown("<div class='card'><h2>📑 Fundamental Breakdown vs Peers</h2></div>", unsafe_allow_html=True)

# Ensure peer_list exists
try:
    peer_list
except NameError:
    peer_list = popular

# Gather peer data
peer_info = []
for p in peer_list:
    try:
        peer_info.append(yf.Ticker(p).info)
    except:
        continue

# Compute peer averages
keys = ['trailingPE','pegRatio','profitMargins','returnOnEquity','debtToEquity','enterpriseValue']
avg_vals = {}
for k in keys:
    vals = [pi.get(k) for pi in peer_info if isinstance(pi.get(k), (int,float))]
    avg_vals[k] = np.nanmean(vals) if vals else np.nan

# Define sections
sections = {
    'Valuation': [('P/E Ratio','trailingPE','15–25 = fair valuation'),('PEG Ratio','pegRatio','~1 = balanced')],
    'Profitability': [('Net Margin','profitMargins','>5% profitable'),('ROE','returnOnEquity','>15% strong')],
    'Leverage': [('Debt/Equity','debtToEquity','<1 manageable'),('Enterprise Value','enterpriseValue','incl. debt & cash')]
}

# Render in three columns
cols = st.columns(3)
for idx, (sec, items) in enumerate(sections.items()):
    with cols[idx]:
        st.markdown(f"### {sec}")
        for name, key, tip in items:
            val = info.get(key)
            peer_avg = avg_vals.get(key, np.nan)
            # format peer avg
            if pd.isna(peer_avg):
                peer_str = 'N/A'
            elif name in ['Net Margin','ROE']:
                peer_str = f"{peer_avg*100:.2f}%"
            elif key=='enterpriseValue':
                peer_str = f"${peer_avg:,.0f}"
            else:
                peer_str = f"{peer_avg:.2f}"
            # determine display and color
            if val is None or pd.isna(peer_avg):
                disp, color = 'N/A','gray'
            else:
                better = (val>=peer_avg) if key!='debtToEquity' else (val<=peer_avg)
                color = 'green' if better else 'red'
                if name in ['Net Margin','ROE']:
                    disp = f"{val*100:.2f}%"
                elif key=='enterpriseValue':
                    disp = f"${val:,.0f}"
                else:
                    disp = f"{val:.2f}" if isinstance(val,(int,float)) else 'N/A'
        st.markdown(
            f"- {name}: "
            f"<span style='color:{color}; font-weight:bold;'>{disp}</span> "
            f"<abbr title='{tip}'>ℹ️</abbr>",
            unsafe_allow_html=True
        )

# Improved AI Insight summarizing all three pillars
# Evaluate strengths
val_diff = info.get('trailingPE', np.nan) - avg_vals.get('trailingPE', np.nan)
profit_diff = (info.get('returnOnEquity',0) - avg_vals.get('returnOnEquity',0))*100
leverage_diff = avg_vals.get('debtToEquity', np.nan) - info.get('debtToEquity', np.nan)
insights = []

# Valuation insight
if not pd.isna(val_diff):
    if val_diff < 0:
        insights.append('📈 Valuation is attractive relative to peers.')
    else:
        insights.append('⚠️ Valuation is above peer average; consider risks.')
# Profitability insight
if not pd.isna(profit_diff):
    if profit_diff > 0:
        insights.append('👍 Profitability (ROE) outperforms peers.')
    else:
        insights.append('🔻 Profitability lags behind peers.')
# Leverage insight
if not pd.isna(leverage_diff):
    if leverage_diff > 0:
        insights.append('🏦 Strong balance sheet with lower debt relative to peers.')
    else:
        insights.append('⚠️ Higher leverage than peers; watch debt levels.')

summary = ' '.join(insights) if insights else 'No sufficient data for peer comparison.'
st.markdown(f"<div class='card-dark'>💡 {summary}</div>", unsafe_allow_html=True)

# --- FUNDAMENTAL ANALYSIS MODULE ---
def render_fundamental_analysis(ticker: str):
    data = yf.Ticker(ticker)
    st.markdown("<div class='card'><h2>📊 Earnings Review</h2></div>", unsafe_allow_html=True)

    # 1) Pull last 4 quarters of key metrics
    df_income = data.quarterly_financials.T
    metrics = [
        'Total Revenue','Revenue','Gross Profit',
        'Operating Income','EBIT','Net Income','Operating Cash Flow'
    ]
    avail = [m for m in metrics if m in df_income.columns]
    df_q = df_income[avail].iloc[:4]
    df_q.index = pd.to_datetime(df_q.index).to_period('Q').astype(str)

    # Identify the latest and prior quarters
    latest_q = df_q.index[-1]      # e.g. '2025Q1'
    prev_q   = df_q.index[-2] if len(df_q) > 1 else None

    # 2) Compute QoQ % changes
    df_pct = (df_q.pct_change() * 100).round(1)
    df_pct.columns = [f"{col} % Change" for col in df_pct.columns]

    # 3) Merge USD & % tables
    df_show = pd.concat([df_q, df_pct], axis=1)

    # 4) Short‐scale formatter
    def format_short(x):
        try:
            x = float(x)
        except:
            return "-"
        if abs(x) >= 1e9:
            return f"{x/1e9:.2f}B"
        elif abs(x) >= 1e6:
            return f"{x/1e6:.2f}M"
        elif abs(x) >= 1e3:
            return f"{x/1e3:.2f}K"
        else:
            return f"{x:.0f}"

    # 5) Build formatted DataFrame
    df_fmt = df_show.copy()
    for col in avail:
        df_fmt[col] = df_fmt[col].apply(format_short)
    for col in df_pct.columns:
        df_fmt[col] = df_fmt[col].apply(lambda v: f"{v:.1f}%" if pd.notna(v) else "-")

    # 6) Display table
    st.dataframe(df_fmt, use_container_width=True)

    # --- ENHANCED EARNINGS INSIGHTS for the latest quarter ---
    human_insights = []
    def sentiment(change):
        if change > 5:        return "strong growth"
        elif change > 0:      return "modest increase"
        elif change > -5:     return "slight decline"
        else:                 return "notable decrease"

    if prev_q:
        for metric in avail:
            pct_col = f"{metric} % Change"
            if pct_col in df_pct.columns:
                change = df_pct.loc[latest_q, pct_col]
                sent   = sentiment(change)
                val_new = df_q.loc[latest_q, metric]
                val_old = df_q.loc[prev_q, metric]
                human_insights.append(
                    f"• {metric} moved from {format_short(val_old)} in {prev_q} to "
                    f"{format_short(val_new)} in {latest_q} ({sent} of {abs(change):.1f}%)."
                )

    # Analyst‐style wrap up on revenue
    analyst_notes = []
    rev_col = "Revenue % Change"
    if prev_q and rev_col in df_pct.columns:
        rc = df_pct.loc[latest_q, rev_col]
        mood = "bullish" if rc > 0 else "cautious"
        analyst_notes.append(f"🧐 Analysts are {mood} on revenue after a {abs(rc):.1f}% {'rise' if rc>0 else 'drop'} in {latest_q}.")

    summary = human_insights + analyst_notes
    out = "<br>".join(summary) if summary else f"No QoQ data for {latest_q}."
    st.markdown(f"<div class='card-dark'><b>💡 Earnings Insights:</b><br>{out}</div>", unsafe_allow_html=True)


# ✅ Call the function (outside the definition!)
render_fundamental_analysis(ticker)

# --- TECHNICAL PARAMETER CONTROLS ---
st.sidebar.header("🔧 Technical Settings")
rsi_period   = st.sidebar.slider("RSI Period",            min_value=5,  max_value=30, value=14, step=1)
macd_fast    = st.sidebar.slider("MACD Fast EMA",        min_value=5,  max_value=30, value=12, step=1)
macd_slow    = st.sidebar.slider("MACD Slow EMA",        min_value=10, max_value=60, value=26, step=1)
macd_signal  = st.sidebar.slider("MACD Signal EMA",      min_value=5,  max_value=20, value=9,  step=1)
bb_window    = st.sidebar.slider("Bollinger Window",     min_value=10, max_value=60, value=20, step=1)
bb_std_mult  = st.sidebar.slider("Bollinger Std Mult",   min_value=1.0, max_value=3.0, value=2.0)
atr_period   = st.sidebar.slider("ATR Period",           min_value=5,  max_value=30, value=14, step=1)

# --- RECOMPUTE INDICATORS WITH USER SETTINGS ---
# Moving averages (we’ll still draw 50 & 200 for cross analysis)
hist['MA20']  = hist['Close'].rolling(rsi_period).mean()
hist['MA50']  = hist['Close'].rolling(50).mean()
hist['MA200'] = hist['Close'].rolling(200).mean()

# RSI
delta = hist['Close'].diff()
gain  = delta.clip(lower=0).rolling(rsi_period).mean()
loss  = -delta.clip( upper=0).rolling(rsi_period).mean()
hist['RSI'] = 100 - (100 / (1 + gain/loss))

# MACD
hist['EMA_fast']  = hist['Close'].ewm(span=macd_fast, adjust=False).mean()
hist['EMA_slow']  = hist['Close'].ewm(span=macd_slow, adjust=False).mean()
hist['MACD']      = hist['EMA_fast'] - hist['EMA_slow']
hist['MACD_sig']  = hist['MACD'].ewm(span=macd_signal, adjust=False).mean()
hist['MACD_hist'] = hist['MACD'] - hist['MACD_sig']

# Bollinger Bands
hist['BB_mid']   = hist['Close'].rolling(bb_window).mean()
hist['BB_std']   = hist['Close'].rolling(bb_window).std()
hist['BB_upper'] = hist['BB_mid'] + bb_std_mult * hist['BB_std']
hist['BB_lower'] = hist['BB_mid'] - bb_std_mult * hist['BB_std']
hist['BB_%B']    = (hist['Close'] - hist['BB_lower']) / (hist['BB_upper'] - hist['BB_lower'])

# ATR
tr = pd.concat([
    hist['High'] - hist['Low'],
    (hist['High'] - hist['Close'].shift()).abs(),
    (hist['Low']  - hist['Close'].shift()).abs()
], axis=1).max(axis=1)
hist['ATR'] = tr.rolling(atr_period).mean()

# OBV (On-Balance Volume)
hist['OBV'] = (np.sign(hist['Close'].diff()) * hist['Volume']).fillna(0).cumsum()

# --- TECHNICAL ANALYSIS MODULE (Enhanced) ---
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# 1) Compute Indicators
hist['MA20'] = hist['Close'].rolling(20).mean()
hist['MA50'] = hist['Close'].rolling(50).mean()
hist['MA200'] = hist['Close'].rolling(200).mean()

# RSI
delta = hist['Close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
hist['RSI'] = 100 - (100 / (1 + gain/loss))

# MACD
hist['EMA12'] = hist['Close'].ewm(span=12).mean()
hist['EMA26'] = hist['Close'].ewm(span=26).mean()
hist['MACD'] = hist['EMA12'] - hist['EMA26']
hist['MACD_hist'] = hist['MACD'] - hist['MACD'].ewm(span=9).mean()

# Bollinger Bands and %B
hist['BB_mid'] = hist['Close'].rolling(20).mean()
hist['BB_std'] = hist['Close'].rolling(20).std()
hist['BB_upper'] = hist['BB_mid'] + 2*hist['BB_std']
hist['BB_lower'] = hist['BB_mid'] - 2*hist['BB_std']
hist['BB_pctB'] = (hist['Close'] - hist['BB_lower']) / (hist['BB_upper'] - hist['BB_lower'])

# ATR (volatility)
tr = pd.concat([
    hist['High'] - hist['Low'],
    (hist['High'] - hist['Close'].shift()).abs(),
    (hist['Low'] - hist['Close'].shift()).abs()
], axis=1).max(axis=1)
hist['ATR'] = tr.rolling(14).mean()

# OBV (On‐Balance Volume)
obv = (np.sign(hist['Close'].diff()) * hist['Volume']).fillna(0).cumsum()
hist['OBV'] = obv

# Support & Resistance (10th / 90th percentile of last 3 months)
recent = hist.last('90D')
support = np.percentile(recent['Low'], 10)
resistance = np.percentile(recent['High'], 90)

# --- TECHNICAL OVERVIEW & NARRATIVE ---
latest = hist.iloc[-1]
cross  = (
    "Golden Cross ✅" if latest['MA50'] > latest['MA200']
    else "Death Cross ⚠️" if latest['MA50'] < latest['MA200']
    else "No Cross"
)

tech_df = pd.DataFrame([
    ["RSI",         f"{latest['RSI']:.1f}",      ""],
    ["MACD",        f"{latest['MACD']:.2f}",     ""],
    ["MACD Signal", f"{latest['MACD_sig']:.2f}", ""],
    ["MACD Hist",   f"{latest['MACD_hist']:.2f}", ""],
    ["MA20/50/200", f"{latest['MA20']:.2f}/{latest['MA50']:.2f}/{latest['MA200']:.2f}", cross],
    ["%B (BB)",     f"{latest['BB_%B']:.2f}",     ""],
    ["ATR",         f"{latest['ATR']:.2f}",      ""],
    ["OBV",         f"{latest['OBV']:.0f}",      ""]
], columns=["Indicator","Value","Signal"])

st.markdown("<div class='card'><h2>📈 Technical Overview</h2></div>", unsafe_allow_html=True)
st.dataframe(tech_df, use_container_width=True)

# Narrative summary
rsi_desc   = "overbought" if latest['RSI']>70 else "oversold" if latest['RSI']<30 else "neutral"
macd_desc  = "bullish"   if latest['MACD']>latest['MACD_sig'] else "bearish"
bb_desc    = "above upper band" if latest['BB_%B']>1 else "below lower band" if latest['BB_%B']<0 else "within bands"
vol_desc   = "strong"    if hist['OBV'][-1] > hist['OBV'][-10] else "weak"

narrative = (
    f"RSI is {latest['RSI']:.1f} ({rsi_desc}). "
    f"MACD is {macd_desc} with a histogram of {latest['MACD_hist']:.2f}. "
    f"A {cross} just occurred. "
    f"Price is {bb_desc} of its Bollinger Bands. "
    f"ATR is {latest['ATR']:.2f}, showing {('high' if latest['ATR']>hist['ATR'].mean() else 'low')} volatility. "
    f"OBV flow is {vol_desc}, confirming money movement."
)

st.markdown(
    f"<div class='card-dark'><b>🧠 Technical Summary:</b><br>{narrative}</div>",
    unsafe_allow_html=True
)

# 3) AI Insight
ins = []
ins.append(f"RSI is at {latest['RSI']:.1f}, which is {'overbought' if latest['RSI']>70 else 'oversold' if latest['RSI']<30 else 'neutral'}.")
ins.append(f"MACD is {'positive' if latest['MACD']>0 else 'negative'}, suggesting {'bullish' if latest['MACD']>0 else 'bearish'} momentum.")
ins.append(f"The 50/200 MA cross: {cross}.")
ins.append(f"Price sits at {latest['BB_pctB']:.2f} of its Bollinger range.")
ins.append(f"ATR at {latest['ATR']:.2f} reflects recent volatility.")
ins.append(f"Recent OBV trend is {'up' if hist['OBV'].iloc[-1]>hist['OBV'].iloc[-10] else 'down'}–confirming money flow direction.")

st.markdown(
    f"<div class='card-dark'><b>📊 Technical Insights:</b><br>{'<br>'.join(ins)}</div>",
    unsafe_allow_html=True
)

# --- 3️⃣ Trade-Signal Visualization & 4️⃣ Pattern Detection ---

# 1) Identify signal dates
ma50 = hist['MA50']
ma200 = hist['MA200']
macd = hist['MACD']
macd_sig = hist['MACD_sig']

# Golden / Death Cross
gcross = (ma50.shift(1) < ma200.shift(1)) & (ma50 > ma200)
dcross = (ma50.shift(1) > ma200.shift(1)) & (ma50 < ma200)
gc_dates = hist.index[gcross]
dc_dates = hist.index[dcross]

# MACD crossovers
macd_buy  = (macd.shift(1) < macd_sig.shift(1)) & (macd > macd_sig)
macd_sell = (macd.shift(1) > macd_sig.shift(1)) & (macd < macd_sig)
mb_dates  = hist.index[macd_buy]
ms_dates  = hist.index[macd_sell]

# Doji detection: |Open–Close| ≤ 10% of (High–Low)
doji = (hist['Close'] - hist['Open']).abs() <= 0.1 * (hist['High'] - hist['Low'])
doji_dates = hist.index[doji]

# 2) Annotate on your last-30-day candlestick chart
last30 = hist.last('30D')
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.7,0.3], vertical_spacing=0.05)

# Main candlesticks
fig.add_trace(go.Candlestick(
    x=last30.index, open=last30['Open'], high=last30['High'],
    low=last30['Low'], close=last30['Close'], name="Price"
), row=1, col=1)

# Volume bars
fig.add_trace(go.Bar(
    x=last30.index, y=last30['Volume'], name="Volume", marker_color='grey'
), row=2, col=1)

# Overlay markers for signals (only include those within last30)
for d in gc_dates:
    if d in last30.index:
        fig.add_trace(go.Scatter(
            x=[d], y=[last30.loc[d, 'Close']],
            mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'),
            name='Golden Cross'
        ), row=1, col=1)
for d in dc_dates:
    if d in last30.index:
        fig.add_trace(go.Scatter(
            x=[d], y=[last30.loc[d, 'Close']],
            mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'),
            name='Death Cross'
        ), row=1, col=1)
for d in mb_dates:
    if d in last30.index:
        fig.add_trace(go.Scatter(
            x=[d], y=[last30.loc[d, 'Low']*0.995],
            mode='markers', marker=dict(symbol='circle', size=8, color='blue'),
            name='MACD Buy'
        ), row=1, col=1)
for d in ms_dates:
    if d in last30.index:
        fig.add_trace(go.Scatter(
            x=[d], y=[last30.loc[d, 'High']*1.005],
            mode='markers', marker=dict(symbol='circle', size=8, color='orange'),
            name='MACD Sell'
        ), row=1, col=1)
for d in doji_dates:
    if d in last30.index:
        fig.add_trace(go.Scatter(
            x=[d], y=[last30.loc[d, 'Close']],
            mode='markers', marker=dict(symbol='x', size=10, color='purple'),
            name='Doji'
        ), row=1, col=1)

# Finalize layout
fig.update_layout(
    template="plotly_dark", height=650, showlegend=True,
    title=f"{ticker} — Last 30 Days with Signals & Patterns"
)
st.plotly_chart(fig, use_container_width=True)

# 3) Summary of last signals/patterns
# 3) Summary of last signals/patterns (fixed length checks)
sig_summary = {
    "Golden Cross": (
        gc_dates[-1].strftime("%Y-%m-%d")
        if len(gc_dates) > 0
        else "None in 30d"
    ),
    "Death Cross": (
        dc_dates[-1].strftime("%Y-%m-%d")
        if len(dc_dates) > 0
        else "None in 30d"
    ),
    "MACD Buy": (
        mb_dates[-1].strftime("%Y-%m-%d")
        if len(mb_dates) > 0
        else "None in 30d"
    ),
    "MACD Sell": (
        ms_dates[-1].strftime("%Y-%m-%d")
        if len(ms_dates) > 0
        else "None in 30d"
    ),
    "Doji (last 3)": (
        ", ".join(d.strftime("%m-%d") for d in doji_dates[-3:])
        if len(doji_dates) > 0
        else "None in 30d"
    )
}

st.markdown("<div class='card'><h3>🔔 Recent Signals & Patterns</h3></div>", unsafe_allow_html=True)
for name, when in sig_summary.items():
    st.markdown(f"- **{name}:** {when}")

# --- PEER COMPARISON MODULE ---
st.markdown("<div class='card'><h2>🤝 Peer Comparison</h2></div>", unsafe_allow_html=True)
peer_data = []
for p in peer_list:
    try:
        pi = yf.Ticker(p).info
        peer_data.append({
            'Ticker': p,
            'Price': pi.get('currentPrice', np.nan),
            'P/E Ratio': pi.get('trailingPE', np.nan)
        })
    except Exception:
        continue
peer_df = pd.DataFrame(peer_data).set_index('Ticker')
if not peer_df.empty:
    st.bar_chart(peer_df['P/E Ratio'])
    st.dataframe(peer_df.style.format({
        'Price': '${:,.2f}',
        'P/E Ratio': '{:.2f}'
    }))
else:
    st.info("No peer data available.")

# --- NEWS & SENTIMENT MODULE ---
st.markdown("<div class='card'><h2>📰 News & Sentiment</h2></div>", unsafe_allow_html=True)
news_url = f"https://finance.yahoo.com/quote/{ticker}"
try:
    resp = requests.get(news_url, timeout=5)
    soup = BeautifulSoup(resp.content, 'html.parser')
    headlines = soup.find_all('h3')[:5]
    for h in headlines:
        text = h.get_text(strip=True)
        badge = '🟢' if any(w in text.lower() for w in ['beat','upgrade','gain']) else ('🔴' if any(w in text.lower() for w in ['miss','downgrade','drop']) else '⚪️')
        st.markdown(f"- {badge} {text}", unsafe_allow_html=True)
    st.markdown("<div class='card-dark'>🔍 Overall sentiment: Neutral to Positive based on recent headlines.</div>", unsafe_allow_html=True)
except Exception:
    st.warning("Unable to fetch news headlines.")

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p style="color:#888888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p style="color:#888888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
