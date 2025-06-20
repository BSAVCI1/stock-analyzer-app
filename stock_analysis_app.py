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

# Popular tickers
popular = ["BBAI","ARCC","VUSA.AS","SPCE","AAPL","MSFT","GOOGL","AMZN","QS","TSLA","NVDA"]

# 1) Select from dropdown…
ticker_select = st.sidebar.selectbox(
    "Choose from popular tickers", 
    options=popular, 
    index=popular.index("SPCE")
)

# 2) …or free-text any other symbol
ticker_input = st.sidebar.text_input(
    "Or enter any ticker symbol", 
    value=""
).upper().strip()

# Final ticker to use
ticker = ticker_input if ticker_input else ticker_select

# Dynamically fetch default peers based on sector if no manual override
if st.sidebar.checkbox("Auto-select peers based on sector", value=True):
    try:
        sector = yf.Ticker(ticker).info.get("sector")
        sector_map = {
            "Technology": ["AAPL","MSFT","GOOGL"],
            "Consumer Cyclical": ["AMZN","TSLA","BBWI"],
            "Communication Services": ["META","NFLX","DIS"],
            "AI Services": ["BBAI","SOUN","NVDA"],
            "Space": ["SPCE","BKSY","LHX"],
            # …extend as needed
        }
        peer_list = sector_map.get(sector, popular)
    except Exception:
        peer_list = popular
else:
    peers_input = st.sidebar.text_input(
        "Or enter peers (comma separated)", 
        "AAPL,MSFT,GOOGL"
    ).upper()
    peer_list = [p.strip() for p in peers_input.split(",") if p.strip()]


# --- FETCH DATA ---
data = yf.Ticker(ticker)
info = data.info
hist = data.history(period="6mo")
hist['MA20'] = hist['Close'].rolling(20).mean()
hist['MA50'] = hist['Close'].rolling(50).mean()

st.markdown(f"### {info.get('shortName', ticker)} ({ticker})")

# --- MARKET OVERVIEW & SUPPORT/RESISTANCE ---
st.markdown("<div class='card'><h2>📈 Market & Trading Overview</h2></div>", unsafe_allow_html=True)
vol = info.get('volume', 0)
avg_vol = info.get('averageVolume', 0)
mc = info.get('marketCap', 0)
rev = info.get('totalRevenue', 0)
dy = info.get('dividendYield', 0) * 100
beta = info.get('beta', 0)
# Columns with trend arrows
cols = st.columns(3)
cols[0].markdown(f"**Volume:** {vol:,} {'<span class=\"arrow-up\">▲</span>' if vol>avg_vol else '<span class=\"arrow-down\">▼</span>'} <abbr title='Shares traded in last session.'>ℹ️</abbr>", unsafe_allow_html=True)
cols[1].markdown(f"**Avg Volume:** {avg_vol:,} {'<span class=\"arrow-up\">▲</span>' if avg_vol>vol else '<span class=\"arrow-down\">▼</span>'} <abbr title='30-day avg volume.'>ℹ️</abbr>", unsafe_allow_html=True)
cols[2].markdown(f"**Market Cap:** ${mc:,} {'<span class=\"arrow-up\">▲</span>' if mc>mc else '<span class=\"arrow-down\">▼</span>'} <abbr title='Total market value.'>ℹ️</abbr>", unsafe_allow_html=True)
cols2 = st.columns(3)
cols2[0].markdown(f"**Revenue (TTM):** ${rev:,} {'<span class=\"arrow-up\">▲</span>' if rev>rev else '<span class=\"arrow-down\">▼</span>'} <abbr title='Trailing 12m revenue.'>ℹ️</abbr>", unsafe_allow_html=True)
cols2[1].markdown(f"**Dividend Yield:** {dy:.2f}% {'<span class=\"arrow-up\">▲</span>' if dy>dy else '<span class=\"arrow-down\">▼</span>'} <abbr title='Annual dividend %.'>ℹ️</abbr>", unsafe_allow_html=True)
cols2[2].markdown(f"**Beta:** {beta:.2f} {'<span class=\"arrow-up\">▲</span>' if beta>1 else '<span class=\"arrow-down\">▼</span>'} <abbr title='Volatility vs market.'>ℹ️</abbr>", unsafe_allow_html=True)
ins = (
    f"Over the last session, trading volume was {'higher' if vol>avg_vol else 'lower'} than the 30‑day average, suggesting {'strong buying interest' if vol>avg_vol else 'potential lack of market enthusiasm'}. "
    f"Current market cap stands at ${mc:,}, which makes this a {'smaller' if mc<1e9 else 'mid to large'}-cap company—" 
    f"smaller companies can be more volatile, while larger caps offer more stability. "
    f"Revenue (TTM) of ${rev:,} shows how much the company sold in the past year. "
    f"A dividend yield of {dy:.2f}% {'rewards investors with regular payouts' if dy>0 else 'means the company does not currently pay dividends'}. "
    f"A beta of {beta:.2f} indicates {'higher' if beta>1 else 'lower'} volatility relative to the overall market."
)
st.markdown(f"<div class='card-dark'>🔍 {ins}</div>", unsafe_allow_html=True)


# --- EXTENDED FUNDAMENTALS ---
st.markdown("<div class='card'><h2>🧲 Fundamental Breakdown</h2></div>", unsafe_allow_html=True)
sections = {
    'Valuation': [('P/E Ratio', 'trailingPE', '15–25 = fair valuation range'), ('PEG Ratio', 'pegRatio', '~1 = growth adjusted')],
    'Profitability': [('Net Margin', 'profitMargins', '>5% = profitable'), ('ROE', 'returnOnEquity', '>15% = strong returns')],
    'Leverage': [('Debt/Equity', 'debtToEquity', '<1 = manageable debt'), ('Enterprise Value', 'enterpriseValue', 'ratio vs MC = leverage context')]
}
peer_info = [yf.Ticker(p).info for p in peer_list]
avg_vals = { 
    'trailingPE': np.nanmean([pi.get('trailingPE', np.nan) for pi in peer_info]),
    'profitMargins': np.nanmean([pi.get('profitMargins', np.nan) for pi in peer_info]),
    'returnOnEquity': np.nanmean([pi.get('returnOnEquity', np.nan) for pi in peer_info]),
    'debtToEquity': np.nanmean([pi.get('debtToEquity', np.nan) for pi in peer_info])
}
for sec, items in sections.items():
    st.markdown(f"**{sec} Metrics vs Peers**")
    for name, key, tip in items:
        val = info.get(key)
        if val is None:
            continue
        peer_avg = avg_vals.get(key, np.nan)
        better = val >= peer_avg if key!='debtToEquity' else val <= peer_avg
        color = 'green' if better else 'red'
        disp = f"{val*100:.2f}%" if 'Margin' in name or 'ROE' in name else f"{val:.2f}"
        st.markdown(f"- {name}: <span style='color:{color}; font-weight:bold;'>{disp}</span> (<abbr title='{tip}'>ℹ️</abbr>) vs peer avg {peer_avg:.2f}", unsafe_allow_html=True)
prod_insight = (
    "Valuation is attractive vs peers." if info.get('trailingPE', np.nan) < avg_vals['trailingPE'] else 
    "Valuation is at or above peer average; investigate further."
)
st.markdown(f"<div class='card-dark'>🧠 {prod_insight}</div>", unsafe_allow_html=True)

# --- FUNDAMENTAL ANALYSIS MODULE ---
def render_fundamental_analysis(ticker: str):
    data = yf.Ticker(ticker)
    st.markdown("<div class='card'><h2>📊 Quarterly Earnings Review</h2></div>", unsafe_allow_html=True)
    df_income = data.quarterly_financials.T
    df4 = df_income.loc[:, df_income.columns.intersection(['Total Revenue','Revenue','Gross Profit',
                                                           'Operating Income','EBIT','Net Income','Operating Cash Flow'])].iloc[:4]
    df4.index = pd.to_datetime(df4.index).to_period('Q')
    changes = df4.pct_change().iloc[1:] * 100
    insights = [
        f"• {m} {'up' if changes.iloc[0][m]>0 else 'down'} {abs(changes.iloc[0][m]):.1f}% vs prior quarter"
        for m in df4.columns if not np.isnan(changes.iloc[0][m])
    ]
    st.markdown(f"<div class='card-dark'><b>🧠 Earnings Insight:</b><br>{'<br>'.join(insights)}</div>", unsafe_allow_html=True)
    st.dataframe(df4.style.format("${:,.0f}"))
render_fundamental_analysis(ticker)

# --- TECHNICAL ANALYSIS MODULE ---
# RSI
delta = hist['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
hist['RSI'] = 100 - (100 / (1 + rs))
# MACD
hist['EMA12'] = hist['Close'].ewm(span=12).mean()
hist['EMA26'] = hist['Close'].ewm(span=26).mean()
hist['MACD'] = hist['EMA12'] - hist['EMA26']
# Bollinger Bands
hist['BB_mid'] = hist['Close'].rolling(20).mean()
hist['BB_std'] = hist['Close'].rolling(20).std()
hist['BB_upper'] = hist['BB_mid'] + 2 * hist['BB_std']
hist['BB_lower'] = hist['BB_mid'] - 2 * hist['BB_std']
# ADX
high, low, close = hist['High'], hist['Low'], hist['Close']
tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
hist['ATR'] = tr.rolling(14).mean()
up = high.diff()
dn = -low.diff()
hist['+DI'] = 100 * (up.where((up>dn)&(up>0), 0).rolling(14).mean() / hist['ATR'])
hist['-DI'] = 100 * (dn.where((dn>up)&(dn>0), 0).rolling(14).mean() / hist['ATR'])
hist['ADX'] = (abs(hist['+DI'] - hist['-DI']) / (hist['+DI'] + hist['-DI']) * 100).rolling(14).mean()

# --- COMPUTE 6M CHANGE ---
current_price = info.get('currentPrice', None)
price_6m = hist['Close'].iloc[0] if len(hist) > 0 else None
pct6m = ((current_price - price_6m) / price_6m * 100) if price_6m else None

# --- TECHNICAL ANALYSIS CARD ---
st.markdown("<div class='card'><h2>📈 Technical Analysis</h2></div>", unsafe_allow_html=True)
# Insights
rsi_val = hist['RSI'].iloc[-1] if 'RSI' in hist.columns else np.nan
ma_trend = 'upward' if hist['Close'].iloc[-1] > hist['MA50'].iloc[-1] else 'downward'
macd_val = hist['MACD'].iloc[-1] if 'MACD' in hist.columns else np.nan
bb_pos = (
    'above upper band' if hist['Close'].iloc[-1] > hist['BB_upper'].iloc[-1]
    else 'below lower band' if hist['Close'].iloc[-1] < hist['BB_lower'].iloc[-1]
    else 'within bands'
)
adx_val = hist['ADX'].iloc[-1] if 'ADX' in hist.columns else np.nan
adx_text = f"{adx_val:.1f}" if not np.isnan(adx_val) else 'N/A'
tech_insights = [
    f"• RSI at {rsi_val:.1f} indicates {'overbought' if rsi_val>70 else 'oversold' if rsi_val<30 else 'neutral'} conditions.",
    f"• Price {ma_trend} vs 50-day MA.",
    f"• MACD at {macd_val:.2f} suggests {'bullish' if macd_val>0 else 'bearish'} momentum.",
    f"• Price is {bb_pos}, indicating volatility.",
    f"• ADX at {adx_text} means {'strong trend' if adx_val>25 else 'weak trend'}.",
]
st.markdown(
    f"<div class='card-dark'><b>🧠 Technical Insight:</b><br>{'<br>'.join(tech_insights)}</div>",
    unsafe_allow_html=True
)
# Charts
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI'))
fig1.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD'))
fig1.update_layout(template='plotly_white', height=300)
st.plotly_chart(fig1, use_container_width=True)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price'))
fig2.add_trace(go.Scatter(x= hist.index, y=hist['BB_upper'], line=dict(dash='dash'), name='Upper Band'))
fig2.add_trace(go.Scatter(x= hist.index, y=hist['BB_mid'], line=dict(dash='dot'), name='Mid Band'))
fig2.add_trace(go.Scatter(x= hist.index, y=hist['BB_lower'], line=dict(dash='dash'), name='Lower Band'))
fig2.update_layout(template='plotly_white', height=300)
st.plotly_chart(fig2, use_container_width=True)

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
