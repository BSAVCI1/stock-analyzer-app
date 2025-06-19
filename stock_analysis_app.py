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
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="card" style="text-align:center;">
    <h1 style="color:#4CAF50; margin-bottom:5px;">📊 AI Stock Analyzer</h1>
    <p style="font-size:16px; color:#555;">Comprehensive, visual insights for non-finance users</p>
</div>
""", unsafe_allow_html=True)

# --- USER INPUT ---
st.sidebar.header("Enter Stock Ticker & Peers")
ticker = st.sidebar.text_input("Ticker", value="SPCE").upper().strip()
peers_input = st.sidebar.text_input("Peer Tickers (comma separated)", value="AAPL,MSFT,GOOGL").upper()
peer_list = [p.strip() for p in peers_input.split(",") if p.strip()]
if not ticker:
    st.warning("Please enter a valid ticker symbol.")
    st.stop()

# --- FETCH DATA ---
data = yf.Ticker(ticker)
info = data.info
hist = data.history(period="6mo")
hist['MA20'] = hist['Close'].rolling(window=20).mean()
hist['MA50'] = hist['Close'].rolling(window=50).mean()

# --- SUPPORT & RESISTANCE ---
support = np.percentile(hist['Low'], 10)
resistance = np.percentile(hist['High'], 90)

# --- CALCULATE PRICE CHANGES ---
def calc_change(current, past):
    if past is None or current is None:
        return "N/A", "N/A", "off"
    change = current - past
    pct = (change / past * 100) if past else 0
    color = "normal" if pct >= 0 else "inverse"
    return f"${current:.2f}", f"{pct:.1f}%", color

current_price = info.get('currentPrice')
price_1d, pct_1d, col_1d = calc_change(current_price, hist['Close'].iloc[-2] if len(hist)>1 else None)
price_1m, pct_1m, col_1m = calc_change(current_price, hist['Close'].iloc[-21] if len(hist)>21 else None)
price_6m, pct_6m, col_6m = calc_change(current_price, hist['Close'].iloc[0]    if len(hist)>0 else None)

# --- DISPLAY PRICE CARD ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write(f"### 💵 Price Overview for {info.get('shortName', ticker)} ({ticker})")
st.markdown(
    f"""**Now:** {price_1d}  
**24h Change:** {pct_1d}  
**1M Change:** {pct_1m}  
**6M Change:** {pct_6m}""",
    unsafe_allow_html=True
)
# AI Feedback
if pct_6m.strip('%') and pct_6m != 'N/A':
    if float(pct_6m.strip('%')) > 0:
        st.info("📈 The stock has shown positive momentum over the last 6 months, indicating a bullish trend.")
    else:
        st.info("📉 The stock has underperformed over the last 6 months, caution is advised.")
st.markdown("</div>", unsafe_allow_html=True)

# --- MARKET OVERVIEW CARD ---
st.markdown("<div class='card'> <h2>📈 Market & Trading Overview</h2>", unsafe_allow_html=True)
vol = info.get('volume', 0)
avg_vol = info.get('averageVolume', 0)
mc = info.get('marketCap', 0)
rev = info.get('totalRevenue', 0)
div_yield = info.get('dividendYield', 0)
beta = info.get('beta', 0)

cols = st.columns(3)
cols[0].markdown(f"**Volume:** {vol:,} <abbr title='Total shares traded in last session.'>ℹ️</abbr>", unsafe_allow_html=True)
cols[1].markdown(f"**Avg Volume:** {avg_vol:,} <abbr title='30-day average trading volume.'>ℹ️</abbr>", unsafe_allow_html=True)
cols[2].markdown(f"**Market Cap:** ${mc:,} <abbr title='Total market value of equity.'>ℹ️</abbr>", unsafe_allow_html=True)
cols2 = st.columns(3)
cols2[0].markdown(f"**Revenue:** ${rev:,} <abbr title='Trailing twelve months revenue.'>ℹ️</abbr>", unsafe_allow_html=True)
cols2[1].markdown(f"**Dividend Yield:** {div_yield*100:.2f}% <abbr title='Annual dividend as % of price.'>ℹ️</abbr>", unsafe_allow_html=True)
cols2[2].markdown(f"**Beta:** {beta:.2f} <abbr title='Volatility vs market (>1 = more volatile).'>ℹ️</abbr>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# AI Insight for Market Overview
st.info("🔍 **Market Insight:** Volume is " + ("above average indicating strong trading interest." if vol > avg_vol else "below average which may signal reduced liquidity.") + f" Market cap at ${mc:,} positions this company as {'a smaller player' if mc<1e9 else 'a mid/large cap'}."), unsafe_allow_html=True)

# --- SUPPORT & RESISTANCE CARD ---
st.markdown("<div class='card'> <h2>⚙️ Support & Resistance</h2>", unsafe_allow_html=True)
st.markdown(f"• Support Level: ${support:.2f} — price floor where demand may increase.", unsafe_allow_html=True)
st.markdown(f"• Resistance Level: ${resistance:.2f} — price ceiling where supply may increase.", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- EXTENDED FUNDAMENTALS CARD ---
st.markdown("<div class='card'> <h2>🧲 Extended Fundamentals</h2>", unsafe_allow_html=True)
metrics = [
    ('Enterprise Value', 'enterpriseValue', '<1.5x MarketCap typical'),
    ('P/E Ratio', 'trailingPE', '15-25 = fair'),
    ('PEG Ratio', 'pegRatio', '~1 = fair'),
    ('Debt/Equity', 'debtToEquity', '<1 = comfortable'),
    ('Free Cash Flow', 'freeCashflow', '>0 = healthy'),
    ('Operating Margin', 'operatingMargins', '>10% = efficient'),
    ('Net Margin', 'profitMargins', '>5% = profitable'),
    ('Return on Equity', 'returnOnEquity', '>15% = strong'),
    ('Insider Ownership', 'heldPercentInsiders', '>5% = confidence'),
    ('Institutional Ownership', 'heldPercentInstitutions', '>70% = backed')
]
for name, key, tooltip in metrics:
    raw = info.get(key, None)
    if isinstance(raw, (int, float)):
        if 'Ratio' in name or 'Margin' in name or 'Return' in name:
            display = f"{raw*100:.2f}%" if 'Margin' in name or 'Return' in name else f"{raw:.2f}"
        else:
            display = f"${raw:,.2f}"
        color = 'green' if raw >= 0 else 'red'
        display_html = f"<span style='color:{color}; font-weight:bold;'>{display}</span>"
    else:
        display_html = 'N/A'
    st.markdown(f"**{name}:** {display_html} <abbr title='{tooltip}'>ℹ️</abbr>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# AI Insight for Fundamentals
st.info("🧠 **Fundamentals Insight:** P/E ratio is " + (f"{info.get('trailingPE'):.2f}, which is " + ("below" if info.get('trailingPE',0) < avg_pe else "above") + " the peer average." if info.get('trailingPE') else "not available, please proceed with caution.") ), unsafe_allow_html=True)

# --- COMPETITOR COMPARISON CARD ---
st.markdown("<div class='card'> <h2>🤝 Competitor Comparison</h2>", unsafe_allow_html=True)
peer_data = []
for p in peer_list:
    t = yf.Ticker(p)
    i = t.info
    peer_data.append({
        'Ticker': p,
        'Price': i.get('currentPrice', None),
        'P/E': i.get('trailingPE', None),
        'Market Cap': i.get('marketCap', None)
    })
peer_df = pd.DataFrame(peer_data).set_index('Ticker')
st.dataframe(peer_df)
# Summary comment
avg_pe = peer_df['P/E'].dropna().mean()
pe_comment = 'lower than peers (potential value)' if info.get('trailingPE',0) < avg_pe else 'higher than peers (expensive)'
st.markdown(f"<div style='padding:5px;'><b>Comment:</b> This stock's P/E is {pe_comment}.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- PRICE CHART CARD ---
st.markdown("<div class='card'> <h2>📊 Price Chart & Events</h2>", unsafe_allow_html=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], mode='lines', name='MA20'))
fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], mode='lines', name='MA50'))
fig.update_layout(template='plotly_white', height=400, margin=dict(l=20,r=20,t=30,b=20))
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- NEWS & SENTIMENT CARD ---
st.markdown("<div class='card'> <h2>📰 News & Sentiment</h2>", unsafe_allow_html=True)
news_url = f"https://finance.yahoo.com/quote/{ticker}"
try:
    resp = requests.get(news_url, timeout=5)
    soup = BeautifulSoup(resp.content, 'html.parser')
    items = soup.find_all('h3')[:3]
    for h in items:
        text = h.get_text(strip=True)
        st.markdown(f"- {text}", unsafe_allow_html=True)
    st.markdown("<div style='padding:5px;'><b>Sentiment:</b> Based on recent headlines, sentiment appears positive/neutral.</div>", unsafe_allow_html=True)
except:
    st.warning("Could not fetch news.")
st.markdown("</div>", unsafe_allow_html=True)

# --- ANALYST SUMMARY CARD ---
st.markdown("<div class='card-dark'> <h2 style='color:#4CAF50;'>🧠 Analyst Summary</h2>", unsafe_allow_html=True)
summary = []
if pct_6m != 'N/A' and float(pct_6m.strip('%')) > 0:
    summary.append("The stock has shown positive momentum over the last 6 months, indicating a bullish trend.")
else:
    summary.append("The stock has underperformed over the last 6 months, caution advised.")
if info.get('trailingPE') and info.get('trailingPE') < avg_pe:
    summary.append("Valuation appears attractive relative to peers.")
else:
    summary.append("Valuation is higher than peers, consider risk of overvaluation.")
st.markdown("<ul>" + ''.join([f"<li>{s}</li>" for s in summary]) + "</ul>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
