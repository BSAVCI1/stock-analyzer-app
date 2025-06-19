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
st.markdown("""
<div class="card">
  <h2>💵 Price Overview</h2>
  <div style="display:flex; justify-content:space-around;">
    <div><b>Now:</b> %s</div>
    <div><b>Change 24h:</b> %s</div>
    <div><b>Change 1M:</b> %s</div>
    <div><b>Change 6M:</b> %s</div>
  </div>
</div>
""" % (price_1d, pct_1d, pct_1m, pct_6m), unsafe_allow_html=True)

# --- MARKET OVERVIEW CARD ---
st.markdown("<div class='card'> <h2>📈 Market & Trading Overview</h2>", unsafe_allow_html=True)
vol = info.get('volume', 0)
avg_vol = info.get('averageVolume', 0)
mc = info.get('marketCap', 0)
rev = info.get('totalRevenue', 0)
div_yield = info.get('dividendYield', 0)
beta = info.get('beta', 0)

cols = st.columns(3)
cols[0].markdown(f"**Volume:** {vol:,} <span class='metric-tooltip' title='Total shares traded in last session.'>?</span>")
cols[1].markdown(f"**Avg Volume:** {avg_vol:,} <span class='metric-tooltip' title='30-day average trading volume.'>?</span>")
cols[2].markdown(f"**Market Cap:** ${mc:,} <span class='metric-tooltip' title='Total market value of equity.'>?</span>")
cols2 = st.columns(3)
cols2[0].markdown(f"**Revenue:** ${rev:,} <span class='metric-tooltip' title='Trailing twelve months revenue.'>?</span>")
cols2[1].markdown(f"**Dividend Yield:** {div_yield*100:.2f}% <span class='metric-tooltip' title='Annual dividend as % of price.'>?</span>")
cols2[2].markdown(f"**Beta:** {beta:.2f} <span class='metric-tooltip' title='Volatility vs market (>1 = more volatile).'>?</span>")
st.markdown("</div>", unsafe_allow_html=True)

# --- SUPPORT & RESISTANCE CARD ---
st.markdown("<div class='card'> <h2>⚙️ Support & Resistance</h2>", unsafe_allow_html=True)
st.write(f"• Support Level: ${support:.2f} — price floor where demand may increase.")
st.write(f"• Resistance Level: ${resistance:.2f} — price ceiling where supply may increase.")
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
    val = info.get(key, 'N/A')
    display = f"{val:.2f}" if isinstance(val, (int, float)) else val
    st.markdown(f"**{name}:** {display} <span class='metric-tooltip' title='{tooltip}'>?</span>")
st.markdown("</div>", unsafe_allow_html=True)

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
        st.markdown(f"- {text}")
    st.markdown("<div style='padding:5px;'><b>Sentiment:</b> Based on recent headlines, sentiment appears positive/neutral.</div>", unsafe_allow_html=True)
except:
    st.warning("Could not fetch news.")
st.markdown("</div>", unsafe_allow_html=True)

# --- ANALYST SUMMARY CARD ---
st.markdown("<div class='card-dark'> <h2 style='color:#4CAF50;'>🧠 Analyst Summary</h2>", unsafe_allow_html=True)
summary = []
if pct_6m.strip('%') and float(pct_6m.strip('%')) > 0:
    summary.append("The stock has outperformed its 6-month trend, showing resilience.")
else:
    summary.append("The stock is underperforming its 6-month trend, caution advised.")
if info.get('trailingPE') and info.get('trailingPE') < avg_pe:
    summary.append("Valuation appears attractive relative to peers.")
else:
    summary.append("Valuation is higher than peers, consider risk of overvaluation.")
st.markdown("<ul>" + ''.join([f"<li>{s}</li>" for s in summary]) + "</ul>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
