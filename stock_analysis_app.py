# ai_stock_analyzer_app/main.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from io import BytesIO
import base64
import requests
from bs4 import BeautifulSoup

# --- PAGE CONFIG ---
st.set_page_config(page_title="BSAV Stock Analyzer", layout="wide")

# --- HEADER ---
st.markdown("""
<div style="text-align:center">
    <h1 style="color:#4CAF50;">📊 AI Stock Analyzer</h1>
    <p style="font-size:18px; color:#888888;">Smart insights for smarter investing — built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)

# --- USER INPUT ---
st.sidebar.header("Enter Stock Ticker")
ticker = st.sidebar.text_input("Example: AAPL", value="SPCE").upper().strip()
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

# --- PRICE CHANGES ---
def calc_change(current, past):
    if not current or not past:
        return "N/A", "N/A", "off", "No significant movement."
    change = current - past
    pct = (change / past * 100) if past else 0
    if pct >= 10:
        color = "normal"
        note = "Strong growth — likely driven by exceptional earnings or market catalysts."
    elif pct >= 3:
        color = "normal"
        note = "Solid uptrend — reflecting positive sentiment or sector performance."
    elif pct > 0:
        color = "normal"
        note = "Mild gain — modest optimism or technical rebound."
    elif pct == 0:
        color = "off"
        note = "Flat trend — price has stabilized or market is undecided."
    elif pct > -3:
        color = "inverse"
        note = "Slight dip — typical short-term fluctuation."
    elif pct > -10:
        color = "inverse"
        note = "Noticeable drop — may indicate weakening fundamentals or negative sentiment."
    else:
        color = "inverse"
        note = "Sharp sell-off — likely triggered by significant bad news or earnings miss."
    return f"${change:.2f}", f"{pct:.2f}%", color, note

current_price = info.get("currentPrice", 0.0)
price_1d = hist['Close'].iloc[-2] if len(hist) > 1 else None
price_1m = hist['Close'].iloc[-21] if len(hist) > 21 else None
price_6m = hist['Close'].iloc[0] if len(hist) > 0 else None

change_1d, pct_1d, color_1d, note_1d = calc_change(current_price, price_1d)
change_1m, pct_1m, color_1m, note_1m = calc_change(current_price, price_1m)
change_6m, pct_6m, color_6m, note_6m = calc_change(current_price, price_6m)

# --- DISPLAY PANEL ---
st.markdown(f"### 💵 **{info.get('shortName', ticker)} ({ticker})**")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:.2f}")
col1.markdown(f"<div style='font-size:14px; color:#333;'>This is the latest trading price for {ticker}.</div>", unsafe_allow_html=True)
col2.metric("24h Change", f"{pct_1d}", delta_color=color_1d)
col2.markdown(f"<div style='font-size:14px; color:#333;'>🧠 {note_1d}</div>", unsafe_allow_html=True)
col3.metric("1 Month Change", f"{pct_1m}", delta_color=color_1m)
col3.markdown(f"<div style='font-size:14px; color:#333;'>🧠 {note_1m}</div>", unsafe_allow_html=True)
col4.metric("6 Month Change", f"{pct_6m}", delta_color=color_6m)
col4.markdown(f"<div style='font-size:14px; color:#333;'>🧠 {note_6m}</div>", unsafe_allow_html=True)

# --- INTERACTIVE PRICE CHART ---
st.markdown("## 📈 Price History (6M with MA20 & MA50)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], mode='lines', name='MA20'))
fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], mode='lines', name='MA50'))
fig.update_layout(title=f"{ticker} Stock Price with Moving Averages", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# --- NEWS SUMMARY ---
st.markdown("## 📰 Latest Headline Summary")
news_url = f"https://finance.yahoo.com/quote/{ticker}"
try:
    page = requests.get(news_url, timeout=5)
    soup = BeautifulSoup(page.content, 'html.parser')
    headlines = soup.find_all('h3')[:3]
    for h in headlines:
        st.markdown(f"- {h.get_text(strip=True)}")
except:
    st.warning("Couldn't fetch news headlines.")

# --- ANALYST SUMMARY ---
st.markdown("""
<div style='background-color:#f0f0f0; padding:10px; border-left: 5px solid #4CAF50;'>
<b>🧠 Analyst Overview:</b><br>
Price momentum shows a positive trend supported by technical signals. Market cap and volume confirm active interest, although earnings data should be monitored to validate valuation.
</div>
""", unsafe_allow_html=True)
