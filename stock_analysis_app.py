# ai_stock_analyzer_app/main.py

import streamlit as st
import yfinance as yf
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="📈 AI Stock Analyzer", layout="wide")

# --- HEADER ---
st.markdown("""
<div style="text-align:center">
    <h1 style="color:#4CAF50;">📊 AI Stock Analyzer</h1>
    <p style="font-size:18px;">Smart insights for smarter investing — built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)

# --- USER INPUT ---
st.sidebar.header("Enter Stock Ticker")
ticker = st.sidebar.text_input("Example: AAPL", value="SPCE").upper()

if not ticker:
    st.warning("Please enter a valid ticker symbol.")
    st.stop()

# --- FETCH DATA ---
data = yf.Ticker(ticker)
info = data.info
hist = data.history(period="1mo")

# --- PRICE CHANGES ---
def calc_change(current, past):
    if not current or not past:
        return "N/A", "N/A"
    change = current - past
    pct = (change / past * 100) if past else 0
    return f"${change:.2f}", f"{pct:.2f}%"

current_price = info.get("currentPrice", 0.0)
price_1d = hist["Close"].iloc[-2] if len(hist) > 1 else None
price_1w = hist["Close"].iloc[-6] if len(hist) > 5 else None
price_1m = hist["Close"].iloc[0] if len(hist) > 0 else None

change_1d, pct_1d = calc_change(current_price, price_1d)
change_1w, pct_1w = calc_change(current_price, price_1w)
change_1m, pct_1m = calc_change(current_price, price_1m)

# --- PRICE PANEL ---
st.markdown(f"### 💵 **{info.get('shortName', ticker)} ({ticker})**")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("24h Change", pct_1d, delta_color="inverse")
col3.metric("1 Week Change", pct_1w)
col4.metric("1 Month Change", pct_1m)

# --- MARKET OVERVIEW ---
st.markdown("## 🧾 Market & Trading Overview")
col1, col2, col3 = st.columns(3)

col1.metric("Volume (Last)", f"{info.get('volume', 0):,}")
col2.metric("Avg Volume", f"{info.get('averageVolume', 0):,}")
col3.metric("Market Cap", f"${info.get('marketCap', 0):,}")

col1.metric("Revenue (TTM)", f"${info.get('totalRevenue', 0):,}")
col2.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get("dividendYield") else "N/A")
col3.metric("Beta", f"{info.get('beta', 'N/A')}")

# --- TECHNICAL INDICATORS ---
st.markdown("## 📊 Technical Indicators")

# RSI
delta = hist['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
hist['RSI'] = 100 - (100 / (1 + rs))

# MACD
hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
hist['MACD'] = hist['EMA12'] - hist['EMA26']

# Moving Averages
hist['MA20'] = hist['Close'].rolling(20).mean()
hist['MA50'] = hist['Close'].rolling(50).mean()

# Visuals
st.line_chart(hist[['Close', 'MA20', 'MA50']], use_container_width=True)
st.line_chart(hist[['RSI', 'MACD']], use_container_width=True)

# --- COMING SOON PANELS ---
st.markdown("## 🧠 Coming Soon: Deep Analysis")
st.info("""
- 🔍 Resistance & Support Levels
- 📐 ADX and Trend Strength
- 🧾 Analyst Ratings and Forecasts
- 📊 Candlestick Patterns
- 🧠 AI-based Technical & Sentiment Summary
- 💼 Financial Health and Risk Profile
- 📈 Valuation Multiples and Profitability Ratios
""")

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p>Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
