# ai_stock_analyzer_app/main.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import requests
from bs4 import BeautifulSoup

# --- PAGE CONFIG ---
st.set_page_config(page_title="📈 BSAV Stock Analyzer", layout="wide")

# --- HEADER ---
st.markdown("""
<div style="text-align:center">
    <h1>📊 AI Stock Analyzer</h1>
    <p style="font-size:18px; color:#333333;">Smart insights for smarter investing — built with ❤️ using Streamlit</p>
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

# --- SUPPORT & RESISTANCE ---
support = np.percentile(hist['Low'], 10)
resistance = np.percentile(hist['High'], 90)

# --- PRICE CHANGES ---
def calc_change(current, past):
    if not current or not past:
        return "N/A", "N/A", "off", "⚪ No significant movement.", "⚪"
    change = current - past
    pct = (change / past * 100) if past else 0
    if pct >= 10:
        color = "normal"
        note = "🟢 Too big increase — strong bullish activity or positive event."
        symbol = "🟢⬆️⬆️"
    elif pct >= 3:
        color = "normal"
        note = "🟢 Moderate increase — market confidence may be building."
        symbol = "🟢⬆️"
    elif pct > 0:
        color = "normal"
        note = "🟢 Slight increase — minor positive sentiment."
        symbol = "🟢↗️"
    elif pct == 0:
        color = "off"
        note = "🟡 Flat — not much change, often reflecting neutrality or consolidation."
        symbol = "🟡➖"
    elif pct > -3:
        color = "inverse"
        note = "🔴 Slight decline — potentially routine fluctuation."
        symbol = "🔴↘️"
    elif pct > -10:
        color = "inverse"
        note = "🔴 Noticeable decline — possible concerns or market correction."
        symbol = "🔴⬇️"
    else:
        color = "inverse"
        note = "🔴 Sharp drop — reaction to negative news or major events."
        symbol = "🔴⬇️⬇️"
    return f"${change:.2f}", f"{pct:.2f}%", color, note, symbol

current_price = info.get("currentPrice", 0.0)
price_1d = hist["Close"].iloc[-2] if len(hist) > 1 else None
price_1m = hist["Close"].iloc[-21] if len(hist) > 21 else None
price_6m = hist["Close"].iloc[0] if len(hist) > 0 else None

change_1d, pct_1d, color_1d, note_1d, symbol_1d = calc_change(current_price, price_1d)
change_1m, pct_1m, color_1m, note_1m, symbol_1m = calc_change(current_price, price_1m)
change_6m, pct_6m, color_6m, note_6m, symbol_6m = calc_change(current_price, price_6m)

# --- DISPLAY PANEL ---
st.markdown(f"### 💵 **{info.get('shortName', ticker)} ({ticker})**")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:.2f}")
col1.markdown(f"<div style='font-size:14px; color:#000000;'>This is the latest trading price for {ticker}.</div>", unsafe_allow_html=True)
col2.metric("24h Change", f"{pct_1d} {symbol_1d}", delta_color=color_1d)
col2.markdown(f"<div style='font-size:14px; color:#000000;'>{note_1d}</div>", unsafe_allow_html=True)
col3.metric("1 Month Change", f"{pct_1m} {symbol_1m}", delta_color=color_1m)
col3.markdown(f"<div style='font-size:14px; color:#000000;'>{note_1m}</div>", unsafe_allow_html=True)
col4.metric("6 Month Change", f"{pct_6m} {symbol_6m}", delta_color=color_6m)
col4.markdown(f"<div style='font-size:14px; color:#000000;'>{note_6m}</div>", unsafe_allow_html=True)

# The rest of the code remains unchanged.
