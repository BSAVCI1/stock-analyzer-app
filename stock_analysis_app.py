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
price_1d = hist["Close"].iloc[-2] if len(hist) > 1 else None
price_1m = hist["Close"].iloc[-21] if len(hist) > 21 else None
price_6m = hist["Close"].iloc[0] if len(hist) > 0 else None

change_1d, pct_1d, color_1d, note_1d = calc_change(current_price, price_1d)
change_1m, pct_1m, color_1m, note_1m = calc_change(current_price, price_1m)
change_6m, pct_6m, color_6m, note_6m = calc_change(current_price, price_6m)

# --- DISPLAY PANEL ---
st.markdown(f"### 💵 **{info.get('shortName', ticker)} ({ticker})**")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:.2f}")
col1.markdown(f"<div style='font-size:14px; color:#333;'>This is the latest trading price for {ticker}.</div>", unsafe_allow_html=True)
col2.metric("24h Change", f"{pct_1d}", delta_color=color_1d)
col2.markdown(f"<div style='font-size:14px; color:#333;'>{note_1d}</div>", unsafe_allow_html=True)
col3.metric("1 Month Change", f"{pct_1m}", delta_color=color_1m)
col3.markdown(f"<div style='font-size:14px; color:#333;'>{note_1m}</div>", unsafe_allow_html=True)
col4.metric("6 Month Change", f"{pct_6m}", delta_color=color_6m)
col4.markdown(f"<div style='font-size:14px; color:#333;'>{note_6m}</div>", unsafe_allow_html=True)

# --- MARKET OVERVIEW ---
st.markdown("## 🧾 Market & Trading Overview")
col1, col2, col3 = st.columns(3)

col1.metric("Volume (Last)", f"{info.get('volume', 0):,}")
col2.metric("Avg Volume", f"{info.get('averageVolume', 0):,}")
col3.metric("Market Cap", f"${info.get('marketCap', 0):,}")

col1.metric("Revenue (TTM)", f"${info.get('totalRevenue', 0):,}")
col2.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get("dividendYield") else "N/A")
col3.metric("Beta", f"{info.get('beta', 'N/A')}")

# --- SUPPORT & RESISTANCE DISPLAY ---
st.markdown("## 🧭 Key Price Levels")
sr_col1, sr_col2 = st.columns(2)
sr_col1.metric("🔻 Support Level", f"${support:.2f}", help="A lower price range where the stock may find buying interest.")
sr_col2.metric("🔺 Resistance Level", f"${resistance:.2f}", help="An upper price range where the stock may face selling pressure.")

# --- TECHNICAL INDICATORS TABLE ---
st.markdown("## 📊 Technical Indicators Summary")
delta = hist['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
hist['RSI'] = 100 - (100 / (1 + rs))

hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
hist['MACD'] = hist['EMA12'] - hist['EMA26']

hist['MA20'] = hist['Close'].rolling(20).mean()
hist['MA50'] = hist['Close'].rolling(50).mean()

tech_df = pd.DataFrame({
    'Indicator': ['RSI (14)', 'MACD', 'MA20', 'MA50'],
    'Value': [
        round(hist['RSI'].iloc[-1], 2),
        round(hist['MACD'].iloc[-1], 2),
        round(hist['MA20'].iloc[-1], 2),
        round(hist['MA50'].iloc[-1], 2)
    ],
    'Interpretation': [
        'Overbought' if hist['RSI'].iloc[-1] > 70 else 'Oversold' if hist['RSI'].iloc[-1] < 30 else 'Neutral',
        'Positive momentum' if hist['MACD'].iloc[-1] > 0 else 'Negative momentum',
        'Trending above short MA' if current_price > hist['MA20'].iloc[-1] else 'Below short MA',
        'Trending above long MA' if current_price > hist['MA50'].iloc[-1] else 'Below long MA'
    ]
})
st.dataframe(tech_df, use_container_width=True)

# --- NEWS & SENTIMENT ---
st.markdown("## 📰 Recent News & Market Sentiment")
news_url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
try:
    res = requests.get(news_url, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")
    headlines = soup.find_all("h3")[:5]
    if headlines:
        for h in headlines:
            link_tag = h.find("a")
            if link_tag and link_tag.text:
                st.markdown(f"- [{link_tag.text}](https://finance.yahoo.com{link_tag['href']})")
    else:
        st.info("No recent headlines available.")
except Exception as e:
    st.warning(f"Unable to fetch news. Reason: {e}")

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p style="color:#888888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
