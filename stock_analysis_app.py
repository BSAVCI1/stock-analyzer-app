import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="📈 BSAV Stock Analyzer", layout="wide")

# --- DARK THEME STYLE ---
st.markdown("""
    <style>
        body, .stApp {
            background-color: #111111;
            color: #E0E0E0;
        }
        .css-1v0mbdj, .css-1cpxqw2, .css-qrbaxs {
            background-color: #1E1E1E;
            color: #E0E0E0;
        }
        .st-bb, .st-bc, .st-bd {
            color: #E0E0E0;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #4CAF50;
        }
        .stMetricValue, .stMetricDelta {
            color: white !important;
        }
        .note {
            font-size: 14px;
            color: #BBBBBB;
            margin-top: -10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div style="text-align:center">
    <h1>📊 AI Stock Analyzer</h1>
    <p style="font-size:18px; color:#BBBBBB;">Smart insights for smarter investing — built with ❤️ using Streamlit</p>
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

# --- DISPLAY PANEL ---
st.markdown(f"### 💵 **{info.get('shortName', ticker)} ({ticker})**")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:.2f}")
col1.markdown(f"<div class='note'>This is the latest trading price for {ticker}.</div>", unsafe_allow_html=True)
col2.metric("24h Change", pct_1d, delta_color="inverse")
col2.markdown("<div class='note'>A positive change may indicate short-term optimism.</div>", unsafe_allow_html=True)
col3.metric("1 Week Change", pct_1w)
col3.markdown("<div class='note'>Reflects weekly trend direction.</div>", unsafe_allow_html=True)
col4.metric("1 Month Change", pct_1m)
col4.markdown("<div class='note'>Shows price momentum over the last month.</div>", unsafe_allow_html=True)

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

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p style="color:#888888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
