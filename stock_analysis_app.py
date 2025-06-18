
# ai_stock_analyzer_app/main.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="📈 AI Stock Analyzer", layout="wide")

# --- DARK THEME STYLE ---
st.markdown("""
    <style>
        body, .stApp {
            background-color: #111111;
            color: #F5F5F5;
        }
        .css-1v0mbdj, .css-1cpxqw2, .css-qrbaxs {
            background-color: #1E1E1E;
            color: #F5F5F5;
        }
        .st-bb, .st-bc, .st-bd {
            color: #F5F5F5;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #4CAF50;
        }
        .stMetricValue, .stMetricDelta {
            color: #F5F5F5 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div style="text-align:center">
    <h1 style="color:#4CAF50;">📊 AI Stock Analyzer</h1>
    <p style="font-size:18px;">Smart insights for smarter investing — built with ❤️ using Streamlit</p>
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

# --- SUPPORT & RESISTANCE DISPLAY ---
st.markdown("## 🧭 Key Price Levels")
sr_col1, sr_col2 = st.columns(2)
sr_col1.metric("🔻 Support Level", f"${support:.2f}", help="A lower price range where the stock may find buying interest.")
sr_col2.metric("🔺 Resistance Level", f"${resistance:.2f}", help="An upper price range where the stock may face selling pressure.")

# --- ANALYST RATINGS & EARNINGS CALENDAR ---
analysts = data.recommendations.dropna() if hasattr(data, 'recommendations') else pd.DataFrame()
next_earnings = data.calendar.get('Earnings Date', [None])[0] if hasattr(data, 'calendar') else None

# --- PDF EXPORT HELPERS ---
def create_pdf_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

def get_table_download_link(fig, filename="report.png"):
    img_buf = create_pdf_image(fig)
    b64 = base64.b64encode(img_buf.read()).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">📥 Download Chart</a>'

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

# --- TECHNICAL INDICATORS ---
st.markdown("## 📊 Technical Indicators")
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

st.line_chart(hist[['Close', 'MA20', 'MA50']], use_container_width=True)
st.line_chart(hist[['RSI', 'MACD']], use_container_width=True)

# --- SUPPORT/RESISTANCE & MOVING AVERAGES CHART ---
st.markdown("### 🔍 Support & Resistance")
fig, ax = plt.subplots(figsize=(10, 4))
hist['Close'].plot(ax=ax, label='Close', color='blue')
hist['MA20'].plot(ax=ax, label='MA20', color='orange')
hist['MA50'].plot(ax=ax, label='MA50', color='green')
ax.axhline(support, linestyle='--', color='red', label='Support')
ax.axhline(resistance, linestyle='--', color='purple', label='Resistance')
ax.legend(loc='upper left')
st.pyplot(fig)
st.markdown(get_table_download_link(fig), unsafe_allow_html=True)

# --- ANALYST RATINGS ---
st.subheader("📋 Analyst Ratings")
if not analysts.empty:
    recent = analysts.groupby(['Firm', 'To Grade']).size().reset_index(name='Count').sort_values('Count', ascending=False)
    st.write(recent.head(5))
else:
    st.info("No recent analyst rating data available.")

# --- EARNINGS CALENDAR ---
st.subheader("🗓️ Upcoming Earnings")
if next_earnings:
    st.info(f"Next earnings date: {next_earnings.date()}")
else:
    st.info("Earnings calendar unavailable.")

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p style="color:#888888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
