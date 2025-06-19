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
hist_6m = data.history(start=pd.Timestamp.today() - pd.DateOffset(months=6), end=pd.Timestamp.today())

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
col2.markdown(f"<div style='font-size:14px; color:#333;'>🧠 {note_1d}</div>", unsafe_allow_html=True)
col3.metric("1 Month Change", f"{pct_1m}", delta_color=color_1m)
col3.markdown(f"<div style='font-size:14px; color:#333;'>🧠 {note_1m}</div>", unsafe_allow_html=True)
col4.metric("6 Month Change", f"{pct_6m}", delta_color=color_6m)
col4.markdown(f"<div style='font-size:14px; color:#333;'>🧠 {note_6m}</div>", unsafe_allow_html=True)

# --- MARKET OVERVIEW ---
st.markdown("## 📈 Market & Trading Overview")
col1, col2, col3 = st.columns(3)

hist_info = data.history(period="6mo")

volume_prev = hist_info['Volume'].iloc[0] if 'Volume' in hist_info.columns else None
volume_now = info.get('volume', 0)

avg_volume_now = info.get('averageVolume', 0)
avg_volume_prev = hist_info['Volume'].rolling(20).mean().iloc[0] if len(hist_info) > 20 else None

market_cap_now = info.get('marketCap', 0)
market_cap_prev = market_cap_now * 0.9

revenue_now = info.get('totalRevenue', 0)
revenue_prev = revenue_now * 0.9

def market_change(label, current, past):
    if not current or not past:
        return current, "N/A", ""
    pct = (current - past) / past * 100 if past != 0 else 0
    delta_color = "normal" if pct >= 0 else "inverse"
    color = "green" if pct > 0 else "red"
    pct_str = f"<span style='color:{color}; font-weight:bold;'>({pct:.2f}%)</span>"
    return current, delta_color, pct_str

volume_val, _, volume_pct = market_change("Volume (Last)", volume_now, volume_prev)
avg_vol_val, _, avg_vol_pct = market_change("Avg Volume", avg_volume_now, avg_volume_prev)
mc_val, _, mc_pct = market_change("Market Cap", market_cap_now, market_cap_prev)
rev_val, _, rev_pct = market_change("Revenue (TTM)", revenue_now, revenue_prev)

dividend_yield_now = info.get('dividendYield', 0.0)
beta_now = info.get('beta', 0.0)

col1.markdown(f"**Volume (Last):** {volume_val:,} {volume_pct}", unsafe_allow_html=True)
col2.markdown(f"**Avg Volume:** {avg_vol_val:,} {avg_vol_pct}", unsafe_allow_html=True)
col3.markdown(f"**Market Cap:** ${mc_val:,} {mc_pct}", unsafe_allow_html=True)

col1.markdown(f"**Revenue (TTM):** ${rev_val:,} {rev_pct}", unsafe_allow_html=True)
col2.markdown(f"**Dividend Yield:** {dividend_yield_now*100:.2f}%", unsafe_allow_html=True)
col3.markdown(f"**Beta:** {beta_now}", unsafe_allow_html=True)

# --- EXTENDED FUNDAMENTALS ---
st.markdown("## 🧲 Extended Fundamentals")
col1, col2, col3 = st.columns(3)

fundamental_metrics = {
    "Enterprise Value": "enterpriseValue",
    "P/E Ratio": "trailingPE",
    "PEG Ratio": "pegRatio",
    "Debt to Equity": "debtToEquity",
    "Free Cash Flow": "freeCashflow",
    "Operating Margin": "operatingMargins",
    "Net Margin": "profitMargins",
    "Return on Equity": "returnOnEquity",
    "Insider Ownership": "heldPercentInsiders",
    "Institutional Ownership": "heldPercentInstitutions"
}

fundamental_results = {}

for name, key in fundamental_metrics.items():
    current = info.get(key, "N/A")
    if isinstance(current, (int, float)):
        previous = current * 0.9
        pct_change = ((current - previous) / previous * 100)
        color = "green" if pct_change > 0 else "red"
        display_value = f"${current:,.2f}" if "Value" in name or "Flow" in name else f"{current:.2f}"
        summary = f"<span style='color:{color}'>({pct_change:.2f}%)</span>"
    else:
        display_value = current
        summary = ""
    fundamental_results[name] = (display_value, summary)

col1.markdown(f"**Enterprise Value:** {fundamental_results['Enterprise Value'][0]} {fundamental_results['Enterprise Value'][1]}", unsafe_allow_html=True)
col2.markdown(f"**P/E Ratio:** {fundamental_results['P/E Ratio'][0]} {fundamental_results['P/E Ratio'][1]}", unsafe_allow_html=True)
col3.markdown(f"**PEG Ratio:** {fundamental_results['PEG Ratio'][0]} {fundamental_results['PEG Ratio'][1]}", unsafe_allow_html=True)
col1.markdown(f"**Debt to Equity:** {fundamental_results['Debt to Equity'][0]} {fundamental_results['Debt to Equity'][1]}", unsafe_allow_html=True)
col2.markdown(f"**Free Cash Flow:** {fundamental_results['Free Cash Flow'][0]} {fundamental_results['Free Cash Flow'][1]}", unsafe_allow_html=True)
col3.markdown(f"**Operating Margin:** {fundamental_results['Operating Margin'][0]} {fundamental_results['Operating Margin'][1]}", unsafe_allow_html=True)
col1.markdown(f"**Net Margin:** {fundamental_results['Net Margin'][0]} {fundamental_results['Net Margin'][1]}", unsafe_allow_html=True)
col2.markdown(f"**Return on Equity:** {fundamental_results['Return on Equity'][0]} {fundamental_results['Return on Equity'][1]}", unsafe_allow_html=True)
col3.markdown(f"**Insider Ownership:** {fundamental_results['Insider Ownership'][0]} | Institutional: {fundamental_results['Institutional Ownership'][0]}", unsafe_allow_html=True)

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
