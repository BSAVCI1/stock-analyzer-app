# ai_stock_analyzer_app/main.py

import streamlit as st
import yfinance as yf
import pandas as pd
import openai
import os

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
hist = data.history(period="1y")

st.header(f"{info.get('shortName', ticker)} ({ticker})")
st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")

# --- FUNDAMENTALS SECTION ---
with st.expander("🔍 Fundamentals Overview"):
    st.subheader("📊 Key Financial Metrics with Insights")

    def interpret_metric(label, value):
        status = ""
        note = ""
        color = "black"

        if label == "PE Ratio":
            if value is None:
                return ("N/A", "No earnings data available", "gray")
            elif value < 15:
                status, note, color = "✅ Good", "Undervalued compared to peers", "green"
            elif value < 30:
                status, note, color = "⚠️ Moderate", "Fairly valued", "orange"
            else:
                status, note, color = "❌ High", "Likely overvalued", "red"

        elif label == "Profit Margin":
            if value is None:
                return ("N/A", "Data not available", "gray")
            elif value > 0.15:
                status, note, color = "✅ Strong", "Healthy profitability", "green"
            elif value > 0:
                status, note, color = "⚠️ Low", "Slim margins", "orange"
            else:
                status, note, color = "❌ Negative", "Losing money", "red"

        elif label == "Debt-to-Equity":
            if value is None:
                return ("N/A", "No data", "gray")
            elif value < 1:
                status, note, color = "✅ Low", "Financially safe", "green"
            elif value < 2:
                status, note, color = "⚠️ Moderate", "Manageable debt", "orange"
            else:
                status, note, color = "❌ High", "Debt-heavy balance sheet", "red"

        return (f"{value:.2f}", note, color)

    metrics = [
        ("PE Ratio", info.get("trailingPE")),
        ("Profit Margin", info.get("profitMargins")),
        ("Debt-to-Equity", info.get("debtToEquity")),
        ("EPS", info.get("trailingEps")),
        ("ROE", info.get("returnOnEquity")),
        ("ROA", info.get("returnOnAssets"))
    ]

    for label, value in metrics:
        display_val, insight, color = interpret_metric(label, value)
        st.markdown(f"<b style='color:{color}'>{label}: {display_val}</b> — <i>{insight}</i>", unsafe_allow_html=True)

# --- TECHNICALS ---
with st.expander("📈 Price Trend and RSI"):
    hist['MA20'] = hist['Close'].rolling(20).mean()
    hist['MA50'] = hist['Close'].rolling(50).mean()
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))

    st.line_chart(hist[['Close', 'MA20', 'MA50']])
    st.line_chart(hist['RSI'])

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p>Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
