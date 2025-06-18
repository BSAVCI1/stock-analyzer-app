import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import openai
import os

# Set up page configuration
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("\U0001F4C8 AI Stock Analyzer")

# User input
st.sidebar.title("Stock Settings")
tickers = st.sidebar.text_input("Enter Stock Tickers (comma separated)", value="AAPL,NVDA").upper().split(",")

st.title("📊 Stock Comparison")

if len(tickers) > 1:
    st.write(f"Comparing: {', '.join(tickers)}")


# Get historical data
hist = data.history(period="6mo")

# Plot candlestick chart
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=hist.index,
    open=hist['Open'],
    high=hist['High'],
    low=hist['Low'],
    close=hist['Close'],
    name='Candlestick'
))
fig.update_layout(title=f'{ticker} Price Chart', xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Technical Indicators
hist['MA20'] = hist['Close'].rolling(20).mean()
st.subheader("📉 Moving Average (20-day)")
st.line_chart(hist[['Close', 'MA20']])

# RSI Calculation
delta = hist['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
hist['RSI'] = 100 - (100 / (1 + rs))
st.subheader("📊 Relative Strength Index (RSI)")
st.line_chart(hist['RSI'])

# Show fundamentals
info = data.info
st.subheader("📊 Fundamentals")
st.write({
    "PE Ratio": info.get("trailingPE"),
    "Market Cap": info.get("marketCap"),
    "Revenue (TTM)": info.get("totalRevenue"),
    "Profit Margin": info.get("profitMargins"),
    "Dividend Yield": info.get("dividendYield")
})

# AI Financial Summary
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_ai_summary(ticker, info):
    prompt = f"Give a financial summary and investment recommendation for {ticker} based on the following info:\n{info}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

if st.button("Get AI Summary"):
    with st.spinner("Analyzing with AI..."):
        ai_summary = get_ai_summary(ticker, info)
        st.markdown("### \U0001F916 AI Summary")
        st.write(ai_summary)
