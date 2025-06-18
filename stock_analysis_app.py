# ai_stock_analyzer_app/main.py

import streamlit as st
import yfinance as yf
import pandas as pd
import openai
import os

st.set_page_config(page_title="📈 AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer")

# User input
st.sidebar.header("Enter Stock Ticker")
ticker = st.sidebar.text_input("Example: SPCE", value="SPCE").upper()

if not ticker:
    st.warning("Please enter a valid ticker symbol.")
    st.stop()

# Fetch data
data = yf.Ticker(ticker)
info = data.info
hist = data.history(period="1y")

st.header(f"{info.get('shortName', ticker)} ({ticker})")
st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")

# Show financials
with st.expander("📊 Fundamental Metrics"):
    fundamentals = {
        "Revenue (TTM)": info.get("totalRevenue"),
        "Profit Margin": info.get("profitMargins"),
        "EPS": info.get("trailingEps"),
        "PE Ratio": info.get("trailingPE"),
        "Forward PE": info.get("forwardPE"),
        "P/S Ratio": info.get("priceToSalesTrailing12Months"),
        "P/B Ratio": info.get("priceToBook"),
        "Book Value": info.get("bookValue"),
        "Beta": info.get("beta"),
        "Debt-to-Equity": info.get("debtToEquity"),
        "Operating Margin": info.get("operatingMargins"),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets")
    }
    df_fundamentals = pd.DataFrame(fundamentals.items(), columns=["Metric", "Value"])
    st.dataframe(df_fundamentals)

# Technical indicators
with st.expander("📈 Technical Indicators"):
    hist['MA20'] = hist['Close'].rolling(20).mean()
    hist['MA50'] = hist['Close'].rolling(50).mean()
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))

    st.line_chart(hist[['Close', 'MA20', 'MA50']])
    st.line_chart(hist['RSI'])

# AI-powered analysis prompts
openai.api_key = os.getenv("OPENAI_API_KEY")

prompts = [
    "Please analyze the most recent quarterly earnings report for {ticker}. Highlight changes in revenue, gross margin, operating margin, net income, cash flow from operations, and major asset or liability movements compared to the last 3 quarters.",
    "Analyze {ticker}’s recent technical indicators—RSI, MACD, 50/200-day moving averages, Bollinger Bands, and ADX—over the last 30 trading days. What are the short-term and medium-term signals?",
    "Summarize the top {ticker}-related news headlines from the past 30 days, and identify which items are likely to influence near-term stock price.",
    "Provide {ticker}’s valuation multiples—P/E, P/S, EV/EBITDA, and PEG ratio. How do these align with its revenue growth and margin profile?",
    "What are {ticker}’s major upcoming catalysts—such as contract wins, AI software updates, partnerships, or regulatory approvals?",
    "Based on the latest technical and fundamental analysis, identify optimal entry points for a long position in {ticker}. Recommend stop-loss and take-profit levels."
]

st.header("🧠 AI-Generated Insights")
if st.button("Generate Analysis"):
    with st.spinner("Contacting GPT-4 for deep analysis..."):
        try:
            for p in prompts:
                query = p.format(ticker=ticker)
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": query}]
                )
                st.subheader(query.split(".")[0])
                st.write(response['choices'][0]['message']['content'])
        except Exception as e:
            st.error(f"An error occurred: {e}")
