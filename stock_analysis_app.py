# ai_stock_analyzer_app/main.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from bs4 import BeautifulSoup
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="BSAV Stock Analyzer", layout="wide")

# --- GLOBAL STYLES ---
st.markdown("""
<style>
.card {background:#ffffff; padding:20px; margin-bottom:20px; border-radius:10px; box-shadow:0 2px 4px rgba(0,0,0,0.1);}
.card-dark {background:#2b2b2b; color:#fff; padding:20px; margin-bottom:20px; border-radius:10px;}
.metric-tooltip {text-decoration:underline; cursor:help;}
.arrow-up {color:green;}
.arrow-down {color:red;}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="card" style="text-align:center;">
    <h1 style="color:#4CAF50; margin-bottom:5px;">📊 AI Stock Analyzer</h1>
    <p style="font-size:16px; color:#555;">Interactive, non-finance friendly insights with action recommendations</p>
</div>
""", unsafe_allow_html=True)

# --- USER INPUT ---
st.sidebar.header("Select Stock & Peers")
popular = ["AAPL","MSFT","GOOGL","AMZN","SPCE","TSLA"]
ticker = st.sidebar.selectbox("Choose Ticker", options=popular, index=popular.index("SPCE"))
peers_input = st.sidebar.text_input("Or enter peers (comma separated)", "AAPL,MSFT,GOOGL").upper()
peer_list = [p.strip() for p in peers_input.split(",") if p.strip()]

# --- FUNDAMENTAL ANALYSIS MODULE ---
def render_fundamental_analysis(ticker: str):
    data = yf.Ticker(ticker)
    st.markdown("<div class='card'><h2>📊 Quarterly Earnings Review</h2></div>", unsafe_allow_html=True)
    try:
        df_income = data.quarterly_financials.T
    except Exception:
        st.error("Unable to fetch quarterly financials.")
        return
    df4 = df_income.iloc[:4][[
        'Total Revenue',
        'Gross Profit',
        'Operating Income',
        'Net Income',
        'Operating Cash Flow'
    ]]
    df4.index = pd.to_datetime(df4.index).to_period('Q')
    st.dataframe(df4.style.format("${:,.0f}"))
    changes = df4.pct_change().iloc[1:] * 100
    insight_lines = []
    for metric in df4.columns:
        pct = changes.loc[df4.index[0], metric]
        direction = 'increase' if pct > 0 else 'decrease'
        insight_lines.append(f"• {metric}: {direction} of {pct:.1f}% vs prior quarter.")
    insight_text = '<br>'.join(insight_lines)
    st.markdown(f"<div class='card-dark'><b>🧠 Earnings Insight:</b><br>{insight_text}</div>", unsafe_allow_html=True)

# --- RENDER QUARTERLY EARNINGS ---
render_fundamental_analysis(ticker)

# --- FETCH PRICES & INDICATORS ---
data = yf.Ticker(ticker)
info = data.info
hist = data.history(period="6mo")
hist['MA20'] = hist['Close'].rolling(window=20).mean()
hist['MA50'] = hist['Close'].rolling(window=50).mean()
support = np.percentile(hist['Low'], 10)
resistance = np.percentile(hist['High'], 90)

# rest of your main code (price, market overview, etc.) follows...

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p style="color:#888888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)

