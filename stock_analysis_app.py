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
    # Prepare insights before table
    desired = ['Total Revenue','Revenue','Gross Profit','Operating Income','EBIT','Net Income','Net Income(Loss)','Operating Cash Flow']
    avail = [c for c in desired if c in df_income.columns]
    if not avail:
        st.error("No standard fields in quarterly data.")
        return
    df4 = df_income[avail].iloc[:4]
    df4.index = pd.to_datetime(df4.index).to_period('Q')
    changes = df4.pct_change().iloc[1:]*100
    insights = []
    for metric in df4.columns:
        pct = changes.iloc[0].get(metric, np.nan)
        if not np.isnan(pct):
            direction = 'increased' if pct>0 else 'decreased'
            insights.append(f"• {metric} {direction} by {pct:.1f}% vs prior quarter.")
    st.markdown(f"<div class='card-dark'><b>🧠 Earnings Insight:</b><br>{'<br>'.join(insights)}</div>", unsafe_allow_html=True)
    # Display financial table
    st.dataframe(df4.style.format("${:,.0f}"))

# Render fundamental analysis
render_fundamental_analysis(ticker)

# --- FETCH DATA & INDICATORS ---
data = yf.Ticker(ticker)
info = data.info
hist = data.history(period="6mo")
hist['MA20'] = hist['Close'].rolling(20).mean()
hist['MA50'] = hist['Close'].rolling(50).mean()
# Technical indicators computation
# RSI
...
# ADX
...

# --- TECHNICAL ANALYSIS CARD ---
st.markdown("<div class='card'><h2>📈 Technical Analysis</h2></div>", unsafe_allow_html=True)
# Prepare technical insights before charts
rsi_val = hist['RSI'].iloc[-1]
ma_trend = 'upward' if hist['Close'].iloc[-1] > hist['MA50'].iloc[-1] else 'downward'
macd_val = hist['MACD'].iloc[-1]
bb_touch = 'above upper band' if hist['Close'].iloc[-1] > hist['BB_upper'].iloc[-1] else 'below lower band' if hist['Close'].iloc[-1] < hist['BB_lower'].iloc[-1] else 'within bands'
adx_val = hist['ADX'].iloc[-1]
adx_text = f"{adx_val:.1f}" if not np.isnan(adx_val) else 'N/A'
tech_insights = []
tech_insights.append(f"• RSI at {rsi_val:.1f} indicates {'overbought' if rsi_val>70 else 'oversold' if rsi_val<30 else 'neutral'} conditions.")
tech_insights.append(f"• Price {ma_trend} relative to 50-day MA.")
tech_insights.append(f"• MACD {macd_val:.2f} suggests {'bullish' if macd_val>0 else 'bearish'} momentum.")
tech_insights.append(f"• Price is {bb_touch}, indicating volatility levels.")
tech_insights.append(f"• ADX at {adx_text} signifies {'strong' if adx_val>25 else 'weak'} trend.")
st.markdown(f"<div class='card-dark'><b>🧠 Technical Insight:</b><br>{'<br>'.join(tech_insights)}</div>", unsafe_allow_html=True)
# RSI & MACD chart
g... # charts code continues

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p style="color:#888888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
