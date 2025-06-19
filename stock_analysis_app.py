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
    # Select available financial fields for last 4 quarters
    desired = [
        'Total Revenue', 'Revenue', 'Gross Profit', 'Operating Income',
        'EBIT', 'Net Income', 'Net Income\(Loss\)', 'Operating Cash Flow'
    ]
    available = [col for col in desired if col in df_income.columns]
    if not available:
        st.error("No standard financial fields found in quarterly data.")
        return
    df4 = df_income[available].iloc[:4]
    df4.index = pd.to_datetime(df4.index).to_period('Q')
    st.dataframe(df4.style.format("${:,.0f}"))
    # Compute QoQ changes
    changes = df4.pct_change().iloc[1:] * 100
    insight_lines = []
    for metric in df4.columns:
        pct = changes.iloc[0].get(metric, None)
        if pct is None or np.isnan(pct):
            continue
        direction = 'increase' if pct > 0 else 'decrease'
        insight_lines.append(f"• {metric}: {direction} of {pct:.1f}% vs prior quarter.")
    insight_text = '<br>'.join(insight_lines)
    st.markdown(f"<div class='card-dark'><b>🧠 Earnings Insight:</b><br>{insight_text}</div>", unsafe_allow_html=True)

# --- RENDER QUARTERLY EARNINGS ---
render_fundamental_analysis(ticker)

# --- FETCH PRICES & INDICATORS ---

# --- FETCH PRICES & INDICATORS ---
data = yf.Ticker(ticker)
info = data.info
hist = data.history(period="6mo")
hist['MA20'] = hist['Close'].rolling(window=20).mean()
hist['MA50'] = hist['Close'].rolling(window=50).mean()
support = np.percentile(hist['Low'], 10)
resistance = np.percentile(hist['High'], 90)

# --- TECHNICAL ANALYSIS MODULE ---
# Compute indicators: RSI, MACD, Bollinger Bands, ADX
# RSI
delta = hist['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
hist['RSI'] = 100 - (100 / (1 + rs))
# MACD
hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
hist['MACD'] = hist['EMA12'] - hist['EMA26']
# Bollinger Bands
hist['BB_mid'] = hist['Close'].rolling(20).mean()
hist['BB_std'] = hist['Close'].rolling(20).std()
hist['BB_upper'] = hist['BB_mid'] + (2 * hist['BB_std'])
hist['BB_lower'] = hist['BB_mid'] - (2 * hist['BB_std'])
# ADX (approximate using TA-Lib style formulas)
# Since no TA-Lib, compute basic ADX manually
high = hist['High']
low = hist['Low']
close = hist['Close']
# True Range
tr1 = high - low
tr2 = (high - close.shift()).abs()
tr3 = (low - close.shift()).abs()
hist['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
hist['ATR'] = hist['TR'].rolling(14).mean()
# +DM, -DM
up_move = high - high.shift()
down_move = low.shift() - low
plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
hist['+DI'] = 100 * (pd.Series(plus_dm).rolling(14).mean() / hist['ATR'])
hist['-DI'] = 100 * (pd.Series(minus_dm).rolling(14).mean() / hist['ATR'])
hist['DX'] = (abs(hist['+DI'] - hist['-DI']) / (hist['+DI'] + hist['-DI'])) * 100
hist['ADX'] = hist['DX'].rolling(14).mean()

# Display indicators card
st.markdown("<div class='card'> <h2>📈 Technical Analysis</h2>", unsafe_allow_html=True)
# RSI & MACD charts
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], mode='lines', name='RSI'))
fig2.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], mode='lines', name='MACD'))
fig2.update_layout(template='plotly_white', height=300)
st.plotly_chart(fig2, use_container_width=True)
# Bollinger and price
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close'))
fig3.add_trace(go.Scatter(x=hist.index, y=hist['BB_upper'], line=dict(dash='dash'), name='BB Upper'))
fig3.add_trace(go.Scatter(x=hist.index, y=hist['BB_mid'], line=dict(dash='dot'), name='BB Mid'))
fig3.add_trace(go.Scatter(x=hist.index, y=hist['BB_lower'], line=dict(dash='dash'), name='BB Lower'))
fig3.update_layout(template='plotly_white', height=300)
st.plotly_chart(fig3, use_container_width=True)
# ADX summary
adx_latest = hist['ADX'].iloc[-1]
adx_comment = 'Strong trend' if adx_latest > 25 else 'Weak trend'
st.markdown(f"<div class='card-dark'>🧠 ADX ({adx_latest:.1f}): {adx_comment}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Continue with Market Overview and subsequent modules
# rest of the code...

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p style="color:#888888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
