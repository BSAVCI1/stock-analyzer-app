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
peer_list = [p.strip() for p in peers_input.split(',') if p.strip()]

# --- FETCH DATA ---
data = yf.Ticker(ticker)
info = data.info
hist = data.history(period="6mo")
hist['MA20'] = hist['Close'].rolling(20).mean()
hist['MA50'] = hist['Close'].rolling(50).mean()

# --- MARKET OVERVIEW & SUPPORT/RESISTANCE ---
st.markdown("<div class='card'><h2>📈 Market & Trading Overview</h2></div>", unsafe_allow_html=True)
vol = info.get('volume', 0)
avg_vol = info.get('averageVolume', 0)
mc = info.get('marketCap', 0)
rev = info.get('totalRevenue', 0)
dy = info.get('dividendYield', 0) * 100
beta = info.get('beta', 0)
cols = st.columns(3)
cols[0].markdown(f"**Volume:** {vol:,} {'<span class=\"arrow-up\">▲</span>' if vol>avg_vol else '<span class=\"arrow-down\">▼</span>'} <abbr title='Shares traded in last session.'>ℹ️</abbr>", unsafe_allow_html=True)
cols[1].markdown(f"**Avg Volume:** {avg_vol:,} <abbr title='30-day avg volume.'>ℹ️</abbr>", unsafe_allow_html=True)
cols[2].markdown(f"**Market Cap:** ${mc:,} <abbr title='Total market value.'>ℹ️</abbr>", unsafe_allow_html=True)
cols2 = st.columns(3)
cols2[0].markdown(f"**Revenue (TTM):** ${rev:,} <abbr title='Trailing 12m revenue.'>ℹ️</abbr>", unsafe_allow_html=True)
cols2[1].markdown(f"**Dividend Yield:** {dy:.2f}% <abbr title='Annual dividend %.'>ℹ️</abbr>", unsafe_allow_html=True)
cols2[2].markdown(f"**Beta:** {beta:.2f} <abbr title='Volatility vs market.'>ℹ️</abbr>", unsafe_allow_html=True)
ins = f"Volume {'above' if vol>avg_vol else 'below'} average; Market cap {'small' if mc<1e9 else 'mid/large'} cap."
st.markdown(f"<div class='card-dark'>🔍 {ins}</div>", unsafe_allow_html=True)

# --- EXTENDED FUNDAMENTALS ---
st.markdown("<div class='card'><h2>🧲 Fundamental Breakdown</h2></div>", unsafe_allow_html=True)
sections = {
    'Valuation': [('P/E Ratio', 'trailingPE', '15–25 fair'), ('PEG Ratio', 'pegRatio', '~1 fair')],
    'Profitability': [('Net Margin', 'profitMargins', '>5% profitable'), ('ROE', 'returnOnEquity', '>15% strong')],
    'Leverage': [('Debt/Equity', 'debtToEquity', '<1 comfortable'), ('Enterprise Value', 'enterpriseValue', '<1.5×MC typical')]
}
for sec, items in sections.items():
    st.markdown(f"**{sec}**")
    for name, key, tip in items:
        raw = info.get(key)
        if isinstance(raw, (int, float)):
            disp = f"{raw*100:.2f}%" if '%' in tip else f"${raw:,.2f}" 
            color = 'green' if raw >= 0 else 'red'
            st.markdown(f"- {name}: <span style='color:{color}; font-weight:bold;'>{disp}</span> <abbr title='{tip}'>ℹ️</abbr>", unsafe_allow_html=True)
avg_peers = np.mean([yf.Ticker(p).info.get('trailingPE', np.nan) for p in peer_list if yf.Ticker(p).info.get('trailingPE')])
fund_insight = 'Attractive valuation compared to peers.' if info.get('trailingPE', np.nan) < avg_peers else 'Valuation above peer median.'
st.markdown(f"<div class='card-dark'>🧠 {fund_insight}</div>", unsafe_allow_html=True)

# --- FUNDAMENTAL ANALYSIS MODULE ---
def render_fundamental_analysis(ticker: str):
    data = yf.Ticker(ticker)
    st.markdown("<div class='card'><h2>📊 Quarterly Earnings Review</h2></div>", unsafe_allow_html=True)
    try:
        df_income = data.quarterly_financials.T
    except Exception:
        st.error("Unable to fetch quarterly financials.")
        return
    desired = ['Total Revenue','Revenue','Gross Profit','Operating Income','EBIT','Net Income','Operating Cash Flow']
    avail = [c for c in desired if c in df_income.columns]
    if not avail:
        st.error("No standard fields in quarterly data.")
        return
    df4 = df_income[avail].iloc[:4]
    df4.index = pd.to_datetime(df4.index).to_period('Q')
    changes = df4.pct_change().iloc[1:] * 100
    insights = [f"• {m} {'increased' if changes.iloc[0][m]>0 else 'decreased'} by {changes.iloc[0][m]:.1f}% vs prior quarter." for m in df4.columns if not np.isnan(changes.iloc[0][m])]
    st.markdown(f"<div class='card-dark'><b>🧠 Earnings Insight:</b><br>{'<br>'.join(insights)}</div>", unsafe_allow_html=True)
    st.dataframe(df4.style.format("${:,.0f}"))
render_fundamental_analysis(ticker)

# --- TECHNICAL ANALYSIS MODULE ---
# RSI
delta = hist['Close'].diff()
...  # unchanged technical code continues

# --- PEER COMPARISON MODULE ---
...  # unchanged

# --- NEWS & SENTIMENT MODULE ---
...  # unchanged

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p style="color:#888888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
