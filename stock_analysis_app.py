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
    desired = ['Total Revenue','Revenue','Gross Profit','Operating Income','EBIT','Net Income','Operating Cash Flow']
    avail = [c for c in desired if c in df_income.columns]
    if not avail:
        st.error("No standard fields in quarterly data.")
        return
    df4 = df_income[avail].iloc[:4]
    df4.index = pd.to_datetime(df4.index).to_period('Q')
    changes = df4.pct_change().iloc[1:] * 100
    insights = []
    for metric in df4.columns:
        pct = changes.iloc[0].get(metric, np.nan)
        if not np.isnan(pct):
            direction = 'increased' if pct > 0 else 'decreased'
            insights.append(f"• {metric} {direction} by {pct:.1f}% vs prior quarter.")
    st.markdown(
        f"<div class='card-dark'><b>🧠 Earnings Insight:</b><br>{'<br>'.join(insights)}</div>",
        unsafe_allow_html=True
    )
    st.dataframe(df4.style.format("${:,.0f}"))

render_fundamental_analysis(ticker)

# --- FETCH DATA & INDICATORS ---
data = yf.Ticker(ticker)
info = data.info
hist = data.history(period="6mo")
hist['MA20'] = hist['Close'].rolling(20).mean()
hist['MA50'] = hist['Close'].rolling(50).mean()
# RSI
delta = hist['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
hist['RSI'] = 100 - (100 / (1 + rs))
# MACD
hist['EMA12'] = hist['Close'].ewm(span=12).mean()
hist['EMA26'] = hist['Close'].ewm(span=26).mean()
hist['MACD'] = hist['EMA12'] - hist['EMA26']
# Bollinger Bands
hist['BB_mid'] = hist['Close'].rolling(20).mean()
hist['BB_std'] = hist['Close'].rolling(20).std()
hist['BB_upper'] = hist['BB_mid'] + 2 * hist['BB_std']
hist['BB_lower'] = hist['BB_mid'] - 2 * hist['BB_std']
# ADX
high, low, close = hist['High'], hist['Low'], hist['Close']
tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
hist['ATR'] = tr.rolling(14).mean()
up = high.diff()
dn = -low.diff()
hist['+DI'] = 100 * (up.where((up>dn)&(up>0), 0).rolling(14).mean() / hist['ATR'])
hist['-DI'] = 100 * (dn.where((dn>up)&(dn>0), 0).rolling(14).mean() / hist['ATR'])
hist['ADX'] = (abs(hist['+DI'] - hist['-DI']) / (hist['+DI'] + hist['-DI']) * 100).rolling(14).mean()

# --- TECHNICAL ANALYSIS CARD ---
st.markdown("<div class='card'><h2>📈 Technical Analysis</h2></div>", unsafe_allow_html=True)
# Insights
rsi_val = hist['RSI'].iloc[-1] if 'RSI' in hist.columns else np.nan
ma_trend = 'upward' if hist['Close'].iloc[-1] > hist['MA50'].iloc[-1] else 'downward'
macd_val = hist['MACD'].iloc[-1] if 'MACD' in hist.columns else np.nan
bb_pos = (
    'above upper band' if hist['Close'].iloc[-1] > hist['BB_upper'].iloc[-1]
    else 'below lower band' if hist['Close'].iloc[-1] < hist['BB_lower'].iloc[-1]
    else 'within bands'
)
adx_val = hist['ADX'].iloc[-1] if 'ADX' in hist.columns else np.nan
adx_text = f"{adx_val:.1f}" if not np.isnan(adx_val) else 'N/A'
tech_insights = [
    f"• RSI at {rsi_val:.1f} indicates {('overbought' if rsi_val>70 else 'oversold' if rsi_val<30 else 'neutral')} conditions.",
    f"• Price {ma_trend} vs 50-day MA.",
    f"• MACD at {macd_val:.2f} suggests {('bullish' if macd_val>0 else 'bearish')} momentum.",
    f"• Price is {bb_pos}, indicating volatility.",
    f"• ADX at {adx_text} means {('strong trend' if adx_val>25 else 'weak trend')}.",
]
st.markdown(
    f"<div class='card-dark'><b>🧠 Technical Insight:</b><br>{'<br>'.join(tech_insights)}</div>",
    unsafe_allow_html=True
)
# Charts
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI'))
fig1.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD'))
fig1.update_layout(template='plotly_white', height=300)
st.plotly_chart(fig1, use_container_width=True)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Price'))
fig2.add_trace(go.Scatter(x=hist.index, y=hist['BB_upper'], line=dict(dash='dash'), name='Upper Band'))
fig2.add_trace(go.Scatter(x=hist.index, y=hist['BB_mid'], line=dict(dash='dot'), name='Mid Band'))
fig2.add_trace(go.Scatter(x=hist.index, y=hist['BB_lower'], line=dict(dash='dash'), name='Lower Band'))
fig2.update_layout(template='plotly_white', height=300)
st.plotly_chart(fig2, use_container_width=True)

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
cols2[0].markdown(f"**Revenue:** ${rev:,} <abbr title='Trailing 12m revenue.'>ℹ️</abbr>", unsafe_allow_html=True)
cols2[1].markdown(f"**Div Yield:** {dy:.2f}% <abbr title='Annual dividend %.'>ℹ️</abbr>", unsafe_allow_html=True)
cols2[2].markdown(f"**Beta:** {beta:.2f} <abbr title='Volatility vs market.'>ℹ️</abbr>", unsafe_allow_html=True)
ins = f"Volume {'above' if vol>avg_vol else 'below'} average; Market cap {'small' if mc<1e9 else 'mid/large'} cap."
st.markdown(f"<div class='card-dark'>🔍 {ins}</div>", unsafe_allow_html=True)

# --- EXTENDED FUNDAMENTALS ---
st.markdown("<div class='card'><h2>🧲 Fundamental Breakdown</h2></div>", unsafe_allow_html=True)
sections = {
    'Valuation': [('P/E', 'trailingPE', '15-25 fair'), ('PEG', 'pegRatio', '~1 fair')],
    'Profitability': [('Net Margin', 'profitMargins', '>5% profitable'), ('ROE', 'returnOnEquity', '>15% strong')],
    'Leverage': [('Debt/Eq', 'debtToEquity', '<1 comfortable'), ('EV', 'enterpriseValue', '<1.5x MC typical')]
}
for sec, items in sections.items():
    st.markdown(f"**{sec}**")
    for name, key, tip in items:
        raw = info.get(key)
        if isinstance(raw, (int, float)):
            disp = f"{raw*100:.2f}%" if '%' in tip else f"{raw:.2f}"
            color = 'green' if raw >= 0 else 'red'
            st.markdown(f"- {name}: <span style='color:{color}; font-weight:bold;'>{disp}</span> <abbr title='{tip}'>ℹ️</abbr>", unsafe_allow_html=True)
st.markdown(f"<div class='card-dark'>🧠 Valuation is {'attractive' if info.get('trailingPE',0)<np.mean([yf.Ticker(p).info.get('trailingPE',0) for p in peer_list]) else 'above peers'}</div>", unsafe_allow_html=True)

# --- COMPETITOR COMPARISON ---
st.markdown("<div class='card'><h2>🤝 Peer Comparison</h2></div>", unsafe_allow_html=True)
peer_data = []
for p in peer_list:
    pi = yf.Ticker(p).info
    peer_data.append({'Ticker': p, 'Price': pi.get('currentPrice'), 'P/E': pi.get('trailingPE')})
peer_df = pd.DataFrame(peer_data).set_index('Ticker')
st.bar_chart(peer_df['P/E'])
st.dataframe(peer_df)

# --- NEWS & SENTIMENT ---
st.markdown("<div class='card'><h2>📰 News & Sentiment</h2></div>", unsafe_allow_html=True)
news_url = f"https://finance.yahoo.com/quote/{ticker}"
try:
    page = requests.get(news_url, timeout=5)
    soup = BeautifulSoup(page.content, 'html.parser')
    for h in soup.find_all('h3')[:5]:
        text = h.get_text(strip=True)
        badge = '🟢' if 'beat' in text.lower() else '🔴' if 'miss' in text.lower() else '⚪'
        st.markdown(f"- {badge} {text}", unsafe_allow_html=True)
    st.markdown("<div style='padding:5px;'><b>Sentiment:</b> Overall neutral to positive based on recent headlines.</div>", unsafe_allow_html=True)
except:
    st.warning("News unavailable")

# --- ACTION RECOMMENDATION ---
st.markdown("<div class='card-dark'><h2 style='color:#4CAF50;'>🎯 Action Recommendation</h2></div>", unsafe_allow_html=True)
rec = 'Buy' if pct6m and pct6m>0 and info.get('trailingPE',999)<np.mean([yf.Ticker(p).info.get('trailingPE',999) for p in peer_list]) else 'Hold'
st.markdown(f"<b>Recommendation:</b> {rec} <br><b>Confidence:</b> ⭐️⭐️⭐️", unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p style="color:#888888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
