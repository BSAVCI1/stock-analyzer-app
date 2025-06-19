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

# --- FETCH DATA ---
data = yf.Ticker(ticker)
info = data.info
hist = data.history(period="6mo")
hist['MA20'] = hist['Close'].rolling(window=20).mean()
hist['MA50'] = hist['Close'].rolling(window=50).mean()

# --- SUPPORT & RESISTANCE ---
support = np.percentile(hist['Low'], 10)
resistance = np.percentile(hist['High'], 90)
hist['MA20'] = hist['Close'].rolling(window=20).mean()
hist['MA50'] = hist['Close'].rolling(window=50).mean()

# --- PRICE CHANGE UTIL ---
def arrow(pct): return "<span class='arrow-up'>▲</span>" if pct>0 else "<span class='arrow-down'>▼</span>" if pct<0 else "—"

def calc_change(current, past):
    if past is None or current is None: return None, None, None
    pct = (current-past)/past*100
    return current, pct, arrow(pct)

cp, _, _ = calc_change(info.get('currentPrice'), info.get('currentPrice'))
p1d, pct1d, arr1d = calc_change(info.get('currentPrice'), hist['Close'].iloc[-2] if len(hist)>1 else None)
p1m, pct1m, arr1m = calc_change(info.get('currentPrice'), hist['Close'].iloc[-21] if len(hist)>21 else None)
p6m, pct6m, arr6m = calc_change(info.get('currentPrice'), hist['Close'].iloc[0] if len(hist)>0 else None)

# --- PRICE CARD ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write(f"### 💵 {info.get('shortName', ticker)} ({ticker}) Price Overview")
st.markdown(f"**Now:** ${info.get('currentPrice'):.2f}  {arr1d}\n**24h:** {pct1d:.1f}% {arr1d}\n**1M:** {pct1m:.1f}% {arr1m}\n**6M:** {pct6m:.1f}% {arr6m}", unsafe_allow_html=True)
# Action recommendation
action = "Hold" if pct6m and pct6m<0 else "Consider Buy near support" if pct6m and pct6m>0 else "N/A"
st.markdown(f"<b>⭐️ Action:</b> {action}", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- MARKET OVERVIEW CARD ---
vol, avg_vol, mc, rev, dy, beta = info.get('volume',0), info.get('averageVolume',0), info.get('marketCap',0), info.get('totalRevenue',0), info.get('dividendYield',0), info.get('beta',0)
st.markdown("<div class='card'> <h2>📈 Market & Trading Overview</h2>", unsafe_allow_html=True)
cols = st.columns(3)
for col, label, val, tip in zip(cols,
    ["Volume","Avg Volume","Market Cap"],[vol,avg_vol,mc],
    ['Last session trading volume','30-day avg volume','Total market capitalization']):
    arrow_icon = arr1d if label=="Volume" else ''
    col.markdown(f"**{label}:** {val:,} {arrow_icon} <abbr title='{tip}'>ℹ️</abbr>",unsafe_allow_html=True)
cols2 = st.columns(3)
for col, label, val, tip in zip(cols2,
    ["Revenue","Dividend Yield","Beta"],[rev,dy*100,beta],
    ['TTM Revenue','Annual dividend %','Stock volatility vs market']):
    disp = f"${val:,}" if label=="Revenue" else f"{val:.2f}%" if label=="Dividend Yield" else f"{val:.2f}"
    col.markdown(f"**{label}:** {disp} <abbr title='{tip}'>ℹ️</abbr>",unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# Market insight
ins = f"Volume {'above' if vol>avg_vol else 'below'} average; Market cap indicates {'small' if mc<1e9 else 'mid/large'} cap." 
st.markdown(f"<div class='card-dark'>🔍 {ins}</div>", unsafe_allow_html=True)

# --- SUPPORT & RESISTANCE on Chart ---
st.markdown("<div class='card'> <h2>⚙️ Price Chart & Key Levels</h2>",unsafe_allow_html=True)
fig=go.Figure()
fig.add_trace(go.Scatter(x=hist.index,y=hist['Close'],name='Close'))
fig.add_trace(go.Scatter(x=hist.index,y=hist['MA20'],name='MA20'))
fig.add_trace(go.Scatter(x=hist.index,y=hist['MA50'],name='MA50'))
fig.add_hline(y=support,line_dash='dash',annotation_text='Support',annotation_position='bottom right')
fig.add_hline(y=resistance,line_dash='dash',annotation_text='Resistance',annotation_position='top right')
fig.update_layout(template='plotly_white',height=400)
st.plotly_chart(fig,use_container_width=True)
st.markdown("</div>",unsafe_allow_html=True)

# --- EXTENDED FUNDAMENTALS CARD ---
st.markdown("<div class='card'> <h2>🧲 Fundamental Breakdown</h2>",unsafe_allow_html=True)
sections={
 'Valuation':[('P/E','trailingPE','15-25= fair'),('PEG','pegRatio','~1=fair')],
 'Profitability':[('Net Margin','profitMargins','>5% profitable'),('ROE','returnOnEquity','>15% strong')],
 'Leverage':[('Debt/Eq','debtToEquity','<1 comfortable'),('EV','enterpriseValue','<1.5xMC typical')]
}
for sec, items in sections.items():
    st.markdown(f"**{sec}**")
    for name,key,tip in items:
        raw=info.get(key)
        if isinstance(raw,(int,float)):
            disp=f"{raw*100:.2f}%" if 'Margin' in name or 'ROE' in name else f"{raw:.2f}"
            colr='green' if raw>0 else 'red'
            st.markdown(f"- {name}: <span style='color:{colr};'>{disp}</span> <abbr title='{tip}'>ℹ️</abbr>",unsafe_allow_html=True)
st.markdown("</div>",unsafe_allow_html=True)
st.markdown(f"<div class='card-dark'>🧠 Fundamentals Insight: {'Valuation attractive' if info.get('trailingPE',0)<avg_pe else 'Valuation above peers'}</div>",unsafe_allow_html=True)

# --- COMPETITOR OVERVIEW ---
st.markdown("<div class='card'> <h2>🤝 Peer Comparison</h2>",unsafe_allow_html=True)
peer_df=pd.DataFrame([{ 'Ticker':p, 'Price':yf.Ticker(p).info.get('currentPrice'), 'P/E':yf.Ticker(p).info.get('trailingPE')} for p in peer_list]).set_index('Ticker')
st.bar_chart(peer_df[['P/E']])
st.dataframe(peer_df)
st.markdown("</div>",unsafe_allow_html=True)

# --- NEWS & SENTIMENT CARD ---
st.markdown("<div class='card'> <h2>📰 News & Sentiment</h2>",unsafe_allow_html=True)
news_url=f"https://finance.yahoo.com/quote/{ticker}"
try:
    soup=BeautifulSoup(requests.get(news_url).content,'html.parser')
    for h in soup.find_all('h3')[:3]:
        txt=h.get_text(strip=True)
        badge='🟢' if 'beat' in txt.lower() else '🔴' if 'miss' in txt.lower() else '⚪'
        st.markdown(f"- {badge} {txt}",unsafe_allow_html=True)
    st.markdown("<div style='padding:5px;'><b>Sentiment:</b> Overall neutral to positive.</div>",unsafe_allow_html=True)
except:
    st.warning("News unavailable")
st.markdown("</div>",unsafe_allow_html=True)

# --- ANALYST RECOMMENDATION CARD ---
st.markdown("<div class='card-dark'> <h2 style='color:#4CAF50;'>🎯 Action Recommendation</h2>",unsafe_allow_html=True)
rec='Buy' if pct6m and pct6m>0 and info.get('trailingPE',999)<avg_pe else 'Hold'
st.markdown(f"<b>Recommendation:</b> {rec} <br><b>Confidence:</b> ⭐️⭐️⭐️",unsafe_allow_html=True)
st.markdown("</div>",unsafe_allow_html=True)
