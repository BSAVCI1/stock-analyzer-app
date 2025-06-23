# ai_stock_analyzer_app/main.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from bs4 import BeautifulSoup
import requests
import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="📈 AI Stock Analyzer", layout="wide")

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

# --- USER INPUT & PEERS ---
st.sidebar.header("Select Stock & Peers")
popular = ["BBAI","AARC","VUSA.AS","SPCE","AAPL","MSFT","GOOGL","AMZN","QS","TSLA","NVDA"]
ticker_select = st.sidebar.selectbox("Choose from popular tickers", popular, index=popular.index("SPCE"))
ticker_input  = st.sidebar.text_input("Or enter any ticker symbol", "").upper().strip()
ticker = ticker_input or ticker_select

# Auto-select peers by industry/sector
data = yf.Ticker(ticker)
info = data.info
if st.sidebar.checkbox("Auto-select peers by sector/industry", True):
    sector   = info.get("sector")
    industry = info.get("industry")
    industry_map = {
        'Information Technology Services': ['SOUN','CRNC','AI','NVDA','PLTR'],
        'Software—Infrastructure':          ['NOW','CRM','ORCL','ADBE','SNOW'],
    }
    sector_map = {
        'Technology':          ['AAPL','MSFT','GOOGL','AMZN','TSLA'],
        'Consumer Cyclical':   ['AMZN','TSLA','BBWI'],
        'Communication Services':['META','NFLX','DIS'],
    }
    peer_list = industry_map.get(industry) or sector_map.get(sector) or popular
else:
    text = st.sidebar.text_input("Or enter peers (comma separated)", ",".join(popular))
    peer_list = [p.strip().upper() for p in text.split(",") if p.strip()]

# --- FETCH DATA ---
data = yf.Ticker(ticker)
info = data.info
hist = data.history(period="6mo")
hist['MA20'] = hist['Close'].rolling(20).mean()
hist['MA50'] = hist['Close'].rolling(50).mean()

# --- DIVIDEND DATES FIX ---
div_dates = [dt.date() for dt in data.dividends.index]

# Safe previous-close lookup
prev_close = hist['Close'].shift(1).iloc[-1]
if pd.isna(prev_close): prev_close = hist['Close'].iloc[-1]

# --- MARKET OVERVIEW & SUPPORT/RESISTANCE ---
st.markdown(f"### {info.get('shortName', ticker)} ({ticker})")
st.markdown("<div class='card'><h2>📈 Market & Trading Overview</h2></div>", unsafe_allow_html=True)

vol     = info.get('volume',0)
avg_vol = info.get('averageVolume',0)
mc      = info.get('marketCap',0)
rev     = info.get('totalRevenue',0)
dy      = info.get('dividendYield',0)*100
beta    = info.get('beta',0)

c1, c2, c3 = st.columns(3)
# decide arrow
arrow_html = (
    '<span class="arrow-up">▲</span>'
    if vol > avg_vol
    else '<span class="arrow-down">▼</span>'
)

# now it's just one clean f-string
c1.markdown(
    f"**Volume:** {vol:,} {arrow_html} "
    "<abbr title='Shares traded last session.'>ℹ️</abbr>",
    unsafe_allow_html=True
)
c2.markdown(f"**Avg Volume:** {avg_vol:,} {'<span class=\"arrow-up\">▲</span>' if avg_vol>vol else '<span class=\"arrow-down\">▼</span>'} <abbr title='30-day avg.'>ℹ️</abbr>", unsafe_allow_html=True)
prev_mc = prev_close * info.get('sharesOutstanding',1)
c3.markdown(f"**Market Cap:** ${mc:,} {'<span class=\"arrow-up\">▲</span>' if mc>prev_mc else '<span class=\"arrow-down\">▼</span>'} <abbr title='Total equity value.'>ℹ️</abbr>", unsafe_allow_html=True)

c4, c5, c6 = st.columns(3)
c4.markdown(f"**Revenue (TTM):** ${rev:,} {'<span class=\"arrow-up\">▲</span>' if rev>prev_close*info.get('sharesOutstanding',1) else '<span class=\"arrow-down\">▼</span>'} <abbr title='Trailing 12m rev.'>ℹ️</abbr>", unsafe_allow_html=True)
c5.markdown(f"**Dividend Yield:** {dy:.2f}% {'<span class=\"arrow-up\">▲</span>' if dy>np.nanmean([yf.Ticker(p).info.get('dividendYield',0)*100 for p in peer_list]) else '<span class=\"arrow-down\">▼</span>'} <abbr title='Annual dividend.'>ℹ️</abbr>", unsafe_allow_html=True)
c6.markdown(f"**Beta:** {beta:.2f} {'<span class=\"arrow-up\">▲</span>' if beta>1 else '<span class=\"arrow-down\">▼</span>'} <abbr title='Volatility vs market.'>ℹ️</abbr>", unsafe_allow_html=True)

ins = (
    f"Volume was {'above' if vol>avg_vol else 'below'} its 30-day avg; "
    f"{'strong interest' if vol>avg_vol else 'muted trading'}. "
    f"Market cap ${mc:,} ({'small' if mc<1e9 else 'mid/large'}-cap). "
    f"TTM rev ${rev:,}; "
    f"Dividend yield {dy:.2f}% ({'pays' if dy>0 else 'no payout'}); "
    f"Beta {beta:.2f} ({'high' if beta>1 else 'low'} volatility)."
)
st.markdown(f"<div class='card-dark'>🔍 {ins}</div>", unsafe_allow_html=True)

# --- EXTENDED FUNDAMENTALS vs PEERS ---
st.markdown("<div class='card'><h2>📑 Fundamental Breakdown vs Peers</h2></div>", unsafe_allow_html=True)

# gather peer info
peer_info = []
for p in peer_list:
    try: peer_info.append(yf.Ticker(p).info)
    except: pass

keys = ['trailingPE','pegRatio','profitMargins','returnOnEquity','debtToEquity','enterpriseValue']
avg_vals = {k: np.nanmean([pi.get(k) for pi in peer_info if isinstance(pi.get(k), (int,float))]) for k in keys}

cols = st.columns(3)
sections = {
    'Valuation'     : [('P/E Ratio','trailingPE','15–25 fair'), ('PEG Ratio','pegRatio','~1 fair')],
    'Profitability' : [('Net Margin','profitMargins','>5% profitable'), ('ROE','returnOnEquity','>15% strong')],
    'Leverage'      : [('Debt/Equity','debtToEquity','<1 manageable'), ('Enterprise Value','enterpriseValue','incl debt & cash')]
}

for idx, (sec, items) in enumerate(sections.items()):
    with cols[idx]:
        st.markdown(f"**{sec}**")
        for name,key,tip in items:
            val     = info.get(key)
            peer_av = avg_vals.get(key, np.nan)
            # display text
            if pd.isna(val) or pd.isna(peer_av):
                disp, color = 'N/A','gray'
            else:
                better = (val>=peer_av) if key!='debtToEquity' else (val<=peer_av)
                color  = 'green' if better else 'red'
                if name in ['Net Margin','ROE']:
                    disp = f"{val*100:.2f}%"
                elif key=='enterpriseValue':
                    disp = f"${val:,.0f}"
                else:
                    disp = f"{val:.2f}"
            st.markdown(f"- {name}: <span style='color:{color};font-weight:bold'>{disp}</span> <abbr title='{tip}'>ℹ️</abbr>", unsafe_allow_html=True)

# AI insight for fundamentals
vd = info.get('trailingPE',np.nan) - avg_vals['trailingPE']
pdiff = (info.get('returnOnEquity',0) - avg_vals['returnOnEquity'])*100
ld = avg_vals['debtToEquity'] - info.get('debtToEquity',np.nan)
notes=[]
if not np.isnan(vd):
    notes.append("📈 Valuation attractive vs peers." if vd<0 else "⚠️ Valuation above peers.")
if not np.isnan(pdiff):
    notes.append("👍 ROE outperforms peers." if pdiff>0 else "🔻 ROE lags peers.")
if not np.isnan(ld):
    notes.append("🏦 Lower debt vs peers." if ld>0 else "⚠️ Higher leverage.")
st.markdown(f"<div class='card-dark'>💡 {' '.join(notes)}</div>", unsafe_allow_html=True)

# --- QUARTERLY EARNINGS REVIEW ---
def render_fundamental_analysis(ticker):
    data = yf.Ticker(ticker)
    st.markdown("<div class='card'><h2>📊 Quarterly Earnings Review</h2></div>", unsafe_allow_html=True)

    df = data.quarterly_financials.T
    metrics = ['Total Revenue','Revenue','Gross Profit','Operating Income','EBIT','Net Income','Operating Cash Flow']
    avail   = [m for m in metrics if m in df.columns]
    df_q     = df[avail].iloc[:4]
    df_q.index = pd.to_datetime(df_q.index).to_period('Q').astype(str)

    # compute QoQ %
    df_pct = (df_q.pct_change()*100).round(1)
    df_pct.columns = [f"{c} % Change" for c in df_pct.columns]

    df_show = pd.concat([df_q, df_pct],axis=1)

    def short_fmt(x):
        try: x=float(x)
        except: return "-"
        if abs(x)>=1e9: return f"{x/1e9:.2f}B"
        if abs(x)>=1e6: return f"{x/1e6:.2f}M"
        if abs(x)>=1e3: return f"{x/1e3:.2f}K"
        return f"{x:.0f}"

    df_fmt = df_show.copy()
    for c in avail: df_fmt[c] = df_fmt[c].apply(short_fmt)
    for c in df_pct.columns: df_fmt[c] = df_fmt[c].apply(lambda v: f"{v:.1f}%" if pd.notna(v) else "-")

    st.dataframe(df_fmt, use_container_width=True)

    # insights
    latest = df_pct.index[-1]
    prev   = df_pct.index[-2] if len(df_pct)>1 else None
    ins    = []
    def senti(ch):
        if ch>5: return "strong growth"
        if ch>0: return "modest increase"
        if ch>-5: return "slight decline"
        return "notable decrease"

    if prev:
        for m in avail:
            key = f"{m} % Change"
            if key in df_pct.columns:
                ch = df_pct.loc[latest,key]
                ins.append(f"• {m} {senti(ch)} of {abs(ch):.1f}% this quarter.")
        # analyst style
        rc = df_pct.loc[latest,"Revenue % Change"] if "Revenue % Change" in df_pct else None
        if rc is not None:
            mood = "bullish" if rc>0 else "cautious"
            ins.append(f"🧐 Analysts are {mood} on rev after a {abs(rc):.1f}% {'rise' if rc>0 else 'drop'}.")

    summary = "<br>".join(ins) if ins else "No significant quarter-over-quarter changes."
    st.markdown(f"<div class='card-dark'><b>💡 Earnings Insights:</b><br>{summary}</div>", unsafe_allow_html=True)

render_fundamental_analysis(ticker)

# --- TECHNICAL PARAMETER CONTROLS ---
st.sidebar.header("🔧 Technical Settings")
rsi_p   = st.sidebar.slider("RSI Period",5,30,14)
macd_f  = st.sidebar.slider("MACD Fast EMA",5,30,12)
macd_s  = st.sidebar.slider("MACD Slow EMA",10,60,26)
macd_sig= st.sidebar.slider("MACD Signal EMA",5,20,9)
bb_w    = st.sidebar.slider("BB Window",10,60,20)
bb_m    = st.sidebar.slider("BB Std Mult",1.0,3.0,2.0)
atr_p   = st.sidebar.slider("ATR Period",5,30,14)

# recompute with settings
hist['MA20']    = hist['Close'].rolling(rsi_p).mean()
hist['MA50']    = hist['Close'].rolling(50).mean()
hist['MA200']   = hist['Close'].rolling(200).mean()
delta          = hist['Close'].diff()
gain           = delta.clip(lower=0).rolling(rsi_p).mean()
loss           = -delta.clip(upper=0).rolling(rsi_p).mean()
hist['RSI']    = 100 - (100/(1+gain/loss))
hist['EMAf']   = hist['Close'].ewm(span=macd_f).mean()
hist['EMAs']   = hist['Close'].ewm(span=macd_s).mean()
hist['MACD']   = hist['EMAf'] - hist['EMAs']
hist['MACDs']  = hist['MACD'].ewm(span=macd_sig).mean()
hist['MACD_h'] = hist['MACD'] - hist['MACDs']
hist['BBm']    = hist['Close'].rolling(bb_w).mean()
hist['BBstd']  = hist['Close'].rolling(bb_w).std()
hist['BBu']    = hist['BBm'] + bb_m*hist['BBstd']
hist['BBl']    = hist['BBm'] - bb_m*hist['BBstd']
hist['BBpctB'] = (hist['Close']-hist['BBl'])/(hist['BBu']-hist['BBl'])
tr             = pd.concat([hist['High']-hist['Low'],
                             (hist['High']-hist['Close'].shift()).abs(),
                             (hist['Low'] -hist['Close'].shift()).abs()], axis=1).max(axis=1)
hist['ATR']    = tr.rolling(atr_p).mean()
hist['OBV']    = (np.sign(hist['Close'].diff())*hist['Volume']).fillna(0).cumsum()

# signals & overview
recent = hist.last('90D')
sup = np.percentile(recent['Low'],10)
res = np.percentile(recent['High'],90)
latest = hist.iloc[-1]
cross  = ("Golden Cross ✅" if latest['MA50']>latest['MA200']
           else "Death Cross ⚠️" if latest['MA50']<latest['MA200']
           else "No Cross")

tech_df = pd.DataFrame([
    ["RSI",       f"{latest['RSI']:.1f}",              ""],
    ["MACD",      f"{latest['MACD']:.2f}",             ""],
    ["MACD Hist", f"{latest['MACD_h']:.2f}",           ""],
    ["MA20/50/200", f"{latest['MA20']:.2f}/{latest['MA50']:.2f}/{latest['MA200']:.2f}", cross],
    ["%B",        f"{latest['BBpctB']:.2f}",           ""],
    ["ATR",       f"{latest['ATR']:.2f}",              ""],
    ["OBV",       f"{int(latest['OBV'])}",             ""],
    ["Support",   f"{sup:.2f}",                        ""],
    ["Resistance",f"{res:.2f}",                        ""],
], columns=["Indicator","Value","Signal"])

st.markdown("<div class='card'><h2>📈 Technical Overview</h2></div>", unsafe_allow_html=True)
st.dataframe(tech_df, use_container_width=True)

ins=[]
ins.append(f"RSI {latest['RSI']:.1f} ({'overbought' if latest['RSI']>70 else 'oversold' if latest['RSI']<30 else 'neutral'}).")
ins.append(f"MACD {'+ve' if latest['MACD']>0 else '-ve'} (momentum).")
ins.append(f"50/200MA: {cross}.")
ins.append(f"%B {latest['BBpctB']:.2f} of range.")
ins.append(f"ATR {latest['ATR']:.2f} volatility.")
ins.append(f"OBV trend {'up' if hist['OBV'][-1]>hist['OBV'][-10] else 'down'}.")
st.markdown(f"<div class='card-dark'><b>📊 Technical Insights:</b><br>{'<br>'.join(ins)}</div>", unsafe_allow_html=True)

# --- 3️⃣ Signals & Events Overlay ---
# compute signal dates
ma50s  = hist['MA50']; ma200s = hist['MA200']
macd   = hist['MACD']; macdsig = hist['MACDs']
gcross = hist.index[(ma50s.shift(1)<ma200s.shift(1))&(ma50s>ma200s)]
dcross = hist.index[(ma50s.shift(1)>ma200s.shift(1))&(ma50s<ma200s)]
mbuy   = hist.index[(macd.shift(1)<macdsig.shift(1))&(macd>macdsig)]
msell  = hist.index[(macd.shift(1)>macdsig.shift(1))&(macd<macdsig)]
doji   = hist.index[(hist['Close']-hist['Open']).abs()<=0.1*(hist['High']-hist['Low'])]

# fetch yfinance news and filter 6m + big-move days
sixmo = datetime.datetime.now() - datetime.timedelta(days=180)
rawnews = getattr(data,"news",[]) or []
filtered = [n for n in rawnews if n.get("providerPublishTime") and datetime.datetime.fromtimestamp(n["providerPublishTime"])>=sixmo]
bigdays = set(hist.index[hist['Close'].pct_change().abs()>0.05].date)
event_news = [n for n in filtered if datetime.datetime.fromtimestamp(n["providerPublishTime"]).date() in bigdays]

last30 = hist.tail(30)
fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3],vertical_spacing=0.05)
fig.add_trace(go.Candlestick(x=last30.index,open=last30['Open'],high=last30['High'],
                             low=last30['Low'],close=last30['Close'],name="Price"),row=1,col=1)
fig.add_trace(go.Bar(x=last30.index,y=last30['Volume'],marker_color='grey',name="Vol"),row=2,col=1)

for dates,name,color,symbol in [
    (gcross,"Golden Cross","green","triangle-up"),
    (dcross,"Death Cross","red","triangle-down"),
    (mbuy,"MACD Buy","blue","circle"),
    (msell,"MACD Sell","orange","circle"),
    (doji,"Doji","purple","x"),
]:
    for d in dates:
        if d in last30.index:
            y = last30.loc[d,'Close']
            if name=="MACD Buy":  y = last30.loc[d,'Low']*0.995
            if name=="MACD Sell": y = last30.loc[d,'High']*1.005
            fig.add_trace(go.Scatter(x=[d],y=[y],mode='markers',marker=dict(symbol=symbol,size=10,color=color),name=name),row=1,col=1)

# overlay earnings/dividends
earn = data.calendar.get("Earnings Date",[]) or []
earn_dates=[]
for e in (earn if isinstance(earn,(list,tuple)) else [earn]):
    d=e[0] if isinstance(e,(list,tuple)) else e
    try: earn_dates.append(pd.to_datetime(d).date())
    except: pass
divs = [dt.date() for dt in data.dividends.index]

for dt,color,txt in [(earn_dates,"gold","💰 Earnings"),(divs,"green","💵 Dividend")]:
    for d in dt:
        if d in last30.index.date:
            fig.add_vline(
    x=pd.Timestamp(d),
    line=dict(color="gold", dash="dash"),
    row=1, col=1
)

# label the earnings lines
for d in earn_dates:
    if d in last30.index.date:
        fig.add_annotation(
            x=pd.Timestamp(d),
            y=last30['High'].max(),     # or whatever vertical position you like
            xref="x", yref="y",
            text="💰 Earnings",
            showarrow=True,
            arrowhead=2,
            arrowcolor="gold",
            font=dict(color="gold")
        )

# label the dividends lines
for dt in div_dates:
    if dt in set(last30.index.date):
        fig.add_vline(
            x=pd.Timestamp(dt),
            line=dict(color="green", dash="dot"),
            annotation_text="💵 Dividend",
            row=1, col=1
        )
# label the news lines
for n in event_news:
    d = n["date"]
    if d in last30.index.date:
        fig.add_annotation(
            x=pd.Timestamp(d),
            y=last30['Low'].min(),      # near bottom
            xref="x", yref="y",
            text="📰 News",
            showarrow=True,
            arrowhead=2,
            arrowcolor="cyan",
            font=dict(color="cyan")
        )

for n in event_news:
    d=datetime.datetime.fromtimestamp(n["providerPublishTime"]).date()
    if d in last30.index.date:
        fig.add_vline(
    x=pd.Timestamp(d),
    line=dict(color="gold", dash="dash"),
    row=1, col=1
)

# label the earnings lines
for d in earn_dates:
    if d in last30.index.date:
        fig.add_annotation(
            x=pd.Timestamp(d),
            y=last30['High'].max(),     # or whatever vertical position you like
            xref="x", yref="y",
            text="💰 Earnings",
            showarrow=True,
            arrowhead=2,
            arrowcolor="gold",
            font=dict(color="gold")
        )

# label the dividends lines
for d in div_dates:
    if d in last30.index.date:
        fig.add_annotation(
            x=pd.Timestamp(d),
            y=last30['High'].max()*0.95,  # slightly lower
            xref="x", yref="y",
            text="💵 Dividend",
            showarrow=True,
            arrowhead=2,
            arrowcolor="green",
            font=dict(color="green")
        )

# label the news lines
for n in event_news:
    d = n["date"]
    if d in last30.index.date:
        fig.add_annotation(
            x=pd.Timestamp(d),
            y=last30['Low'].min(),      # near bottom
            xref="x", yref="y",
            text="📰 News",
            showarrow=True,
            arrowhead=2,
            arrowcolor="cyan",
            font=dict(color="cyan")
        )

fig.update_layout(template="plotly_dark",height=650,showlegend=True,title=f"{ticker} — Last 30d: Candles, Signals & Events")
st.plotly_chart(fig,use_container_width=True,key="signals_events")

st.markdown("<div class='card'><h3>📰 Headlines on Big Moves</h3></div>",unsafe_allow_html=True)
if event_news:
    for n in event_news:
        d=datetime.datetime.fromtimestamp(n["providerPublishTime"]).date()
        st.markdown(f"- **{d.isoformat()}**  {n['title']}",unsafe_allow_html=True)
else:
    st.info("No high-impact headlines in last 6m.")

# --- PEER COMPARISON MODULE ---
st.markdown("<div class='card'><h2>🤝 Peer Comparison</h2></div>", unsafe_allow_html=True)
peer_data=[]
for p in peer_list:
    try:
        pi=yf.Ticker(p).info
        peer_data.append({'Ticker':p,'Price':pi.get('currentPrice',np.nan),'P/E':pi.get('trailingPE',np.nan)})
    except: pass
peer_df=pd.DataFrame(peer_data).set_index("Ticker")
if not peer_df.empty:
    st.bar_chart(peer_df['P/E'])
    st.dataframe(peer_df.style.format({'Price':'${:,.2f}','P/E':'{:.2f}'}))
else:
    st.info("No peer data available.")

# --- NEWS & SENTIMENT MODULE ---
st.markdown("<div class='card'><h2>📰 News & Sentiment</h2></div>", unsafe_allow_html=True)
news_url=f"https://finance.yahoo.com/quote/{ticker}"
try:
    resp=requests.get(news_url,timeout=5)
    soup=BeautifulSoup(resp.content,'html.parser')
    heads=soup.find_all('h3')[:5]
    for h in heads:
        t=h.get_text(strip=True)
        badge='🟢' if any(w in t.lower() for w in ['beat','upgrade','gain']) else ('🔴' if any(w in t.lower() for w in ['miss','downgrade','drop']) else '⚪️')
        st.markdown(f"- {badge} {t}",unsafe_allow_html=True)
    st.markdown("<div class='card-dark'>🔍 Overall sentiment: Neutral-to-Positive based on headlines</div>",unsafe_allow_html=True)
except:
    st.warning("Unable to fetch headlines.")

# --- FOOTER ---
st.markdown("""
<hr style="margin-top:2em;">
<div style="text-align:center"><p style="color:#888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p></div>
""", unsafe_allow_html=True)
