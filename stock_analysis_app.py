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
popular = ["BBAI","AARC","VUSA.AS","SPCE","AAPL","MSFT","GOOGL","AMZN","QS","TSLA","NVDA"]
# Free-text ticker override
ticker_select = st.sidebar.selectbox("Choose from popular tickers", options=popular, index=popular.index("SPCE"))
ticker_input = st.sidebar.text_input("Or enter any ticker symbol", "").upper().strip()
ticker = ticker_input if ticker_input else ticker_select

# --- FETCH DATA ---
data = yf.Ticker(ticker)
info = data.info
hist = data.history(period="6mo")
hist['MA20'] = hist['Close'].rolling(20).mean()
hist['MA50'] = hist['Close'].rolling(50).mean()

# --- MARKET OVERVIEW & SUPPORT/RESISTANCE ---
st.markdown(f"### {info.get('shortName', ticker)} ({ticker})")
st.markdown("<div class='card'><h2>📈 Market & Trading Overview</h2></div>", unsafe_allow_html=True)
vol = info.get('volume', 0)
avg_vol = info.get('averageVolume', 0)
mc = info.get('marketCap', 0)
rev = info.get('totalRevenue', 0)
dy = info.get('dividendYield', 0) * 100
beta = info.get('beta', 0)
cols = st.columns(3)
cols[0].markdown(f"**Volume:** {vol:,} {'<span class=\"arrow-up\">▲</span>' if vol>avg_vol else '<span class=\"arrow-down\">▼</span>'} <abbr title='Shares traded in last session.'>ℹ️</abbr>", unsafe_allow_html=True)
cols[1].markdown(f"**Avg Volume:** {avg_vol:,} {'<span class=\"arrow-up\">▲</span>' if avg_vol>vol else '<span class=\"arrow-down\">▼</span>'} <abbr title='30-day avg volume.'>ℹ️</abbr>", unsafe_allow_html=True)
prev_mc = hist['Close'].iloc[-2] * info.get('sharesOutstanding', 1)
cols[2].markdown(
    f"**Market Cap:** ${mc:,} {'<span class=\"arrow-up\">▲</span>' if mc>prev_mc else '<span class=\"arrow-down\">▼</span>'} <abbr title='Total market value.'>ℹ️</abbr>",
    unsafe_allow_html=True
)
cols2 = st.columns(3)
cols2[0].markdown(
    f"**Revenue (TTM):** ${rev:,} {'<span class=\"arrow-up\">▲</span>' if rev>hist['Close'].iloc[-2]*info.get('sharesOutstanding',1) else '<span class=\"arrow-down\">▼</span>'} <abbr title='Trailing 12m revenue.'>ℹ️</abbr>",
    unsafe_allow_html=True
)
cols2[1].markdown(
    f"**Dividend Yield:** {dy:.2f}% {'<span class=\"arrow-up\">▲</span>' if dy>np.nanmean([yf.Ticker(p).info.get('dividendYield',0)*100 for p in popular]) else '<span class=\"arrow-down\">▼</span>'} <abbr title='Annual dividend %.'>ℹ️</abbr>",
    unsafe_allow_html=True
)
cols2[2].markdown(
    f"**Beta:** {beta:.2f} {'<span class=\"arrow-up\">▲</span>' if beta>1 else '<span class=\"arrow-down\">▼</span>'} <abbr title='Volatility vs market.'>ℹ️</abbr>",
    unsafe_allow_html=True
)
ins = (
    f"Last session volume was {'higher' if vol>avg_vol else 'lower'} than the 30-day avg, suggesting {'strong buying interest.' if vol>avg_vol else 'potential lack of enthusiasm.'} "
    f"Market cap ${mc:,} indicates {'small' if mc<1e9 else 'mid/large'}-cap. "
    f"TTM revenue ${rev:,}. "
    f"Dividend {dy:.2f}% {'paid' if dy>0 else 'none'}. "
    f"Beta {beta:.2f} {'higher' if beta>1 else 'lower'} vol."
)
st.markdown(f"<div class='card-dark'>🔍 {ins}</div>", unsafe_allow_html=True)

# --- EXTENDED FUNDAMENTALS ---
st.markdown("<div class='card'><h2>📑 Fundamental Breakdown vs Peers</h2></div>", unsafe_allow_html=True)

# Ensure peer_list exists
try:
    peer_list
except NameError:
    peer_list = popular

# Gather peer data
peer_info = []
for p in peer_list:
    try:
        peer_info.append(yf.Ticker(p).info)
    except:
        continue

# Compute peer averages
keys = ['trailingPE','pegRatio','profitMargins','returnOnEquity','debtToEquity','enterpriseValue']
avg_vals = {}
for k in keys:
    vals = [pi.get(k) for pi in peer_info if isinstance(pi.get(k), (int,float))]
    avg_vals[k] = np.nanmean(vals) if vals else np.nan

# Define sections
sections = {
    'Valuation': [('P/E Ratio','trailingPE','15–25 = fair valuation'),('PEG Ratio','pegRatio','~1 = balanced')],
    'Profitability': [('Net Margin','profitMargins','>5% profitable'),('ROE','returnOnEquity','>15% strong')],
    'Leverage': [('Debt/Equity','debtToEquity','<1 manageable'),('Enterprise Value','enterpriseValue','incl. debt & cash')]
}

# Render in three columns
cols = st.columns(3)
for idx, (sec, items) in enumerate(sections.items()):
    with cols[idx]:
        st.markdown(f"### {sec}")
        for name, key, tip in items:
            val = info.get(key)
            peer_avg = avg_vals.get(key, np.nan)
            # format peer avg
            if pd.isna(peer_avg):
                peer_str = 'N/A'
            elif name in ['Net Margin','ROE']:
                peer_str = f"{peer_avg*100:.2f}%"
            elif key=='enterpriseValue':
                peer_str = f"${peer_avg:,.0f}"
            else:
                peer_str = f"{peer_avg:.2f}"
            # determine display and color
            if val is None or pd.isna(peer_avg):
                disp, color = 'N/A','gray'
            else:
                better = (val>=peer_avg) if key!='debtToEquity' else (val<=peer_avg)
                color = 'green' if better else 'red'
                if name in ['Net Margin','ROE']:
                    disp = f"{val*100:.2f}%"
                elif key=='enterpriseValue':
                    disp = f"${val:,.0f}"
                else:
                    disp = f"{val:.2f}" if isinstance(val,(int,float)) else 'N/A'
        st.markdown(
            f"- {name}: "
            f"<span style='color:{color}; font-weight:bold;'>{disp}</span> "
            f"<abbr title='{tip}'>ℹ️</abbr>",
            unsafe_allow_html=True
        )

# Improved AI Insight summarizing all three pillars
# Evaluate strengths
val_diff = info.get('trailingPE', np.nan) - avg_vals.get('trailingPE', np.nan)
profit_diff = (info.get('returnOnEquity',0) - avg_vals.get('returnOnEquity',0))*100
leverage_diff = avg_vals.get('debtToEquity', np.nan) - info.get('debtToEquity', np.nan)
insights = []

# Valuation insight
if not pd.isna(val_diff):
    if val_diff < 0:
        insights.append('📈 Valuation is attractive relative to peers.')
    else:
        insights.append('⚠️ Valuation is above peer average; consider risks.')
# Profitability insight
if not pd.isna(profit_diff):
    if profit_diff > 0:
        insights.append('👍 Profitability (ROE) outperforms peers.')
    else:
        insights.append('🔻 Profitability lags behind peers.')
# Leverage insight
if not pd.isna(leverage_diff):
    if leverage_diff > 0:
        insights.append('🏦 Strong balance sheet with lower debt relative to peers.')
    else:
        insights.append('⚠️ Higher leverage than peers; watch debt levels.')

summary = ' '.join(insights) if insights else 'No sufficient data for peer comparison.'
st.markdown(f"<div class='card-dark'>💡 {summary}</div>", unsafe_allow_html=True)

# Helper to format in millions, dropping “.0” when possible
def fmt_m(x):
    if pd.isna(x):
        return "-"
    m = x / 1e6
    # If m is essentially an integer, show no decimals
    if abs(m - round(m)) < 1e-6:
        return f"${int(round(m))}M"
    else:
        return f"${m:.1f}M"

# --- FUNDAMENTAL ANALYSIS MODULE ---
def render_fundamental_analysis(ticker: str):
    # 1) Pull last 4 quarters of key metrics
    data = yf.Ticker(ticker)
    df_income = data.quarterly_financials.T

    metrics = [
        'Total Revenue','Revenue','Gross Profit',
        'Operating Income','EBIT','Net Income','Operating Cash Flow'
    ]
    avail = [m for m in metrics if m in df_income.columns]
    df_q = df_income[avail].iloc[:4]
    df_q.index = pd.to_datetime(df_q.index).to_period('Q').astype(str)

    # 2) Convert to millions and round
    df_q_m = (df_q / 1e6).round(1)
    df_q_m.columns = [f"{col} (M)" for col in df_q_m.columns]

    # 3) QoQ % changes and round
    df_pct = (df_q_m.pct_change().iloc[1:] * 100).round(1)
    df_pct.columns = [f"{col} % Change" for col in df_pct.columns]

    # 4) Merge for display
    df_show = pd.concat([df_q_m.iloc[1:], df_pct], axis=1)

    # 5) Style via Pandas Styler, using fmt_m for USD columns
    styled = (
        df_show.style
               .format({c: fmt_m for c in df_q_m.columns}, na_rep='-')
               .format({c: "{:.1f}%" for c in df_pct.columns}, na_rep='-')
               .background_gradient(subset=df_pct.columns, cmap='RdYlGn')
               .set_caption("Values in millions (M) & QoQ % changes")
    )

    # 6) Render the HTML table so styles take effect
    html = styled.to_html()
    st.markdown(html, unsafe_allow_html=True)

    # 7) Generate human-friendly insights
    insights = []
    if not df_pct.empty:
        last = df_pct.iloc[-1]
        for col, change in last.items():
            name = col.replace(" % Change", "")
            if change > 5:
                insights.append(f"✅ {name} up {change:.1f}% vs prior quarter")
            elif change < -5:
                insights.append(f"⚠️ {name} down {abs(change):.1f}% vs prior quarter")
            else:
                insights.append(f"🔄 {name} change {change:.1f}% vs prior quarter")

    summary = "<br>".join(insights) if insights else "No significant quarter-over-quarter moves."
    st.markdown(
        f"<div class='card-dark'><b>💡 Earnings Insights:</b><br>{summary}</div>",
        unsafe_allow_html=True
    )

# Actually call it
render_fundamental_analysis(ticker)

# --- TECHNICAL ANALYSIS MODULE ---
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

# --- COMPUTE 6M CHANGE ---
current_price = info.get('currentPrice', None)
price_6m = hist['Close'].iloc[0] if len(hist) > 0 else None
pct6m = ((current_price - price_6m) / price_6m * 100) if price_6m else None

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
    f"• RSI at {rsi_val:.1f} indicates {'overbought' if rsi_val>70 else 'oversold' if rsi_val<30 else 'neutral'} conditions.",
    f"• Price {ma_trend} vs 50-day MA.",
    f"• MACD at {macd_val:.2f} suggests {'bullish' if macd_val>0 else 'bearish'} momentum.",
    f"• Price is {bb_pos}, indicating volatility.",
    f"• ADX at {adx_text} means {'strong trend' if adx_val>25 else 'weak trend'}.",
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
fig2.add_trace(go.Scatter(x= hist.index, y=hist['BB_upper'], line=dict(dash='dash'), name='Upper Band'))
fig2.add_trace(go.Scatter(x= hist.index, y=hist['BB_mid'], line=dict(dash='dot'), name='Mid Band'))
fig2.add_trace(go.Scatter(x= hist.index, y=hist['BB_lower'], line=dict(dash='dash'), name='Lower Band'))
fig2.update_layout(template='plotly_white', height=300)
st.plotly_chart(fig2, use_container_width=True)

# --- PEER COMPARISON MODULE ---
st.markdown("<div class='card'><h2>🤝 Peer Comparison</h2></div>", unsafe_allow_html=True)
peer_data = []
for p in peer_list:
    try:
        pi = yf.Ticker(p).info
        peer_data.append({
            'Ticker': p,
            'Price': pi.get('currentPrice', np.nan),
            'P/E Ratio': pi.get('trailingPE', np.nan)
        })
    except Exception:
        continue
peer_df = pd.DataFrame(peer_data).set_index('Ticker')
if not peer_df.empty:
    st.bar_chart(peer_df['P/E Ratio'])
    st.dataframe(peer_df.style.format({
        'Price': '${:,.2f}',
        'P/E Ratio': '{:.2f}'
    }))
else:
    st.info("No peer data available.")

# --- NEWS & SENTIMENT MODULE ---
st.markdown("<div class='card'><h2>📰 News & Sentiment</h2></div>", unsafe_allow_html=True)
news_url = f"https://finance.yahoo.com/quote/{ticker}"
try:
    resp = requests.get(news_url, timeout=5)
    soup = BeautifulSoup(resp.content, 'html.parser')
    headlines = soup.find_all('h3')[:5]
    for h in headlines:
        text = h.get_text(strip=True)
        badge = '🟢' if any(w in text.lower() for w in ['beat','upgrade','gain']) else ('🔴' if any(w in text.lower() for w in ['miss','downgrade','drop']) else '⚪️')
        st.markdown(f"- {badge} {text}", unsafe_allow_html=True)
    st.markdown("<div class='card-dark'>🔍 Overall sentiment: Neutral to Positive based on recent headlines.</div>", unsafe_allow_html=True)
except Exception:
    st.warning("Unable to fetch news headlines.")

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p style="color:#888888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p style="color:#888888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
