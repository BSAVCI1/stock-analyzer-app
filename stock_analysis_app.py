make sure that extended fundamentals has accurate comparison data of 6month ago





# --- SUPPORT & RESISTANCE DISPLAY ---
st.markdown("## 🧭 Key Price Levels")
sr_col1, sr_col2 = st.columns(2)
sr_col1.metric("🔻 Support Level", f"${support:.2f}", help="A lower price range where the stock may find buying interest.")
sr_col2.metric("🔺 Resistance Level", f"${resistance:.2f}", help="An upper price range where the stock may face selling pressure.")

# --- TECHNICAL INDICATORS TABLE ---
st.markdown("## 📊 Technical Indicators Summary")
delta = hist['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
hist['RSI'] = 100 - (100 / (1 + rs))

hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
hist['MACD'] = hist['EMA12'] - hist['EMA26']

hist['MA20'] = hist['Close'].rolling(20).mean()
hist['MA50'] = hist['Close'].rolling(50).mean()

tech_df = pd.DataFrame({
    'Indicator': ['RSI (14)', 'MACD', 'MA20', 'MA50'],
    'Value': [
        round(hist['RSI'].iloc[-1], 2),
        round(hist['MACD'].iloc[-1], 2),
        round(hist['MA20'].iloc[-1], 2),
        round(hist['MA50'].iloc[-1], 2)
    ],
    'Interpretation': [
        'Overbought' if hist['RSI'].iloc[-1] > 70 else 'Oversold' if hist['RSI'].iloc[-1] < 30 else 'Neutral',
        'Positive momentum' if hist['MACD'].iloc[-1] > 0 else 'Negative momentum',
        'Trending above short MA' if current_price > hist['MA20'].iloc[-1] else 'Below short MA',
        'Trending above long MA' if current_price > hist['MA50'].iloc[-1] else 'Below long MA'
    ]
})
st.dataframe(tech_df, use_container_width=True)

# --- NEWS & SENTIMENT ---
st.markdown("## 📰 Recent News & Market Sentiment")
news_url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
try:
    res = requests.get(news_url, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")
    headlines = soup.find_all("h3")[:5]
    if headlines:
        for h in headlines:
            link_tag = h.find("a")
            if link_tag and link_tag.text:
                st.markdown(f"- [{link_tag.text}](https://finance.yahoo.com{link_tag['href']})")
    else:
        st.info("No recent headlines available.")
except Exception as e:
    st.warning(f"Unable to fetch news. Reason: {e}")

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 2em;">
<div style="text-align:center">
    <p style="color:#888888;">Created by <b>BSAVCI1</b> • Powered by Streamlit & Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
