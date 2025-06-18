st.title("📊 Stock Comparison")

tickers = st.sidebar.text_input("Enter Stock Tickers (comma separated)", value="AAPL,NVDA").upper().split(",")

comparison_data = []

for t in tickers:
    stock = yf.Ticker(t.strip())
    info = stock.info

    comparison_data.append({
        "Ticker": t.strip(),
        "Price": info.get("currentPrice"),
        "Market Cap": info.get("marketCap"),
        "PE Ratio": info.get("trailingPE"),
        "Revenue (TTM)": info.get("totalRevenue"),
        "Profit Margin": info.get("profitMargins"),
        "Dividend Yield": info.get("dividendYield"),
    })

df = pd.DataFrame(comparison_data)
st.dataframe(df)

if st.button("🧠 Compare with AI"):
    prompt = f"Compare these stocks based on this data:\n{df.to_markdown(index=False)}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    st.markdown("### 🤖 AI Comparison Summary")
    st.write(response['choices'][0]['message']['content'])
