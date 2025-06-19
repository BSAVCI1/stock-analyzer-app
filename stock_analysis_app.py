# Fundamental Analysis Module Implementation for AI Stock Analyzer

import streamlit as st
import yfinance as yf
import pandas as pd

# --- FUNDAMENTAL ANALYSIS CARD ---
def render_fundamental_analysis(ticker: str):
    data = yf.Ticker(ticker)
    # Fetch quarterly income statement
    try:
        df_income = data.quarterly_financials.T  # columns: Revenue, Gross Profit, Operating Income, Net Income, ...
    except Exception:
        st.error("Unable to fetch quarterly financials.")
        return

    # Select last 4 quarters
    df4 = df_income.iloc[:4][[
        'Total Revenue',
        'Gross Profit',
        'Operating Income',
        'Net Income',
        'Operating Cash Flow'
    ]]
    df4.index = pd.to_datetime(df4.index).to_period('Q')

    # Render comparison table
    st.markdown("<div class='card'><h2>📊 Quarterly Earnings Review</h2></div>", unsafe_allow_html=True)
    st.dataframe(df4.style.format("${:,.0f}"))

    # Insights: change from previous quarters
    changes = df4.pct_change().iloc[1:] * 100
    latest = df4.iloc[0]
    insight_lines = []
    for metric in df4.columns:
        pct = changes.loc[df4.index[0], metric]
        direction = 'increase' if pct > 0 else 'decrease'
        insight_lines.append(f"• {metric}: {direction} of {pct:.1f}% vs prior quarter.")
    insight_text = '<br>'.join(insight_lines)
    st.markdown(f"<div class='card-dark'><b>🧠 Earnings Insight:</b><br>{insight_text}</div>", unsafe_allow_html=True)

# --- USAGE in main.py ---
# from fundamental_analysis_module import render_fundamental_analysis
# render_fundamental_analysis(ticker)
