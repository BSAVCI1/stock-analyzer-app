"""Detailed user guide for both Streamlit applications."""

from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Stock Platform Guide",
    page_icon="ℹ️",
    layout="wide",
)


def render_sections(
    sections: tuple[
        tuple[str, str, str, str],
        ...,
    ],
) -> None:
    for (
        title,
        provides,
        reading,
        caution,
    ) in sections:
        with st.expander(
            title,
            expanded=False,
        ):
            st.markdown(
                "#### What this area provides"
            )

            st.markdown(provides)

            st.markdown(
                "#### How to read it"
            )

            st.markdown(reading)

            st.markdown(
                "#### Important context"
            )

            st.markdown(caution)


st.title("ℹ️ Stock Platform Guide")

st.caption(
    "A practical guide to the Stock Analyzer and "
    "the Paper Portfolio & Reliability dashboard."
)

left, right = st.columns(2)

with left:
    if st.button(
        "📈 Open Stock Analyzer",
        width="stretch",
        key="guide_stock_analyzer",
    ):
        st.switch_page(
            "stock_analysis_app.py"
        )

with right:
    if st.button(
        "📊 Open Paper Portfolio",
        width="stretch",
        key="guide_paper_portfolio",
    ):
        st.switch_page(
            "pages/2_Paper_Portfolio.py"
        )


stock_tab, portfolio_tab = st.tabs(
    (
        "📈 Stock Analyzer Guide",
        "📊 Paper Portfolio Guide",
    )
)


with stock_tab:
    st.header("Stock Analyzer")

    st.markdown(
        """
The Stock Analyzer is the research and decision-support
application. It loads current market information, applies
the configured fundamental and technical analysis, and
produces a traceable trading-expert report.

A useful reading order is:

1. Validate the selected stock and market-data date.
2. Review the market and fundamental overview.
3. Review technical conditions and market regime.
4. Read the strategy result and weighted score.
5. Read conflicts, risks, invalidation levels and evidence.
6. Use the chart, peers and news as supporting context.
"""
    )

    stock_sections = (
        (
            "1. Stock selection and peer group",
            """
The sidebar lets you choose a popular ticker or enter
another market symbol. The peer group is used for relative
fundamental comparisons. Automatic peers are selected from
available sector or industry information, while manual
selection lets you provide your own comparison set.
""",
            """
Confirm that the normalized ticker, instrument type,
exchange and currency match the security you intended to
analyze. A weak or unrelated peer group can make relative
comparisons less meaningful.
""",
            """
Changing the ticker triggers a new market-data request.
Peer comparison does not mean the companies have identical
business models or risk profiles.
""",
        ),
        (
            "2. Market & Trading Overview",
            """
This section summarizes current price information,
instrument type, exchange, reporting currency, latest
market date, support and resistance context, and other
high-level market facts.
""",
            """
Start by checking the latest market date. Then compare the
current price with the displayed support, resistance,
moving-average and volatility context. A price near support
may still be weak; the wider trend and regime must agree.
""",
            """
The values are analytical reference points rather than
guaranteed reversal or breakout levels.
""",
        ),
        (
            "3. Fundamental Breakdown vs Peers",
            """
This area compares available valuation, profitability,
growth, balance-sheet and operating metrics with the
selected peer group.
""",
            """
Read each metric together with its unit and direction.
Lower valuation is not automatically better, and stronger
growth may come with higher valuation or financial risk.
Look for consistency across several related indicators.
""",
            """
Provider coverage differs by instrument and exchange.
Missing values are shown as unavailable rather than being
estimated by the application.
""",
        ),
        (
            "4. Earnings and financial statements",
            """
This section displays available quarterly earnings,
revenue and financial-statement trends from the external
data provider.
""",
            """
Look for trend consistency, acceleration, deterioration
and unusual one-period changes. Compare recent quarters
instead of interpreting one quarter in isolation.
""",
            """
Reporting calendars, accounting currencies and provider
availability can differ between companies.
""",
        ),
        (
            "5. Technical settings",
            """
The sidebar controls parameters such as RSI, MACD, moving
averages, Bollinger Bands and ATR. These parameters affect
the displayed technical calculations.
""",
            """
Use the defaults as the validated baseline. Changing one
parameter can alter signals and should be treated as a new
analytical configuration rather than a cosmetic change.
""",
            """
Repeatedly changing parameters until a preferred result
appears introduces selection bias.
""",
        ),
        (
            "6. Technical Signals Table",
            """
This table exposes the underlying indicator values and
their interpreted technical state.
""",
            """
Read indicator values together. Momentum, trend,
volatility and volume can disagree. A strong conclusion
should normally be supported by several independent
components rather than one indicator.
""",
            """
Technical indicators describe historical price behavior.
They do not know future prices.
""",
        ),
        (
            "7. Technical Overview and market regime",
            """
The technical overview summarizes trend direction,
momentum, volatility and the detected market regime.
""",
            """
The market regime provides context for which strategy may
be suitable. A trend strategy is less reliable in a
sideways environment, while a mean-reversion setup may be
less suitable during a strong directional move.
""",
            """
Regime classification is deterministic but remains an
estimate based on the available market history.
""",
        ),
        (
            "8. Trading Expert recommendation",
            """
The trading-expert section combines the supported
strategies, weighted components, evidence, conflicts,
market regime and risk controls into a final
decision-support report.
""",
            """
Read the recommendation together with the score,
confidence, evidence and conflicts. Confidence measures
agreement within the model; it is not the probability of a
profitable trade.
""",
            """
A high score does not override invalidation levels,
position sizing, liquidity rules or portfolio-risk limits.
The output is designed for paper-trading validation.
""",
        ),
        (
            "9. Strategy details and weighted components",
            """
This area shows how each supported strategy contributed to
the result and exposes the weighted calculation trace.
""",
            """
Review which components supported the conclusion and which
opposed it. A result with several strong conflicts deserves
more caution than a similar score with broad agreement.
""",
            """
Comparing scores from different threshold configurations
may be misleading unless the configuration is also
recorded.
""",
        ),
        (
            "10. Entry, stop, targets and invalidation",
            """
Where the analysis supports a paper setup, this section
shows the planned entry area, stop level, targets,
reward-to-risk relationship and conditions that invalidate
the idea.
""",
            """
The stop is the predefined point where the original thesis
is no longer valid. Targets describe planned reward areas.
Always interpret the potential reward relative to the
distance and capital at risk.
""",
            """
These are model-generated paper-trading levels, not live
broker orders.
""",
        ),
        (
            "11. Price chart",
            """
The chart displays recent candles, volume and selected
technical overlays.
""",
            """
Use the chart to verify that the written conclusion is
consistent with visible price structure. Check trend,
gaps, volatility expansion, support tests and volume.
""",
            """
Visual chart interpretation is supporting evidence and
should not replace the traceable calculation.
""",
        ),
        (
            "12. News and peer comparison",
            """
The final areas provide recent external news and broader
peer context.
""",
            """
Use news to identify events that may explain volatility or
invalidate assumptions. Use peers to understand whether a
move is company-specific or shared by the wider industry.
""",
            """
Headlines may be delayed, incomplete or unrelated to the
specific market move. The analytical engine does not treat
a headline as verified financial advice.
""",
        ),
    )

    render_sections(stock_sections)


with portfolio_tab:
    st.header(
        "Paper Portfolio & Reliability"
    )

    st.markdown(
        """
The Paper Portfolio dashboard is the operational,
read-only application. It does not download current market
data and cannot create, change or submit orders.

It reads the persisted SQLite records produced by the
scanner, paper-execution engine, scheduler and notification
service.

A useful reading order is:

1. Confirm the database and paper account.
2. Check reconciliation.
3. Review open positions and pending orders.
4. Review closed trades and their decision evidence.
5. Review equity and performance.
6. Review scans and operational reliability.
7. Use Provenance to verify each displayed value.
"""
    )

    portfolio_sections = (
        (
            "1. Dashboard source",
            """
The sidebar identifies the SQLite database and paper
account being displayed. The recent-record controls limit
how many scans, execution runs and jobs are loaded.
""",
            """
Always verify the account ID before interpreting the
portfolio. Clicking Refresh reloads persisted records from
SQLite.
""",
            """
Refresh does not run a scan, execute a paper order or
contact an external data provider.
""",
        ),
        (
            "2. Top portfolio metrics",
            """
The top cards show cash balance, available cash, open
positions, pending orders, realized net P&L and account
reconciliation.
""",
            """
Cash balance is the persisted account balance. Available
cash subtracts reserved cash. Realized P&L includes only
closed paper trades. Reconciled means stored cash matches
the ledger-derived balance.
""",
            """
An empty new account correctly shows zero positions,
orders and realized P&L.
""",
        ),
        (
            "3. Overview",
            """
The Overview tab shows the complete account record and the
cash-reconciliation calculation.
""",
            """
The reconciliation difference should be zero. A non-zero
difference means the account balance and ledger entries do
not agree and should be investigated before another cycle.
""",
            """
Values such as `0E-8` are Decimal representations of exact
zero, not a financial discrepancy.
""",
        ),
        (
            "4. Positions & Orders",
            """
This tab displays open paper positions and orders that are
waiting for a paper fill or another lifecycle action.
""",
            """
For each position, review the symbol, quantity, entry,
stop, targets, opening time and expiry. For each order,
review reserved cash, idempotency key and status.
""",
            """
The dashboard cannot cancel, fill or modify an order. It
only displays persisted records.
""",
        ),
        (
            "5. Trades & Evidence",
            """
This tab displays closed-trade history and the signal
evidence linked to pending orders, positions and completed
trades.
""",
            """
Read the entry and exit price, net P&L, costs, return,
holding period and exit reason. Then inspect the original
strategy, regime, score, confidence, threshold version and
evidence.
""",
            """
A profitable result does not prove that the original
decision process was sound. Evidence and rule compliance
remain important.
""",
        ),
        (
            "6. Equity & Performance",
            """
This area displays the persisted equity curve, realized
trade metrics, costs, win rate, average return, total
return and maximum drawdown.
""",
            """
The equity curve shows how account equity changed at each
persisted execution snapshot. Drawdown measures the decline
from a previous equity peak. Performance breakdowns group
closed trades by strategy, instrument, regime and threshold
version.
""",
            """
With no execution snapshots, latest equity and drawdown
correctly display as unavailable.
""",
        ),
        (
            "7. Scans & Strategy",
            """
This tab shows recent market scans, every persisted scan
result, candidate ranking and closed-trade performance
grouped by analytical dimensions.
""",
            """
Check scan status, requested versus processed symbols,
rejections, signals and created orders. For results, review
release eligibility, ranking, score, confidence and
rejection reasons.
""",
            """
A completed scan can legitimately produce no order when
candidates fail release, liquidity, data-quality or
portfolio-risk rules.
""",
        ),
        (
            "8. Reliability",
            """
This area summarizes scans, execution runs, scheduled jobs,
notifications and system events.
""",
            """
Review total, successful, failed and pending records.
Then inspect detailed execution errors, scheduler failures,
notification delivery errors and high-severity events.
""",
            """
A reliability percentage based on very few records should
not be treated as a long-term performance measure.
""",
        ),
        (
            "9. Provenance",
            """
The Provenance tab identifies the SQLite tables, record
IDs, filters and deterministic calculation behind each
dashboard section.
""",
            """
Use this page when investigating a displayed number.
Record IDs connect the value to its persisted account,
signal, trade, scan, execution, job or system-event record.
""",
            """
A source record count of zero means the section is based on
an empty persisted dataset, not hidden application state.
""",
        ),
        (
            "10. Dashboard metadata",
            """
Metadata records the selected account, display limits,
read-only status and persisted-data source.
""",
            """
Confirm that `source` is `persisted_sqlite_records` and
`read_only` is true.
""",
            """
Metadata describes the dashboard request. It does not
represent a trading recommendation.
""",
        ),
    )

    render_sections(portfolio_sections)


st.divider()

st.info(
    "Both applications are designed for analysis and "
    "paper-trading validation. Neither page submits a "
    "real-money order."
)
