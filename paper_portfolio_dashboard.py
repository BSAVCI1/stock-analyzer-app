"""Read-only Streamlit dashboard for persisted paper records."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.portfolio_dashboard import (
    PortfolioDashboardRepository,
    PortfolioDashboardService,
    broker_reconciliation_item_rows,
    broker_reconciliation_summary_rows,
    closed_trade_rows,
    decision_trace_rows,
    equity_rows,
    execution_run_rows,
    format_money,
    format_percent,
    job_rows,
    metric_cards,
    notification_rows,
    open_position_rows,
    pending_order_rows,
    performance_breakdown_rows,
    provenance_rows,
    reliability_rows,
    scan_result_rows,
    scan_rows,
    system_event_rows,
)


st.set_page_config(
    page_title="Paper Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
)


def show_table(
    rows,
    *,
    empty_message: str,
) -> None:
    if not rows:
        st.info(empty_message)
        return

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )


st.title("📊 Paper Portfolio & Reliability")
st.caption(
    "Read-only view of persisted SQLite records. "
    "No market-data request, broker connection, or "
    "order submission occurs from this dashboard."
)


navigation_left, navigation_right = st.columns(2)

with navigation_left:
    if st.button(
        "📈 Back to Stock Analyzer",
        width="stretch",
        key="open_stock_analyzer",
    ):
        st.switch_page(
            "stock_analysis_app.py"
        )

with navigation_right:
    if st.button(
        "ℹ️ How to Read Both Apps",
        width="stretch",
        key="open_portfolio_guide",
    ):
        st.switch_page(
            "pages/3_App_Guide.py"
        )


default_database = os.getenv(
    "PAPER_DATABASE_PATH",
    "data/paper_trading.db",
)

default_account = os.getenv(
    "PAPER_ACCOUNT_ID",
    "",
)

with st.sidebar:
    st.header("Dashboard source")

    database_value = st.text_input(
        "SQLite database",
        value=default_database,
    ).strip()

    account_id = st.text_input(
        "Paper account ID",
        value=default_account,
    ).strip()

    recent_scan_limit = st.number_input(
        "Recent scans",
        min_value=1,
        max_value=500,
        value=20,
        step=1,
    )

    recent_execution_limit = st.number_input(
        "Recent execution runs",
        min_value=1,
        max_value=500,
        value=50,
        step=1,
    )

    recent_job_limit = st.number_input(
        "Recent scheduled jobs",
        min_value=1,
        max_value=500,
        value=50,
        step=1,
    )

    st.button(
        "Refresh persisted records",
        width="stretch",
    )


if not account_id:
    st.info(
        "Enter an existing paper account ID in "
        "the sidebar or set PAPER_ACCOUNT_ID."
    )

    st.stop()


database_path = Path(database_value)

if not database_path.exists():
    st.error(
        "The configured SQLite database does not "
        f"exist: {database_path}"
    )

    st.stop()


try:
    repository = PortfolioDashboardRepository(
        database_path
    )

    service = PortfolioDashboardService(
        repository
    )

    snapshot = service.build_snapshot(
        account_id,
        recent_scan_limit=int(
            recent_scan_limit
        ),
        recent_execution_limit=int(
            recent_execution_limit
        ),
        recent_job_limit=int(
            recent_job_limit
        ),
    )

except Exception as exc:
    st.error(
        f"{type(exc).__name__}: {exc}"
    )

    st.stop()


st.caption(
    f"Account: `{snapshot.account.account_id}` · "
    f"Generated: "
    f"`{snapshot.generated_at.isoformat()}` · "
    f"Database: `{database_path}`"
)


cards = metric_cards(snapshot)

for start in range(
    0,
    len(cards),
    3,
):
    card_columns = st.columns(3)

    for column, card in zip(
        card_columns,
        cards[start:start + 3],
    ):
        with column:
            st.metric(
                card["label"],
                card["value"],
            )

            st.caption(
                "Source: "
                f"`{card['source_table']}`"
            )


(
    overview_tab,
    positions_tab,
    trades_tab,
    equity_tab,
    scans_tab,
    reliability_tab,
    provenance_tab,
) = st.tabs(
    (
        "Overview",
        "Positions & Orders",
        "Trades & Evidence",
        "Equity & Performance",
        "Scans & Strategy",
        "Reliability",
        "Provenance",
    )
)


with overview_tab:
    st.subheader("Account")

    account_rows = (
        {
            "account_id":
            snapshot.account.account_id,
            "name":
            snapshot.account.name,
            "base_currency":
            snapshot.account.base_currency,
            "status":
            snapshot.account.status.value,
            "starting_balance":
            str(
                snapshot.account
                .starting_balance
            ),
            "cash_balance":
            str(
                snapshot.account
                .cash_balance
            ),
            "reserved_cash":
            str(
                snapshot.account
                .reserved_cash
            ),
            "created_at":
            snapshot.account
            .created_at
            .isoformat(),
            "updated_at":
            snapshot.account
            .updated_at
            .isoformat(),
        },
    )

    show_table(
        account_rows,
        empty_message=(
            "No account record is available."
        ),
    )

    st.subheader("Cash reconciliation")

    reconciliation = (
        snapshot.reconciliation
    )

    reconciliation_rows = (
        {
            "account_id":
            reconciliation.account_id,
            "stored_cash_balance":
            str(
                reconciliation
                .stored_cash_balance
            ),
            "ledger_cash_balance":
            str(
                reconciliation
                .ledger_cash_balance
            ),
            "difference":
            str(
                reconciliation.difference
            ),
            "reconciled":
            reconciliation.reconciled,
        },
    )

    show_table(
        reconciliation_rows,
        empty_message=(
            "No reconciliation record "
            "is available."
        ),
    )

    if reconciliation.reconciled:
        st.success(
            "Stored cash reconciles to the "
            "persisted ledger."
        )
    else:
        st.error(
            "Stored cash does not reconcile "
            "to the persisted ledger."
        )


with positions_tab:
    st.subheader("Open positions")

    show_table(
        open_position_rows(snapshot),
        empty_message=(
            "There are no open paper positions."
        ),
    )

    st.subheader("Pending paper orders")

    show_table(
        pending_order_rows(snapshot),
        empty_message=(
            "There are no pending paper orders."
        ),
    )


with trades_tab:
    st.subheader("Closed-trade history")

    show_table(
        closed_trade_rows(snapshot),
        empty_message=(
            "There are no closed paper trades."
        ),
    )

    st.subheader(
        "Decision evidence and exit reasons"
    )

    show_table(
        decision_trace_rows(snapshot),
        empty_message=(
            "No persisted lifecycle records "
            "have linked decision evidence."
        ),
    )


with equity_tab:
    performance = snapshot.performance
    equity = snapshot.equity_performance

    performance_columns = st.columns(8)

    performance_values = (
        (
            "Closed trades",
            str(performance.trade_count),
        ),
        (
            "Win rate",
            format_percent(
                performance.win_rate_pct
            ),
        ),
        (
            "Gross P&L",
            format_money(
                performance.gross_pnl,
                snapshot.account
                .base_currency,
            ),
        ),
        (
            "Transaction costs",
            format_money(
                performance.total_costs,
                snapshot.account
                .base_currency,
            ),
        ),
        (
            "Net P&L after costs",
            format_money(
                performance.net_pnl,
                snapshot.account
                .base_currency,
            ),
        ),
        (
            "Expectancy per trade",
            format_money(
                performance.expectancy,
                snapshot.account
                .base_currency,
            ),
        ),
        (
            "Latest equity",
            (
                format_money(
                    equity.latest_equity,
                    snapshot.account
                    .base_currency,
                )
                if equity.latest_equity
                is not None
                else "N/A"
            ),
        ),
        (
            "Maximum drawdown",
            (
                format_percent(
                    equity
                    .maximum_drawdown_pct
                )
                if equity.point_count
                else "N/A"
            ),
        ),
    )

    for column, (
        label,
        value,
    ) in zip(
        performance_columns,
        performance_values,
    ):
        with column:
            st.metric(label, value)

    st.subheader("Persisted equity curve")

    equity_data = pd.DataFrame(
        equity_rows(snapshot)
    )

    if equity_data.empty:
        st.info(
            "No persisted equity snapshots "
            "are available."
        )
    else:
        figure = go.Figure()

        figure.add_trace(
            go.Scatter(
                x=equity_data[
                    "captured_at"
                ],
                y=equity_data[
                    "equity"
                ],
                mode="lines+markers",
                name="Equity",
            )
        )

        figure.add_trace(
            go.Scatter(
                x=equity_data[
                    "captured_at"
                ],
                y=equity_data[
                    "cash_balance"
                ],
                mode="lines",
                name="Cash",
            )
        )

        figure.add_trace(
            go.Scatter(
                x=equity_data[
                    "captured_at"
                ],
                y=equity_data[
                    "market_value"
                ],
                mode="lines",
                name="Market value",
            )
        )

        figure.update_layout(
            xaxis_title="Captured at",
            yaxis_title=(
                snapshot.account
                .base_currency
            ),
            legend_title="Persisted value",
            margin={
                "l": 20,
                "r": 20,
                "t": 20,
                "b": 20,
            },
        )

        st.plotly_chart(
            figure,
            width="stretch",
        )

        show_table(
            tuple(
                equity_data.to_dict(
                    orient="records"
                )
            ),
            empty_message=(
                "No persisted equity snapshots "
                "are available."
            ),
        )

    st.subheader(
        "Performance breakdowns"
    )

    show_table(
        performance_breakdown_rows(
            snapshot
        ),
        empty_message=(
            "No closed trades are available "
            "for breakdown analysis."
        ),
    )


with scans_tab:
    st.subheader("Recent market scans")

    show_table(
        scan_rows(snapshot),
        empty_message=(
            "No persisted market scans "
            "are available."
        ),
    )

    st.subheader(
        "Scan results and candidates"
    )

    show_table(
        scan_result_rows(snapshot),
        empty_message=(
            "No persisted scan results "
            "are available."
        ),
    )

    st.subheader(
        "Strategy, instrument, regime, "
        "and threshold results"
    )

    show_table(
        performance_breakdown_rows(
            snapshot
        ),
        empty_message=(
            "No closed-trade breakdowns "
            "are available."
        ),
    )


with reliability_tab:
    st.subheader(
        "Operational reliability"
    )

    show_table(
        reliability_rows(snapshot),
        empty_message=(
            "No persisted reliability "
            "records are available."
        ),
    )

    st.subheader(
        "Broker-paper reconciliation"
    )

    st.caption(
        "This section reads only the latest "
        "persisted reconciliation result. "
        "It does not contact a broker."
    )

    show_table(
        broker_reconciliation_summary_rows(
            snapshot
        ),
        empty_message=(
            "No persisted broker-paper "
            "reconciliation run is available."
        ),
    )

    st.subheader(
        "Unresolved broker differences"
    )

    show_table(
        broker_reconciliation_item_rows(
            snapshot,
            unresolved_only=True,
        ),
        empty_message=(
            "No unresolved broker-paper "
            "differences are persisted."
        ),
    )

    st.subheader("Execution runs")

    show_table(
        execution_run_rows(snapshot),
        empty_message=(
            "No persisted execution runs "
            "are available."
        ),
    )

    st.subheader("Scheduled jobs")

    show_table(
        job_rows(snapshot),
        empty_message=(
            "No persisted scheduled jobs "
            "are available."
        ),
    )

    st.subheader(
        "Notification delivery"
    )

    show_table(
        notification_rows(snapshot),
        empty_message=(
            "No persisted notifications "
            "are available."
        ),
    )

    st.subheader("System events")

    show_table(
        system_event_rows(snapshot),
        empty_message=(
            "No persisted system events "
            "are available."
        ),
    )


with provenance_tab:
    st.subheader(
        "Displayed-value provenance"
    )

    st.caption(
        "Every dashboard section identifies "
        "its persisted source tables, record "
        "IDs, filters, and calculations."
    )

    show_table(
        provenance_rows(snapshot),
        empty_message=(
            "No section provenance "
            "is available."
        ),
    )

    with st.expander(
        "Dashboard metadata"
    ):
        st.json(
            dict(snapshot.metadata)
        )
