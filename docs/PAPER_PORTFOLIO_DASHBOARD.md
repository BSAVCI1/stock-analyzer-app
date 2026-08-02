# Paper Portfolio Dashboard

The paper portfolio dashboard is a separate, read-only
Streamlit application backed entirely by persisted SQLite
records.

It does not:

- download current market data
- connect to a broker
- submit or modify orders
- change paper portfolio records
- dispatch notifications

## Start the dashboard

Configure the persistent database and an existing paper
account:

    export PAPER_DATABASE_PATH=data/paper_trading.db
    export PAPER_ACCOUNT_ID=ACC-REPLACE-ME

Start the application:

    streamlit run paper_portfolio_dashboard.py

The dashboard normally opens at:

    http://localhost:8501

The database and account can also be entered in the
dashboard sidebar.

## Dashboard sections

### Overview

Displays the persisted account record, cash balances and
ledger reconciliation.

### Positions and orders

Displays open paper positions and pending paper orders,
including their persisted identifiers, prices, targets and
expiry timestamps.

### Trades and evidence

Displays closed trades, exit reasons, decision evidence,
strategy, regime, threshold version and associated signal
records.

### Equity and performance

Displays the persisted equity curve, realised performance,
costs, drawdown and closed-trade breakdowns.

### Scans and strategy

Displays recent scanner runs, scan results, candidate
ranking and results grouped by strategy, instrument,
market regime and threshold version.

### Reliability

Displays persisted execution runs, scheduled jobs,
notification outcomes and system events.

### Provenance

Every dashboard section identifies:

- source SQLite tables
- persisted record identifiers
- applied filters
- deterministic calculations

The dashboard service does not use hidden Streamlit session
state to calculate portfolio results.

## Refreshing

The Refresh button reruns the Streamlit application and
reloads records from SQLite.

The dashboard does not run a market scan or execution cycle.
Those remain separate scheduled-job commands.

## Deployment

Use the same persistent database volume as the paper
automation service.

A temporary CI runner or ephemeral container without a
persistent database volume is not suitable for production
dashboard deployment.

Example:

    streamlit run paper_portfolio_dashboard.py \
      --server.headless=true \
      --server.port=8501

## Safety boundary

This dashboard is part of the internal paper-trading system.
It contains no live-trading adapter, broker credentials or
real-money order path.

## Integrated application navigation

For navigation between the Stock Analyzer, Paper Portfolio
and App Guide, use the Stock Analyzer as the primary
Streamlit entry point:

    streamlit run stock_analysis_app.py

The buttons at the top of each application switch between
the registered Streamlit pages without requiring a second
server.
