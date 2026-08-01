# BSAVCI Trading Expert — gated implementation roadmap

The project will advance one checkpoint at a time. The next checkpoint is not
implemented until the previous checkpoint passes its automated and manual gate.

## P0 — Correctness and stability

### P0.1 — Isolated market-data foundation **(included in this package)**

Deliverables:
- validated ticker input
- safe price-history loading
- metadata fallbacks
- controlled errors
- 15-minute Streamlit cache in a diagnostic app
- unit tests and GitHub Actions

Gate:
- 11 unit tests pass
- AAPL, SPCE, SXR8.DE and VWCE.DE load
- invalid symbol is handled without a traceback

### P0.2 — Integrate the data layer into the production app

Deliverables:
- replace repeated direct `yf.Ticker(...).info` calls
- one validated snapshot per selected ticker
- remove repeated downloads during Streamlit reruns
- application-level loading and error states

Gate:
- existing dashboard loads for all acceptance tickers
- one ticker failure cannot crash the whole app
- no repeated provider calls for the same ticker within the cache window

### P0.3 — Correct technical-history and indicator calculations

Deliverables:
- at least two years of daily data for MA200
- pure indicator functions for MA, EMA, RSI, MACD, Bollinger Bands, ATR and OBV
- sufficient-history checks
- remove deprecated date-selection logic

Gate:
- indicator unit tests pass against fixed fixtures
- MA200 is never displayed with fewer than 200 observations
- no NaN-derived Golden/Death Cross classification

### P0.4 — Correct fundamentals and quarterly analysis

Deliverables:
- metric-specific peer evaluation rules
- chronological quarter sorting
- QoQ and YoY calculations
- explicit stock-versus-ETF handling
- missing-value handling

Gate:
- higher/lower interpretation is correct per metric
- latest quarter is correctly identified
- ETFs do not receive invalid corporate-fundamental scores

### P0.5 — Consolidate charts, news and currency handling

Deliverables:
- display the created Plotly chart
- one news service instead of duplicate modules
- remove fixed sentiment result
- instrument currency, exchange and timestamp display
- remove duplicated imports and broad bare exceptions

Gate:
- one coherent chart and one news panel
- no hard-coded dollar sign for EUR instruments
- all provider errors become controlled warnings

### P0.6 — P0 regression gate

Deliverables:
- full P0 test suite
- Streamlit smoke test
- documented acceptance checklist

Gate:
- P0 tests and GitHub Actions green
- all five acceptance cases pass

## P1 — Trading-expert signal engine

### P1.1 — Canonical analysis data model

Deliverables:
- immutable indicator snapshot
- signal enums: BUY, WATCH, HOLD, REDUCE, SELL
- strategy result and evidence objects

Gate:
- objects validate required inputs and reject inconsistent values

### P1.2 — Market-regime classifier

Deliverables:
- bullish, bearish, sideways and high-volatility regimes
- trend slope, price location and volatility inputs

Gate:
- deterministic fixture-based classification tests pass

### P1.3 — Trend-pullback setup

Deliverables:
- setup detection
- confirmation and invalidation rules
- evidence list

Gate:
- known positive, negative and ambiguous scenarios pass

### P1.4 — Breakout setup

Deliverables:
- range breakout detection
- volume confirmation
- false-breakout filters

Gate:
- no breakout is emitted without close and volume confirmation

### P1.5 — Mean-reversion setup

Deliverables:
- enabled only in suitable regimes
- support-zone and reversal confirmation

Gate:
- strategy is vetoed in a strong bearish trend

### P1.6 — Scoring and conflict resolution

Deliverables:
- weighted trend, setup, momentum, volume, volatility and fundamental scores
- one final recommendation instead of concatenated contradictory signals

Gate:
- identical inputs always produce the same score and decision
- conflict cases resolve according to documented priority rules

### P1.7 — Risk manager and paper-order generator

Deliverables:
- entry zone
- ATR/structure-based stop
- target levels
- reward-to-risk calculation
- signal expiry
- risk vetoes

Gate:
- BUY cannot be issued below the minimum reward-to-risk threshold
- every BUY/SELL decision has a defined invalidation point

### P1.8 — Trading Expert dashboard

Deliverables:
- decision card
- evidence and conflicts
- paper order only
- no broker connection

Gate:
- all recommendations are traceable to deterministic calculations

## P2 — Backtesting and validation

### P2.1 — Backtest event and trade model

Deliverables:
- signal, order, fill, position and closed-trade records

Gate:
- lifecycle tests cover entry, exit, stop, target and expiry

### P2.2 — Next-session execution engine

Deliverables:
- no same-bar/look-ahead execution
- next-open or predefined-limit fill rules

Gate:
- signals cannot use future bars

### P2.3 — Costs, slippage and position sizing

Deliverables:
- configurable fees and slippage
- capital and allocation constraints

Gate:
- account balance reconciles exactly after every test trade

### P2.4 — Performance metrics

Deliverables:
- return, annualised return, drawdown, win rate, profit factor, exposure,
  average holding period and Sharpe ratio
- buy-and-hold comparison

Gate:
- metrics match hand-calculated fixtures

### P2.5 — Train/test and walk-forward validation

Deliverables:
- in-sample and out-of-sample separation
- rolling walk-forward tests
- parameter stability view

Gate:
- no strategy is promoted using only in-sample performance

### P2.6 — Strategy acceptance report

Deliverables:
- performance by instrument and market regime
- accept/reject decision with reasons

Gate:
- strategy must meet agreed return, drawdown, trade-count and stability criteria

### P2.7 — P2 release gate

Deliverables:
- regression suite across P0-P2
- documented limitations
- approved signal thresholds

Gate:
- only validated strategies become eligible for later alert scheduling

## Deferred until after P0-P2

- OpenAI or any other AI-platform connection
- broker connection and live order execution
- background alerts and scheduled watchlist scans

## P3 — Autonomous paper portfolio and monitoring

P3 remains strictly paper-only. Broker connectivity and real-money execution
remain deferred until the internal paper portfolio has demonstrated reliable
operation and acceptable validated performance.

### P3.1 — Persistent automated paper portfolio

Deliverables:
- SQLite account and cash ledger
- persistent signals, orders, fills, positions and closed trades
- idempotent order creation
- exact account reconciliation
- persistent audit events and notification queue
- strategy, threshold and application version tracking

Gate:
- account creation, reservation, fill, position closure and realised P&L
  reconcile exactly
- repeating the same order request cannot create a duplicate purchase
- every material state change has an audit record

### P3.2 — Automatic market scanner

Deliverables:
- configured stock universe
- liquidity, price-history and data-quality filters
- batch deterministic analysis
- candidate ranking
- scan history and rejection reasons

Gate:
- every candidate and rejection is traceable to validated data and rules
- only P2 release-eligible strategies can create candidate orders

### P3.3 — Automated paper execution and monitoring

Deliverables:
- portfolio-level risk approval
- automatic pending paper buys
- automatic next-session fills
- stop, target, expiry and signal-reversal monitoring
- automatic paper sells
- duplicate-order and stale-data protection
- kill switch and portfolio circuit breakers

Gate:
- no same-session or future-data execution
- cash, reserved capital, open risk and realised P&L reconcile after every run
- repeated jobs remain idempotent

### P3.4 — Notifications and scheduled jobs

Deliverables:
- Telegram and/or email notifications
- buy and sell confirmations
- risk-rejection and system-failure alerts
- daily portfolio summary
- weekly performance and reliability report
- exchange-calendar-aware scheduled scans

Gate:
- notification delivery status and failures are persisted
- missed or repeated scheduler runs cannot duplicate orders

### P3.5 — Paper portfolio and reliability dashboard

Deliverables:
- account, cash and open-position overview
- pending orders and trade history
- decision evidence and exit reasons
- equity curve and performance metrics
- results by strategy, instrument, regime and threshold version
- scan, notification and system reliability metrics

Gate:
- every displayed value can be traced to persisted records

### P3.6 — Broker paper-account adapter

Deliverables:
- common execution-adapter interface
- internal paper adapter
- broker paper-account adapter
- account, order and position reconciliation
- live adapter explicitly disabled

Gate:
- internal and broker-paper records reconcile without unresolved differences
- no live credentials or live-order path are enabled

### P3.7 — P3 release gate

Deliverables:
- complete P0–P3 regression suite
- operational reliability report
- paper-trading performance report
- documented limitations and approved operating configuration

Gate:
- automated paper trading must demonstrate stable, reconciled operation
- live trading remains ineligible until separately reviewed and approved

## Deferred until after P3

- real-money broker execution
- fully automatic live trading
- options, leverage and short selling
- AI-generated changes to deterministic scores, orders or risk decisions
