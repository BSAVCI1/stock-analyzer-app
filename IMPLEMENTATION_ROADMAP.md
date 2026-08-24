# Smart Investment Bot — End-to-End Implementation Roadmap v3.1

**Reviewed:** 4 August 2026
**Document type:** Consolidated governing roadmap from P0 through P7
**Current product mode:** Paper-only autonomous investment bot under development
**Operational portfolio target:** EUR 2,000
**Target and hard maximum order value:** EUR 100
**Primary strategies:** Swing and medium-term
**Notifications:** Email and Telegram
**Broker integration:** None in P0-P6; IBKR is a fee/execution reference and optional manual execution venue only

## 1. Document purpose and roadmap governance

This document is the single end-to-end roadmap for the Smart Investment Bot. It consolidates completed foundations, the current operational-validation gap, the approved autonomous paper-bot plan, the unattended validation pilot, optional manual IBKR comparison, and any future broker-integration path.

Roadmap governance rules:
- Every future planning update must preserve the complete P0-P7 view rather than publishing only the next phase.
- A checkpoint is not promoted until its automated and manual gate passes.
- Completed phases remain visible because later design decisions depend on their assumptions and evidence.
- Configuration and strategy changes are versioned; evidence from different versions is not silently combined.
- Paper-only and deny-by-default behaviour remain mandatory through P6.
- P7 is optional and requires a separate written go/no-go decision.

## 2. Executive product charter

The product is an always-on, evidence-driven smart investment bot that observes configured markets, maintains ranked watchlists, identifies qualified swing and medium-term opportunities, creates and manages paper positions, and sends actionable notifications through email and Telegram.

The bot will use a new EUR 2,000 operational paper account. Each normal order targets EUR 100 and may never exceed EUR 100 in portfolio currency. The order must be reduced or rejected when risk, fees, spread, FX, liquidity, fractional-share or cash constraints make EUR 100 inappropriate.

No IBKR API connection is planned during P0-P6. The user may manually copy selected action tickets into IBKR. Those manual trades are journalled separately and never change the official paper-performance history.

## 3. Approved scope and reviewed changes

| Area | Earlier direction | Consolidated approved direction |
|---|---|---|
| Product | Analysis dashboard and manual post-close cycle | Always-on autonomous paper investment bot |
| Strategies | Intraday, swing and medium-term discussed | Swing and medium-term only |
| Portfolio | USD 10,000 default test account | Preserve test account; create EUR 2,000 operational account |
| Order sizing | Percentage-based defaults | EUR 100 target and hard ceiling with risk cap |
| Broker | Provider-neutral future path | No connection through P6; IBKR reference and optional manual venue |
| Notifications | Email after initial Telegram issue | Email and Telegram, routed by event type and severity |
| Runtime | Commands in Codespaces | Always-on deployed service with scheduler and recovery |
| Proof | Regression and one operational cycle | Multi-month operational, actionability and performance evidence |
| Live trading | Possible later extension | Deferred optional P7 with separate approval and gates |

## 4. End-to-end status summary

| Phase | Name | Status | Current decision |
|---|---|---|---|
| P0 | Correctness and stability | Complete | Retain as correctness foundation |
| P1 | Trading-expert signal engine | Complete | Retain as deterministic decision engine |
| P2 | Backtesting and validation | Complete | Retain as validation and promotion control |
| P3 | Operational paper-trading platform | Complete — INTERNAL_ONLY release ready | Preserve evidence and begin lessons-learned-driven P4 work |
| P4 | Autonomous smart investment bot | Planned | Next engineering phase |
| P5 | Unattended paper-validation pilot | Planned | Planned unattended paper pilot |
| P6 | Manual IBKR decision-support pilot | Planned | Planned optional manual IBKR comparison |
| P7 | Optional future broker integration | Deferred / optional | Deferred; no work without separate approval |

### Reviewed repository evidence

- P3.1 persistent paper portfolio: commit `2c9f7a7`.
- P3.2 automatic scanner: commit `e3522a3`.
- P3.3 automated paper execution: commit `359beb6`.
- P3.4 jobs and notifications: commit `677bcbb`.
- P3.5 dashboard: commit `35cf3b5`; integrated navigation: `31cfb94`.
- P3.6 adapters and reconciliation: commit `9cbb415`.
- P3.7 operational release gate: `2c100c0`; internal-only profile: `4cc89bb`.
- Latest reviewed automated baseline: 389 passing tests, Automated tests #45.
- P3 operational gap closed on 3 August 2026 through a genuine XNYS market cycle, persisted scan/execution/job evidence, successful application email delivery, exact account reconciliation and same-session duplicate protection.
- P3 release gate result: `READY`, exit code `0`, INTERNAL_ONLY profile, live trading disabled.
- Operational evidence IDs: job `JOB-97037d6098fa45e78b76d287c383a729`, scan `SCAN-87a968a0c4964682a2fe80520bee8f18`, execution run `RUN-94e0b3a4eea3437db0047a534cba1ba7`.

## 5. Product policy baseline

| Policy | Approved baseline |
|---|---|
| Trading mode | Paper only through P6 |
| Operational starting balance | EUR 2,000 |
| Target order value | EUR 100 |
| Hard order ceiling | EUR 100 in portfolio currency |
| Initial planned-loss cap | EUR 10 per trade, configurable and subject to validation |
| Initial open-position cap | 5 |
| Initial invested-exposure cap | EUR 500 |
| Initial new-position cap | 2 per day |
| Strategies | Swing and medium-term only |
| Leverage, shorts, options, CFDs, crypto | Disabled |
| Averaging down | Disabled |
| Broker connectivity | None through P6 |
| Manual IBKR trades | Optional, user-entered, separately journalled |
| Notification channels | Email and Telegram |

## 6. Strategy operating model

### Swing horizon
- Expected holding period: approximately 2-15 trading sessions.
- Daily and hourly structure with intraday confirmation used only to improve timing, not to create day trades.
- Primary initial autonomous strategy because it fits the existing deterministic trend-pullback foundation and the EUR 100 cost sensitivity.
- Every position records entry, stop, targets, expiry and thesis invalidation.

### Medium-term horizon
- Expected holding period: approximately 3-16 weeks.
- Daily and weekly structure with stronger fundamental and regime confirmation.
- Lower turnover, wider stops and stricter event-risk review.
- Order notional is reduced below EUR 100 when the risk cap requires it.

### Strategy separation rules
- No day-trading strategy or same-day mandatory exit is in scope.
- Signals, orders, positions, trades, thresholds and performance carry the strategy horizon and version.
- Swing and medium-term results are never aggregated to promote a weak strategy.
- Parameter changes create a new evidence cohort.

## 7. Detailed P0-P7 roadmap

## P0 — Correctness and stability

**Status:** Complete
**Purpose:** Establish a reliable market-data and analytical foundation before any strategy or automation is trusted.
**Phase exit outcome:** The production application loads supported instruments safely, calculates indicators and fundamentals correctly, and passes a documented regression gate.

### P0.1 — Isolated market-data foundation

Create a validated, fault-tolerant market-data boundary.

**Deliverables**
- Validated ticker input
- Safe price-history loading
- Metadata fallbacks
- Controlled provider errors
- 15-minute Streamlit cache in a diagnostic app
- Unit tests and GitHub Actions

**Gate**
- AAPL, SPCE, SXR8.DE and VWCE.DE load
- Invalid symbols fail without a traceback
- Initial unit-test gate passes

**Status:** Complete

### P0.2 — Production data-layer integration

Replace repeated direct provider calls with one validated snapshot per ticker.

**Deliverables**
- Single market-data snapshot per selected ticker
- Application loading and controlled error states
- Removal of repeated downloads during Streamlit reruns

**Gate**
- Acceptance tickers load in the production dashboard
- One ticker failure cannot crash the application
- No repeated provider calls inside the cache window

**Status:** Complete

### P0.3 — Technical history and indicator correctness

Ensure technical outputs are calculated from sufficient, correctly ordered history.

**Deliverables**
- At least two years of daily history for MA200
- Pure MA, EMA, RSI, MACD, Bollinger Band, ATR and OBV functions
- Sufficient-history checks
- Removal of deprecated date-selection logic

**Gate**
- Indicator fixtures match expected values
- MA200 is never shown with fewer than 200 observations
- No NaN-derived Golden/Death Cross classification

**Status:** Complete

### P0.4 — Fundamentals and quarterly analysis

Make peer and quarterly analysis directionally and chronologically correct.

**Deliverables**
- Metric-specific peer evaluation rules
- Chronological quarter sorting
- QoQ and YoY calculations
- Stock-versus-ETF handling
- Explicit missing-value handling

**Gate**
- Higher/lower interpretation is correct per metric
- Latest quarter is correctly identified
- ETFs do not receive invalid corporate-fundamental scores

**Status:** Complete

### P0.5 — Charts, news and currency handling

Consolidate market context into one coherent, currency-aware presentation.

**Deliverables**
- Rendered Plotly chart
- Single news service
- Removal of fixed sentiment output
- Instrument currency, exchange and timestamp display
- Controlled provider warnings

**Gate**
- One coherent chart and news panel
- No hard-coded USD symbol for EUR instruments
- Provider errors are warnings rather than crashes

**Status:** Complete

### P0.6 — P0 regression gate

Prove the correctness baseline before strategy work advances.

**Deliverables**
- Full P0 test suite
- Streamlit smoke test
- Documented manual acceptance checklist

**Gate**
- P0 tests and GitHub Actions are green
- All acceptance cases pass

**Status:** Complete

## P1 — Trading-expert signal engine

**Status:** Complete
**Purpose:** Turn validated data into deterministic, traceable investment decisions and paper-order plans.
**Phase exit outcome:** Every recommendation is reproducible, explains supporting and opposing evidence, and produces paper-only risk levels when an actionable setup survives all gates.

### P1.1 — Canonical analysis model

Define immutable, internally consistent objects for analysis and decisions.

**Deliverables**
- Indicator snapshot
- BUY, WATCH, HOLD, REDUCE and SELL signal enums
- Strategy result and evidence objects

**Gate**
- Objects reject missing or inconsistent inputs
- Identical inputs serialize consistently

**Status:** Complete

### P1.2 — Market-regime classifier

Classify bullish, bearish, sideways and high-volatility environments.

**Deliverables**
- Trend slope, price-location and volatility inputs
- Deterministic regime classification

**Gate**
- Fixture-based classifications pass
- Regime output is reproducible

**Status:** Complete

### P1.3 — Trend-pullback setup

Detect pullbacks within eligible trends with explicit confirmation and invalidation.

**Deliverables**
- Setup detector
- Confirmation rules
- Invalidation rules
- Traceable evidence

**Gate**
- Positive, negative and ambiguous fixtures pass

**Status:** Complete

### P1.4 — Breakout setup

Detect confirmed range breakouts while filtering false breaks.

**Deliverables**
- Range-breakout detection
- Volume confirmation
- False-breakout filters

**Gate**
- No breakout is emitted without close and volume confirmation

**Status:** Complete

### P1.5 — Mean-reversion setup

Allow reversal setups only in suitable regimes.

**Deliverables**
- Support-zone logic
- Reversal confirmation
- Regime vetoes

**Gate**
- Mean reversion is vetoed in a strong bearish trend

**Status:** Complete

### P1.6 — Scoring and conflict resolution

Combine multiple analytical dimensions into one decision rather than contradictory fragments.

**Deliverables**
- Weighted trend, setup, momentum, volume, volatility and fundamental components
- Priority rules for conflicts
- One final recommendation

**Gate**
- Identical inputs produce identical scores
- Conflict cases resolve according to documented precedence

**Status:** Complete

### P1.7 — Risk manager and paper-order generator

Translate an eligible signal into a bounded paper plan.

**Deliverables**
- Entry zone
- ATR/structure-based stop
- Targets
- Reward-to-risk calculation
- Signal expiry
- Risk vetoes

**Gate**
- No BUY below the minimum reward-to-risk threshold
- Every BUY/SELL has an invalidation point
- Failed risk checks become vetoed HOLD

**Status:** Complete

### P1.8 — Trading Expert dashboard

Expose the complete deterministic decision trace to the user.

**Deliverables**
- Decision card
- Evidence and conflicts
- Entry, stop, targets and invalidation
- Paper-order presentation only

**Gate**
- All recommendations are traceable
- No broker connection or live order submission

**Status:** Complete

## P2 — Backtesting and validation

**Status:** Complete
**Purpose:** Prove that strategies and lifecycle assumptions work without future-data leakage and after realistic execution effects.
**Phase exit outcome:** Only strategies with accepted out-of-sample evidence and approved thresholds become eligible for later automated scanning and alerts.

### P2.1 — Backtest event and trade model

Represent the complete signal-to-closed-trade lifecycle.

**Deliverables**
- Signal, order, fill, position and closed-trade records
- Entry, exit, stop, target and expiry lifecycle tests

**Gate**
- All lifecycle paths are fixture-tested

**Status:** Complete

### P2.2 — Next-session execution engine

Prevent same-bar and look-ahead execution.

**Deliverables**
- Next-open or predefined-limit fill rules
- Chronological execution sequencing

**Gate**
- Signals cannot use future bars
- Same-bar fills are prohibited

**Status:** Complete

### P2.3 — Costs, slippage and position sizing

Model execution friction and capital constraints.

**Deliverables**
- Configurable fees and slippage
- Capital and allocation constraints
- Deterministic sizing

**Gate**
- Cash and ledger reconcile after every test trade
- Cost fixtures match hand calculations

**Status:** Complete

### P2.4 — Performance and benchmark comparison

Measure strategy results consistently and against a relevant passive alternative.

**Deliverables**
- Return, annualised return, drawdown, win rate, profit factor, exposure, average holding period and Sharpe ratio
- Buy-and-hold comparison
- Breakdowns by instrument and regime

**Gate**
- Metrics match hand-calculated fixtures

**Status:** Complete

### P2.5 — Train/test and walk-forward validation

Separate model development from evidence used for promotion.

**Deliverables**
- In-sample and out-of-sample separation
- Rolling walk-forward tests
- Parameter-stability view

**Gate**
- No strategy is promoted using only in-sample performance

**Status:** Complete

### P2.6 — Strategy acceptance report

Produce a documented accept/reject decision with reasons.

**Deliverables**
- Performance by instrument and regime
- Trade-count and stability checks
- Explicit acceptance reasons

**Gate**
- Strategy meets agreed return, drawdown, trade-count and stability criteria

**Status:** Complete

### P2.7 — P2 release gate

Control eligibility for future automation.

**Deliverables**
- Regression evidence across P0-P2
- Approved signal-threshold manifest
- Documented limitations

**Gate**
- Only validated strategies receive alert-scheduling eligibility
- Threshold changes require revalidation

**Status:** Complete
**Evidence reviewed**
- Release-gate baseline established
- Approved deterministic threshold manifest

## P3 — Operational paper-trading platform

**Status:** Complete — INTERNAL_ONLY release ready
**Purpose:** Convert validated strategies into a persistent, observable and safely operated paper-trading system.
**Phase exit outcome:** The system can persist accounts and orders, scan markets, run paper execution, schedule jobs, dispatch notifications, reconcile state and evaluate operational readiness. One genuine end-to-end operating cycle is still required to close the phase.

### P3.1 — Persistent automated paper portfolio

Create an auditable paper account with idempotent order and ledger lifecycle.

**Deliverables**
- Persistent accounts, balances, ledger, signals, orders, fills, positions and trades
- Risk and allocation constraints
- Idempotent order creation
- Account reconciliation

**Gate**
- Duplicate requests cannot create duplicate orders
- Cash and ledger reconcile exactly
- Portfolio remains paper-only

**Status:** Complete
**Evidence reviewed**
- Commit 2c9f7a7 — Add persistent automated paper portfolio

### P3.2 — Automatic deterministic market scanner

Scan configured symbols and persist every candidate decision.

**Deliverables**
- Release-gate eligibility lookup
- Liquidity and data-quality filters
- Candidate ranking
- Persisted scan and result records
- Idempotent scan keys

**Gate**
- Every requested symbol is processed or rejected with a reason
- A completed scan may legitimately create zero orders
- Duplicate scan keys are safe

**Status:** Complete
**Evidence reviewed**
- Commit e3522a3 — Add automatic deterministic market scanner

### P3.3 — Automated paper execution and monitoring

Advance eligible paper orders and positions through their lifecycle.

**Deliverables**
- Next-session simulated fills
- Position monitoring
- Stop, target, expiry and invalidation handling
- Persisted execution runs and system events

**Gate**
- Execution is deterministic and paper-only
- No future data is used
- Every failure is persisted and observable

**Status:** Complete
**Evidence reviewed**
- Commit 359beb6 — Add automated paper execution and monitoring

### P3.4 — Exchange-aware scheduled jobs and notifications

Wrap the scanner and execution engine in repeatable operational jobs.

**Deliverables**
- Market-cycle and weekly-report jobs
- Exchange-calendar awareness
- Internal event fan-out
- Email and Telegram sender support
- Retryable persisted notifications

**Gate**
- Jobs are idempotent
- Notifications are deduplicated and persisted
- Failed deliveries remain visible and retryable

**Status:** Complete
**Evidence reviewed**
- Commit 677bcbb — Add notifications and exchange-aware scheduled jobs

### P3.5 — Traceable portfolio dashboard and navigation

Provide a read-only operational view over persisted SQLite records.

**Deliverables**
- Account, reconciliation, orders, positions and trades
- Equity and performance
- Scans, jobs, notifications and system events
- Provenance and source-record traceability
- Integrated app navigation and guide

**Gate**
- Dashboard cannot create, modify or submit orders
- Every displayed value is traceable to persisted records

**Status:** Complete
**Evidence reviewed**
- Commit 35cf3b5 — Add traceable paper portfolio dashboard
- Commit 31cfb94 — Add integrated app navigation and user guide

### P3.6 — Provider-neutral broker-paper adapters and reconciliation

Define an optional future integration boundary without making broker connectivity mandatory.

**Deliverables**
- Internal execution adapter
- Provider-neutral broker-paper adapter contract
- Safety configuration and transport abstraction
- Persisted broker reconciliation runs and items
- Broker reconciliation CLI and dashboard status

**Gate**
- Live trading remains disabled
- Adapter layer has no direct network dependency
- Reconciliation differences are explicit and auditable

**Status:** Complete
**Evidence reviewed**
- Commit 9cbb415 — Add broker-paper execution adapters and reconciliation
- GitHub Actions #43 green; 360-test baseline at completion

### P3.7 — Operational release gate and internal-only profile

Require genuine regression and operational evidence before promotion.

**Deliverables**
- P3 release report
- Checks for account reconciliation, broker reconciliation, scans, execution runs, scheduled jobs, notifications and system events
- INTERNAL_ONLY and BROKER_PAPER profiles
- CLI exit codes for READY, BLOCKED and ERROR

**Gate**
- Live trading always blocks
- Missing broker reconciliation does not block INTERNAL_ONLY
- Actual broker differences always block
- Release is READY only when all required operational evidence passes

**Status:** Implemented — operational validation pending
**Evidence reviewed**
- Commit 2c100c0 — Add P3 operational release gate
- Commit 4cc89bb — Add internal-only P3 release profile
- 389 passing tests; Automated tests #45
- Current gap: genuine scans, execution runs, jobs and delivered notifications


## 8. P3 closure evidence and lessons learned

### 8.1 Formal closure decision

P3 is operationally complete under the `INTERNAL_ONLY` paper-trading profile. The release gate returned `READY` with exit code `0`; all required P0-P3 regression and operational checks passed, broker reconciliation was correctly non-blocking, and live trading remained disabled.

### 8.2 Genuine operational evidence

| Evidence area | Observed result | Decision |
|---|---|---|
| Market cycle | XNYS session 2026-08-03 completed | Pass |
| Symbols processed | 30 | Pass |
| Scan | 1 successful persisted scan | Pass |
| Execution run | 1 successful persisted execution run | Pass |
| Scheduled job | 1 completed persisted job | Pass |
| Notifications | 1 EMAIL notification sent, 0 failed | Pass |
| Portfolio | USD 10,000, no positions/orders, reconciled | Pass |
| System events | 1 successful observed event | Pass |
| Duplicate protection | Same-session rerun returned `duplicate: true`; totals remained unchanged | Pass |
| Regression | 389 tests, Automated tests #45 | Pass |
| Release gate | `READY`, exit code `0` | Pass |
| Live trading | Disabled | Pass |

### 8.3 Lessons learned

1. **One-command orchestration works.** `market-cycle` created the job, scan, execution evidence and application notification in one run.
2. **Notification semantics need clearer documentation.** The market cycle delivered the email immediately; a later `dispatch` correctly processed zero pending notifications.
3. **Idempotency is effective.** Re-running the same session reused the original job, scan and execution IDs and did not create another email or order.
4. **Configuration loading is fragile for manual users.** A fresh shell failed until `.paper-automation.env` was sourced. P4 must load and validate configuration automatically.
5. **No forced trade is a valid outcome.** The system processed 30 symbols and created zero candidates, proving rule discipline rather than operational failure.
6. **Decision transparency is insufficient when no candidate qualifies.** P4 should show ranked near-qualifiers, rejection reasons and WATCH candidates.
7. **Codespaces is suitable for development, not unattended operation.** P4 needs an always-on scheduler, health checks, restart recovery and automatic notification dispatch.
8. **One successful cycle closes P3 but does not prove investment sustainability.** Long-run reliability, actionability and cost-adjusted performance remain P5 gates.
9. **The USD 10,000 account is now audit evidence.** It must be preserved; P4 creates a separate EUR 2,000 operational account.
10. **Channel evidence must remain independent.** Email passed P3. Telegram returns in P4 and receives its own delivery, retry and failure-isolation tests.

### 8.4 P4 requirements created from P3 lessons

- Automatic startup configuration validation with a clear readiness report.
- Always-on deployment outside manual Codespaces sessions.
- Scheduler for swing and medium-term observation cycles.
- Explicit notification delivery mode, pending count and retry state.
- Ranked watchlist output even when no order is created.
- Per-symbol rejection reasons and near-qualification distance.
- Separate EUR 2,000 paper account with EUR 100 hard order ceiling.
- Email and Telegram channel isolation, deduplication and retry tests.
- Preserved P3 evidence and immutable release record.
- P5 sustainability evaluation based on multiple sessions and completed paper trades.

## P4 — Autonomous smart investment bot

**Status:** Planned
**Purpose:** Reconfigure and extend P3 into an always-on, paper-only investment bot focused on swing and medium-term opportunities.
**Phase exit outcome:** A production-like service automatically observes configured markets, maintains ranked watchlists, creates and manages EUR-denominated paper positions, and sends actionable email and Telegram alerts without any broker connection.

### P4.0 — Product charter and configuration reset

Make the new product direction explicit and versioned.

**Deliverables**
- End-to-end P0-P7 roadmap becomes the governing document
- Portfolio, strategy, cost, scheduling and notification policies become versioned configuration
- Paper-only and deny-by-default invariants
- Preserve the existing USD 10,000 test account and history

**Gate**
- Configuration can be printed without exposing secrets
- No setting enables live execution
- Regression suite remains green

**Completion evidence**
- Versioned product policy: `config/product_policy_v1.json`, policy version `p4.0-1`
- Strict loader and validator: `src/product_config.py`
- Database-independent safe-print command: `python -m src.jobs.cli product-config`
- Exact-schema validation rejects unknown and sensitive configuration keys
- Paper-only, live-execution-disabled and broker-API-disabled invariants are enforced
- EUR 2,000 operational portfolio direction and EUR 100 hard order ceiling are versioned
- Swing and medium-term horizons are enabled; intraday and day trading are prohibited
- Leverage, shorts, options, CFDs and crypto remain prohibited
- Historical P3 account `ACC-495a2ae778834fc4a2c14d24e66ef41e` is explicitly preserved
- Secret-isolation validation passed
- Full regression evidence: 394 tests passed
- Diff integrity check passed

**Status:** Complete

**Dependencies:** P3 operational cycle
**Relative effort:** S

### P4.1 — EUR 2,000 account and fixed-notional sizing

Create the operational paper portfolio and enforce the approved order policy.

**Deliverables**
- New EUR 2,000 paper account and account ID
- Sizing mode FIXED_NOTIONAL_WITH_RISK_CAP
- EUR 100 target and hard order ceiling
- Initial EUR 10 planned-loss cap per trade
- Initial five-position and EUR 500 invested-exposure caps
- Smaller orders when risk, cost, liquidity, fractional-share or cash constraints require it
- Multi-currency valuation and reconciliation

**Gate**
- No order exceeds EUR 100 in portfolio currency
- Planned loss does not exceed the configured cap
- No whole-share rounding above the ceiling
- Cash and ledger reconcile after every lifecycle

**Completion evidence**
- Operational account `ACC-749ca5703d214ef0b91f87b825e88849` created with EUR 2,000 starting and available cash
- Operational account is ACTIVE with zero reserved cash, zero open positions and zero pending orders at the P4.1 baseline
- Product policy version `p4.1-1` defines `FIXED_NOTIONAL_WITH_RISK_CAP`
- EUR 100 target and hard maximum order value are persisted in account controls
- EUR 10 maximum planned loss, five-position cap and EUR 500 invested-exposure cap are persisted
- Deterministic sizing rejects quantities that breach notional, risk, cash, exposure or position limits
- Whole-share sizing uses floor semantics and does not round above the EUR 100 ceiling
- Multi-currency signal, order, fill, position and closed-trade provenance is persisted through schema version 8
- Lifecycle cash, fees and P&L are converted into portfolio currency with explicit FX provenance
- Execution-engine sizing and portfolio valuation operate in portfolio currency
- Runtime uses `YahooFXRateProvider`, with direct-pair and inverse-pair handling and no silent cross-currency 1:1 fallback
- Account cash and ledger reconcile exactly with zero difference
- Lifecycle FX and reconciliation tests cover reservation, fill and close behavior
- P4.1 regression reached 452 passing tests locally and in a clean CI-like environment
- Latest runtime-FX checkpoint commit `a61ac29` was pushed to `main`; GitHub Automated tests #58 showed success
- Operational bootstrap baseline database SHA256 was `45364e65e2f95bd4b016c8332ed37410fe8ed494d8ce9917546e21792401a22e`
- The P4 operational database contains no reconstructed P3 scan, job, execution, notification or trading rows
- Historical P3 account `ACC-495a2ae778834fc4a2c14d24e66ef41e` remains preserved in the versioned policy and immutable P3 release evidence

**Status:** Complete
**Dependencies:** P4.0
**Relative effort:** M

### P4.2 — IBKR reference cost profile

Model the user’s actual IBKR economics without connecting the account.

**Deliverables**
- Versioned commission profile
- Minimum commission, exchange and regulatory fees
- Fractional-share and minimum-notional rules
- FX conversion cost model
- Round-trip cost and net reward-to-risk calculation
- Manual update workflow after IBKR pricing changes

**Gate**
- Hand-calculated examples match
- Uneconomic EUR 100 trades are rejected
- No credentials or connectivity code are required

**Completion evidence**
- Reference profile `config/ibkr_reference_costs_v2.json` is versioned as `ibkr-reference-2026-08-09-v2`
- Confirmed operational pricing plan is `FIXED`; historical profile v1 remains inactive
- Product policy `p4.2-3` enables the IBKR cost gate and FIXED pricing while API connectivity remains disabled
- US commission minimums, regulatory-cost components, fractional-share rules and FX reference costs are modeled
- Round-trip economics calculate gross and cost-adjusted net reward-to-risk before order creation
- Deterministic hand-calculated economics matched the implementation and demonstrated an uneconomic cost-adjusted trade
- All paper lifecycle fee paths use the authoritative IBKR estimator when a pricing plan is selected
- Runtime `build_runtime()` loads the validated active product-policy cost configuration
- Per-trade FX conversion remains disabled; EUR/USD funding remains a manual portfolio-level activity and USD sale proceeds may remain in USD
- Current operational execution is USD-quoted US equities; non-USD lifecycle execution remains fail-closed
- Temporary EUR 2,000 / USD-quoted acceptance proof created a two-share USD 50 order with EUR 90 notional and EUR 5.40 planned loss
- The accepted proof reserved and booked IBKR FIXED fees and reconciled successfully through close
- An uneconomic candidate created zero orders and was rejected with `IBKRCostGateRejected`
- Final full regression passed 488 tests
- Lifecycle checkpoint `7bc7597` and runtime-activation checkpoint `b0c762d` were pushed to `main`
- Operational database SHA256 remained `45364e65e2f95bd4b016c8332ed37410fe8ed494d8ce9917546e21792401a22e`
- No credentials, IBKR API connectivity, broker execution or live trading capability were introduced
- Formal closure evidence is recorded in `docs/P4_2_OPERATIONAL_CLOSURE.md`

**Status:** Complete
**Dependencies:** IBKR meeting details; P4.1
**Relative effort:** M

### P4.3 — Swing and medium-term strategy separation

Treat the two horizons as independent products with separate evidence.

**Deliverables**
- StrategyHorizon values SWING and MEDIUM_TERM
- Separate timeframes, expiry, confirmation, holding and exit policies
- Horizon and strategy version on every signal, order, position and trade
- Independent acceptance and performance reporting
- Explicit prohibition of day-trading orders

**Gate**
- Identical inputs and configuration produce identical decisions
- Each horizon has independent fixtures
- No intraday/day-trading strategy can create an order

**Evidence**
- Strategy-horizon provenance checkpoint: commit `039e860`
- Versioned horizon-policy checkpoint: commit `cba94a7`
- Independent runtime holding clocks: commit `32e1abf`
- Independent acceptance and performance cohorts: PR #2
- Full regression evidence: 503 tests passed in Automated tests #79

**Status:** Complete
**Dependencies:** P4.0
**Relative effort:** L

### P4.4 — Eligible universe and ranked watchlist

Continuously identify the most actionable liquid instruments without forcing trades.

**Deliverables**
- Configurable universe of approximately 50-100 liquid stocks and broad ETFs
- Price, liquidity, data-quality, event-risk and fractional-eligibility filters
- Separate swing and medium-term ranking
- Watch, prepare, actionable, reject and stale states
- Reason codes and score decomposition
- User-curated inclusion and exclusion lists

**Gate**
- Every symbol has a persisted outcome
- Stale or incomplete data cannot create an order
- Zero actionable candidates is a valid result
- Ranking is deterministic

**Evidence**
- Versioned universe governance and curated inclusion/exclusion: PR #3
- Explicit watchlist states and decomposed deterministic scores: PR #4
- Independent swing and medium-term watchlist ranking: PR #5
- Fractional-feasibility, event-risk and structured filter-reason gates: PR #6
- Full regression evidence: 510 tests passed in Automated tests #91

**Status:** Complete
**Dependencies:** P4.2-P4.3
**Relative effort:** L

### P4.5 — Exchange-aware autonomous scheduler

Run the bot without manual terminal commands.

**Deliverables**
- Startup health check
- Pre-session universe refresh
- Periodic swing scans during configured market windows
- Daily medium-term scan after close
- Position-monitoring cadence
- Automatic notification dispatch
- Post-close reconciliation and daily report
- Weekly performance report

**Gate**
- Duplicate schedules remain idempotent
- Closed exchanges do not run market jobs
- Missed jobs are visible and safely recoverable
- Scheduler timezone and calendar tests pass

**Evidence**
- Versioned exchange-aware autonomous schedule policy: PR #7
- Replay-safe due-window orchestration and failure isolation: PR #8
- Scanner, execution, notification and reporting service wiring: PR #9
- Persistent checkpoints, bounded recovery and missed-job evidence: PR #10
- Full regression evidence: 531 tests passed in Automated tests #107

**Status:** Complete
**Dependencies:** P4.4
**Relative effort:** L

### P4.6 — Paper order and position management

Manage qualified swing and medium-term ideas through a realistic paper lifecycle.

**Deliverables**
- Cost-aware order proposal
- Fractional quantity calculation
- Entry expiry and cancellation
- Paper fills
- Stop, target and thesis-invalidation exits
- Corporate-action and earnings-event policy
- Capital reservation and release

**Gate**
- No order without valid entry, stop, target, expiry and net reward-to-risk
- No position exceeds portfolio policy
- Execution remains independent of manually copied IBKR trades

**Evidence**
- Cost-aware, fractional US long paper-order proposal contract: PR #11
- Proposal-backed persistent order creation and capital reservation: PR #12
- Deterministic managed stop, target, thesis, regime and time-exit policy: PR #13
- Versioned earnings and corporate-action event-risk policy: PR #14
- Existing persistent fill, expiry, cancellation and reconciliation lifecycle retained
- Full regression evidence: 560 tests passed in Automated tests #123

**Status:** Complete
**Dependencies:** P4.1-P4.5
**Relative effort:** L

### P4.7 — Email and Telegram action channels

Deliver useful, non-duplicative and auditable notifications.

**Deliverables**
- Telegram restored for concise startup, opportunity, fill, exit and failure alerts
- Email for complete action tickets, daily and weekly reports, and detailed failures
- Channel routing by event severity
- Deduplication, retry and delivery evidence
- Secret-safe health checks

**Gate**
- Both channels pass direct and application-level delivery tests
- No duplicate alert for the same event/channel
- Notification failures are visible and retryable
- Secrets never appear in logs or reports

**Evidence**
- Versioned severity and purpose-based channel routing with idempotent fan-out: PR #15
- Channel-specific action templates, payload redaction and secret-safe health checks: PR #16
- Bounded exponential-backoff retries and complete delivery evidence: PR #17
- Direct Telegram/SMTP and end-to-end application delivery contracts: PR #18
- Full regression evidence: 580 tests passed in Automated tests #135

**Status:** Complete
**Dependencies:** P4.5-P4.6
**Relative effort:** M

### P4.8 — Always-on deployment

Move runtime operation out of an interactive Codespace.

**Deliverables**
- Containerized service
- Managed scheduler or worker
- Persistent database volume and backups
- Environment-secret management
- Health endpoint and heartbeat
- Controlled deployment and rollback

**Gate**
- Service survives terminal closure and restarts
- Database persists across redeployments
- Backup and restore test succeeds
- Deployment does not expose credentials

**Implementation evidence**
- Non-root container, persistent data boundary, health endpoints and disk-backed worker heartbeat contract: PR #19
- Managed non-overlapping scheduler with deterministic run keys, graceful shutdown and lifecycle heartbeats: PR #20
- Verified online SQLite backup, checksum manifest and controlled atomic restore: PR #21
- File-mounted secret resolution with strict deployment mode and notification integration: PR #22
- Health-gated promotion, automatic rollback contracts, hardened Compose and restart-persistence verification: PR #23
- Portable combined health/worker runtime and harmless local-PC validation profile: PR #24
- Warning-free Intel Mac/Monterey local validation path and verified operating guide: PR #25
- Deny-by-default managed internal-paper adapter, isolated account bootstrap and local paper profile: PR #26
- Full automated deployment evidence passed: unit tests, image build, health, Compose, restart persistence, harmless local profile and managed internal-paper adapter
- User-device acceptance recorded in `P4_8_LOCAL_DEPLOYMENT_ACCEPTANCE.md`: health, heartbeat, persistence, duplicate protection, reconciled simulated account and controlled shutdown all passed

**Optional future external-platform gate**
- Provision externally only if later selected, then capture provider-specific restart, backup/restore, rollback and secret-mount evidence

**Status:** Complete
**Dependencies:** P4.5-P4.7
**Relative effort:** L

### P4.9 — Reliability, recovery and kill switch

Ensure failure causes safe inactivity rather than uncontrolled behaviour.

**Deliverables**
- Global pause/kill switch (operator CLI, persistent state, audit trail and
  recovery runbook implemented in P4.9.1)
- Per-strategy pause (selective entry block, pending-order cancellation,
  operator CLI and audit trail implemented in P4.9.2)
- Stale-data circuit breaker (account-wide entry preflight, persistent trip
  and verified automatic recovery implemented in P4.9.3)
- Reconciliation circuit breaker (persistent pre/post-execution trip,
  idempotent audit and verified automatic recovery implemented in P4.9.4)
- Daily and weekly loss-limit pause (persistent period-locked entry pauses,
  configurable 3%/5% defaults and audited rollover recovery implemented in
  P4.9.5)
- Provider outage handling (provider-stage classification, persistent
  fail-closed entry breaker, scheduler-cycle retry and verified clean-provider
  recovery implemented in P4.9.6)
- Restart recovery and idempotent replay (persistent interrupted-job
  recovery evidence plus duplicate-safe scan, execution and notification
  replay implemented in P4.9.7)
- Operational incident log (persistent lifecycle, named-operator CLI,
  immutable audit timeline and root-cause closure evidence implemented in
  P4.9.8)

**Gate**
- Any critical invariant failure blocks new orders
- Restart creates no duplicates
- Kill switch is tested end to end
- Recovery steps are documented

**Status:** Complete
**Dependencies:** P4.8
**Relative effort:** L

### P4.10 — Actionability and sustainability analytics

Measure whether the bot creates useful decisions, not merely activity.

**Deliverables**
- Performance after commissions, spread, slippage and FX (persisted-trade
  gross, fee, slippage, total-cost, net and expectancy baseline implemented in
  P4.10.1)
- Separate swing and medium-term results (combined horizon and strategy-version
  cohorts implemented in P4.10.1)
- Benchmark and cash comparison (immutable portfolio-currency benchmark
  observations, aligned price-return comparison and nominal-cash baseline
  implemented in P4.10.2)
- Expectancy, drawdown, profit factor and concentration (cost-adjusted
  expectancy and profit factor implemented in P4.10.1; drawdown retained from
  persisted equity snapshots; immutable position valuations, symbol weights,
  top-three weight and HHI concentration implemented in P4.10.3)
- Watchlist conversion and stale-signal rate (version-safe watch/preparation
  episodes, later actionable conversion, abandoned/open outcomes and expired
  unordered-signal rate implemented in P4.10.4)
- Alert usefulness and manual-copy journal
- Operational reliability metrics

**Gate**
- Metrics are reproducible from persisted records
- Results are not mixed across strategy versions
- No headline metric hides transaction costs

**Status:** In progress
**Dependencies:** P4.2-P4.9
**Relative effort:** M

### P4.11 — P4 release gate

Control entry into unattended paper validation.

**Deliverables**
- P0-P4 regression evidence
- Paper-only invariant checks
- EUR portfolio-policy checks
- Scheduler and deployment evidence
- Email and Telegram delivery evidence
- Recovery and kill-switch evidence
- Strategy-horizon acceptance

**Gate**
- Gate is READY only when required evidence passes
- Any live-execution capability blocks release
- Operational failures remain visible and blocking

**Status:** Planned
**Dependencies:** P4.0-P4.10
**Relative effort:** M

## P5 — Unattended paper-validation pilot

**Status:** Planned
**Purpose:** Run the autonomous bot for long enough to establish operational reliability, actionability and cost-adjusted performance.
**Phase exit outcome:** The bot has a statistically and operationally credible paper record. Strategies are promoted, revised or retired independently; insufficient evidence does not become approval.

### P5.1 — Controlled pilot launch

Start the unattended pilot with frozen configuration versions.

**Deliverables**
- Freeze portfolio, universe, strategy, cost and notification versions
- Record pilot start date and baseline
- Daily automated health and reconciliation evidence

**Gate**
- No unreviewed parameter changes during an evaluation window
- Every configuration change starts a new evidence cohort

**Status:** Planned
**Dependencies:** P4.11 READY

### P5.2 — Operational-quality evidence

Demonstrate dependable scheduling, data and communications.

**Deliverables**
- At least 60 genuine market sessions
- Job completion and timeliness
- Data freshness and provider availability
- Email and Telegram delivery rates
- Restart and recovery incidents
- Zero unexplained reconciliation differences

**Gate**
- No duplicate orders
- No material unresolved notification failure
- No silent missed session
- Critical incidents are closed with root cause

**Status:** Planned

### P5.3 — Strategy-performance evidence

Evaluate both horizons after all realistic costs.

**Deliverables**
- Minimum 30 closed paper trades in total
- Separate swing and medium-term cohorts
- Net expectancy, drawdown, profit factor, hit rate and holding period
- Benchmark and cash comparison
- Regime and instrument breakdowns

**Gate**
- Positive net expectancy is required for promotion
- Maximum drawdown stays within approved tolerance
- One exceptional trade cannot dominate the result
- A strategy with insufficient sample remains probationary

**Status:** Planned

### P5.4 — Actionability and recommendation quality

Measure whether alerts are clear and usable.

**Deliverables**
- Action-ticket completeness
- Alert timeliness
- Watchlist-to-order conversion
- False-positive and stale-signal rates
- User usefulness feedback
- Optional manual-copy decisions

**Gate**
- Actionable alerts include symbol, rationale, entry, stop, targets, expiry, costs and risk
- Low-value noise is reduced without hiding failures

**Status:** Planned

### P5.5 — Strategy promotion, revision or retirement

Make an explicit decision for each horizon.

**Deliverables**
- PROMOTE, CONTINUE_PROBATION, REVISE or RETIRE decision
- Reasons and supporting evidence
- Required remediation for non-promotion

**Gate**
- Swing and medium-term are decided independently
- No strategy is promoted on aggregate results alone

**Status:** Planned

### P5.6 — P5 sustainability gate

Decide whether the product creates durable value in paper mode.

**Deliverables**
- Consolidated operational, performance and actionability report
- Risk review
- Cost-model review against observed manual IBKR examples where available

**Gate**
- Gate can approve continued paper operation without approving broker integration
- Any material integrity issue blocks progression

**Status:** Planned

## P6 — Manual IBKR decision-support pilot

**Status:** Planned
**Purpose:** Compare selected paper recommendations with manually entered IBKR trades while keeping the official paper record independent.
**Phase exit outcome:** The team understands real execution friction and whether action tickets remain valuable when manually applied. The platform still has no broker connection.

### P6.1 — Manual action-ticket workflow

Define how the user may copy selected ideas without implying automatic execution.

**Deliverables**
- Complete action ticket
- Explicit paper-only label
- Manual accept/decline/skip journal
- No broker credentials

**Gate**
- User decision does not alter the official paper strategy record

**Status:** Planned
**Dependencies:** P5.6; user discretion

### P6.2 — Manual IBKR trade journal

Record real entries and exits entered outside the platform.

**Deliverables**
- Actual quantity, price, timestamp, commission, FX and exit
- Reason for copying or skipping
- Link to originating paper signal

**Gate**
- Manual records are separate from paper execution tables
- No credentials or statements are required

**Status:** Planned

### P6.3 — Paper-versus-manual execution comparison

Quantify the difference between modelled and observed execution.

**Deliverables**
- Entry and exit slippage
- Actual versus modelled fees
- FX differences
- Fill feasibility
- Timing and human-delay effects

**Gate**
- Comparisons use matched signal IDs and timestamps
- Differences do not rewrite paper history

**Status:** Planned

### P6.4 — Decision-support value assessment

Assess whether the bot improves human investment decisions.

**Deliverables**
- Clarity, timing and confidence feedback
- Copied versus skipped outcome analysis
- Behavioural and operational observations

**Gate**
- No conclusion relies only on copied winners
- Skipped ideas remain part of the analysis

**Status:** Planned

### P6.5 — P6 release decision

Decide whether to continue as decision support, remain paper-only, or begin a separate P7 feasibility review.

**Deliverables**
- Consolidated evidence report
- Security and risk recommendation
- Explicit go/no-go owner decision

**Gate**
- No automatic broker integration is implied by good performance
- A separate approval is mandatory for P7

**Status:** Planned

## P7 — Optional future broker integration

**Status:** Deferred / optional
**Purpose:** Provide a guarded path only if paper sustainability and manual decision-support evidence justify a separate integration programme.
**Phase exit outcome:** No P7 work begins without explicit approval. Any future live capability progresses from read-only shadow mode to manual approval and only then, potentially, to tightly capped autonomous execution.

### P7.0 — Formal go/no-go and scope approval

Open a separate programme rather than silently extending paper mode.

**Deliverables**
- Business owner approval
- Security, regulatory and account-control review
- Defined broker, account type, markets and instruments
- Updated risk appetite

**Gate**
- Written approval exists
- P5 and P6 evidence is accepted
- Live budget and loss limits are explicitly approved

**Status:** Deferred / optional

### P7.1 — Security and broker integration foundation

Build the broker boundary with least privilege and complete auditability.

**Deliverables**
- Secret vault integration
- Read-only and trading permission separation
- Broker sandbox/paper environment
- Rate limits, retries and audit logs

**Gate**
- No credentials in repository or logs
- Permission scope is independently reviewed
- Failure defaults to no new order

**Status:** Deferred / optional

### P7.2 — Read-only shadow reconciliation

Observe the broker without submitting orders.

**Deliverables**
- Cash, positions and orders read-only
- Internal-versus-broker reconciliation
- Latency and data mismatch evidence

**Gate**
- No order endpoint is enabled
- Differences are blocking and explained

**Status:** Deferred / optional

### P7.3 — Manual-approval order mode

Prepare orders in the platform but require explicit user approval.

**Deliverables**
- Approval interface
- Order preview with fees and risk
- Expiry and cancellation
- Post-fill reconciliation

**Gate**
- No order without fresh explicit approval
- Maximum order remains capped by approved policy
- Duplicate protection passes

**Status:** Deferred / optional

### P7.4 — Guarded micro-live automation

Consider limited autonomous execution only after manual-approval evidence.

**Deliverables**
- Small fixed notional
- Maximum open exposure
- Daily and weekly loss stops
- Allowed instruments only
- Immediate kill switch

**Gate**
- Live execution is disabled by default
- Independent release gate is READY
- Rollback and incident drills pass

**Status:** Deferred / optional

### P7.5 — Production operations and oversight

Operate with full monitoring and governance.

**Deliverables**
- 24/7 health monitoring
- Broker outage procedures
- Daily reconciliation
- Incident management
- Periodic access review

**Gate**
- No unresolved cash or position difference
- All live actions are fully auditable

**Status:** Deferred / optional

### P7.6 — P7 release and continuation gate

Require periodic reapproval of live capability.

**Deliverables**
- Live regression and operational evidence
- Security review
- Risk and performance review
- Renewal or shutdown decision

**Gate**
- Any material breach disables live mode
- Continued operation requires explicit renewal

**Status:** Deferred / optional

## 8. Autonomous daily operating cycle

1. Service startup, configuration validation and health check.
2. Paper-account and ledger reconciliation.
3. Exchange calendar, timezone and market-data freshness validation.
4. Pre-session universe refresh and watchlist ranking.
5. Periodic swing scans during configured market windows.
6. Medium-term scan after the relevant market close.
7. Cost, liquidity, risk and portfolio gates.
8. Paper-order creation, simulated fill and position monitoring.
9. Email and Telegram event fan-out.
10. Post-close reconciliation, daily report and weekly review.

The system is expected to do nothing when no candidate passes. Inactivity is a valid risk-controlled outcome.

## 9. Notification design

| Event | Telegram | Email |
|---|---|---|
| Startup / heartbeat | Concise status | Optional detailed health summary |
| Watchlist change | High-value changes only | Ranked watchlist summary |
| Actionable opportunity | Concise action ticket | Full rationale, costs, risk and evidence |
| Paper order / fill / exit | Yes | Yes |
| Daily report | Optional summary | Full report |
| Weekly report | Optional headline | Full performance and reliability report |
| Critical failure / reconciliation difference | Immediate | Detailed context and recovery steps |

Every notification is persisted, channel-specific, deduplicated, retryable and linked to its source event. A channel being configured is not considered proof of delivery; application-level sent evidence is required.

## 10. Cost and manual IBKR policy

- IBKR is not connected to the application through P6.
- Fees, minimum commissions, fractional-share rules, FX and market-data details gathered from the user’s account are stored as a versioned reference profile.
- Each order reports estimated entry cost, exit cost, FX cost, spread/slippage allowance, planned loss and net reward-to-risk.
- A paper trade is rejected when realistic costs make it uneconomic.
- Manual IBKR orders remain the user’s independent decision. The app records them only when the user chooses to journal them.
- The official strategy scorecard always uses the autonomous paper portfolio, preventing selective manual execution from biasing the bot’s evaluation.

## 11. Cross-phase quality, safety and governance requirements

- Determinism: identical data and configuration produce the same decision.
- Provenance: every displayed metric and recommendation links to persisted records and configuration versions.
- No future-data leakage in analysis, backtesting or simulated execution.
- Idempotency for scans, jobs, orders, dispatch and recovery.
- Safe failure: stale data, reconciliation differences, missing configuration or critical provider failures block new orders.
- Secrets remain outside the repository and are never printed.
- No live-order code path is enabled through P6.
- All phase gates include automated regression and manual operational acceptance.
- Results are reported after fees, spread, slippage and FX.
- Strategy promotion requires independent evidence by horizon and configuration version.

## 12. Key risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Small EUR 100 orders are consumed by fees | False profitability | Versioned IBKR cost profile and net reward-to-risk gate |
| Codespace stops or terminal closes | Missed scans and alerts | Always-on deployment in P4.8 |
| Delayed or stale market data | Invalid decisions | Freshness gate and circuit breaker |
| Overfitting | Unsustainable results | Walk-forward validation, frozen cohorts and independent strategy gates |
| Too many alerts | User ignores important events | Severity routing, deduplication and actionability metrics |
| Duplicate jobs or orders | Incorrect portfolio | Idempotency keys, replay tests and reconciliation |
| Manual IBKR copying biases evaluation | Misleading results | Independent paper source of truth and separate journal |
| Premature broker integration | Financial and security risk | P7 deferred with separate written approval |

## 13. Immediate execution plan

### Close P3
1. Run one genuine market cycle after a valid session.
2. Dispatch application notifications through the configured channels.
3. Confirm persisted jobs, scans, execution runs and sent notifications.
4. Run the P3 release gate using the 389-test Automated tests #45 evidence.
5. Do not tag the phase as ready unless the gate returns READY.

### Start P4
1. Commit this P0-P7 roadmap as the governing roadmap.
2. Preserve the existing USD 10,000 account as test history.
3. Create the new EUR 2,000 operational paper account.
4. Implement EUR 100 fixed-notional-with-risk-cap sizing.
5. Capture the user’s IBKR fee and account rules as a reference profile.
6. Separate swing and medium-term strategy configuration and evidence.
7. Restore and application-test Telegram alongside email.
8. Build the autonomous scheduler, deployment, recovery and kill switch.
9. Pass the P4 release gate before beginning the P5 pilot.

## 14. Decision register

| Decision | Approved position |
|---|---|
| Roadmap scope | Always maintained end to end from P0 through P7 |
| Product identity | Smart autonomous investment bot, not only an analysis dashboard |
| Strategy focus | Swing and medium-term; no day trading |
| Paper budget | EUR 2,000 |
| Order value | EUR 100 target and hard ceiling |
| Real broker connection | None through P6 |
| IBKR role | Fee reference and optional manual execution |
| Notifications | Email and Telegram |
| Proof standard | Operational reliability, actionability and cost-adjusted sustainability |
| Live integration | Optional deferred P7 with separate approval |

## 15. Definition of programme success

The programme succeeds when the bot reliably produces a small number of clear, timely and economically sensible swing and medium-term actions; manages the EUR 2,000 paper portfolio without integrity failures; reports all decisions and costs transparently; and demonstrates sustainable cost-adjusted value over a meaningful unattended evidence period.

Profit alone is insufficient. The system must also prove deterministic reasoning, low operational failure, controlled drawdown, useful notifications, realistic execution assumptions and safe inactivity when no opportunity qualifies.
