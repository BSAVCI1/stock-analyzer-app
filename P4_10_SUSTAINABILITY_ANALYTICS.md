# P4.10 sustainability analytics

## Evidence contract

The analytics are derived from persisted paper records. They do not infer live
broker performance and do not convert activity into an investment-quality
claim. Empty or small cohorts remain valid but insufficient evidence.

## P4.10.1 cost-adjusted performance baseline

Closed-trade performance is calculated in the portfolio currency already
persisted on each trade. Entry and exit FX rates are therefore reflected in
the recorded P&L. Every headline displays these values together:

- gross P&L before recorded transaction costs;
- entry and exit commissions in `fees`;
- recorded execution-price impact in `slippage`;
- total transaction costs as fees plus slippage; and
- net P&L after those costs.

Expectancy is net P&L divided by closed-trade count. Profit factor is the sum
of positive net trades divided by the absolute sum of negative net trades. It
is `null` when there are neither net gains nor losses and infinite when gains
exist without a loss. Cost drag is total transaction costs divided by the
absolute gross P&L and is unavailable when gross P&L is zero.

The persisted equity curve remains the source for portfolio drawdown; trade
P&L is not substituted for missing equity snapshots.

## Version-safe cohorts

Every closed trade is grouped by a combined cohort:

```text
strategy_horizon|strategy_version
```

Examples are `SWING|p4.3-swing-v1` and
`MEDIUM_TERM|p4.3-medium-v1`. Missing provenance is shown explicitly as
`UNKNOWN`; it is never silently assigned to another cohort. Each cohort shows
gross P&L, fees, slippage, total costs, net P&L, expectancy and profit factor,
with the contributing persisted trade IDs attached as provenance.

This baseline intentionally does not yet supply watchlist conversion, alert
usefulness or manual-copy journal metrics. Those remain
separate P4.10 slices so their data contracts can be tested without weakening
the cost and version boundaries above.

## P4.10.2 benchmark and nominal-cash comparison

Benchmark observations are immutable persisted evidence. Each record stores
the symbol, timestamp, quote-currency close, quote-to-portfolio FX rate,
portfolio-currency price and source. Repeating the same observation is
idempotent; conflicting values for the same account, symbol and timestamp are
rejected instead of overwriting history.

Record an operator-verified observation when an automated source has not yet
been selected:

```bash
python -m src.jobs.cli benchmark record VWCE.DE \
  --captured-at 2026-08-19T20:00:00+00:00 \
  --quote-currency EUR \
  --close-price 100.00 \
  --fx-rate 1 \
  --source "operator-verified-close"
```

List persisted evidence:

```bash
python -m src.jobs.cli benchmark list VWCE.DE
```

The dashboard compares each symbol independently. A comparison requires at
least two observations at different timestamps and two equity snapshots
aligned at or before those endpoints. If this evidence is missing, the result
is explicitly marked insufficient.

The benchmark result is portfolio-currency price return, so the stored FX rate
is included. It is not yet a dividend-adjusted total return. The cash baseline
is nominal 0%; interest and inflation are not assumed. The report displays:

- account return over the aligned equity window;
- benchmark portfolio-currency price return;
- nominal cash return;
- excess return versus the benchmark; and
- excess return versus nominal cash.

Multiple benchmark symbols may be recorded and are never blended into one
headline. Benchmark selection policy and automated observation capture remain
future configuration work.

## P4.10.3 portfolio concentration

Concentration uses immutable position-valuation evidence rather than entry
cost or an inferred price. Each observation records the open position,
quantity, close price, quote-to-portfolio FX rate, portfolio-currency market
value, timestamp and source. Repeating identical evidence is idempotent;
conflicting evidence is rejected.

```bash
python -m src.jobs.cli position-valuation record POS-123 \
  --captured-at 2026-08-24T20:00:00+00:00 \
  --quote-currency USD --close-price 230.00 --fx-rate 0.86 \
  --source "operator-verified-close"
```

A result is reported only when the latest timestamp covers every currently
open position and an equity snapshot exists at or before that timestamp.
Positions in the same symbol are combined. The read-only dashboard displays
each symbol's share of invested market value and total equity, the largest
symbol weight, top-three weight, invested share of equity and the
Herfindahl-Hirschman Index (HHI). No diversification judgment or policy limit
is inferred; the slice provides reproducible evidence only.

## P4.10.4 watchlist conversion and stale signals

Watchlist conversion is measured as a persisted decision journey, not as the
number of rows in one scan. For each symbol and combined strategy cohort
(`strategy_horizon|strategy_version`), the metric opens an episode when the
scanner first records `WATCH` or `PREPARE`. A later `ACTIONABLE` result in the
same cohort converts the episode. `REJECT` or `STALE` closes it as abandoned;
an unresolved episode remains open. Repeated watch results do not inflate the
denominator, and strategy versions are never blended.

The stale-signal rate has a separate contract. A signal is mature when it has
expired by the report timestamp or already produced an order. A mature signal
is stale when no persisted order references it. Signals that have not yet
expired are excluded from both numerator and denominator. The dashboard shows
the conversion funnel by cohort and the aggregate stale-signal rate, with the
supporting scan-result, signal and order IDs retained as provenance.

These metrics describe actionability only. They do not claim that an order was
profitable or that a rejected watchlist idea was incorrect.

## P4.10.5 alert usefulness and manual-copy journal

Operator feedback is immutable evidence attached to one successfully sent
notification. Each assessment records whether the alert was useful, what the
operator did (`COPIED_AS_IS`, `COPIED_MODIFIED`, `DISMISSED` or `NO_ACTION`),
the named operator, rationale, timestamp and—when copied—the paper-broker
reference. Identical retries are idempotent and conflicting reassessments are
rejected rather than overwriting history.

```bash
python -m src.jobs.cli alert-feedback record NOT-123 \
  --usefulness USEFUL --manual-action COPIED_MODIFIED \
  --operator "Salih AVCI" \
  --rationale "Copied after reducing quantity." \
  --broker-reference "IBKR-PAPER-123" \
  --recorded-at 2026-08-24T20:15:00+00:00
```

The dashboard reports three separate measures: assessment coverage over sent
notifications, usefulness over assessed alerts, and manual-copy rate over
assessed alerts. Unassessed notifications are not silently treated as
not-useful. A copy decision is evidence of operator action only; it is not
treated as broker execution, strategy success or investment performance.

## P4.10.6 operational reliability

Operational reliability is calculated across the complete persisted account
history rather than the dashboard's recent-row display limits. The report
keeps each denominator visible and separate:

- successful terminal jobs divided by all terminal jobs;
- jobs started within five minutes of `scheduled_for` divided by all jobs;
- completed market cycles carrying both scan and execution IDs divided by all
  completed market cycles; and
- sent notifications divided by sent-plus-failed terminal notifications.

Average and maximum non-negative start delay are also retained, together with
the count of error/critical system events and the exact observation window.
Pending jobs and notifications remain visible in source records but do not
enter terminal-success denominators. Empty evidence returns `null`, never an
invented 100% success rate.

These measures are operational evidence, not performance evidence. No
composite score is created that could hide a weak component behind stronger
ones. With this slice, the P4.10 actionability and sustainability analytics
deliverables are complete.
