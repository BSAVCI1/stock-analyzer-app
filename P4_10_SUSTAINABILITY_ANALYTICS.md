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

This baseline intentionally does not yet supply concentration, watchlist
conversion, alert usefulness or manual-copy journal metrics. Those remain
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
