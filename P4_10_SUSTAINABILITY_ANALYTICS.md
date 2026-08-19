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

This baseline intentionally does not yet supply benchmark/cash comparison,
concentration, watchlist conversion, alert usefulness or manual-copy journal
metrics. Those remain separate P4.10 slices so their data contracts can be
tested without weakening the cost and version boundaries above.
