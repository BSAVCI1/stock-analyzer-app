# P2 Release Gate

## Scope

The P2 release gate covers the deterministic chain:

1. signal
2. paper order
3. next-session simulated fill
4. position
5. exit and closed trade
6. costs, slippage and settlement
7. performance and benchmark comparison
8. walk-forward validation
9. strategy acceptance
10. alert-scheduling eligibility

Eligibility does not schedule alerts or perform any external action.

## Regression requirement

A strategy is eligible only when:

- the complete P0–P2 regression suite passes
- P0, P1 and P2 are all included in regression evidence
- the strategy acceptance report is accepted
- the committed signal-threshold manifest is approved
- the documented limitations remain visible

The automated GitHub workflow runs the complete pytest suite on pushes and
pull requests targeting `main`.

## Approved signal thresholds

The approved deterministic defaults are stored in:

`config/approved_signal_thresholds.json`

The manifest contains:

- market-regime thresholds
- trend-pullback thresholds
- breakout thresholds
- mean-reversion thresholds
- recommendation thresholds
- score weights
- risk-management thresholds

Changing any approved value requires:

1. regenerating or deliberately editing the manifest
2. rerunning the complete P0–P2 regression suite
3. repeating walk-forward validation
4. rebuilding the strategy acceptance report
5. obtaining a new P2 release-gate decision

## Limitations

The following functionality remains deferred:

- OpenAI or another AI-platform connection
- broker connection and live order execution
- background alerts and scheduled watchlist scans

The release gate produces eligibility only. It does not create an alert,
schedule a scan, submit an order or connect to a broker.

## Eligibility rule

Only a strategy with:

- an accepted deterministic strategy acceptance report
- passing P0–P2 regression evidence
- coverage for P0, P1 and P2
- an approved threshold manifest

can receive `alert_scheduling_eligible = true`.

Rejected or incomplete strategies remain ineligible.
