# P3 Release Gate

## Scope

The P3 release gate covers the complete deterministic paper-trading
operating chain:

1. persistent paper account and ledger
2. automatic market scanning
3. automated internal paper execution
4. exchange-aware scheduled jobs
5. notification persistence and delivery status
6. read-only portfolio and reliability dashboard
7. provider-neutral broker-paper adapter
8. account, order and position reconciliation
9. complete P0-P3 regression evidence
10. operational release decision

The gate does not enable live trading, contact a live broker or submit a
live order.

## Regression requirement

A P3 release can be marked `READY` only when:

- the complete P0-P3 regression suite passes
- P0, P1, P2 and P3 are included in regression evidence
- the test count is recorded
- the workflow producing the evidence is identified

The CLI requires explicit `--regression-passed` attestation. Omitting
that option produces a blocked release result.

GitHub Actions runs the complete pytest suite on pushes and pull
requests targeting `main`.

## Operational reliability requirement

The persisted operational report must contain passing evidence for:

- account reconciliation
- broker-paper reconciliation
- market scans
- execution runs
- scheduled jobs
- notifications
- system events

A required category with no persisted observations is reported as
`NOT_OBSERVED`, not as a successful check.

Failed, pending or otherwise unresolved records block the release.

## Broker-paper requirement

The latest persisted broker-paper reconciliation must:

- exist
- have status `MATCHED`
- contain no mismatched records
- contain no records missing internally
- contain no records missing from the broker-paper account
- have zero unresolved differences

The release command reads only persisted reconciliation evidence. It
does not call a broker transport.

## Live-trading prohibition

The execution-adapter descriptor must report:

`live_trading_enabled = false`

A live-enabled descriptor blocks the release regardless of all other
evidence.

No live credentials or live-order execution path form part of P3.

## CLI command

The read-only report is generated with:

```bash
python -m src.jobs.cli \
  p3-release-status \
  --database data/paper_trading.db \
  --account-id <PAPER_ACCOUNT_ID> \
  --regression-passed \
  --test-count <TEST_COUNT> \
  --workflow "Automated tests <RUN>" \
  --at <TIMEZONE_AWARE_TIMESTAMP>
```

Exit codes:

- `0`: release status is `READY`
- `1`: evidence was evaluated successfully, but release status is
  `BLOCKED`
- `2`: invalid arguments, missing account, malformed evidence or another
  execution error

`<PAPER_ACCOUNT_ID>`, `<TEST_COUNT>`, `<RUN>` and the timestamp are
placeholders and must be replaced with actual values.

## Safety boundary

The P3 release command:

- performs no database write
- starts no scan or scheduled job
- sends no notification
- submits no paper or live order
- cancels no order
- contacts no broker
- changes no runtime configuration

It converts persisted evidence into a deterministic release decision.
