# P4.9 recovery and kill-switch operations

## Safety contract

The global kill switch is stored in the paper database for each account. It
survives application and container restarts. While active, execution cycles:

- block and cancel new paper entries;
- continue monitoring open positions and processing protective exits;
- persist the block reason in the execution run; and
- record operator activation and deactivation in the system-event log.

The commands below affect only the account and database selected by the
runtime environment or the explicit `--database` and `--account-id` options.

## Inspect the switch

```bash
python -m src.jobs.cli kill-switch status
```

The safe field to check is `new_orders_allowed`. It is `false` whenever the
switch is active. The normal `status` command also includes the same state.

## Activate immediately

```bash
python -m src.jobs.cli kill-switch activate \
  --reason "Describe the observed risk or incident" \
  --operator "your-name"
```

Repeat the status command and confirm:

```json
{
  "active": true,
  "new_orders_allowed": false
}
```

Repeating activation is idempotent and does not replace the original reason
or add a duplicate audit event.

## Investigate and recover

1. Keep the switch active.
2. Check application health and the latest job status.
3. Confirm the paper account reconciles and inspect pending orders, open
   positions, execution errors and system events.
4. Correct the underlying issue and run the relevant tests.
5. Restart the service if needed, then confirm the switch remains active.
6. Deactivate only after a named operator has reviewed the evidence.

## Deactivate after review

```bash
python -m src.jobs.cli kill-switch deactivate \
  --reason "Issue corrected; reconciliation and checks passed" \
  --operator "your-name"
```

Confirm `active` is `false` and `new_orders_allowed` is `true`. Deactivation
does not replay old jobs automatically; normal scheduler idempotency remains
in force.

## Docker Compose example

For the managed local paper profile, prefix each command with:

```bash
docker-compose -f compose.yaml -f compose.paper-local.yaml exec -T app
```

For example:

```bash
docker-compose -f compose.yaml -f compose.paper-local.yaml exec -T app \
  python -m src.jobs.cli kill-switch status
```

## Per-strategy pause

A strategy pause is narrower than the global kill switch. It cancels pending
entries and blocks new entries only for the named strategy. Other strategies
continue normally, and protective monitoring and exits remain active for
positions that were opened by the paused strategy.

Inspect all persisted strategy controls:

```bash
python -m src.jobs.cli strategy-pause status
```

Pause one strategy:

```bash
python -m src.jobs.cli strategy-pause activate trend_pullback \
  --reason "Strategy evidence requires review" \
  --operator "your-name"
```

Confirm that `active` is `true` and `new_entries_allowed` is `false`. Repeating
the activation is idempotent and does not add another audit event.

Resume only after review:

```bash
python -m src.jobs.cli strategy-pause deactivate trend_pullback \
  --reason "Review completed and strategy approved" \
  --operator "your-name"
```

The same Docker Compose prefix documented above can be placed before each
strategy-pause command on the managed local paper profile.

## Reconciliation circuit breaker

The reconciliation breaker is automatic and account-wide. Every execution
run compares the account's stored cash balance with the complete ledger before
processing any market action. A mismatch fails the run, persists a
`RECONCILIATION` breaker, records the balances and difference in the audit
metadata, and blocks new entries across process and container restarts.

Inspect the persisted state with the common breaker command:

```bash
python -m src.jobs.cli circuit-breaker status
```

Do not clear the breaker manually or edit the paper database. Investigate the
underlying ledger or account-writing fault and restore from verified evidence
when required. The breaker recovers automatically only when a later execution
run positively verifies that stored cash and ledger cash match exactly. A
repeated mismatch is idempotent and does not create duplicate trip events.

## Daily and weekly realised-loss pauses

The account has two automatic realised-loss controls. The daily limit keeps
the existing configurable default of 3% of starting balance. The weekly limit
has a configurable default of 5%. For the EUR 2,000 operational paper account,
these defaults are EUR 60 per UTC day and EUR 100 per UTC week.

When closed-trade net P&L reaches either limit, the corresponding
`LOSS_LIMIT_DAILY` or `LOSS_LIMIT_WEEKLY` breaker blocks new entries and
cancels pending entries. Protective monitoring and exits remain active. The
pause is locked for the rest of that UTC day or Monday-to-Sunday UTC week; a
later improvement in realised P&L does not reopen entries inside the same
period.

The pause cannot be cleared manually. On the first execution run in a new
period, it recovers automatically only when the new period is within its loss
limit. Trip and recovery transitions are persisted and auditable through:

```bash
python -m src.jobs.cli circuit-breaker status
```

## Stale-data circuit breaker

The stale-data breaker is automatic and account-wide. Before any pending or
new entry is processed, the execution engine checks every relevant entry-data
timestamp against the configured freshness limit. If one critical input is
missing or stale, the breaker:

- blocks all new entries for the account;
- cancels pending entries before they can fill;
- persists across process and container restarts;
- records one auditable trip event; and
- remains active when no fresh evidence is available.

Inspect the state:

```bash
python -m src.jobs.cli circuit-breaker status
```

The breaker cannot be cleared manually. It recovers automatically only when a
later execution cycle positively verifies at least one relevant entry input
and all checked entry inputs are fresh. Recovery is persisted and recorded as
a separate system event. Protective position monitoring is not disabled by
the entry circuit breaker.

For the managed local paper profile:

```bash
docker-compose -f compose.yaml -f compose.paper-local.yaml exec -T app \
  python -m src.jobs.cli circuit-breaker status
```
