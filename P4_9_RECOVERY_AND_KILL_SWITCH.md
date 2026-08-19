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
