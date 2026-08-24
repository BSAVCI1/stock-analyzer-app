# P4 release gate

## Purpose

The P4 gate controls entry into the unattended paper-validation pilot. It only
evaluates evidence already produced by tests and paper operations. It cannot
run a market cycle, contact a broker or enable live execution.

## Contract

The version-1 JSON manifest requires:

- a unique release ID, timestamp and paper account ID;
- passing regression evidence covering P0, P1, P2, P3 and P4;
- one evidence-backed result for every required P4 check;
- `execution_mode` equal to `PAPER`;
- `live_execution_enabled` equal to `false`; and
- zero unresolved operational failures.

Required checks are:

1. `paper_only_invariants`
2. `eur_portfolio_policy`
3. `scheduler_deployment`
4. `email_delivery`
5. `telegram_delivery`
6. `recovery_controls`
7. `kill_switch`
8. `strategy_horizon_acceptance`

A passing check must carry at least one traceable evidence ID. `FAIL` and
`NOT_OBSERVED` are always blocking and must explain why. Missing, duplicate or
unknown checks make the manifest invalid rather than silently weakening the
gate.

## Evaluation

Start from the deliberately blocked example and replace each placeholder with
genuine evidence:

```bash
python -m src.jobs.cli p4-release-status \
  --manifest config/p4_release_evidence.example.json
```

Exit code `0` means `READY`, `1` means valid but `BLOCKED`, and `2` means the
manifest itself is invalid. Any non-paper execution mode, enabled live
capability or unresolved operational failure blocks release independently of
the other checks.

This contract is P4.11.1. Later P4.11 slices will produce and assemble the
specific evidence; the example is not an approval and must not be promoted by
changing labels without the underlying records.
