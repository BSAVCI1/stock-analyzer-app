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

## Static policy evidence (P4.11.2)

Produce the first two gate checks directly from the versioned product policy:

```bash
python -m src.jobs.cli p4-policy-evidence \
  --policy config/product_policy_v1.json
```

The command verifies paper-only, disabled live and broker connectivity,
deny-by-default behaviour, prohibited instrument classes, the EUR 2,000
portfolio, EUR 100 target/hard ceiling, and approved risk/exposure limits. A
pass carries a deterministic SHA-256 evidence ID over the canonical policy.
Any missing or changed invariant is emitted as explicit `FAIL` evidence and
returns exit code `1`.

## Scheduler and deployment evidence (P4.11.3)

Record genuine health, completed-cycle, restart and persistent-storage IDs in
a copy of the deliberately blocked example, then evaluate it:

```bash
python -m src.jobs.cli p4-scheduler-evidence \
  --evidence config/p4_scheduler_evidence.example.json
```

Both the validated Mac local-device runtime and a future external always-on
runtime are supported. The check requires paper mode, an enabled managed
scheduler, healthy container/liveness/readiness/worker observations, verified
restart recovery, verified persistent storage and at least one completed paper
cycle. Labels without evidence IDs cannot pass. The repository example is
intentionally blocked and is not release evidence.

## Notification delivery evidence (P4.11.4)

Email and Telegram are evaluated independently from persisted application-level
sent records:

```bash
python -m src.jobs.cli p4-notification-evidence \
  --evidence config/p4_notification_evidence.example.json
```

Each channel requires persisted, channel-specific, deduplicated and retryable
delivery records linked to a source reference. At least one `SENT` record with
a timezone-aware delivery time and attempt count is mandatory; configured
credentials or channels alone are not proof. Pending or failed delivery blocks
that channel. The repository example deliberately contains no sent records and
therefore returns a blocking result.

## Recovery and kill-switch evidence (P4.11.5)

Recovery controls and the global kill switch are evaluated independently:

```bash
python -m src.jobs.cli p4-recovery-evidence \
  --evidence config/p4_recovery_evidence.example.json
```

Recovery evidence requires verified restart recovery, idempotent replay, stale
data and reconciliation breakers, loss-limit pause, provider-outage handling,
closed incidents and no unresolved critical incident. Kill-switch evidence
requires a named operator and reason, verified activation, blocked new orders,
pending-order policy, persisted audit history, verified recovery and an
`INACTIVE` final state. Each category requires traceable evidence IDs. The
repository example is deliberately blocked and cannot approve release.

## Strategy-horizon acceptance evidence (P4.11.6)

Swing and medium-term acceptance is evaluated independently:

```bash
python -m src.jobs.cli p4-horizon-evidence \
  --evidence config/p4_horizon_evidence.example.json
```

The check requires exactly one decision for each enabled horizon and the
approved `p4.3-swing-v1` and `p4.3-medium-term-v1` versions. Each must be
accepted with out-of-sample, walk-forward, transaction-cost, minimum-trade and
parameter-stability evidence, plus traceable acceptance, validation and
threshold IDs. Aggregate results, duplicate horizons, version mismatches,
rejections or missing evidence block release. The repository example is
deliberately blocked.
