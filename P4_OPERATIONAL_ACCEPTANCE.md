# P4 operational acceptance rehearsal

This procedure is the final P4 rehearsal on the target Mac. It is paper-only,
does not authorize broker connectivity, and must not be satisfied with example
or invented identifiers. P5 may start only when the final command reports
`"safe_to_start_p5": true`.

## 1. Update and start the local paper runtime

```bash
cd ~/Documents/stock-analyzer-app
git pull --ff-only
colima start --cpu 2 --memory 3 --disk 20
docker-compose -f compose.yaml -f compose.paper-local.yaml -f compose.p4-acceptance.yaml build
docker-compose -f compose.yaml -f compose.paper-local.yaml -f compose.p4-acceptance.yaml up -d
docker-compose -f compose.yaml -f compose.paper-local.yaml -f compose.p4-acceptance.yaml ps
```

Require the app to be `healthy`. Then verify all three runtime checks:

```bash
curl http://127.0.0.1:8080/health/live
curl http://127.0.0.1:8080/health/ready
curl http://127.0.0.1:8080/health/worker
```

## 2. Collect genuine evidence

Confirm `status` reports account `ACC-P4-EUR-2000`, EUR base currency and a
starting balance of exactly EUR 2,000. The bootstrap fails closed if an
existing account with that ID does not match the configured currency or
starting balance. The earlier `ACC-LOCAL-DEVICE` account remains preserved as
deployment-test history and must not be used for P4 release evidence.

Create private working copies of the seven example files. Do not commit
credentials, message destinations, database contents, or other secrets.
Replace example identifiers only with records observed from this release and
the same paper account.

- `release`: release ID, paper account, PAPER mode, live execution disabled,
  and the current unresolved-failure count.
- `regression`: the successful GitHub Actions run URL, run ID, commit, covered
  phases, test count, and completion time.
- `scheduler`: healthy runtime, completed managed cycle, restart proof, worker
  heartbeat, and persistent-volume proof.
- `notifications`: one genuine application-level SENT record for email and one
  for Telegram, with their source references and retry/deduplication state.
- `recovery`: completed restart/replay, circuit-breaker, provider-outage,
  incident-closure, and global kill-switch drills in paper mode.
- `horizons`: independent accepted swing and medium-term reports.
- `policy`: use the approved repository product policy unchanged.

Operational drills must use the documented paper controls. Stop and investigate
any unexpected order, live capability, failed notification, unhealthy service,
or unresolved incident; do not edit evidence to hide it.

Use `notification-probe` once per configured channel to create a genuine,
persisted delivery record without waiting for a trading opportunity. The probe
requires a named operator and reason, uses the normal sender and records the
provider message ID. Credentials remain mounted from private files.

## 3. Assemble and rehearse the gate

Use paths outside the repository for private evidence when practical:

```bash
mkdir -p ~/Documents/stock-analyzer-p4-evidence
```

After populating those JSON files, assemble the manifest:

```bash
docker-compose -f compose.yaml -f compose.paper-local.yaml -f compose.p4-acceptance.yaml run --rm --no-deps \
  -v "$HOME/Documents/stock-analyzer-p4-evidence:/evidence:ro" app \
  python -m src.jobs.cli p4-assemble-evidence \
  --release /evidence/release.json \
  --policy /app/config/product_policy_v1.json \
  --regression /evidence/regression.json \
  --scheduler /evidence/scheduler.json \
  --notifications /evidence/notifications.json \
  --recovery /evidence/recovery.json \
  --horizons /evidence/horizons.json \
  > ~/Documents/stock-analyzer-p4-evidence/assembled-release.json
```

The command mounts the private evidence directory read-only and saves only the
assembled result on the Mac. Rehearse that result in the same container image:

```bash
docker-compose -f compose.yaml -f compose.paper-local.yaml -f compose.p4-acceptance.yaml run --rm --no-deps \
  -v "$HOME/Documents/stock-analyzer-p4-evidence:/evidence:ro" app \
  python -m src.jobs.cli p4-release-rehearsal \
  --manifest /evidence/assembled-release.json
```

Exit code `1` and `BLOCKED` are expected until every genuine requirement is
present. Follow `next_actions` and repeat. Exit code `0`, `READY`, and
`safe_to_start_p5: true` together authorize the separate P5 launch decision.

## 4. Stop safely when finished

```bash
docker-compose -f compose.yaml -f compose.paper-local.yaml -f compose.p4-acceptance.yaml down
colima stop
```
