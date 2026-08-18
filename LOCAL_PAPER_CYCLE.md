# Local internal-paper cycle

This profile connects the managed deployment worker to the project's real
internal paper-trading orchestration. It remains isolated from IBKR and live
trading.

## Safety boundaries

- The adapter is disabled unless `BSAVCI_PAPER_CYCLE_ENABLED=true`.
- Broker and live-trading flags are explicitly rejected.
- No Telegram or email credentials are required.
- The dedicated local account and database remain in the Docker volume.
- The local account uses simulated money only.

Use `docker-compose` with Colima. Docker Desktop users may replace it with
`docker compose`.

## Build

```bash
docker-compose -f compose.yaml -f compose.paper-local.yaml build
```

## Create the isolated local paper account

```bash
docker-compose -f compose.yaml -f compose.paper-local.yaml run --rm --no-deps app python -m src.deployment.bootstrap
```

The command should print `ACC-LOCAL-DEVICE`. Repeating it is safe and does
not reset the simulated balance.

## Start

```bash
docker-compose -f compose.yaml -f compose.paper-local.yaml up -d
```

## Check

```bash
docker-compose -f compose.yaml -f compose.paper-local.yaml ps
curl http://127.0.0.1:8080/health/live
curl http://127.0.0.1:8080/health/ready
curl http://127.0.0.1:8080/health/worker
docker-compose -f compose.yaml -f compose.paper-local.yaml exec -T app python -m src.jobs.cli status
```

The worker checks every five minutes. The exchange-aware orchestration policy
runs jobs only when they are due and safely records duplicates and restarts.

## Stop without deleting simulated data

```bash
docker-compose -f compose.yaml -f compose.paper-local.yaml down
colima stop
```

Do not add `--volumes` unless the dedicated local paper database should be
deleted intentionally.
