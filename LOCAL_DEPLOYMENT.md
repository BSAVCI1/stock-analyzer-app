# Local PC validation

This profile tests the deployment runtime without placing paper or live
orders and without sending notifications.

## Prerequisite

Install and start Docker Desktop.

## Start

```bash
docker compose -f compose.yaml -f compose.local.yaml up --build -d
```

## Check status

```bash
docker compose -f compose.yaml -f compose.local.yaml ps
curl http://127.0.0.1:8080/health/live
curl http://127.0.0.1:8080/health/ready
curl http://127.0.0.1:8080/health/worker
```

The worker endpoint should report a current heartbeat. The local validation
cycle writes only to `deployment_validation.db` in the persistent Docker
volume. It does not invoke the trading engine.

## Restart test

```bash
docker compose -f compose.yaml -f compose.local.yaml restart
```

After restart, repeat the status checks. The Docker volume and validation
database remain in place.

## Stop without deleting data

```bash
docker compose -f compose.yaml -f compose.local.yaml down
```

Do not add `--volumes` unless you intentionally want to delete the local
validation data.

## View logs

```bash
docker compose -f compose.yaml -f compose.local.yaml logs --tail=100 app
```

A real paper-cycle adapter and notification secrets are enabled only after
local infrastructure validation is accepted.
