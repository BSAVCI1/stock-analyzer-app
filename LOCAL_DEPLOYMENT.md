# Local device validation

This profile validates the portable deployment runtime without placing paper
or live orders, sending notifications, or requiring broker credentials. It
writes only to `deployment_validation.db` in a persistent Docker volume.

## Choose a container runtime

### Supported macOS or Windows

Install and start Docker Desktop. Use `docker compose` in the commands below.

### Intel Mac on macOS Monterey

Current Docker Desktop releases do not support Monterey. The verified local
alternative is Colima with the Docker command-line tools.

Install Homebrew packages:

```bash
brew install colima
brew install docker
brew install docker-compose
```

Colima uses QEMU on an Intel Monterey Mac. If Homebrew cannot build the current
QEMU release with Monterey's Apple compiler, install the official Monterey
package from MacPorts and then install QEMU:

```bash
sudo /opt/local/bin/port install qemu
```

Make MacPorts available in future Terminal sessions:

```bash
printf '\nexport PATH="/opt/local/bin:/opt/local/sbin:$PATH"\n' >> ~/.bash_profile
source ~/.bash_profile
```

Start Colima with conservative settings suitable for an 8 GB Intel Mac:

```bash
colima start --cpu 2 --memory 3 --disk 20
```

The commands below use the standalone `docker-compose` command verified with
Colima. Docker Desktop users may replace `docker-compose` with
`docker compose`.

## Start

```bash
docker-compose -f compose.yaml -f compose.local.yaml up --build -d
```

## Check status

```bash
docker-compose -f compose.yaml -f compose.local.yaml ps
curl http://127.0.0.1:8080/health/live
curl http://127.0.0.1:8080/health/ready
curl http://127.0.0.1:8080/health/worker
```

Expected results:

- the container status is `healthy`;
- liveness reports the process as running;
- readiness reports the database as available;
- the worker reports a current heartbeat.

The local validation cycle never invokes the trading engine.

## Confirm persistent validation cycles

```bash
docker-compose -f compose.yaml -f compose.local.yaml exec -T app python -c "import sqlite3; c=sqlite3.connect('/app/data/deployment_validation.db'); print(c.execute('SELECT COUNT(*) FROM deployment_validation_cycles').fetchone()[0]); c.close()"
```

The result should be greater than zero and should continue increasing while
the profile runs.

## Restart test

```bash
docker-compose -f compose.yaml -f compose.local.yaml restart
docker-compose -f compose.yaml -f compose.local.yaml ps
```

After restart, repeat the health checks and the cycle-count command. The
Docker volume and validation database must remain in place.

## View logs

```bash
docker-compose -f compose.yaml -f compose.local.yaml logs --tail=100 app
```

## Stop without deleting data

```bash
docker-compose -f compose.yaml -f compose.local.yaml down
```

Do not add `--volumes` unless you intentionally want to delete the local
validation data. Colima users can then release its resources:

```bash
colima stop
```

A real paper-cycle adapter and notification secrets are enabled only after
local infrastructure validation is accepted.
