# P4.8 local deployment acceptance

## Scope

The portable deployment baseline was validated on an Intel Mac running macOS
Monterey using Colima, QEMU, Docker CLI and Docker Compose.

The test used a dedicated Docker volume and the isolated
`ACC-LOCAL-DEVICE` simulated account. It did not use the operational
database, IBKR connectivity, broker credentials, live orders, Telegram or
email delivery.

## Harmless infrastructure profile

Accepted evidence:

- container image built successfully;
- service reached `healthy`;
- liveness reported the process as running;
- readiness reported the database as available;
- worker heartbeat was current and idle;
- 119 harmless validation cycles were persisted;
- restart returned the service to `healthy`;
- validation data survived restart;
- shutdown preserved the Docker volume.

## Managed internal-paper profile

Accepted evidence:

- dedicated simulated account bootstrapped idempotently;
- account used EUR base currency and EUR 100,000 simulated starting cash;
- managed worker reached `healthy`;
- worker heartbeat was current and idle;
- one XNYS market cycle for 2026-08-17 completed successfully;
- scan and internal-paper execution both completed without errors;
- zero candidates produced zero forced trades;
- the repeated cycle returned `duplicate: true` with the same job, scan and
  execution identifiers;
- restart returned the service to `healthy`;
- the completed job and reconciled portfolio survived restart;
- notifications remained unconfigured and empty;
- broker and live-trading flags remained disabled;
- controlled shutdown preserved the simulated database.

## Acceptance decision

P4.8's local always-on deployment baseline is accepted. The application can
run on the user's device without an interactive Codespaces terminal and can
later be moved to an external container platform using the same portable
runtime contracts.

External provisioning is deferred unless the user selects a provider. Any
future external deployment requires separate restart, backup/restore,
rollback and secret-mount evidence on that provider.
