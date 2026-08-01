# Paper Automation Operations

The scheduled-job layer remains paper-only. It does not connect to a broker or submit real-money orders.

## Prerequisites

Use a persistent host or mounted volume for the SQLite database. Temporary CI runners must not be used as the production scheduler.

The configured account must already exist in the paper database.

Copy the environment template and load the values into the process environment:

    cp config/paper_automation.env.example .paper-automation.env
    set -a
    source .paper-automation.env
    set +a

Do not commit the populated `.paper-automation.env` file.

## Release eligibility

The runtime denies release eligibility when `PAPER_RELEASE_ELIGIBLE_STRATEGIES` is empty.

Only strategies that completed the project's release gate should be added:

    export PAPER_RELEASE_ELIGIBLE_STRATEGIES=trend_pullback

This operational allowlist does not replace the P2 evidence and release review.

## Commands

Run the post-close scan and paper-execution cycle:

    python -m src.jobs.cli market-cycle

Run with an explicit deterministic timestamp:

    python -m src.jobs.cli market-cycle \
      --at 2026-08-03T21:15:00+00:00

Generate the weekly report:

    python -m src.jobs.cli weekly-report

Dispatch queued notifications and retry failures:

    python -m src.jobs.cli dispatch
    python -m src.jobs.cli dispatch --retry-failed

Inspect account and reliability status:

    python -m src.jobs.cli status

The database and account can also be passed directly:

    python -m src.jobs.cli status \
      --database data/paper_trading.db \
      --account-id ACC-REPLACE-ME

## Scheduler example

The following example assumes the server uses UTC.

A 21:15 UTC weekday run is after the regular NYSE close during both US standard and daylight-saving time. The internal exchange calendar still verifies the session and market close before doing any work.

    15 21 * * 1-5 cd /path/to/stock-analyzer-app && /usr/bin/python3 -m src.jobs.cli market-cycle >> logs/market-cycle.log 2>&1
    30 21 * * 5 cd /path/to/stock-analyzer-app && /usr/bin/python3 -m src.jobs.cli weekly-report >> logs/weekly-report.log 2>&1
    0 * * * * cd /path/to/stock-analyzer-app && /usr/bin/python3 -m src.jobs.cli dispatch --retry-failed >> logs/notification-dispatch.log 2>&1

Repeated invocations cannot duplicate the same post-close market cycle or weekly report because the persisted job keys, scan keys, and execution keys are deterministic.

## Exit codes

- `0`: completed successfully, safely skipped, or duplicate
- `1`: completed with errors or failed delivery
- `2`: configuration or command failure

## Notification safety

Telegram tokens and SMTP credentials are read only from the environment. Status output reports configured channel names but never prints credentials.

Notification delivery attempts, successful sends, and failures remain persisted in SQLite.
