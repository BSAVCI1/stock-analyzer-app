"""Command-line interface for scheduled paper jobs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import (
    datetime,
    timezone,
)
from decimal import Decimal
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Sequence

from src.backtest import RegressionEvidence
from src.portfolio_dashboard import (
    PortfolioDashboardRepository,
    PortfolioDashboardService,
)
from src.release_gate import (
    build_operational_reliability_report,
    evaluate_p3_release_gate,
)

from src.execution_adapters import (
    BrokerReconciliationItemStatus,
    BrokerReconciliationRepository,
    BrokerReconciliationRunStatus,
)
from .models import (
    JobStatus,
    ScheduledJobReport,
)
from .runtime import (
    PaperJobRuntime,
    build_runtime,
    load_runtime_settings,
)


def _json_default(
    value: object,
) -> object:
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    raise TypeError(
        "Unsupported JSON value: "
        f"{type(value).__name__}."
    )


def _write_json(
    payload: object,
    *,
    stream=None,
) -> None:
    target = stream or sys.stdout

    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        file=target,
    )


def _parse_datetime(
    value: str | None,
) -> datetime:
    if value is None:
        return datetime.now(
            timezone.utc
        )

    cleaned = value.strip()

    if cleaned.endswith("Z"):
        cleaned = (
            cleaned[:-1] + "+00:00"
        )

    parsed = datetime.fromisoformat(
        cleaned
    )

    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
    ):
        raise ValueError(
            "--at must include a timezone, "
            "for example "
            "2026-08-03T21:15:00+00:00."
        )

    return parsed.astimezone(
        timezone.utc
    )


def _add_runtime_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--database",
        help=(
            "SQLite database path. "
            "Defaults to PAPER_DATABASE_PATH."
        ),
    )

    parser.add_argument(
        "--account-id",
        help=(
            "Paper account ID. Defaults to "
            "PAPER_ACCOUNT_ID."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.jobs.cli",
        description=(
            "Exchange-aware automated "
            "paper-trading jobs."
        ),
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    market = commands.add_parser(
        "market-cycle",
        help=(
            "Run the post-close market scan "
            "and paper execution cycle."
        ),
    )

    _add_runtime_arguments(market)

    market.add_argument(
        "--at",
        help=(
            "Timezone-aware ISO timestamp. "
            "Defaults to the current UTC time."
        ),
    )

    weekly = commands.add_parser(
        "weekly-report",
        help=(
            "Generate the weekly report after "
            "the final exchange session."
        ),
    )

    _add_runtime_arguments(weekly)

    weekly.add_argument(
        "--at",
        help=(
            "Timezone-aware ISO timestamp. "
            "Defaults to the current UTC time."
        ),
    )

    dispatch = commands.add_parser(
        "dispatch",
        help=(
            "Fan out internal events and send "
            "pending notifications."
        ),
    )

    _add_runtime_arguments(dispatch)

    dispatch.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Retry persisted FAILED "
            "notifications."
        ),
    )

    p3_release = commands.add_parser(
        "p3-release-status",
        help=(
            "Build the read-only P3 release "
            "report from persisted operational "
            "evidence."
        ),
    )

    _add_runtime_arguments(
        p3_release
    )

    p3_release.add_argument(
        "--at",
        help=(
            "Timezone-aware ISO timestamp used "
            "as the report generation time. "
            "Defaults to current UTC time."
        ),
    )

    p3_release.add_argument(
        "--regression-passed",
        action="store_true",
        help=(
            "Explicitly attest that the complete "
            "P0-P3 regression suite passed."
        ),
    )

    p3_release.add_argument(
        "--test-count",
        type=int,
        required=True,
        help=(
            "Number of tests in the complete "
            "attested regression run."
        ),
    )

    p3_release.add_argument(
        "--workflow",
        default="Automated tests",
        help=(
            "Name or identifier of the workflow "
            "that produced the regression "
            "evidence."
        ),
    )

    broker_status = commands.add_parser(
        "broker-reconciliation-status",
        help=(
            "Show the latest persisted "
            "broker-paper reconciliation "
            "without contacting a broker."
        ),
    )

    _add_runtime_arguments(
        broker_status
    )

    status = commands.add_parser(
        "status",
        help=(
            "Show paper account, job and "
            "notification status."
        ),
    )

    _add_runtime_arguments(status)

    return parser


def _runtime_from_args(
    args,
) -> PaperJobRuntime:
    settings = load_runtime_settings(
        database_path=args.database,
        account_id=args.account_id,
    )

    return build_runtime(settings)


def _job_payload(
    report: ScheduledJobReport,
) -> dict[str, object]:
    return {
        "job_run_id":
        report.job.job_run_id,
        "job_key": report.job.job_key,
        "job_type":
        report.job.job_type.value,
        "status":
        report.job.status.value,
        "scheduled_for":
        report.job.scheduled_for,
        "started_at":
        report.job.started_at,
        "completed_at":
        report.job.completed_at,
        "exchange_code":
        report.job.exchange_code,
        "session_date": (
            report.session
            .session_date
            .isoformat()
            if report.session
            is not None
            else None
        ),
        "duplicate": report.duplicate,
        "skipped_reason":
        report.skipped_reason,
        "scan_id": report.job.scan_id,
        "execution_run_id":
        report.job.execution_run_id,
        "queued_notifications":
        report.job.queued_notifications,
        "sent_notifications":
        report.job.sent_notifications,
        "failed_notifications":
        report.job.failed_notifications,
        "metadata":
        dict(report.job.metadata),
        "error_message":
        report.job.error_message,
    }


def _job_exit_code(
    report: ScheduledJobReport,
) -> int:
    if report.job.status in {
        JobStatus.FAILED,
        JobStatus.COMPLETED_WITH_ERRORS,
    }:
        return 1

    return 0


def _run_market_cycle(
    runtime: PaperJobRuntime,
    args,
) -> int:
    report = (
        runtime.job_service
        .run_market_cycle(
            account_id=(
                runtime.settings.account_id
            ),
            scheduled_for=(
                _parse_datetime(args.at)
            ),
        )
    )

    _write_json(
        _job_payload(report)
    )

    return _job_exit_code(report)


def _run_weekly_report(
    runtime: PaperJobRuntime,
    args,
) -> int:
    report = (
        runtime.job_service
        .run_weekly_report(
            account_id=(
                runtime.settings.account_id
            ),
            scheduled_for=(
                _parse_datetime(args.at)
            ),
        )
    )

    _write_json(
        _job_payload(report)
    )

    return _job_exit_code(report)


def _run_dispatch(
    runtime: PaperJobRuntime,
    args,
) -> int:
    at = datetime.now(
        timezone.utc
    )

    fanned_out = (
        runtime.notification_service
        .fan_out_internal(
            runtime.settings.account_id,
            channels=(
                runtime
                .notification_channels
            ),
            created_at=at,
        )
    )

    report = (
        runtime.notification_service
        .dispatch_pending(
            runtime.settings.account_id,
            include_failed=(
                args.retry_failed
            ),
            attempted_at=at,
        )
    )

    _write_json(
        {
            "fanned_out": fanned_out,
            "processed":
            report.processed,
            "sent": report.sent,
            "failed": report.failed,
            "skipped": report.skipped,
            "sent_notification_ids":
            report.sent_notification_ids,
            "failed_notification_ids":
            report.failed_notification_ids,
        }
    )

    return 1 if report.failed else 0


def _broker_reconciliation_run_payload(
    run,
) -> dict[str, object]:
    return {
        "reconciliation_run_id":
        run.reconciliation_run_id,
        "account_id": run.account_id,
        "reconciliation_key":
        run.reconciliation_key,
        "provider": run.provider,
        "broker_account_id":
        run.broker_account_id,
        "status": run.status.value,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "account_item_count":
        run.account_item_count,
        "order_item_count":
        run.order_item_count,
        "position_item_count":
        run.position_item_count,
        "matched_item_count":
        run.matched_item_count,
        "mismatched_item_count":
        run.mismatched_item_count,
        "missing_internal_item_count":
        run.missing_internal_item_count,
        "missing_broker_item_count":
        run.missing_broker_item_count,
        "unresolved_item_count":
        run.unresolved_item_count,
        "reconciled": run.reconciled,
        "metadata": dict(run.metadata),
        "error_message":
        run.error_message,
    }


def _broker_reconciliation_item_payload(
    item,
) -> dict[str, object]:
    return {
        "reconciliation_item_id":
        item.reconciliation_item_id,
        "category": item.category.value,
        "comparison_key":
        item.comparison_key,
        "status": item.status.value,
        "internal_reference_ids":
        item.internal_reference_ids,
        "broker_reference_ids":
        item.broker_reference_ids,
        "differences":
        dict(item.differences),
        "message": item.message,
        "created_at": item.created_at,
        "metadata": dict(item.metadata),
    }


def _run_broker_reconciliation_status(
    runtime: PaperJobRuntime,
) -> int:
    account_id = (
        runtime.settings.account_id
    )

    repository = (
        BrokerReconciliationRepository(
            runtime.settings.database_path
        )
    )

    run = repository.latest_run(
        account_id
    )

    if run is None:
        _write_json(
            {
                "account_id": account_id,
                "status": "NOT_RUN",
                "latest_run": None,
                "unresolved_items": (),
                "message": (
                    "No persisted broker-paper "
                    "reconciliation run exists."
                ),
            }
        )

        return 2

    items = repository.list_items(
        run.reconciliation_run_id
    )

    unresolved = tuple(
        item
        for item in items
        if item.status
        is not
        BrokerReconciliationItemStatus
        .MATCH
    )

    _write_json(
        {
            "account_id": account_id,
            "status": run.status.value,
            "latest_run":
            _broker_reconciliation_run_payload(
                run
            ),
            "unresolved_items": tuple(
                _broker_reconciliation_item_payload(
                    item
                )
                for item in unresolved
            ),
        }
    )

    if (
        run.status
        is BrokerReconciliationRunStatus
        .MATCHED
    ):
        return 0

    return 1


def _p3_operational_check_payload(
    check,
) -> dict[str, object]:
    return {
        "name": check.name,
        "status": check.status.value,
        "observed_count":
        check.observed_count,
        "failed_count":
        check.failed_count,
        "details": check.details,
    }


def _run_p3_release_status(
    runtime: PaperJobRuntime,
    args,
) -> int:
    generated_at = _parse_datetime(
        args.at
    )

    repository = (
        PortfolioDashboardRepository(
            runtime.settings.database_path
        )
    )

    snapshot = (
        PortfolioDashboardService(
            repository
        ).build_snapshot(
            runtime.settings.account_id,
            generated_at=generated_at,
        )
    )

    regression_evidence = (
        RegressionEvidence(
            passed=bool(
                args.regression_passed
            ),
            test_count=args.test_count,
            covered_phases=(
                "P0",
                "P1",
                "P2",
                "P3",
            ),
            workflow=args.workflow,
        )
    )

    operational_reliability = (
        build_operational_reliability_report(
            snapshot,
            execution_descriptor=(
                runtime
                .execution_adapter
                .descriptor
            ),
        )
    )

    report = evaluate_p3_release_gate(
        regression_evidence=(
            regression_evidence
        ),
        operational_reliability=(
            operational_reliability
        ),
    )

    _write_json(
        {
            "status": report.status.value,
            "release_ready":
            report.release_ready,
            "generated_at":
            operational_reliability
            .generated_at,
            "account_id":
            operational_reliability
            .account_id,
            "regression": {
                "passed":
                regression_evidence.passed,
                "test_count":
                regression_evidence
                .test_count,
                "covered_phases":
                regression_evidence
                .covered_phases,
                "workflow":
                regression_evidence
                .workflow,
            },
            "operational_reliability": {
                "passed":
                operational_reliability
                .passed,
                "broker_reconciliation_run_id":
                operational_reliability
                .broker_reconciliation_run_id,
                "unresolved_broker_differences":
                operational_reliability
                .unresolved_broker_differences,
                "live_trading_enabled":
                operational_reliability
                .live_trading_enabled,
                "checks": tuple(
                    _p3_operational_check_payload(
                        check
                    )
                    for check
                    in operational_reliability
                    .checks
                ),
            },
            "reasons": report.reasons,
        }
    )

    return (
        0
        if report.release_ready
        else 1
    )


def _run_status(
    runtime: PaperJobRuntime,
) -> int:
    account_id = (
        runtime.settings.account_id
    )

    account = (
        runtime.paper_repository
        .get_account(account_id)
    )

    reconciliation = (
        runtime.paper_repository
        .reconcile_account(account_id)
    )

    positions = (
        runtime.paper_repository
        .list_open_positions(account_id)
    )

    pending_orders = (
        runtime.paper_repository
        .list_pending_orders(account_id)
    )

    trades = (
        runtime.paper_repository
        .list_closed_trades(account_id)
    )

    jobs = (
        runtime.job_repository
        .list_jobs(account_id)
    )

    notifications = (
        runtime.paper_repository
        .list_notifications(account_id)
    )

    job_status_counts = Counter(
        job.status.value
        for job in jobs
    )

    notification_status_counts = Counter(
        notification.status.value
        for notification
        in notifications
    )

    notification_channel_counts = Counter(
        notification.channel.value
        for notification
        in notifications
    )

    latest_job = (
        jobs[-1]
        if jobs
        else None
    )

    _write_json(
        {
            "database_path":
            runtime.settings.database_path,
            "account": {
                "account_id":
                account.account_id,
                "name": account.name,
                "base_currency":
                account.base_currency,
                "status":
                account.status.value,
                "starting_balance":
                account.starting_balance,
                "cash_balance":
                account.cash_balance,
                "reserved_cash":
                account.reserved_cash,
            },
            "portfolio": {
                "open_positions":
                len(positions),
                "pending_orders":
                len(pending_orders),
                "closed_trades":
                len(trades),
                "reconciled":
                reconciliation.reconciled,
            },
            "jobs": {
                "total": len(jobs),
                "status_counts":
                dict(job_status_counts),
                "latest": (
                    {
                        "job_run_id":
                        latest_job.job_run_id,
                        "job_type":
                        latest_job
                        .job_type
                        .value,
                        "status":
                        latest_job
                        .status
                        .value,
                        "scheduled_for":
                        latest_job
                        .scheduled_for,
                        "completed_at":
                        latest_job
                        .completed_at,
                    }
                    if latest_job
                    is not None
                    else None
                ),
            },
            "notifications": {
                "total":
                len(notifications),
                "status_counts":
                dict(
                    notification_status_counts
                ),
                "channel_counts":
                dict(
                    notification_channel_counts
                ),
                "configured_channels": [
                    channel.value
                    for channel
                    in runtime
                    .notification_channels
                ],
            },
            "release_configuration": {
                "eligible_strategies":
                runtime.settings
                .release_eligible_strategies,
                "deny_by_default": (
                    not bool(
                        runtime.settings
                        .release_eligible_strategies
                    )
                ),
            },
        }
    )

    return 0


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        runtime = _runtime_from_args(
            args
        )

        if args.command == "market-cycle":
            return _run_market_cycle(
                runtime,
                args,
            )

        if args.command == "weekly-report":
            return _run_weekly_report(
                runtime,
                args,
            )

        if args.command == "dispatch":
            return _run_dispatch(
                runtime,
                args,
            )

        if args.command == "p3-release-status":
            return _run_p3_release_status(
                runtime,
                args,
            )

        if args.command == "broker-reconciliation-status":
            return _run_broker_reconciliation_status(
                runtime
            )

        if args.command == "status":
            return _run_status(runtime)

        parser.error(
            f"Unsupported command: "
            f"{args.command}."
        )

    except Exception as exc:
        _write_json(
            {
                "status": "ERROR",
                "error_type":
                type(exc).__name__,
                "error_message": str(exc),
            },
            stream=sys.stderr,
        )

        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
