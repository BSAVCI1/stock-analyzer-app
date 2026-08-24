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

from src.product_config import (
    DEFAULT_PRODUCT_POLICY_PATH,
    load_product_policy,
    safe_product_policy_payload,
)

from src.paper import (
    AlertUsefulness,
    IncidentSeverity,
    IncidentStatus,
    ManualAlertAction,
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

    p3_release.add_argument(
        "--require-broker-reconciliation",
        action="store_true",
        help=(
            "Require a persisted matched external "
            "broker-paper reconciliation. Omit "
            "for the internal-only P3 profile."
        ),
    )

    product_config = commands.add_parser(
        "product-config",
        help=(
            "Print validated, secret-free "
            "P4 product policy."
        ),
    )

    product_config.add_argument(
        "--config",
        default=str(
            DEFAULT_PRODUCT_POLICY_PATH
        ),
        help=(
            "Versioned product policy JSON "
            "path."
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

    kill_switch = commands.add_parser(
        "kill-switch",
        help="Inspect or change the global execution kill switch.",
    )
    _add_runtime_arguments(kill_switch)
    kill_switch.add_argument(
        "action",
        choices=("status", "activate", "deactivate"),
    )
    kill_switch.add_argument(
        "--reason",
        help="Required reason for activate or deactivate.",
    )
    kill_switch.add_argument(
        "--operator",
        help="Required operator identity for a state change.",
    )

    strategy_pause = commands.add_parser(
        "strategy-pause",
        help="Inspect or change a named strategy entry pause.",
    )
    _add_runtime_arguments(strategy_pause)
    strategy_pause.add_argument(
        "action",
        choices=("status", "activate", "deactivate"),
    )
    strategy_pause.add_argument(
        "strategy",
        nargs="?",
        help="Strategy name; required for a state change.",
    )
    strategy_pause.add_argument(
        "--reason",
        help="Required reason for activate or deactivate.",
    )
    strategy_pause.add_argument(
        "--operator",
        help="Required operator identity for a state change.",
    )

    circuit_breaker = commands.add_parser(
        "circuit-breaker",
        help="Inspect persisted automatic circuit breakers.",
    )
    _add_runtime_arguments(circuit_breaker)
    circuit_breaker.add_argument(
        "action",
        choices=("status",),
    )

    incident = commands.add_parser(
        "incident",
        help="Open, review, update or resolve operational incidents.",
    )
    _add_runtime_arguments(incident)
    incident.add_argument(
        "action",
        choices=("list", "show", "open", "update", "resolve"),
    )
    incident.add_argument(
        "incident_id",
        nargs="?",
        help="Incident ID; required for show, update and resolve.",
    )
    incident.add_argument("--title")
    incident.add_argument(
        "--severity",
        choices=tuple(level.value for level in IncidentSeverity),
    )
    incident.add_argument("--summary")
    incident.add_argument(
        "--status",
        choices=tuple(state.value for state in IncidentStatus),
    )
    incident.add_argument("--note")
    incident.add_argument("--root-cause")
    incident.add_argument("--resolution")
    incident.add_argument("--operator")
    incident.add_argument("--reference-type")
    incident.add_argument("--reference-id")

    benchmark = commands.add_parser(
        "benchmark",
        help="Record or list persisted benchmark observations.",
    )
    _add_runtime_arguments(benchmark)
    benchmark.add_argument(
        "action",
        choices=("record", "list"),
    )
    benchmark.add_argument("symbol", nargs="?")
    benchmark.add_argument("--captured-at")
    benchmark.add_argument("--quote-currency")
    benchmark.add_argument("--close-price")
    benchmark.add_argument("--fx-rate")
    benchmark.add_argument("--source")

    valuation = commands.add_parser(
        "position-valuation",
        help="Record or list immutable open-position valuation evidence.",
    )
    _add_runtime_arguments(valuation)
    valuation.add_argument("action", choices=("record", "list"))
    valuation.add_argument("position_id", nargs="?")
    valuation.add_argument("--captured-at")
    valuation.add_argument("--quote-currency")
    valuation.add_argument("--close-price")
    valuation.add_argument("--fx-rate")
    valuation.add_argument("--source")

    feedback = commands.add_parser(
        "alert-feedback",
        help="Record or list immutable operator feedback for sent alerts.",
    )
    _add_runtime_arguments(feedback)
    feedback.add_argument("action", choices=("record", "list"))
    feedback.add_argument("notification_id", nargs="?")
    feedback.add_argument(
        "--usefulness", choices=tuple(item.value for item in AlertUsefulness)
    )
    feedback.add_argument(
        "--manual-action", choices=tuple(item.value for item in ManualAlertAction)
    )
    feedback.add_argument("--operator")
    feedback.add_argument("--rationale")
    feedback.add_argument("--broker-reference")
    feedback.add_argument("--recorded-at")

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
            broker_reconciliation_required=(
                args
                .require_broker_reconciliation
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
                "broker_reconciliation_required":
                operational_reliability
                .broker_reconciliation_required,
                "release_profile": (
                    "BROKER_PAPER"
                    if (
                        operational_reliability
                        .broker_reconciliation_required
                    )
                    else "INTERNAL_ONLY"
                ),
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


def _run_product_config(
    args,
) -> int:
    path = Path(args.config)

    policy = load_product_policy(
        path
    )

    _write_json(
        safe_product_policy_payload(
            policy,
            source_path=path,
        )
    )

    return 0


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

    incidents = (
        runtime.paper_repository
        .list_incidents(account_id)
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

    kill_switch = runtime.automation_repository.get_control(
        account_id,
        at=datetime.now(timezone.utc),
    )
    strategy_pauses = (
        runtime.automation_repository.list_strategy_pauses(
            account_id,
            active_only=True,
        )
    )
    circuit_breakers = (
        runtime.automation_repository.list_circuit_breakers(
            account_id,
            active_only=True,
        )
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
            "incidents": {
                "total": len(incidents),
                "open": sum(
                    incident.status
                    is not IncidentStatus.RESOLVED
                    for incident in incidents
                ),
                "critical_open": sum(
                    incident.status
                    is not IncidentStatus.RESOLVED
                    and incident.severity
                    is IncidentSeverity.CRITICAL
                    for incident in incidents
                ),
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
            "kill_switch": {
                "active": kill_switch.kill_switch_active,
                "reason": kill_switch.kill_switch_reason,
                "updated_at": kill_switch.updated_at,
                "new_orders_allowed": (
                    not kill_switch.kill_switch_active
                ),
            },
            "strategy_pauses": [
                {
                    "strategy": pause.strategy,
                    "reason": pause.reason,
                    "changed_by": pause.changed_by,
                    "changed_at": pause.changed_at,
                }
                for pause in strategy_pauses
            ],
            "circuit_breakers": [
                _circuit_breaker_payload(state)
                for state in circuit_breakers
            ],
        }
    )

    return 0


def _run_kill_switch(
    runtime: PaperJobRuntime,
    args,
) -> int:
    account_id = runtime.settings.account_id

    at = datetime.now(timezone.utc)
    before = runtime.automation_repository.get_control(
        account_id,
        at=at,
    )

    if args.action == "status":
        state = before
        changed = False
    else:
        if not args.reason or not args.operator:
            raise ValueError(
                "--reason and --operator are required for "
                "kill-switch state changes."
            )
        state = runtime.automation_repository.set_kill_switch(
            account_id,
            active=args.action == "activate",
            reason=args.reason,
            changed_by=args.operator,
            updated_at=at,
        )
        changed = (
            state.kill_switch_active
            is not before.kill_switch_active
        )

    _write_json(
        {
            "account_id": state.account_id,
            "active": state.kill_switch_active,
            "reason": state.kill_switch_reason,
            "changed_by": (
                args.operator if changed else None
            ),
            "updated_at": state.updated_at,
            "changed": changed,
            "new_orders_allowed": (
                not state.kill_switch_active
            ),
        }
    )
    return 0


def _strategy_pause_payload(pause) -> dict[str, object]:
    return {
        "account_id": pause.account_id,
        "strategy": pause.strategy,
        "active": pause.active,
        "reason": pause.reason,
        "changed_by": pause.changed_by,
        "changed_at": pause.changed_at,
        "new_entries_allowed": not pause.active,
    }


def _run_strategy_pause(
    runtime: PaperJobRuntime,
    args,
) -> int:
    account_id = runtime.settings.account_id
    repository = runtime.automation_repository

    if args.action == "status":
        pauses = repository.list_strategy_pauses(account_id)
        if args.strategy:
            target = args.strategy.strip().lower()
            pauses = tuple(
                pause
                for pause in pauses
                if pause.strategy == target
            )
        _write_json(
            {
                "account_id": account_id,
                "strategy_pauses": [
                    _strategy_pause_payload(pause)
                    for pause in pauses
                ],
                "active_strategies": [
                    pause.strategy
                    for pause in pauses
                    if pause.active
                ],
            }
        )
        return 0

    if not args.strategy or not args.reason or not args.operator:
        raise ValueError(
            "strategy, --reason and --operator are required "
            "for strategy-pause state changes."
        )

    before = repository.get_strategy_pause(
        account_id,
        args.strategy,
    )
    pause = repository.set_strategy_pause(
        account_id,
        strategy=args.strategy,
        active=args.action == "activate",
        reason=args.reason,
        changed_by=args.operator,
        changed_at=datetime.now(timezone.utc),
    )
    payload = _strategy_pause_payload(pause)
    payload["changed"] = (
        before is None or before.active is not pause.active
    )
    _write_json(payload)
    return 0


def _circuit_breaker_payload(state) -> dict[str, object]:
    return {
        "account_id": state.account_id,
        "breaker_type": state.breaker_type,
        "scope": state.scope,
        "active": state.active,
        "reason": state.reason,
        "tripped_at": state.tripped_at,
        "recovered_at": state.recovered_at,
        "metadata": dict(state.metadata),
        "updated_at": state.updated_at,
        "new_entries_allowed": not state.active,
    }


def _run_circuit_breaker_status(
    runtime: PaperJobRuntime,
) -> int:
    account_id = runtime.settings.account_id
    states = runtime.automation_repository.list_circuit_breakers(
        account_id
    )
    _write_json(
        {
            "account_id": account_id,
            "circuit_breakers": [
                _circuit_breaker_payload(state)
                for state in states
            ],
            "active_breaker_types": [
                state.breaker_type
                for state in states
                if state.active
            ],
        }
    )
    return 0


def _incident_payload(incident) -> dict[str, object]:
    return {
        "incident_id": incident.incident_id,
        "account_id": incident.account_id,
        "title": incident.title,
        "severity": incident.severity,
        "status": incident.status,
        "summary": incident.summary,
        "root_cause": incident.root_cause,
        "resolution": incident.resolution,
        "reference_type": incident.reference_type,
        "reference_id": incident.reference_id,
        "opened_by": incident.opened_by,
        "opened_at": incident.opened_at,
        "updated_by": incident.updated_by,
        "updated_at": incident.updated_at,
        "resolved_by": incident.resolved_by,
        "resolved_at": incident.resolved_at,
    }


def _run_incident(runtime: PaperJobRuntime, args) -> int:
    repository = runtime.paper_repository
    account_id = runtime.settings.account_id

    if args.action == "list":
        status = (
            IncidentStatus(args.status)
            if args.status
            else None
        )
        incidents = repository.list_incidents(
            account_id,
            status=status,
        )
        _write_json(
            {
                "account_id": account_id,
                "incidents": [
                    _incident_payload(incident)
                    for incident in incidents
                ],
                "total": len(incidents),
            }
        )
        return 0

    if args.action == "show":
        if not args.incident_id:
            raise ValueError("incident_id is required for show.")
        incident = repository.get_incident(args.incident_id)
        if incident.account_id != account_id:
            raise ValueError("Incident does not belong to the selected account.")
        timeline = tuple(
            event
            for event in repository.list_system_events(account_id)
            if (
                event.reference_type == "INCIDENT"
                and event.reference_id == incident.incident_id
            )
        )
        payload = _incident_payload(incident)
        payload["timeline"] = [
            {
                "event_type": event.event_type,
                "severity": event.severity,
                "message": event.message,
                "metadata": dict(event.metadata),
                "created_at": event.created_at,
            }
            for event in timeline
        ]
        _write_json(payload)
        return 0

    if args.action == "open":
        if not all((args.title, args.severity, args.summary, args.operator)):
            raise ValueError(
                "--title, --severity, --summary and --operator are required for open."
            )
        incident = repository.open_incident(
            account_id=account_id,
            title=args.title,
            severity=IncidentSeverity(args.severity),
            summary=args.summary,
            opened_by=args.operator,
            reference_type=args.reference_type,
            reference_id=args.reference_id,
        )
    elif args.action == "update":
        if not args.incident_id or not args.operator or not args.note:
            raise ValueError(
                "incident_id, --operator and --note are required for update."
            )
        if args.status == IncidentStatus.RESOLVED.value:
            raise ValueError("Use resolve to close an incident.")
        existing = repository.get_incident(args.incident_id)
        if existing.account_id != account_id:
            raise ValueError(
                "Incident does not belong to the selected account."
            )
        incident = repository.update_incident(
            args.incident_id,
            changed_by=args.operator,
            note=args.note,
            summary=args.summary,
            severity=(
                IncidentSeverity(args.severity)
                if args.severity
                else None
            ),
            status=(
                IncidentStatus(args.status)
                if args.status
                else None
            ),
        )
    else:
        if not all(
            (
                args.incident_id,
                args.root_cause,
                args.resolution,
                args.operator,
            )
        ):
            raise ValueError(
                "incident_id, --root-cause, --resolution and --operator "
                "are required for resolve."
            )
        existing = repository.get_incident(args.incident_id)
        if existing.account_id != account_id:
            raise ValueError(
                "Incident does not belong to the selected account."
            )
        incident = repository.resolve_incident(
            args.incident_id,
            root_cause=args.root_cause,
            resolution=args.resolution,
            resolved_by=args.operator,
        )

    if incident.account_id != account_id:
        raise ValueError("Incident does not belong to the selected account.")
    _write_json(_incident_payload(incident))
    return 0


def _benchmark_payload(observation) -> dict[str, object]:
    return {
        "observation_id": observation.observation_id,
        "account_id": observation.account_id,
        "symbol": observation.symbol,
        "captured_at": observation.captured_at,
        "quote_currency": observation.quote_currency,
        "close_price": observation.close_price,
        "fx_rate": observation.fx_rate,
        "portfolio_price": observation.portfolio_price,
        "source": observation.source,
    }


def _run_benchmark(runtime: PaperJobRuntime, args) -> int:
    repository = runtime.paper_repository
    account_id = runtime.settings.account_id

    if args.action == "list":
        observations = repository.list_benchmark_observations(
            account_id,
            symbol=args.symbol,
        )
        _write_json(
            {
                "account_id": account_id,
                "observations": [
                    _benchmark_payload(item)
                    for item in observations
                ],
                "total": len(observations),
            }
        )
        return 0

    if not all(
        (
            args.symbol,
            args.captured_at,
            args.quote_currency,
            args.close_price,
            args.fx_rate,
            args.source,
        )
    ):
        raise ValueError(
            "symbol, --captured-at, --quote-currency, --close-price, "
            "--fx-rate and --source are required for record."
        )

    captured_at = datetime.fromisoformat(args.captured_at)
    if captured_at.tzinfo is None or captured_at.utcoffset() is None:
        raise ValueError("--captured-at must be timezone-aware.")
    observation = repository.save_benchmark_observation(
        account_id=account_id,
        symbol=args.symbol,
        captured_at=captured_at,
        quote_currency=args.quote_currency,
        close_price=args.close_price,
        fx_rate=args.fx_rate,
        source=args.source,
    )
    _write_json(_benchmark_payload(observation))
    return 0


def _position_valuation_payload(observation) -> dict[str, object]:
    return {
        "observation_id": observation.observation_id,
        "account_id": observation.account_id,
        "position_id": observation.position_id,
        "symbol": observation.symbol,
        "captured_at": observation.captured_at,
        "quote_currency": observation.quote_currency,
        "close_price": observation.close_price,
        "fx_rate": observation.fx_rate,
        "quantity": observation.quantity,
        "market_value_portfolio": observation.market_value_portfolio,
        "source": observation.source,
    }


def _run_position_valuation(runtime: PaperJobRuntime, args) -> int:
    repository = runtime.paper_repository
    account_id = runtime.settings.account_id
    if args.action == "list":
        observations = repository.list_position_valuation_observations(account_id)
        if args.position_id:
            observations = tuple(
                item for item in observations if item.position_id == args.position_id
            )
        _write_json({
            "account_id": account_id,
            "observations": [_position_valuation_payload(item) for item in observations],
            "total": len(observations),
        })
        return 0
    if not all((args.position_id, args.captured_at, args.quote_currency,
                args.close_price, args.fx_rate, args.source)):
        raise ValueError(
            "position_id, --captured-at, --quote-currency, --close-price, "
            "--fx-rate and --source are required for record."
        )
    captured_at = datetime.fromisoformat(args.captured_at)
    if captured_at.tzinfo is None or captured_at.utcoffset() is None:
        raise ValueError("--captured-at must be timezone-aware.")
    observation = repository.save_position_valuation_observation(
        account_id=account_id, position_id=args.position_id,
        captured_at=captured_at, quote_currency=args.quote_currency,
        close_price=args.close_price, fx_rate=args.fx_rate, source=args.source,
    )
    _write_json(_position_valuation_payload(observation))
    return 0


def _alert_feedback_payload(entry) -> dict[str, object]:
    return {
        "journal_id": entry.journal_id,
        "account_id": entry.account_id,
        "notification_id": entry.notification_id,
        "usefulness": entry.usefulness.value,
        "manual_action": entry.manual_action.value,
        "operator": entry.operator,
        "rationale": entry.rationale,
        "broker_reference": entry.broker_reference,
        "recorded_at": entry.recorded_at,
    }


def _run_alert_feedback(runtime: PaperJobRuntime, args) -> int:
    repository = runtime.paper_repository
    account_id = runtime.settings.account_id
    if args.action == "list":
        entries = repository.list_alert_feedback(account_id)
        if args.notification_id:
            entries = tuple(
                item for item in entries
                if item.notification_id == args.notification_id
            )
        _write_json({
            "account_id": account_id,
            "entries": [_alert_feedback_payload(item) for item in entries],
            "total": len(entries),
        })
        return 0
    if not all((args.notification_id, args.usefulness, args.manual_action,
                args.operator, args.rationale, args.recorded_at)):
        raise ValueError(
            "notification_id, --usefulness, --manual-action, --operator, "
            "--rationale and --recorded-at are required for record."
        )
    recorded_at = datetime.fromisoformat(args.recorded_at)
    if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
        raise ValueError("--recorded-at must be timezone-aware.")
    entry = repository.record_alert_feedback(
        account_id=account_id,
        notification_id=args.notification_id,
        usefulness=AlertUsefulness(args.usefulness),
        manual_action=ManualAlertAction(args.manual_action),
        operator=args.operator,
        rationale=args.rationale,
        broker_reference=args.broker_reference,
        recorded_at=recorded_at,
    )
    _write_json(_alert_feedback_payload(entry))
    return 0


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "product-config":
            return _run_product_config(
                args
            )

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

        if args.command == "kill-switch":
            return _run_kill_switch(runtime, args)

        if args.command == "strategy-pause":
            return _run_strategy_pause(runtime, args)

        if args.command == "circuit-breaker":
            return _run_circuit_breaker_status(runtime)

        if args.command == "incident":
            return _run_incident(runtime, args)

        if args.command == "benchmark":
            return _run_benchmark(runtime, args)

        if args.command == "position-valuation":
            return _run_position_valuation(runtime, args)

        if args.command == "alert-feedback":
            return _run_alert_feedback(runtime, args)

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
