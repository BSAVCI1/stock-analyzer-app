"""Pure presentation helpers for the portfolio dashboard."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Mapping

from .models import (
    PortfolioDashboardSnapshot,
    Provenance,
)


def _enum_value(
    value: object,
) -> object:
    if isinstance(value, Enum):
        return value.value

    return value


def _timestamp(
    value: datetime | None,
) -> str | None:
    if value is None:
        return None

    return value.isoformat()


def _decimal_text(
    value: object,
) -> str:
    return str(Decimal(str(value)))


def format_money(
    value: object,
    currency: str,
) -> str:
    amount = Decimal(str(value))

    return (
        f"{amount:,.2f} "
        f"{str(currency).upper()}"
    )


def format_percent(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return f"{value:.2f}%"


def metric_cards(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[Mapping[str, object], ...]:
    account = snapshot.account
    performance = snapshot.performance

    available_cash = (
        account.cash_balance
        - account.reserved_cash
    )

    return (
        {
            "label": "Cash balance",
            "value": format_money(
                account.cash_balance,
                account.base_currency,
            ),
            "source_table":
            "paper_accounts",
            "record_id":
            account.account_id,
        },
        {
            "label": "Available cash",
            "value": format_money(
                available_cash,
                account.base_currency,
            ),
            "source_table":
            "paper_accounts",
            "record_id":
            account.account_id,
        },
        {
            "label": "Open positions",
            "value": str(
                len(
                    snapshot
                    .open_positions
                )
            ),
            "source_table":
            "paper_positions",
            "record_id": None,
        },
        {
            "label": "Pending orders",
            "value": str(
                len(
                    snapshot
                    .pending_orders
                )
            ),
            "source_table":
            "paper_orders",
            "record_id": None,
        },
        {
            "label": "Realised net P&L",
            "value": format_money(
                performance.net_pnl,
                account.base_currency,
            ),
            "source_table":
            "paper_closed_trades",
            "record_id": None,
        },
        {
            "label": "Reconciled",
            "value": (
                "Yes"
                if snapshot
                .reconciliation
                .reconciled
                else "No"
            ),
            "source_table":
            "paper_ledger_entries",
            "record_id":
            account.account_id,
        },
    )


def open_position_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "position_id":
            position.position_id,
            "order_id":
            position.order_id,
            "fill_id":
            position.fill_id,
            "symbol": position.symbol,
            "side": _enum_value(
                position.side
            ),
            "quantity": _decimal_text(
                position.quantity
            ),
            "entry_price": _decimal_text(
                position.entry_price
            ),
            "stop_price": _decimal_text(
                position.stop_price
            ),
            "targets": ", ".join(
                _decimal_text(target)
                for target
                in position.targets
            ),
            "opened_at": _timestamp(
                position.opened_at
            ),
            "expires_at": _timestamp(
                position.expires_at
            ),
            "status": _enum_value(
                position.status
            ),
        }
        for position
        in snapshot.open_positions
    )


def pending_order_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "order_id": order.order_id,
            "signal_id":
            order.signal_id,
            "idempotency_key":
            order.idempotency_key,
            "symbol": order.symbol,
            "side": _enum_value(
                order.side
            ),
            "quantity": _decimal_text(
                order.quantity
            ),
            "entry_low": _decimal_text(
                order.entry_low
            ),
            "entry_high": _decimal_text(
                order.entry_high
            ),
            "stop_price": _decimal_text(
                order.stop_price
            ),
            "targets": ", ".join(
                _decimal_text(target)
                for target
                in order.targets
            ),
            "reserved_cash":
            _decimal_text(
                order.reserved_cash
            ),
            "status": _enum_value(
                order.status
            ),
            "created_at": _timestamp(
                order.created_at
            ),
            "expires_at": _timestamp(
                order.expires_at
            ),
        }
        for order
        in snapshot.pending_orders
    )


def closed_trade_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "trade_id": trade.trade_id,
            "position_id":
            trade.position_id,
            "signal_id":
            trade.signal_id,
            "symbol": trade.symbol,
            "strategy": trade.strategy,
            "market_regime":
            trade.market_regime,
            "entry_time": _timestamp(
                trade.entry_time
            ),
            "entry_price":
            _decimal_text(
                trade.entry_price
            ),
            "exit_time": _timestamp(
                trade.exit_time
            ),
            "exit_price":
            _decimal_text(
                trade.exit_price
            ),
            "exit_reason":
            _enum_value(
                trade.exit_reason
            ),
            "quantity": _decimal_text(
                trade.quantity
            ),
            "gross_pnl":
            _decimal_text(
                trade.gross_pnl
            ),
            "fees": _decimal_text(
                trade.fees
            ),
            "slippage": _decimal_text(
                trade.slippage
            ),
            "net_pnl": _decimal_text(
                trade.net_pnl
            ),
            "return_pct":
            trade.return_pct,
            "holding_seconds":
            trade.holding_seconds,
        }
        for trade
        in snapshot.closed_trades
    )


def decision_trace_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "reference_type":
            trace.reference_type,
            "reference_id":
            trace.reference_id,
            "signal_id":
            trace.signal_id,
            "symbol": trace.symbol,
            "strategy":
            trace.strategy,
            "recommendation":
            trace.recommendation,
            "market_regime":
            trace.market_regime,
            "score": trace.score,
            "confidence":
            trace.confidence,
            "reward_to_risk":
            trace.reward_to_risk,
            "threshold_version":
            trace.threshold_version,
            "app_version":
            trace.app_version,
            "exit_reason":
            trace.exit_reason,
            "evidence": " | ".join(
                trace.evidence
            ),
            "conflicts": " | ".join(
                trace.conflicts
            ),
            "source_tables":
            ", ".join(
                trace.provenance
                .source_tables
            ),
            "source_record_ids":
            ", ".join(
                trace.provenance
                .record_ids
            ),
        }
        for trace
        in snapshot.decision_traces
    )


def equity_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    snapshots = sorted(
        snapshot.equity_snapshots,
        key=lambda item: (
            item.captured_at,
            item.snapshot_id,
        ),
    )

    return tuple(
        {
            "snapshot_id":
            item.snapshot_id,
            "run_id": item.run_id,
            "captured_at":
            item.captured_at,
            "cash_balance":
            float(item.cash_balance),
            "reserved_cash":
            float(item.reserved_cash),
            "market_value":
            float(item.market_value),
            "equity":
            float(item.equity),
        }
        for item in snapshots
    )


def performance_breakdown_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "dimension":
            row.dimension,
            "key": row.key,
            "trade_count":
            row.trade_count,
            "winning_trades":
            row.winning_trades,
            "losing_trades":
            row.losing_trades,
            "gross_pnl": _decimal_text(
                row.gross_pnl
            ),
            "fees": _decimal_text(
                row.total_fees
            ),
            "slippage": _decimal_text(
                row.total_slippage
            ),
            "total_costs": _decimal_text(
                row.total_costs
            ),
            "net_pnl": _decimal_text(
                row.net_pnl
            ),
            "expectancy": _decimal_text(
                row.expectancy
            ),
            "profit_factor":
            row.profit_factor,
            "average_return_pct":
            row.average_return_pct,
            "source_record_ids":
            ", ".join(
                row.provenance
                .record_ids
            ),
        }
        for row in snapshot.breakdowns
    )


def benchmark_comparison_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "symbol": row.symbol,
            "observation_count": row.observation_count,
            "sufficient_evidence": row.sufficient_evidence,
            "reason": row.reason,
            "period_started_at": _timestamp(
                row.period_started_at
            ) if row.period_started_at else None,
            "period_ended_at": _timestamp(
                row.period_ended_at
            ) if row.period_ended_at else None,
            "account_return_pct": row.account_return_pct,
            "benchmark_return_pct": row.benchmark_return_pct,
            "cash_return_pct": row.cash_return_pct,
            "excess_vs_benchmark_pct": row.excess_vs_benchmark_pct,
            "excess_vs_cash_pct": row.excess_vs_cash_pct,
            "source_record_ids": ", ".join(
                row.provenance.record_ids
            ),
        }
        for row in snapshot.benchmark_comparisons
    )


def concentration_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    summary = snapshot.concentration
    return tuple(
        {
            "captured_at": _timestamp(summary.captured_at),
            "symbol": row.symbol,
            "market_value": _decimal_text(row.market_value),
            "portfolio_weight_pct": row.portfolio_weight_pct,
            "equity_weight_pct": row.equity_weight_pct,
            "position_ids": ", ".join(row.position_ids),
            "sufficient_evidence": summary.sufficient_evidence,
        }
        for row in summary.holdings
    )


def actionability_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "cohort": row.key,
            "watchlist_entries": row.watchlist_entries,
            "converted_entries": row.converted_entries,
            "open_entries": row.open_entries,
            "abandoned_entries": row.abandoned_entries,
            "conversion_rate_pct": row.conversion_rate_pct,
        }
        for row in snapshot.actionability.cohorts
    )


def alert_feedback_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "journal_id": item.journal_id,
            "notification_id": item.notification_id,
            "usefulness": item.usefulness.value,
            "manual_action": item.manual_action.value,
            "operator": item.operator,
            "rationale": item.rationale,
            "broker_reference": item.broker_reference,
            "recorded_at": _timestamp(item.recorded_at),
        }
        for item in snapshot.alert_feedback
    )


def scan_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "scan_id":
            report.scan.scan_id,
            "scan_key":
            report.scan.scan_key,
            "universe":
            report.scan.universe,
            "status":
            report.scan.status.value,
            "started_at":
            _timestamp(
                report.scan.started_at
            ),
            "completed_at":
            _timestamp(
                report.scan.completed_at
            ),
            "requested_count":
            report.scan.requested_count,
            "processed_count":
            report.scan.processed_count,
            "rejected_count":
            report.scan.rejected_count,
            "signal_count":
            report.scan.signal_count,
            "order_count":
            report.scan.order_count,
            "app_version":
            report.scan.app_version,
            "error_message":
            report.scan.error_message,
        }
        for report
        in snapshot.scan_reports
    )


def scan_result_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    rows = []

    for report in snapshot.scan_reports:
        for result in report.results:
            rows.append(
                {
                    "result_id":
                    result.result_id,
                    "scan_id":
                    result.scan_id,
                    "symbol":
                    result.symbol,
                    "status":
                    result.status.value,
                    "recommendation":
                    result.recommendation,
                    "strategy":
                    result.strategy,
                    "score": result.score,
                    "confidence":
                    result.confidence,
                    "market_regime":
                    result.market_regime,
                    "reward_to_risk":
                    result.reward_to_risk,
                    "release_eligible":
                    result.release_eligible,
                    "rank_score":
                    result.rank_score,
                    "rank_position":
                    result.rank_position,
                    "signal_id":
                    result.signal_id,
                    "reasons":
                    " | ".join(
                        result.reasons
                    ),
                }
            )

    return tuple(rows)


def execution_run_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "run_id": run.run_id,
            "run_key": run.run_key,
            "scan_id": run.scan_id,
            "status":
            run.status.value,
            "started_at":
            _timestamp(run.started_at),
            "completed_at":
            _timestamp(
                run.completed_at
            ),
            "created_orders":
            run.created_orders,
            "filled_orders":
            run.filled_orders,
            "expired_orders":
            run.expired_orders,
            "cancelled_orders":
            run.cancelled_orders,
            "closed_positions":
            run.closed_positions,
            "rejected_entries":
            run.rejected_entries,
            "error_count":
            run.error_count,
            "entry_block_reasons":
            " | ".join(
                run.entry_block_reasons
            ),
            "app_version":
            run.app_version,
            "error_message":
            run.error_message,
        }
        for run
        in snapshot.execution_runs
    )


def job_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "job_run_id":
            job.job_run_id,
            "job_key": job.job_key,
            "job_type":
            job.job_type.value,
            "status":
            job.status.value,
            "scheduled_for":
            _timestamp(
                job.scheduled_for
            ),
            "started_at":
            _timestamp(job.started_at),
            "completed_at":
            _timestamp(
                job.completed_at
            ),
            "exchange_code":
            job.exchange_code,
            "scan_id": job.scan_id,
            "execution_run_id":
            job.execution_run_id,
            "queued_notifications":
            job.queued_notifications,
            "sent_notifications":
            job.sent_notifications,
            "failed_notifications":
            job.failed_notifications,
            "error_message":
            job.error_message,
        }
        for job in snapshot.jobs
    )


def notification_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "notification_id":
            item.notification_id,
            "event_type":
            item.event_type,
            "reference_type":
            item.reference_type,
            "reference_id":
            item.reference_id,
            "channel":
            item.channel.value,
            "status":
            item.status.value,
            "created_at":
            _timestamp(
                item.created_at
            ),
            "sent_at":
            _timestamp(item.sent_at),
            "error_message":
            item.error_message,
        }
        for item in snapshot.notifications
    )


def system_event_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "event_id":
            event.event_id,
            "event_type":
            event.event_type,
            "severity":
            event.severity,
            "reference_type":
            event.reference_type,
            "reference_id":
            event.reference_id,
            "message":
            event.message,
            "created_at":
            _timestamp(
                event.created_at
            ),
        }
        for event
        in snapshot.system_events
    )


def reliability_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    reliability = snapshot.reliability

    metrics = (
        reliability.scans,
        reliability.execution_runs,
        reliability.scheduled_jobs,
        reliability.notifications,
        reliability.system_events,
    )

    return tuple(
        {
            "name": metric.name,
            "total": metric.total,
            "successful":
            metric.successful,
            "failed": metric.failed,
            "pending_or_other":
            metric.pending_or_other,
            "success_rate_pct":
            metric.success_rate_pct,
            "source_table":
            ", ".join(
                metric.provenance
                .source_tables
            ),
            "source_record_count":
            metric.provenance
            .record_count,
        }
        for metric in metrics
    )


def provenance_row(
    section: str,
    provenance: Provenance,
) -> dict[str, object]:
    return {
        "section": section,
        "source_tables":
        ", ".join(
            provenance.source_tables
        ),
        "record_count":
        provenance.record_count,
        "record_ids":
        ", ".join(
            provenance.record_ids
        ),
        "filters":
        " | ".join(
            provenance.filters
        ),
        "calculation":
        provenance.calculation,
    }


def provenance_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    return tuple(
        provenance_row(
            item.section,
            item.provenance,
        )
        for item
        in snapshot.section_provenance
    )


def broker_reconciliation_summary_rows(
    snapshot: PortfolioDashboardSnapshot,
) -> tuple[dict[str, object], ...]:
    """Describe the latest persisted reconciliation run."""

    run = (
        snapshot
        .broker_reconciliation_run
    )

    if run is None:
        return ()

    return (
        {
            "reconciliation_run_id":
            run.reconciliation_run_id,
            "provider": run.provider,
            "broker_account_id":
            run.broker_account_id,
            "status": run.status.value,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "account_items":
            run.account_item_count,
            "order_items":
            run.order_item_count,
            "position_items":
            run.position_item_count,
            "matched":
            run.matched_item_count,
            "mismatched":
            run.mismatched_item_count,
            "missing_internal":
            run.missing_internal_item_count,
            "missing_broker":
            run.missing_broker_item_count,
            "unresolved":
            run.unresolved_item_count,
            "reconciled": run.reconciled,
            "error_message":
            run.error_message,
        },
    )


def broker_reconciliation_item_rows(
    snapshot: PortfolioDashboardSnapshot,
    *,
    unresolved_only: bool = True,
) -> tuple[dict[str, object], ...]:
    """Return persisted broker comparison evidence."""

    items = (
        snapshot
        .broker_reconciliation_items
    )

    if unresolved_only:
        items = tuple(
            item
            for item in items
            if item.status.value != "MATCH"
        )

    return tuple(
        {
            "reconciliation_item_id":
            item.reconciliation_item_id,
            "category":
            item.category.value,
            "comparison_key":
            item.comparison_key,
            "status": item.status.value,
            "message": item.message,
            "internal_reference_ids":
            ", ".join(
                item.internal_reference_ids
            ),
            "broker_reference_ids":
            ", ".join(
                item.broker_reference_ids
            ),
            "differences":
            dict(item.differences),
            "created_at":
            item.created_at,
        }
        for item in items
    )
