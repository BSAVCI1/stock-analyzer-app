"""Deterministic metrics derived from persisted records."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Callable, Iterable, Mapping

from src.automation import (
    EquitySnapshot,
    ExecutionRun,
)
from src.jobs import JobRun
from src.paper import (
    ClosedPaperTrade,
    NotificationRecord,
    PaperAccount,
    PersistedSignal,
    SystemEventRecord,
)
from src.scanner import MarketScanReport

from .models import (
    EquityPerformance,
    PerformanceBreakdown,
    PerformanceSummary,
    Provenance,
    ReliabilityMetric,
    ReliabilitySummary,
)


ZERO = Decimal("0")


def make_provenance(
    *,
    tables: Iterable[str],
    record_ids: Iterable[str] = (),
    filters: Iterable[str] = (),
    calculation: str = "",
) -> Provenance:
    return Provenance(
        source_tables=tuple(
            dict.fromkeys(tables)
        ),
        record_ids=tuple(
            dict.fromkeys(record_ids)
        ),
        filters=tuple(
            dict.fromkeys(filters)
        ),
        calculation=calculation,
    )


def calculate_performance(
    trades: tuple[
        ClosedPaperTrade,
        ...,
    ],
) -> PerformanceSummary:
    wins = tuple(
        trade
        for trade in trades
        if trade.net_pnl > ZERO
    )

    losses = tuple(
        trade
        for trade in trades
        if trade.net_pnl < ZERO
    )

    breakeven = tuple(
        trade
        for trade in trades
        if trade.net_pnl == ZERO
    )

    gross_pnl = sum(
        (
            trade.gross_pnl
            for trade in trades
        ),
        ZERO,
    )

    net_pnl = sum(
        (
            trade.net_pnl
            for trade in trades
        ),
        ZERO,
    )

    total_fees = sum(
        (
            trade.fees
            for trade in trades
        ),
        ZERO,
    )

    total_slippage = sum(
        (
            trade.slippage
            for trade in trades
        ),
        ZERO,
    )

    gross_profit = sum(
        (
            trade.net_pnl
            for trade in wins
        ),
        ZERO,
    )

    gross_loss = abs(
        sum(
            (
                trade.net_pnl
                for trade in losses
            ),
            ZERO,
        )
    )

    trade_count = len(trades)

    win_rate = (
        len(wins) / trade_count * 100
        if trade_count
        else 0.0
    )

    average_return = (
        sum(
            trade.return_pct
            for trade in trades
        )
        / trade_count
        if trade_count
        else 0.0
    )

    profit_factor = (
        float(
            gross_profit / gross_loss
        )
        if gross_loss > ZERO
        else (
            None
            if gross_profit == ZERO
            else float("inf")
        )
    )

    return PerformanceSummary(
        trade_count=trade_count,
        winning_trades=len(wins),
        losing_trades=len(losses),
        breakeven_trades=(
            len(breakeven)
        ),
        win_rate_pct=win_rate,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        total_fees=total_fees,
        total_slippage=(
            total_slippage
        ),
        average_return_pct=(
            average_return
        ),
        best_trade_net_pnl=(
            max(
                trade.net_pnl
                for trade in trades
            )
            if trades
            else None
        ),
        worst_trade_net_pnl=(
            min(
                trade.net_pnl
                for trade in trades
            )
            if trades
            else None
        ),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        provenance=make_provenance(
            tables=(
                "paper_closed_trades",
            ),
            record_ids=(
                trade.trade_id
                for trade in trades
            ),
            calculation=(
                "Aggregated persisted closed-"
                "trade P&L, costs and returns."
            ),
        ),
    )


def calculate_equity_performance(
    account: PaperAccount,
    snapshots: tuple[
        EquitySnapshot,
        ...,
    ],
) -> EquityPerformance:
    if not snapshots:
        return EquityPerformance(
            point_count=0,
            latest_equity=None,
            peak_equity=None,
            lowest_equity=None,
            total_return=None,
            total_return_pct=None,
            maximum_drawdown=None,
            maximum_drawdown_pct=None,
            provenance=make_provenance(
                tables=(
                    "paper_equity_snapshots",
                    "paper_accounts",
                ),
                record_ids=(
                    account.account_id,
                ),
                filters=(
                    f"account_id={account.account_id}",
                ),
                calculation=(
                    "No persisted equity "
                    "snapshots are available."
                ),
            ),
        )

    ordered = tuple(
        sorted(
            snapshots,
            key=lambda item: (
                item.captured_at,
                item.snapshot_id,
            ),
        )
    )

    latest = ordered[-1].equity
    peak = max(
        item.equity
        for item in ordered
    )

    lowest = min(
        item.equity
        for item in ordered
    )

    running_peak = ordered[0].equity
    maximum_drawdown = ZERO
    maximum_drawdown_pct = 0.0

    for item in ordered:
        if item.equity > running_peak:
            running_peak = item.equity

        drawdown = (
            running_peak - item.equity
        )

        drawdown_pct = (
            float(
                drawdown
                / running_peak
                * Decimal("100")
            )
            if running_peak > ZERO
            else 0.0
        )

        if drawdown > maximum_drawdown:
            maximum_drawdown = drawdown

        if (
            drawdown_pct
            > maximum_drawdown_pct
        ):
            maximum_drawdown_pct = (
                drawdown_pct
            )

    total_return = (
        latest
        - account.starting_balance
    )

    total_return_pct = (
        float(
            total_return
            / account.starting_balance
            * Decimal("100")
        )
        if account.starting_balance
        != ZERO
        else None
    )

    return EquityPerformance(
        point_count=len(ordered),
        latest_equity=latest,
        peak_equity=peak,
        lowest_equity=lowest,
        total_return=total_return,
        total_return_pct=(
            total_return_pct
        ),
        maximum_drawdown=(
            maximum_drawdown
        ),
        maximum_drawdown_pct=(
            maximum_drawdown_pct
        ),
        provenance=make_provenance(
            tables=(
                "paper_equity_snapshots",
                "paper_accounts",
            ),
            record_ids=(
                (
                    account.account_id,
                    *(
                        item.snapshot_id
                        for item in ordered
                    ),
                )
            ),
            calculation=(
                "Chronological persisted "
                "equity curve, running peak "
                "and peak-to-trough drawdown."
            ),
        ),
    )


def _breakdown(
    *,
    dimension: str,
    trades: tuple[
        ClosedPaperTrade,
        ...,
    ],
    key_lookup: Callable[
        [ClosedPaperTrade],
        str,
    ],
) -> tuple[
    PerformanceBreakdown,
    ...,
]:
    grouped: dict[
        str,
        list[ClosedPaperTrade],
    ] = defaultdict(list)

    for trade in trades:
        key = (
            str(key_lookup(trade))
            .strip()
            or "UNKNOWN"
        )

        grouped[key].append(trade)

    rows = []

    for key in sorted(grouped):
        members = tuple(grouped[key])

        wins = sum(
            trade.net_pnl > ZERO
            for trade in members
        )

        losses = sum(
            trade.net_pnl < ZERO
            for trade in members
        )

        net_pnl = sum(
            (
                trade.net_pnl
                for trade in members
            ),
            ZERO,
        )

        average_return = (
            sum(
                trade.return_pct
                for trade in members
            )
            / len(members)
        )

        rows.append(
            PerformanceBreakdown(
                dimension=dimension,
                key=key,
                trade_count=len(members),
                winning_trades=wins,
                losing_trades=losses,
                net_pnl=net_pnl,
                average_return_pct=(
                    average_return
                ),
                provenance=(
                    make_provenance(
                        tables=(
                            "paper_closed_trades",
                        ),
                        record_ids=(
                            trade.trade_id
                            for trade
                            in members
                        ),
                        calculation=(
                            f"Closed trades "
                            f"grouped by "
                            f"{dimension}={key}."
                        ),
                    )
                ),
            )
        )

    return tuple(rows)


def calculate_breakdowns(
    trades: tuple[
        ClosedPaperTrade,
        ...,
    ],
    signals: Mapping[
        str,
        PersistedSignal,
    ],
) -> tuple[
    PerformanceBreakdown,
    ...,
]:
    strategy = _breakdown(
        dimension="strategy",
        trades=trades,
        key_lookup=lambda trade:
        trade.strategy,
    )

    horizon = _breakdown(
        dimension="strategy_horizon",
        trades=trades,
        key_lookup=lambda trade: (
            trade.strategy_horizon.value
            if trade.strategy_horizon
            is not None
            else "UNKNOWN"
        ),
    )

    strategy_version = _breakdown(
        dimension="strategy_version",
        trades=trades,
        key_lookup=lambda trade: (
            trade.strategy_version
            or "UNKNOWN"
        ),
    )

    instrument = _breakdown(
        dimension="instrument",
        trades=trades,
        key_lookup=lambda trade:
        trade.symbol,
    )

    regime = _breakdown(
        dimension="market_regime",
        trades=trades,
        key_lookup=lambda trade:
        trade.market_regime,
    )

    threshold = _breakdown(
        dimension="threshold_version",
        trades=trades,
        key_lookup=lambda trade: (
            signals[trade.signal_id]
            .threshold_version
            if trade.signal_id
            in signals
            else "UNKNOWN"
        ),
    )

    return (
        *strategy,
        *horizon,
        *strategy_version,
        *instrument,
        *regime,
        *threshold,
    )


def _reliability_metric(
    *,
    name: str,
    source_table: str,
    records,
    id_lookup,
    status_lookup,
    successful_statuses: set[str],
    failed_statuses: set[str],
) -> ReliabilityMetric:
    statuses = tuple(
        str(status_lookup(record))
        .strip()
        .upper()
        for record in records
    )

    successful = sum(
        status in successful_statuses
        for status in statuses
    )

    failed = sum(
        status in failed_statuses
        for status in statuses
    )

    total = len(statuses)

    other = (
        total
        - successful
        - failed
    )

    completed = (
        successful + failed
    )

    success_rate = (
        successful
        / completed
        * 100
        if completed
        else None
    )

    return ReliabilityMetric(
        name=name,
        total=total,
        successful=successful,
        failed=failed,
        pending_or_other=other,
        success_rate_pct=success_rate,
        provenance=make_provenance(
            tables=(source_table,),
            record_ids=(
                id_lookup(record)
                for record in records
            ),
            calculation=(
                f"Persisted {name} statuses "
                "classified as successful, "
                "failed or pending/other."
            ),
        ),
    )


def calculate_reliability(
    *,
    scans: tuple[
        MarketScanReport,
        ...,
    ],
    execution_runs: tuple[
        ExecutionRun,
        ...,
    ],
    jobs: tuple[JobRun, ...],
    notifications: tuple[
        NotificationRecord,
        ...,
    ],
    system_events: tuple[
        SystemEventRecord,
        ...,
    ],
) -> ReliabilitySummary:
    scan_metric = _reliability_metric(
        name="scans",
        source_table="paper_scans",
        records=scans,
        id_lookup=lambda report:
        report.scan.scan_id,
        status_lookup=lambda report:
        report.scan.status.value,
        successful_statuses={
            "COMPLETED",
        },
        failed_statuses={
            "FAILED",
            "COMPLETED_WITH_ERRORS",
        },
    )

    execution_metric = (
        _reliability_metric(
            name="execution_runs",
            source_table=(
                "paper_execution_runs"
            ),
            records=execution_runs,
            id_lookup=lambda run:
            run.run_id,
            status_lookup=lambda run:
            run.status.value,
            successful_statuses={
                "COMPLETED",
            },
            failed_statuses={
                "FAILED",
                "COMPLETED_WITH_ERRORS",
            },
        )
    )

    job_metric = _reliability_metric(
        name="scheduled_jobs",
        source_table="paper_job_runs",
        records=jobs,
        id_lookup=lambda job:
        job.job_run_id,
        status_lookup=lambda job:
        job.status.value,
        successful_statuses={
            "COMPLETED",
            "SKIPPED",
        },
        failed_statuses={
            "FAILED",
            "COMPLETED_WITH_ERRORS",
        },
    )

    notification_metric = (
        _reliability_metric(
            name="notifications",
            source_table=(
                "paper_notifications"
            ),
            records=notifications,
            id_lookup=lambda notification:
            notification.notification_id,
            status_lookup=lambda notification:
            notification.status.value,
            successful_statuses={
                "SENT",
            },
            failed_statuses={
                "FAILED",
            },
        )
    )

    event_metric = _reliability_metric(
        name="system_events",
        source_table=(
            "paper_system_events"
        ),
        records=system_events,
        id_lookup=lambda event:
        event.event_id,
        status_lookup=lambda event: (
            "FAILED"
            if event.severity
            .strip()
            .upper()
            in {
                "ERROR",
                "CRITICAL",
            }
            else "COMPLETED"
        ),
        successful_statuses={
            "COMPLETED",
        },
        failed_statuses={
            "FAILED",
        },
    )

    return ReliabilitySummary(
        scans=scan_metric,
        execution_runs=(
            execution_metric
        ),
        scheduled_jobs=job_metric,
        notifications=(
            notification_metric
        ),
        system_events=event_metric,
    )
