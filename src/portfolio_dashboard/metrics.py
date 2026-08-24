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
    AlertFeedbackJournalEntry,
    AlertUsefulness,
    BenchmarkObservation,
    ClosedPaperTrade,
    NotificationRecord,
    NotificationStatus,
    ManualAlertAction,
    PaperAccount,
    PersistedSignal,
    PaperPositionRecord,
    PaperOrderRecord,
    PositionValuationObservation,
    SystemEventRecord,
)
from src.scanner import MarketScanReport
from src.scanner import WatchlistState

from .models import (
    EquityPerformance,
    BenchmarkComparison,
    ConcentrationHolding,
    ConcentrationSummary,
    ActionabilityCohort,
    ActionabilitySummary,
    AlertUsefulnessSummary,
    PerformanceBreakdown,
    PerformanceSummary,
    Provenance,
    ReliabilityMetric,
    ReliabilitySummary,
)


ZERO = Decimal("0")


def calculate_alert_usefulness(
    notifications: tuple[NotificationRecord, ...],
    feedback: tuple[AlertFeedbackJournalEntry, ...],
) -> AlertUsefulnessSummary:
    sent = tuple(
        item for item in notifications if item.status is NotificationStatus.SENT
    )
    sent_ids = {item.notification_id for item in sent}
    assessed = tuple(
        item for item in feedback if item.notification_id in sent_ids
    )
    useful = sum(
        item.usefulness is AlertUsefulness.USEFUL for item in assessed
    )
    copied_as_is = sum(
        item.manual_action is ManualAlertAction.COPIED_AS_IS for item in assessed
    )
    copied_modified = sum(
        item.manual_action is ManualAlertAction.COPIED_MODIFIED for item in assessed
    )
    dismissed = sum(
        item.manual_action is ManualAlertAction.DISMISSED for item in assessed
    )
    no_action = sum(
        item.manual_action is ManualAlertAction.NO_ACTION for item in assessed
    )
    return AlertUsefulnessSummary(
        sent_alerts=len(sent), assessed_alerts=len(assessed),
        useful_alerts=useful, not_useful_alerts=len(assessed) - useful,
        copied_as_is=copied_as_is, copied_modified=copied_modified,
        dismissed=dismissed, no_action=no_action,
        assessment_coverage_pct=(
            len(assessed) / len(sent) * 100 if sent else None
        ),
        usefulness_rate_pct=(
            useful / len(assessed) * 100 if assessed else None
        ),
        manual_copy_rate_pct=(
            (copied_as_is + copied_modified) / len(assessed) * 100
            if assessed else None
        ),
        provenance=make_provenance(
            tables=("paper_notifications", "paper_alert_feedback_journal"),
            record_ids=(
                *(item.notification_id for item in sent),
                *(item.journal_id for item in assessed),
            ),
            calculation=(
                "Useful and manual-copy assessments divided by assessed sent "
                "notification records; coverage divided by all sent records."
            ),
        ),
    )


def _cohort_key(item) -> str:
    horizon = (
        item.strategy_horizon.value
        if item.strategy_horizon is not None else "UNKNOWN"
    )
    return f"{horizon}|{item.strategy_version or 'UNKNOWN'}"


def calculate_actionability(
    *, signals: tuple[PersistedSignal, ...],
    orders: tuple[PaperOrderRecord, ...],
    scans: tuple[MarketScanReport, ...],
    at,
) -> ActionabilitySummary:
    """Calculate version-safe watchlist conversion and stale-signal rates."""
    observed_signals = tuple(item for item in signals if item.generated_at <= at)
    observed_orders = tuple(item for item in orders if item.created_at <= at)
    ordered_signal_ids = {item.signal_id for item in observed_orders}
    matured = tuple(
        item for item in observed_signals
        if item.expires_at <= at or item.signal_id in ordered_signal_ids
    )
    stale = tuple(
        item for item in matured if item.signal_id not in ordered_signal_ids
    )

    grouped_results = defaultdict(list)
    result_ids = []
    for report in scans:
        for result in report.results:
            if result.processed_at <= at:
                grouped_results[(result.symbol, _cohort_key(result))].append(result)
                result_ids.append(result.result_id)

    cohort_counts = defaultdict(lambda: [0, 0, 0, 0])
    for (_, cohort), results in grouped_results.items():
        open_episode = False
        for result in sorted(results, key=lambda x: (x.processed_at, x.result_id)):
            state = result.watchlist_state
            if state in (WatchlistState.WATCH, WatchlistState.PREPARE):
                if not open_episode:
                    cohort_counts[cohort][0] += 1
                    open_episode = True
            elif state is WatchlistState.ACTIONABLE:
                if open_episode:
                    cohort_counts[cohort][1] += 1
                    open_episode = False
            elif state in (WatchlistState.REJECT, WatchlistState.STALE):
                if open_episode:
                    cohort_counts[cohort][3] += 1
                open_episode = False
        if open_episode:
            cohort_counts[cohort][2] += 1

    cohorts = tuple(
        ActionabilityCohort(
            key=key,
            watchlist_entries=values[0],
            converted_entries=values[1],
            open_entries=values[2],
            abandoned_entries=values[3],
            conversion_rate_pct=(
                values[1] / values[0] * 100 if values[0] else None
            ),
        )
        for key, values in sorted(cohort_counts.items())
    )
    entries = sum(item.watchlist_entries for item in cohorts)
    converted = sum(item.converted_entries for item in cohorts)
    open_entries = sum(item.open_entries for item in cohorts)
    abandoned = sum(item.abandoned_entries for item in cohorts)
    return ActionabilitySummary(
        generated_at=at,
        watchlist_entries=entries,
        converted_entries=converted,
        open_entries=open_entries,
        abandoned_entries=abandoned,
        conversion_rate_pct=(converted / entries * 100 if entries else None),
        signal_count=len(observed_signals),
        matured_signal_count=len(matured),
        ordered_signal_count=sum(
            1 for item in matured if item.signal_id in ordered_signal_ids
        ),
        stale_signal_count=len(stale),
        stale_signal_rate_pct=(
            len(stale) / len(matured) * 100 if matured else None
        ),
        cohorts=cohorts,
        provenance=make_provenance(
            tables=("paper_scan_results", "paper_signals", "paper_orders"),
            record_ids=(
                *result_ids,
                *(item.signal_id for item in matured),
                *(item.order_id for item in observed_orders),
            ),
            filters=(f"observed_at<={at.isoformat()}",),
            calculation=(
                "Watch/preparation episodes converting to actionable within the same "
                "strategy-horizon/version cohort; expired unordered signals divided "
                "by matured signals."
            ),
        ),
    )


def calculate_concentration(
    positions: tuple[PaperPositionRecord, ...],
    observations: tuple[PositionValuationObservation, ...],
    equity_snapshots: tuple[EquitySnapshot, ...],
) -> ConcentrationSummary:
    open_ids = {item.position_id for item in positions}
    latest_at = max((item.captured_at for item in observations), default=None)
    latest = tuple(
        item for item in observations
        if item.captured_at == latest_at and item.position_id in open_ids
    )
    valued_ids = {item.position_id for item in latest}
    reason = None
    equity_snapshot = None
    if not positions:
        reason = "No open positions are available for concentration analysis."
    elif latest_at is None:
        reason = "No persisted position valuations are available."
    elif valued_ids != open_ids:
        reason = "The latest valuation timestamp does not cover every open position."
    else:
        aligned = tuple(
            item for item in equity_snapshots if item.captured_at <= latest_at
        )
        equity_snapshot = max(
            aligned, key=lambda item: (item.captured_at, item.snapshot_id),
            default=None,
        )
        if equity_snapshot is None:
            reason = "No equity snapshot is aligned to the valuation timestamp."
        elif equity_snapshot.equity <= ZERO:
            reason = "Aligned portfolio equity must be positive."

    sufficient = reason is None
    grouped: dict[str, list[PositionValuationObservation]] = defaultdict(list)
    for item in latest:
        grouped[item.symbol].append(item)
    total = sum((item.market_value_portfolio for item in latest), ZERO)
    if sufficient and total <= ZERO:
        sufficient = False
        reason = "Invested market value must be positive."

    holdings = ()
    invested_equity_pct = largest_weight = top_three = hhi = None
    largest_symbol = None
    if sufficient:
        rows = []
        for symbol, items in grouped.items():
            value = sum((item.market_value_portfolio for item in items), ZERO)
            weight = float(value / total * Decimal("100"))
            rows.append(ConcentrationHolding(
                symbol=symbol,
                market_value=value,
                portfolio_weight_pct=weight,
                equity_weight_pct=float(
                    value / equity_snapshot.equity * Decimal("100")
                ),
                position_ids=tuple(sorted(item.position_id for item in items)),
            ))
        holdings = tuple(sorted(rows, key=lambda row: (-row.portfolio_weight_pct, row.symbol)))
        largest_symbol = holdings[0].symbol
        largest_weight = holdings[0].portfolio_weight_pct
        top_three = sum(row.portfolio_weight_pct for row in holdings[:3])
        hhi = sum((row.portfolio_weight_pct / 100) ** 2 for row in holdings)
        invested_equity_pct = float(total / equity_snapshot.equity * Decimal("100"))

    return ConcentrationSummary(
        sufficient_evidence=sufficient, reason=reason, captured_at=latest_at,
        position_count=len(positions), symbol_count=len(grouped),
        invested_market_value=total,
        equity=(equity_snapshot.equity if equity_snapshot else None),
        invested_equity_pct=invested_equity_pct,
        largest_symbol=largest_symbol,
        largest_symbol_weight_pct=largest_weight,
        top_three_weight_pct=top_three, hhi=hhi, holdings=holdings,
        provenance=make_provenance(
            tables=("paper_position_valuation_observations", "paper_positions", "paper_equity_snapshots"),
            record_ids=(
                *(item.observation_id for item in latest),
                *((equity_snapshot.snapshot_id,) if equity_snapshot else ()),
            ),
            calculation="Latest complete position-valuation set grouped by symbol and aligned to persisted portfolio equity.",
        ),
    )


def calculate_benchmark_comparisons(
    observations: tuple[BenchmarkObservation, ...],
    equity_snapshots: tuple[EquitySnapshot, ...],
) -> tuple[BenchmarkComparison, ...]:
    grouped: dict[str, list[BenchmarkObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.symbol].append(observation)

    snapshots = tuple(
        sorted(
            equity_snapshots,
            key=lambda item: (item.captured_at, item.snapshot_id),
        )
    )
    comparisons = []

    for symbol in sorted(grouped):
        marks = tuple(
            sorted(
                grouped[symbol],
                key=lambda item: (item.captured_at, item.observation_id),
            )
        )
        record_ids = tuple(mark.observation_id for mark in marks)
        reason = None
        start = marks[0] if marks else None
        end = marks[-1] if marks else None
        start_equity = None
        end_equity = None

        if len(marks) < 2:
            reason = "At least two benchmark observations are required."
        elif start is not None and end is not None and start.captured_at == end.captured_at:
            reason = "Benchmark observations must span two timestamps."
        else:
            start_candidates = tuple(
                item for item in snapshots
                if item.captured_at <= start.captured_at
            )
            end_candidates = tuple(
                item for item in snapshots
                if item.captured_at <= end.captured_at
            )
            start_equity = start_candidates[-1] if start_candidates else None
            end_equity = end_candidates[-1] if end_candidates else None
            if start_equity is None or end_equity is None:
                reason = "Equity snapshots do not bracket the benchmark period."
            elif start_equity.snapshot_id == end_equity.snapshot_id:
                reason = "Two aligned equity snapshots are required."
            elif start_equity.equity <= ZERO or start.portfolio_price <= ZERO:
                reason = "Starting equity and benchmark price must be positive."

        sufficient = reason is None
        account_return = None
        benchmark_return = None
        if sufficient:
            account_return = float(
                (end_equity.equity - start_equity.equity)
                / start_equity.equity
                * Decimal("100")
            )
            benchmark_return = float(
                (end.portfolio_price - start.portfolio_price)
                / start.portfolio_price
                * Decimal("100")
            )

        comparisons.append(
            BenchmarkComparison(
                symbol=symbol,
                observation_count=len(marks),
                sufficient_evidence=sufficient,
                reason=reason,
                period_started_at=(start.captured_at if start else None),
                period_ended_at=(end.captured_at if end else None),
                account_return_pct=account_return,
                benchmark_return_pct=benchmark_return,
                cash_return_pct=0.0,
                excess_vs_benchmark_pct=(
                    account_return - benchmark_return
                    if sufficient else None
                ),
                excess_vs_cash_pct=(account_return if sufficient else None),
                provenance=make_provenance(
                    tables=("paper_benchmark_observations", "paper_equity_snapshots"),
                    record_ids=(
                        *record_ids,
                        *((start_equity.snapshot_id, end_equity.snapshot_id) if sufficient else ()),
                    ),
                    calculation=(
                        "Portfolio-currency benchmark price return and nominal "
                        "cash comparison aligned to persisted equity snapshots."
                    ),
                ),
            )
        )

    return tuple(comparisons)


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

    total_costs = (
        total_fees
        + total_slippage
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

    expectancy = (
        net_pnl / trade_count
        if trade_count
        else ZERO
    )

    cost_drag_pct = (
        float(
            total_costs
            / abs(gross_pnl)
            * Decimal("100")
        )
        if gross_pnl != ZERO
        else None
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
        total_costs=total_costs,
        expectancy=expectancy,
        cost_drag_pct=cost_drag_pct,
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

        gross_pnl = sum(
            (
                trade.gross_pnl
                for trade in members
            ),
            ZERO,
        )

        total_fees = sum(
            (
                trade.fees
                for trade in members
            ),
            ZERO,
        )

        total_slippage = sum(
            (
                trade.slippage
                for trade in members
            ),
            ZERO,
        )

        gross_profit = sum(
            (
                trade.net_pnl
                for trade in members
                if trade.net_pnl > ZERO
            ),
            ZERO,
        )

        gross_loss = abs(
            sum(
                (
                    trade.net_pnl
                    for trade in members
                    if trade.net_pnl < ZERO
                ),
                ZERO,
            )
        )

        profit_factor = (
            float(gross_profit / gross_loss)
            if gross_loss > ZERO
            else (
                None
                if gross_profit == ZERO
                else float("inf")
            )
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
                gross_pnl=gross_pnl,
                total_fees=total_fees,
                total_slippage=total_slippage,
                total_costs=(
                    total_fees
                    + total_slippage
                ),
                net_pnl=net_pnl,
                expectancy=(
                    net_pnl
                    / len(members)
                ),
                profit_factor=profit_factor,
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

    strategy_cohort = _breakdown(
        dimension="strategy_cohort",
        trades=trades,
        key_lookup=lambda trade: (
            f"{trade.strategy_horizon.value if trade.strategy_horizon is not None else 'UNKNOWN'}"
            f"|{trade.strategy_version or 'UNKNOWN'}"
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
        *strategy_cohort,
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
