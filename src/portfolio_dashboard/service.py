"""Build a complete traceable portfolio dashboard snapshot."""

from __future__ import annotations

from datetime import datetime, timezone

from src.paper import PersistedSignal

from .metrics import (
    calculate_breakdowns,
    calculate_equity_performance,
    calculate_performance,
    calculate_reliability,
    make_provenance,
)
from .models import (
    DecisionTrace,
    PortfolioDashboardSnapshot,
    SectionProvenance,
)
from .repository import (
    PortfolioDashboardRepository,
)


class PortfolioDashboardService:
    def __init__(
        self,
        repository:
        PortfolioDashboardRepository,
    ) -> None:
        self.repository = repository

    @staticmethod
    def _trace(
        *,
        reference_type: str,
        reference_id: str,
        signal: PersistedSignal,
        source_tables: tuple[
            str,
            ...,
        ],
        source_ids: tuple[
            str,
            ...,
        ],
        exit_reason: str | None,
    ) -> DecisionTrace:
        return DecisionTrace(
            reference_type=reference_type,
            reference_id=reference_id,
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            strategy=signal.strategy,
            recommendation=(
                signal.recommendation
            ),
            market_regime=(
                signal.market_regime
            ),
            score=signal.score,
            confidence=signal.confidence,
            reward_to_risk=(
                signal.reward_to_risk
            ),
            threshold_version=(
                signal.threshold_version
            ),
            app_version=(
                signal.app_version
            ),
            evidence=signal.evidence,
            conflicts=signal.conflicts,
            exit_reason=exit_reason,
            provenance=make_provenance(
                tables=(
                    *source_tables,
                    "paper_signals",
                ),
                record_ids=(
                    *source_ids,
                    signal.signal_id,
                ),
                calculation=(
                    "Decision record joined "
                    "to its persisted signal "
                    "and evidence."
                ),
            ),
        )

    def build_snapshot(
        self,
        account_id: str,
        *,
        generated_at:
        datetime | None = None,
        recent_scan_limit: int = 20,
        recent_execution_limit: int = 50,
        recent_job_limit: int = 50,
    ) -> PortfolioDashboardSnapshot:
        if recent_scan_limit < 1:
            raise ValueError(
                "recent_scan_limit must be "
                "positive."
            )

        if recent_execution_limit < 1:
            raise ValueError(
                "recent_execution_limit must "
                "be positive."
            )

        if recent_job_limit < 1:
            raise ValueError(
                "recent_job_limit must be "
                "positive."
            )

        at = (
            generated_at
            or datetime.now(timezone.utc)
        )

        if (
            at.tzinfo is None
            or at.utcoffset() is None
        ):
            raise ValueError(
                "generated_at must be "
                "timezone-aware."
            )

        account = (
            self.repository.get_account(
                account_id
            )
        )

        reconciliation = (
            self.repository
            .reconcile_account(account_id)
        )

        signals = (
            self.repository.list_signals(
                account_id
            )
        )

        signals_by_id = {
            signal.signal_id: signal
            for signal in signals
        }

        pending_orders = (
            self.repository
            .list_pending_orders(
                account_id
            )
        )

        open_positions = (
            self.repository
            .list_open_positions(
                account_id
            )
        )

        closed_trades = (
            self.repository
            .list_closed_trades(
                account_id
            )
        )

        all_scan_reports = (
            self.repository
            .list_scan_reports(
                account_id
            )
        )

        scan_reports = tuple(
            reversed(
                all_scan_reports[
                    -recent_scan_limit:
                ]
            )
        )

        all_execution_runs = (
            self.repository
            .list_execution_runs(
                account_id
            )
        )

        execution_runs = tuple(
            reversed(
                all_execution_runs[
                    -recent_execution_limit:
                ]
            )
        )

        equity_snapshots = (
            self.repository
            .list_equity_snapshots(
                account_id
            )
        )

        all_jobs = (
            self.repository.list_jobs(
                account_id
            )
        )

        jobs = tuple(
            reversed(
                all_jobs[
                    -recent_job_limit:
                ]
            )
        )

        notifications = (
            self.repository
            .list_notifications(
                account_id
            )
        )

        system_events = (
            self.repository
            .list_system_events(
                account_id
            )
        )

        decision_traces = []

        for order in pending_orders:
            signal = signals_by_id.get(
                order.signal_id
            )

            if signal is None:
                continue

            decision_traces.append(
                self._trace(
                    reference_type=(
                        "PENDING_ORDER"
                    ),
                    reference_id=(
                        order.order_id
                    ),
                    signal=signal,
                    source_tables=(
                        "paper_orders",
                    ),
                    source_ids=(
                        order.order_id,
                    ),
                    exit_reason=None,
                )
            )

        for position in open_positions:
            order = (
                self.repository.paper
                .get_order(
                    position.order_id
                )
            )

            signal = signals_by_id.get(
                order.signal_id
            )

            if signal is None:
                continue

            decision_traces.append(
                self._trace(
                    reference_type=(
                        "OPEN_POSITION"
                    ),
                    reference_id=(
                        position.position_id
                    ),
                    signal=signal,
                    source_tables=(
                        "paper_positions",
                        "paper_orders",
                    ),
                    source_ids=(
                        position.position_id,
                        order.order_id,
                    ),
                    exit_reason=None,
                )
            )

        for trade in closed_trades:
            signal = signals_by_id.get(
                trade.signal_id
            )

            if signal is None:
                continue

            decision_traces.append(
                self._trace(
                    reference_type=(
                        "CLOSED_TRADE"
                    ),
                    reference_id=(
                        trade.trade_id
                    ),
                    signal=signal,
                    source_tables=(
                        "paper_closed_trades",
                    ),
                    source_ids=(
                        trade.trade_id,
                    ),
                    exit_reason=(
                        trade.exit_reason.value
                    ),
                )
            )

        traces = tuple(
            sorted(
                decision_traces,
                key=lambda item: (
                    item.reference_type,
                    item.reference_id,
                ),
            )
        )

        performance = (
            calculate_performance(
                closed_trades
            )
        )

        equity_performance = (
            calculate_equity_performance(
                account,
                equity_snapshots,
            )
        )

        breakdowns = (
            calculate_breakdowns(
                closed_trades,
                signals_by_id,
            )
        )

        reliability = (
            calculate_reliability(
                scans=scan_reports,
                execution_runs=(
                    execution_runs
                ),
                jobs=jobs,
                notifications=(
                    notifications
                ),
                system_events=(
                    system_events
                ),
            )
        )

        sections = (
            SectionProvenance(
                section="account",
                provenance=(
                    make_provenance(
                        tables=(
                            "paper_accounts",
                        ),
                        record_ids=(
                            account.account_id,
                        ),
                    )
                ),
            ),
            SectionProvenance(
                section="reconciliation",
                provenance=(
                    make_provenance(
                        tables=(
                            "paper_accounts",
                            "paper_ledger_entries",
                        ),
                        record_ids=(
                            account.account_id,
                        ),
                        filters=(
                            f"account_id={account_id}",
                        ),
                        calculation=(
                            "Stored account cash "
                            "minus the sum of all "
                            "persisted ledger "
                            "entries."
                        ),
                    )
                ),
            ),
            SectionProvenance(
                section="positions",
                provenance=(
                    make_provenance(
                        tables=(
                            "paper_positions",
                        ),
                        record_ids=(
                            position.position_id
                            for position
                            in open_positions
                        ),
                        filters=(
                            f"account_id={account_id}",
                            "status=OPEN",
                        ),
                    )
                ),
            ),
            SectionProvenance(
                section="pending_orders",
                provenance=(
                    make_provenance(
                        tables=(
                            "paper_orders",
                        ),
                        record_ids=(
                            order.order_id
                            for order
                            in pending_orders
                        ),
                        filters=(
                            f"account_id={account_id}",
                            "status=PENDING",
                        ),
                    )
                ),
            ),
            SectionProvenance(
                section="closed_trades",
                provenance=(
                    performance.provenance
                ),
            ),
            SectionProvenance(
                section="decision_traces",
                provenance=(
                    make_provenance(
                        tables=(
                            "paper_signals",
                            "paper_orders",
                            "paper_positions",
                            "paper_closed_trades",
                        ),
                        record_ids=(
                            trace.reference_id
                            for trace in traces
                        ),
                        calculation=(
                            "Persisted lifecycle "
                            "records joined to "
                            "signal evidence."
                        ),
                    )
                ),
            ),
            SectionProvenance(
                section="equity",
                provenance=(
                    equity_performance
                    .provenance
                ),
            ),
            SectionProvenance(
                section="scans",
                provenance=(
                    make_provenance(
                        tables=(
                            "paper_scans",
                            "paper_scan_results",
                        ),
                        record_ids=(
                            report.scan.scan_id
                            for report
                            in scan_reports
                        ),
                        filters=(
                            f"account_id={account_id}",
                        ),
                    )
                ),
            ),
            SectionProvenance(
                section="execution",
                provenance=(
                    reliability
                    .execution_runs
                    .provenance
                ),
            ),
            SectionProvenance(
                section="jobs",
                provenance=(
                    reliability
                    .scheduled_jobs
                    .provenance
                ),
            ),
            SectionProvenance(
                section="notifications",
                provenance=(
                    reliability
                    .notifications
                    .provenance
                ),
            ),
            SectionProvenance(
                section="system_events",
                provenance=(
                    reliability
                    .system_events
                    .provenance
                ),
            ),
        )

        return PortfolioDashboardSnapshot(
            generated_at=(
                at.astimezone(
                    timezone.utc
                )
            ),
            account=account,
            reconciliation=(
                reconciliation
            ),
            open_positions=(
                open_positions
            ),
            pending_orders=(
                pending_orders
            ),
            closed_trades=(
                closed_trades
            ),
            decision_traces=traces,
            equity_snapshots=(
                equity_snapshots
            ),
            performance=performance,
            equity_performance=(
                equity_performance
            ),
            breakdowns=breakdowns,
            scan_reports=scan_reports,
            execution_runs=(
                execution_runs
            ),
            jobs=jobs,
            notifications=(
                notifications
            ),
            system_events=(
                system_events
            ),
            reliability=reliability,
            section_provenance=sections,
            metadata={
                "account_id": account_id,
                "recent_scan_limit":
                recent_scan_limit,
                "recent_execution_limit":
                recent_execution_limit,
                "recent_job_limit":
                recent_job_limit,
                "source": (
                    "persisted_sqlite_records"
                ),
                "read_only": True,
            },
        )
