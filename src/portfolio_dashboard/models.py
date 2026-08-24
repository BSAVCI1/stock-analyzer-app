"""Immutable read models for the paper portfolio dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Mapping

from src.execution_adapters import (
    BrokerReconciliationItem,
    BrokerReconciliationRun,
)
from src.automation import (
    EquitySnapshot,
    ExecutionRun,
)
from src.jobs import JobRun
from src.paper import (
    AlertFeedbackJournalEntry,
    BenchmarkObservation,
    AccountReconciliation,
    ClosedPaperTrade,
    NotificationRecord,
    PaperAccount,
    PaperOrderRecord,
    PaperPositionRecord,
    PositionValuationObservation,
    SystemEventRecord,
)
from src.scanner import MarketScanReport


@dataclass(frozen=True, slots=True)
class Provenance:
    """Persisted sources supporting a displayed value."""

    source_tables: tuple[str, ...]
    record_ids: tuple[str, ...] = ()
    filters: tuple[str, ...] = ()
    calculation: str = ""

    @property
    def record_count(self) -> int:
        return len(self.record_ids)


@dataclass(frozen=True, slots=True)
class SectionProvenance:
    section: str
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    reference_type: str
    reference_id: str

    signal_id: str
    symbol: str
    strategy: str
    recommendation: str
    market_regime: str

    score: float
    confidence: float
    reward_to_risk: float

    threshold_version: str
    app_version: str

    evidence: tuple[str, ...]
    conflicts: tuple[str, ...]

    exit_reason: str | None

    provenance: Provenance


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    trade_count: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int

    win_rate_pct: float

    gross_pnl: Decimal
    net_pnl: Decimal
    total_fees: Decimal
    total_slippage: Decimal
    total_costs: Decimal
    expectancy: Decimal
    cost_drag_pct: float | None

    average_return_pct: float
    best_trade_net_pnl: Decimal | None
    worst_trade_net_pnl: Decimal | None

    gross_profit: Decimal
    gross_loss: Decimal
    profit_factor: float | None

    provenance: Provenance


@dataclass(frozen=True, slots=True)
class EquityPerformance:
    point_count: int

    latest_equity: Decimal | None
    peak_equity: Decimal | None
    lowest_equity: Decimal | None

    total_return: Decimal | None
    total_return_pct: float | None

    maximum_drawdown: Decimal | None
    maximum_drawdown_pct: float | None

    provenance: Provenance


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    symbol: str
    observation_count: int
    sufficient_evidence: bool
    reason: str | None
    period_started_at: datetime | None
    period_ended_at: datetime | None
    account_return_pct: float | None
    benchmark_return_pct: float | None
    cash_return_pct: float
    excess_vs_benchmark_pct: float | None
    excess_vs_cash_pct: float | None
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class ConcentrationHolding:
    symbol: str
    market_value: Decimal
    portfolio_weight_pct: float
    equity_weight_pct: float
    position_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConcentrationSummary:
    sufficient_evidence: bool
    reason: str | None
    captured_at: datetime | None
    position_count: int
    symbol_count: int
    invested_market_value: Decimal
    equity: Decimal | None
    invested_equity_pct: float | None
    largest_symbol: str | None
    largest_symbol_weight_pct: float | None
    top_three_weight_pct: float | None
    hhi: float | None
    holdings: tuple[ConcentrationHolding, ...]
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class ActionabilityCohort:
    key: str
    watchlist_entries: int
    converted_entries: int
    open_entries: int
    abandoned_entries: int
    conversion_rate_pct: float | None


@dataclass(frozen=True, slots=True)
class ActionabilitySummary:
    generated_at: datetime
    watchlist_entries: int
    converted_entries: int
    open_entries: int
    abandoned_entries: int
    conversion_rate_pct: float | None
    signal_count: int
    matured_signal_count: int
    ordered_signal_count: int
    stale_signal_count: int
    stale_signal_rate_pct: float | None
    cohorts: tuple[ActionabilityCohort, ...]
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class AlertUsefulnessSummary:
    sent_alerts: int
    assessed_alerts: int
    useful_alerts: int
    not_useful_alerts: int
    copied_as_is: int
    copied_modified: int
    dismissed: int
    no_action: int
    assessment_coverage_pct: float | None
    usefulness_rate_pct: float | None
    manual_copy_rate_pct: float | None
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class PerformanceBreakdown:
    dimension: str
    key: str

    trade_count: int
    winning_trades: int
    losing_trades: int

    gross_pnl: Decimal
    total_fees: Decimal
    total_slippage: Decimal
    total_costs: Decimal
    net_pnl: Decimal
    expectancy: Decimal
    profit_factor: float | None
    average_return_pct: float

    provenance: Provenance


@dataclass(frozen=True, slots=True)
class ReliabilityMetric:
    name: str

    total: int
    successful: int
    failed: int
    pending_or_other: int

    success_rate_pct: float | None

    provenance: Provenance


@dataclass(frozen=True, slots=True)
class ReliabilitySummary:
    scans: ReliabilityMetric
    execution_runs: ReliabilityMetric
    scheduled_jobs: ReliabilityMetric
    notifications: ReliabilityMetric
    system_events: ReliabilityMetric


@dataclass(frozen=True, slots=True)
class PortfolioDashboardSnapshot:
    generated_at: datetime

    account: PaperAccount
    reconciliation: AccountReconciliation

    broker_reconciliation_run: (
        BrokerReconciliationRun | None
    )

    broker_reconciliation_items: tuple[
        BrokerReconciliationItem,
        ...,
    ]

    open_positions: tuple[
        PaperPositionRecord,
        ...,
    ]

    pending_orders: tuple[
        PaperOrderRecord,
        ...,
    ]

    closed_trades: tuple[
        ClosedPaperTrade,
        ...,
    ]

    decision_traces: tuple[
        DecisionTrace,
        ...,
    ]

    equity_snapshots: tuple[
        EquitySnapshot,
        ...,
    ]

    benchmark_observations: tuple[
        BenchmarkObservation,
        ...,
    ]

    benchmark_comparisons: tuple[
        BenchmarkComparison,
        ...,
    ]

    position_valuation_observations: tuple[
        PositionValuationObservation, ...,
    ]
    concentration: ConcentrationSummary
    actionability: ActionabilitySummary
    alert_feedback: tuple[AlertFeedbackJournalEntry, ...]
    alert_usefulness: AlertUsefulnessSummary

    performance: PerformanceSummary
    equity_performance: EquityPerformance

    breakdowns: tuple[
        PerformanceBreakdown,
        ...,
    ]

    scan_reports: tuple[
        MarketScanReport,
        ...,
    ]

    execution_runs: tuple[
        ExecutionRun,
        ...,
    ]

    jobs: tuple[JobRun, ...]

    notifications: tuple[
        NotificationRecord,
        ...,
    ]

    system_events: tuple[
        SystemEventRecord,
        ...,
    ]

    reliability: ReliabilitySummary

    section_provenance: tuple[
        SectionProvenance,
        ...,
    ]

    metadata: Mapping[str, object]

    def provenance_for(
        self,
        section: str,
    ) -> Provenance:
        for item in self.section_provenance:
            if item.section == section:
                return item.provenance

        raise KeyError(
            f"Unknown dashboard section: "
            f"{section}."
        )
