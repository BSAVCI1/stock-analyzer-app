"""Immutable read models for the paper portfolio dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Mapping

from src.automation import (
    EquitySnapshot,
    ExecutionRun,
)
from src.jobs import JobRun
from src.paper import (
    AccountReconciliation,
    ClosedPaperTrade,
    NotificationRecord,
    PaperAccount,
    PaperOrderRecord,
    PaperPositionRecord,
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
class PerformanceBreakdown:
    dimension: str
    key: str

    trade_count: int
    winning_trades: int
    losing_trades: int

    net_pnl: Decimal
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
