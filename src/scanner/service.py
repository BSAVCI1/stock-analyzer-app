"""Automatic deterministic market-scanner orchestration."""

from __future__ import annotations

from dataclasses import (
    asdict,
    replace,
)
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from src.analysis import Signal
from src.backtest import P2ReleaseGateReport
from src.data import (
    MarketSnapshot,
    load_market_snapshot,
)
from src.paper import PaperTradingService
from src.strategy import (
    StrategyHorizon,
    coerce_strategy_horizon,
    normalise_strategy_version,
)

from .analysis import (
    run_deterministic_scanner_analysis,
)
from .filters import (
    evaluate_market_snapshot,
)
from .models import (
    DataQualityMetrics,
    MarketScanReport,
    ScannerAnalysisOutcome,
    ScannerThresholds,
    ScanResult,
    ScanResultStatus,
    StockUniverse,
    WatchlistState,
)
from .ranking import (
    calculate_candidate_rank,
)
from .repository import ScannerRepository


SnapshotLoader = Callable[
    [str],
    MarketSnapshot,
]

AnalysisRunner = Callable[
    [MarketSnapshot],
    ScannerAnalysisOutcome,
]

ReleaseGateLookup = Callable[
    [str],
    P2ReleaseGateReport | None,
]


def _new_result_id() -> str:
    return f"RES-{uuid4().hex}"


def _candidate_quote_currency(
    result: ScanResult,
) -> str:
    value = str(
        result.metadata.get(
            "currency"
        )
        or ""
    ).strip().upper()

    if (
        len(value) != 3
        or not value.isalpha()
    ):
        raise ValueError(
            f"{result.symbol} candidate "
            "requires a valid three-letter "
            "quote currency."
        )

    return value


def _evidence_strings(
    evidence: tuple[
        dict[str, object],
        ...,
    ],
) -> tuple[str, ...]:
    results: list[str] = []

    for item in evidence:
        code = str(
            item.get("code")
            or "EVIDENCE"
        ).strip()

        detail = ""

        for key in (
            "reason",
            "message",
            "explanation",
            "description",
        ):
            value = item.get(key)

            if value:
                detail = str(value).strip()
                break

        results.append(
            f"{code}: {detail}"
            if detail
            else code
        )

    return tuple(results)


class AutomaticMarketScanner:
    """Discover and rank paper-order candidates."""

    def __init__(
        self,
        *,
        scanner_repository: ScannerRepository,
        paper_service: PaperTradingService,
        release_gate_lookup: ReleaseGateLookup,
        thresholds: ScannerThresholds | None = None,
        snapshot_loader: SnapshotLoader | None = None,
        analysis_runner: AnalysisRunner = (
            run_deterministic_scanner_analysis
        ),
        app_version: str = "v0.3.1-p3.1",
    ) -> None:
        self.scanner_repository = (
            scanner_repository
        )

        self.paper_service = paper_service

        self.release_gate_lookup = (
            release_gate_lookup
        )

        self.thresholds = (
            thresholds
            or ScannerThresholds()
        )

        self.snapshot_loader = (
            snapshot_loader
            or (
                lambda symbol:
                load_market_snapshot(
                    symbol,
                    min_rows=(
                        self.thresholds
                        .minimum_history_rows
                    ),
                )
            )
        )

        self.analysis_runner = (
            analysis_runner
        )

        self.app_version = app_version

    def _result_from_metrics(
        self,
        *,
        scan_id: str,
        account_id: str,
        symbol: str,
        status: ScanResultStatus,
        processed_at: datetime,
        metrics: DataQualityMetrics | None,
        outcome: ScannerAnalysisOutcome | None,
        release_eligible: bool,
        rank_score: float | None,
        reasons: tuple[str, ...],
        score_components: (
            dict[str, float] | None
        ) = None,
        metadata: dict[str, object] | None = None,
    ) -> ScanResult:
        order = (
            outcome.order
            if outcome is not None
            else None
        )

        if (
            status
            is ScanResultStatus.ORDER_CANDIDATE
        ):
            watchlist_state = (
                WatchlistState.ACTIONABLE
            )
        elif (
            status
            is ScanResultStatus.RELEASE_INELIGIBLE
        ):
            watchlist_state = (
                WatchlistState.PREPARE
            )
        elif (
            status is ScanResultStatus.WATCH
        ):
            watchlist_state = (
                WatchlistState.WATCH
            )
        elif (
            status
            is ScanResultStatus.DATA_REJECTED
            and metrics is not None
            and metrics.staleness_days
            > self.thresholds
            .maximum_staleness_days
        ):
            watchlist_state = (
                WatchlistState.STALE
            )
        else:
            watchlist_state = (
                WatchlistState.REJECT
            )

        components = dict(
            score_components or {}
        )

        return ScanResult(
            result_id=_new_result_id(),
            scan_id=scan_id,
            account_id=account_id,
            symbol=symbol,
            status=status,
            processed_at=processed_at,
            data_as_of=(
                metrics.data_as_of
                if metrics is not None
                else None
            ),
            history_rows=(
                metrics.history_rows
                if metrics is not None
                else 0
            ),
            latest_price=(
                metrics.latest_price
                if metrics is not None
                else None
            ),
            average_volume=(
                metrics.average_volume
                if metrics is not None
                else None
            ),
            average_dollar_volume=(
                metrics.average_dollar_volume
                if metrics is not None
                else None
            ),
            recommendation=(
                outcome.recommendation.value
                if outcome is not None
                else None
            ),
            strategy=(
                outcome.strategy
                if outcome is not None
                else None
            ),
            score=(
                outcome.score
                if outcome is not None
                else None
            ),
            confidence=(
                outcome.confidence
                if outcome is not None
                else None
            ),
            market_regime=(
                outcome.market_regime
                if outcome is not None
                else None
            ),
            reward_to_risk=(
                order.reward_to_risk
                if order is not None
                else None
            ),
            release_eligible=(
                release_eligible
            ),
            rank_score=rank_score,
            rank_position=None,
            signal_id=None,
            reasons=reasons,
            evidence=(
                outcome.evidence
                if outcome is not None
                else ()
            ),
            watchlist_state=watchlist_state,
            score_components=components,
            metadata=(
                {
                    "watchlist_state":
                    watchlist_state.value,
                    "rank_score_components":
                    components,
                    "quote_type":
                    metrics.quote_type,
                    "currency":
                    metrics.currency,
                    "exchange":
                    metrics.exchange,
                    "staleness_days":
                    metrics.staleness_days,
                    "fractional_eligible":
                    metrics.fractional_eligible,
                    "next_event_at": (
                        metrics.next_event_at
                        .isoformat()
                        if metrics.next_event_at
                        is not None
                        else None
                    ),
                    "filter_reason_codes":
                    list(
                        metrics
                        .filter_reason_codes
                    ),
                    "provider_warnings":
                    list(
                        metrics
                        .provider_warnings
                    ),
                    "provider_load_succeeded": True,
                    "analysis_warnings":
                    list(
                        outcome.warnings
                        if outcome
                        is not None
                        else ()
                    ),
                    **dict(metadata or {}),
                }
                if metrics is not None
                else {
                    "watchlist_state":
                    watchlist_state.value,
                    "rank_score_components":
                    components,
                    **dict(metadata or {}),
                }
            ),
        )

    def run_scan(
        self,
        *,
        account_id: str,
        universe: StockUniverse,
        started_at: datetime | None = None,
        scan_key: str = "",
        strategy_horizon: (
            StrategyHorizon | str | None
        ) = None,
        strategy_version: str | None = None,
    ) -> MarketScanReport:
        horizon = coerce_strategy_horizon(
            strategy_horizon
        )
        version = normalise_strategy_version(
            strategy_version
        )

        if (
            (horizon is None)
            != (version is None)
        ):
            raise ValueError(
                "strategy_horizon and "
                "strategy_version must be "
                "provided together."
            )

        at = (
            started_at
            or datetime.now(timezone.utc)
        )

        scan, created = (
            self.scanner_repository
            .start_scan(
                account_id=account_id,
                universe=universe,
                configuration={
                    **asdict(
                        self.thresholds
                    ),
                    "strategy_horizon": (
                        horizon.value
                        if horizon is not None
                        else None
                    ),
                    "strategy_version": version,
                    "universe_policy": {
                        "policy_version": (
                            universe.policy_version
                        ),
                        "included_symbols": list(
                            universe.included_symbols
                        ),
                        "excluded_symbols": list(
                            universe.excluded_symbols
                        ),
                        "effective_count": len(
                            universe.symbols
                        ),
                    },
                },
                app_version=(
                    self.app_version
                ),
                started_at=at,
                scan_key=scan_key,
            )
        )

        if not created:
            return (
                self.scanner_repository
                .get_report(scan.scan_id)
            )

        results: list[ScanResult] = []

        outcomes: dict[
            str,
            ScannerAnalysisOutcome,
        ] = {}

        for symbol in universe.symbols:
            processed_at = datetime.now(
                timezone.utc
            )

            metrics: (
                DataQualityMetrics | None
            ) = None

            outcome: (
                ScannerAnalysisOutcome | None
            ) = None

            try:
                try:
                    snapshot = self.snapshot_loader(symbol)
                except Exception as exc:
                    results.append(
                        self._result_from_metrics(
                            scan_id=scan.scan_id,
                            account_id=account_id,
                            symbol=symbol,
                            status=ScanResultStatus.SCAN_ERROR,
                            processed_at=processed_at,
                            metrics=None,
                            outcome=None,
                            release_eligible=False,
                            rank_score=None,
                            reasons=(
                                "Market-data provider failed with "
                                f"{type(exc).__name__}: {exc}",
                            ),
                            metadata={
                                "failure_stage":
                                "MARKET_DATA_PROVIDER",
                                "provider_load_succeeded": False,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            },
                        )
                    )
                    continue

                metrics, data_reasons = (
                    evaluate_market_snapshot(
                        snapshot,
                        thresholds=(
                            self.thresholds
                        ),
                        scan_started_at=at,
                    )
                )

                if data_reasons:
                    results.append(
                        self._result_from_metrics(
                            scan_id=scan.scan_id,
                            account_id=(
                                account_id
                            ),
                            symbol=symbol,
                            status=(
                                ScanResultStatus
                                .DATA_REJECTED
                            ),
                            processed_at=(
                                processed_at
                            ),
                            metrics=metrics,
                            outcome=None,
                            release_eligible=(
                                False
                            ),
                            rank_score=None,
                            reasons=(
                                data_reasons
                            ),
                        )
                    )

                    continue

                outcome = self.analysis_runner(
                    snapshot
                )

                outcomes[symbol] = outcome

                if (
                    outcome.recommendation
                    is Signal.WATCH
                ):
                    results.append(
                        self._result_from_metrics(
                            scan_id=scan.scan_id,
                            account_id=(
                                account_id
                            ),
                            symbol=symbol,
                            status=(
                                ScanResultStatus
                                .WATCH
                            ),
                            processed_at=(
                                processed_at
                            ),
                            metrics=metrics,
                            outcome=outcome,
                            release_eligible=(
                                False
                            ),
                            rank_score=None,
                            reasons=(
                                "Recommendation is "
                                "WATCH; no order "
                                "candidate was created.",
                            ),
                        )
                    )

                    continue

                if (
                    outcome.recommendation
                    is not Signal.BUY
                ):
                    results.append(
                        self._result_from_metrics(
                            scan_id=scan.scan_id,
                            account_id=(
                                account_id
                            ),
                            symbol=symbol,
                            status=(
                                ScanResultStatus
                                .ANALYSIS_REJECTED
                            ),
                            processed_at=(
                                processed_at
                            ),
                            metrics=metrics,
                            outcome=outcome,
                            release_eligible=(
                                False
                            ),
                            rank_score=None,
                            reasons=(
                                "Recommendation is "
                                f"{outcome.recommendation.value}; "
                                "only BUY can become "
                                "an entry candidate.",
                            ),
                        )
                    )

                    continue

                if outcome.order is None:
                    reasons = (
                        outcome.risk_vetoes
                        or (
                            "BUY recommendation did "
                            "not produce an approved "
                            "paper order.",
                        )
                    )

                    results.append(
                        self._result_from_metrics(
                            scan_id=scan.scan_id,
                            account_id=(
                                account_id
                            ),
                            symbol=symbol,
                            status=(
                                ScanResultStatus
                                .ANALYSIS_REJECTED
                            ),
                            processed_at=(
                                processed_at
                            ),
                            metrics=metrics,
                            outcome=outcome,
                            release_eligible=(
                                False
                            ),
                            rank_score=None,
                            reasons=reasons,
                        )
                    )

                    continue

                if not outcome.order.paper_only:
                    results.append(
                        self._result_from_metrics(
                            scan_id=scan.scan_id,
                            account_id=(
                                account_id
                            ),
                            symbol=symbol,
                            status=(
                                ScanResultStatus
                                .ANALYSIS_REJECTED
                            ),
                            processed_at=(
                                processed_at
                            ),
                            metrics=metrics,
                            outcome=outcome,
                            release_eligible=(
                                False
                            ),
                            rank_score=None,
                            reasons=(
                                "Order is not marked "
                                "paper-only.",
                            ),
                        )
                    )

                    continue

                release_report = (
                    self.release_gate_lookup(
                        outcome.strategy
                    )
                )

                if (
                    release_report is None
                    or not bool(
                        release_report
                        .alert_scheduling_eligible
                    )
                ):
                    release_reasons = (
                        tuple(
                            release_report.reasons
                        )
                        if release_report
                        is not None
                        else (
                            "No P2 release-gate "
                            "report exists for "
                            f"{outcome.strategy}.",
                        )
                    )

                    results.append(
                        self._result_from_metrics(
                            scan_id=scan.scan_id,
                            account_id=(
                                account_id
                            ),
                            symbol=symbol,
                            status=(
                                ScanResultStatus
                                .RELEASE_INELIGIBLE
                            ),
                            processed_at=(
                                processed_at
                            ),
                            metrics=metrics,
                            outcome=outcome,
                            release_eligible=(
                                False
                            ),
                            rank_score=None,
                            reasons=(
                                release_reasons
                            ),
                        )
                    )

                    continue

                rank = (
                    calculate_candidate_rank(
                        outcome
                    )
                )

                results.append(
                    self._result_from_metrics(
                        scan_id=scan.scan_id,
                        account_id=account_id,
                        symbol=symbol,
                        status=(
                            ScanResultStatus
                            .ORDER_CANDIDATE
                        ),
                        processed_at=(
                            processed_at
                        ),
                        metrics=metrics,
                        outcome=outcome,
                        release_eligible=True,
                        rank_score=rank.total,
                        score_components=(
                            rank.as_dict()
                        ),
                        reasons=tuple(
                            release_report.reasons
                        ),
                    )
                )

            except Exception as exc:
                results.append(
                    self._result_from_metrics(
                        scan_id=scan.scan_id,
                        account_id=account_id,
                        symbol=symbol,
                        status=(
                            ScanResultStatus
                            .SCAN_ERROR
                        ),
                        processed_at=(
                            processed_at
                        ),
                        metrics=metrics,
                        outcome=outcome,
                        release_eligible=False,
                        rank_score=None,
                        reasons=(
                            "Scanner failed with "
                            f"{type(exc).__name__}: "
                            f"{exc}",
                        ),
                        metadata={
                            "failure_stage": "SCANNER_PROCESSING",
                            "provider_load_succeeded": True,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                )

        results = [
            replace(
                result,
                strategy_horizon=horizon,
                strategy_version=version,
            )
            for result in results
        ]

        candidate_results = sorted(
            (
                result
                for result in results
                if result.status
                is ScanResultStatus
                .ORDER_CANDIDATE
            ),
            key=lambda result: (
                -float(
                    result.rank_score or 0
                ),
                -float(
                    result.confidence or 0
                ),
                result.symbol,
            ),
        )

        rank_by_symbol = {
            result.symbol: position
            for position, result
            in enumerate(
                candidate_results,
                start=1,
            )
        }

        ranked_results = [
            replace(
                result,
                rank_position=(
                    rank_by_symbol.get(
                        result.symbol
                    )
                ),
            )
            for result in results
        ]

        final_results: list[ScanResult] = []

        for result in ranked_results:
            if (
                result.status
                is not ScanResultStatus
                .ORDER_CANDIDATE
            ):
                final_results.append(result)
                continue

            outcome = outcomes[result.symbol]
            order = outcome.order

            try:
                signal_id = (
                    f"SIG-{scan.scan_id}-"
                    f"{result.symbol}"
                )

                signal = (
                    self.paper_service
                    .persist_signal(
                        account_id=account_id,
                        scan_id=scan.scan_id,
                        signal_id=signal_id,
                        symbol=result.symbol,
                        quote_currency=(
                            _candidate_quote_currency(
                                result
                            )
                        ),
                        generated_at=(
                            outcome.generated_at
                        ),
                        expires_at=(
                            order.expires_at
                        ),
                        strategy=(
                            outcome.strategy
                        ),
                        strategy_horizon=horizon,
                        strategy_version=version,
                        recommendation="BUY",
                        market_regime=(
                            outcome
                            .market_regime
                        ),
                        score=outcome.score,
                        confidence=(
                            outcome.confidence
                        ),
                        reward_to_risk=(
                            order
                            .reward_to_risk
                        ),
                        entry_low=(
                            order.entry_low
                        ),
                        entry_high=(
                            order.entry_high
                        ),
                        stop_price=(
                            order.stop_price
                        ),
                        targets=(
                            order.targets
                        ),
                        evidence=(
                            _evidence_strings(
                                tuple(
                                    dict(value)
                                    for value
                                    in outcome
                                    .evidence
                                )
                            )
                        ),
                        conflicts=(
                            outcome.risk_vetoes
                        ),
                    )
                )

                final_results.append(
                    replace(
                        result,
                        signal_id=(
                            signal.signal_id
                        ),
                    )
                )

            except Exception as exc:
                final_results.append(
                    replace(
                        result,
                        status=(
                            ScanResultStatus
                            .SCAN_ERROR
                        ),
                        release_eligible=False,
                        rank_score=None,
                        rank_position=None,
                        score_components={},
                        reasons=(
                            "Candidate signal could "
                            "not be persisted: "
                            f"{type(exc).__name__}: "
                            f"{exc}",
                        ),
                    )
                )

        successful_candidates = sorted(
            (
                result
                for result in final_results
                if result.status
                is ScanResultStatus
                .ORDER_CANDIDATE
            ),
            key=lambda result: (
                -float(
                    result.rank_score or 0
                ),
                -float(
                    result.confidence or 0
                ),
                result.symbol,
            ),
        )

        final_ranks = {
            result.symbol: position
            for position, result
            in enumerate(
                successful_candidates,
                start=1,
            )
        }

        final_results = [
            replace(
                result,
                rank_position=(
                    final_ranks.get(
                        result.symbol
                    )
                    if result.status
                    is ScanResultStatus
                    .ORDER_CANDIDATE
                    else None
                ),
            )
            for result in final_results
        ]

        for result in final_results:
            self.scanner_repository.save_result(
                result
            )

        completed_at = datetime.now(
            timezone.utc
        )

        self.scanner_repository.complete_scan(
            scan.scan_id,
            completed_at=completed_at,
        )

        return (
            self.scanner_repository
            .get_report(scan.scan_id)
        )
