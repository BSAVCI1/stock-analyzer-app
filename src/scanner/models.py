"""Domain models for deterministic automatic market scans."""

from __future__ import annotations

from src.strategy import StrategyHorizon

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Mapping

from src.analysis import PaperOrder, Signal
from src.data import normalise_symbol


def _required_text(
    name: str,
    value: object,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    result = value.strip()

    if not result:
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return result


def _aware_datetime(
    name: str,
    value: object,
) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(
            f"{name} must be a datetime."
        )

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{name} must be timezone-aware."
        )

    return value


def _finite_number(
    name: str,
    value: object,
) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be finite."
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be finite."
        ) from exc

    if not isfinite(result):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


class ScanStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_ERRORS = (
        "COMPLETED_WITH_ERRORS"
    )
    FAILED = "FAILED"


class WatchlistState(str, Enum):
    STALE = "STALE"
    REJECT = "REJECT"
    WATCH = "WATCH"
    PREPARE = "PREPARE"
    ACTIONABLE = "ACTIONABLE"


class ScanResultStatus(str, Enum):
    DATA_REJECTED = "DATA_REJECTED"
    ANALYSIS_REJECTED = "ANALYSIS_REJECTED"
    RELEASE_INELIGIBLE = (
        "RELEASE_INELIGIBLE"
    )
    WATCH = "WATCH"
    ORDER_CANDIDATE = "ORDER_CANDIDATE"
    SCAN_ERROR = "SCAN_ERROR"


@dataclass(frozen=True, slots=True)
class StockUniverse:
    """Named, deterministic collection of governed symbols."""

    name: str
    symbols: tuple[str, ...]
    description: str = ""
    policy_version: str = (
        "legacy-universe-v1"
    )
    included_symbols: tuple[str, ...] = ()
    excluded_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _required_text(
            "name",
            self.name,
        )
        policy_version = _required_text(
            "policy_version",
            self.policy_version,
        )

        def normalise(
            values: tuple[str, ...],
        ) -> tuple[str, ...]:
            result: list[str] = []
            seen: set[str] = set()

            for raw_symbol in values:
                symbol = normalise_symbol(
                    raw_symbol
                )

                if symbol in seen:
                    continue

                seen.add(symbol)
                result.append(symbol)

            return tuple(result)

        normalised = normalise(
            self.symbols
        )
        included = normalise(
            self.included_symbols
        )
        excluded = normalise(
            self.excluded_symbols
        )

        if not normalised:
            raise ValueError(
                "Stock universe cannot be empty."
            )

        if len(normalised) > 100:
            raise ValueError(
                "Stock universe cannot exceed "
                "100 symbols."
            )

        overlap = (
            set(included)
            & set(excluded)
        )

        if overlap:
            raise ValueError(
                "Included and excluded symbols "
                "must be disjoint."
            )

        leaked = (
            set(normalised)
            & set(excluded)
        )

        if leaked:
            raise ValueError(
                "Excluded symbols cannot appear "
                "in the effective universe."
            )

        object.__setattr__(
            self,
            "name",
            name,
        )
        object.__setattr__(
            self,
            "policy_version",
            policy_version,
        )
        object.__setattr__(
            self,
            "symbols",
            normalised,
        )
        object.__setattr__(
            self,
            "included_symbols",
            included,
        )
        object.__setattr__(
            self,
            "excluded_symbols",
            excluded,
        )
        object.__setattr__(
            self,
            "description",
            str(self.description or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class ScannerThresholds:
    """Approved scanner data-quality constraints."""

    minimum_history_rows: int = 200
    maximum_staleness_days: int = 7

    minimum_price: float = 5.0
    minimum_average_volume: float = 500_000.0
    minimum_average_dollar_volume: float = (
        10_000_000.0
    )

    liquidity_lookback_sessions: int = 20

    allowed_quote_types: tuple[str, ...] = (
        "EQUITY",
        "ETF",
    )

    def __post_init__(self) -> None:
        for name in (
            "minimum_history_rows",
            "maximum_staleness_days",
            "liquidity_lookback_sessions",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        for name in (
            "minimum_price",
            "minimum_average_volume",
            "minimum_average_dollar_volume",
        ):
            value = _finite_number(
                name,
                getattr(self, name),
            )

            if value <= 0:
                raise ValueError(
                    f"{name} must be greater than zero."
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        quote_types = tuple(
            _required_text(
                "allowed quote type",
                value,
            ).upper()
            for value in self.allowed_quote_types
        )

        if not quote_types:
            raise ValueError(
                "allowed_quote_types cannot be empty."
            )

        object.__setattr__(
            self,
            "allowed_quote_types",
            quote_types,
        )


@dataclass(frozen=True, slots=True)
class DataQualityMetrics:
    symbol: str
    data_as_of: datetime
    history_rows: int

    latest_price: float
    average_volume: float
    average_dollar_volume: float

    quote_type: str
    currency: str
    exchange: str

    staleness_days: int
    provider_warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScannerAnalysisOutcome:
    """Scanner-safe projection of TradingExpertReport."""

    symbol: str
    generated_at: datetime

    strategy: str
    recommendation: Signal
    score: float
    confidence: float

    market_regime: str
    regime_confidence: float

    order: PaperOrder | None
    risk_vetoes: tuple[str, ...]

    evidence: tuple[
        Mapping[str, object],
        ...,
    ]

    warnings: tuple[str, ...] = ()
    strategy_horizon: StrategyHorizon | None = None
    strategy_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "symbol",
            normalise_symbol(self.symbol),
        )

        object.__setattr__(
            self,
            "generated_at",
            _aware_datetime(
                "generated_at",
                self.generated_at,
            ),
        )

        object.__setattr__(
            self,
            "strategy",
            _required_text(
                "strategy",
                self.strategy,
            ),
        )

        if not isinstance(
            self.recommendation,
            Signal,
        ):
            raise ValueError(
                "recommendation must be a Signal."
            )

        score = _finite_number(
            "score",
            self.score,
        )

        confidence = _finite_number(
            "confidence",
            self.confidence,
        )

        regime_confidence = _finite_number(
            "regime_confidence",
            self.regime_confidence,
        )

        if not 0 <= confidence <= 1:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        if not 0 <= regime_confidence <= 1:
            raise ValueError(
                "regime_confidence must be between "
                "0 and 1."
            )

        object.__setattr__(
            self,
            "score",
            score,
        )

        object.__setattr__(
            self,
            "confidence",
            confidence,
        )

        object.__setattr__(
            self,
            "regime_confidence",
            regime_confidence,
        )

        object.__setattr__(
            self,
            "market_regime",
            _required_text(
                "market_regime",
                self.market_regime,
            ).upper(),
        )

        object.__setattr__(
            self,
            "risk_vetoes",
            tuple(
                str(value).strip()
                for value in self.risk_vetoes
                if str(value).strip()
            ),
        )

        object.__setattr__(
            self,
            "evidence",
            tuple(
                dict(value)
                for value in self.evidence
            ),
        )

        object.__setattr__(
            self,
            "warnings",
            tuple(
                str(value).strip()
                for value in self.warnings
                if str(value).strip()
            ),
        )


@dataclass(frozen=True, slots=True)
class MarketScan:
    scan_id: str
    account_id: str
    scan_key: str

    universe: str
    status: ScanStatus

    started_at: datetime
    completed_at: datetime | None

    requested_count: int
    processed_count: int
    rejected_count: int
    signal_count: int
    order_count: int

    configuration: Mapping[str, object]
    app_version: str
    error_message: str | None


@dataclass(frozen=True, slots=True)
class ScanResult:
    result_id: str
    scan_id: str
    account_id: str
    symbol: str

    status: ScanResultStatus
    processed_at: datetime

    data_as_of: datetime | None
    history_rows: int

    latest_price: float | None
    average_volume: float | None
    average_dollar_volume: float | None

    recommendation: str | None
    strategy: str | None
    score: float | None
    confidence: float | None
    market_regime: str | None
    reward_to_risk: float | None

    release_eligible: bool

    rank_score: float | None
    rank_position: int | None

    signal_id: str | None

    reasons: tuple[str, ...]
    evidence: tuple[
        Mapping[str, object],
        ...,
    ]

    metadata: Mapping[str, object] = field(
        default_factory=dict
    )
    strategy_horizon: StrategyHorizon | None = None
    strategy_version: str | None = None
    watchlist_state: WatchlistState | None = None
    score_components: Mapping[
        str,
        float,
    ] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        state = self.watchlist_state

        if (
            state is not None
            and not isinstance(
                state,
                WatchlistState,
            )
        ):
            try:
                state = WatchlistState(
                    state
                )
            except ValueError as exc:
                raise ValueError(
                    "Unsupported watchlist_state."
                ) from exc

        components = {
            str(key): _finite_number(
                f"score_components.{key}",
                value,
            )
            for key, value
            in self.score_components.items()
        }

        if components:
            expected = {
                "analysis_score",
                "confidence",
                "reward_to_risk",
                "regime_confidence",
            }

            if set(components) != expected:
                raise ValueError(
                    "score_components contains "
                    "unexpected or missing keys."
                )

            if any(
                value < 0
                for value in components.values()
            ):
                raise ValueError(
                    "score_components cannot "
                    "be negative."
                )

            if self.rank_score is None:
                raise ValueError(
                    "score_components requires "
                    "rank_score."
                )

            if abs(
                sum(components.values())
                - self.rank_score
            ) > 0.000001:
                raise ValueError(
                    "score_components must sum "
                    "to rank_score."
                )

        object.__setattr__(
            self,
            "watchlist_state",
            state,
        )
        object.__setattr__(
            self,
            "score_components",
            components,
        )


@dataclass(frozen=True, slots=True)
class MarketScanReport:
    scan: MarketScan
    results: tuple[ScanResult, ...]

    @property
    def candidates(
        self,
    ) -> tuple[ScanResult, ...]:
        return tuple(
            sorted(
                (
                    result
                    for result in self.results
                    if result.status
                    is ScanResultStatus.ORDER_CANDIDATE
                ),
                key=lambda result: (
                    result.rank_position
                    if result.rank_position is not None
                    else 10**9,
                    result.symbol,
                ),
            )
        )
