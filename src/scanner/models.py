"""Domain models for deterministic automatic market scans."""

from __future__ import annotations

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
    """Named, deterministic collection of symbols."""

    name: str
    symbols: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        name = _required_text(
            "name",
            self.name,
        )

        normalised: list[str] = []
        seen: set[str] = set()

        for raw_symbol in self.symbols:
            symbol = normalise_symbol(raw_symbol)

            if symbol in seen:
                continue

            seen.add(symbol)
            normalised.append(symbol)

        if not normalised:
            raise ValueError(
                "Stock universe cannot be empty."
            )

        object.__setattr__(
            self,
            "name",
            name,
        )

        object.__setattr__(
            self,
            "symbols",
            tuple(normalised),
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
