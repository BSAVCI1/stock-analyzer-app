"""Persistent paper-portfolio domain records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from enum import Enum
from math import isfinite
from typing import Mapping

from src.backtest import PositionSide


MONEY_QUANTUM = Decimal("0.00000001")


def money(value: object) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("Money value must be numeric.")

    try:
        result = (
            value
            if isinstance(value, Decimal)
            else Decimal(str(value))
        )
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Money value must be numeric.") from exc

    if not result.is_finite():
        raise ValueError("Money value must be finite.")

    return result.quantize(
        MONEY_QUANTUM,
        rounding=ROUND_HALF_EVEN,
    )


def positive_money(
    name: str,
    value: object,
) -> Decimal:
    result = money(value)

    if result <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )

    return result


def non_negative_money(
    name: str,
    value: object,
) -> Decimal:
    result = money(value)

    if result < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return result


def aware_datetime(
    name: str,
    value: object,
) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{name} must be timezone-aware."
        )

    return value


def finite_number(
    name: str,
    value: object,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc

    if not isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


class AccountStatus(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    CLOSED = "CLOSED"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    CLOSED = "CLOSED"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class NotificationStatus(str, Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"


class NotificationChannel(str, Enum):
    INTERNAL = "INTERNAL"
    EMAIL = "EMAIL"
    TELEGRAM = "TELEGRAM"


class PaperExitReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TARGET = "TARGET"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    TIME_EXIT = "TIME_EXIT"
    REGIME_INVALIDATION = "REGIME_INVALIDATION"
    PORTFOLIO_RISK = "PORTFOLIO_RISK"
    MANUAL = "MANUAL"
    SYSTEM = "SYSTEM"


@dataclass(frozen=True, slots=True)
class PaperAccount:
    account_id: str
    name: str
    base_currency: str

    starting_balance: Decimal
    cash_balance: Decimal
    reserved_cash: Decimal

    status: AccountStatus
    created_at: datetime
    updated_at: datetime

    @property
    def available_cash(self) -> Decimal:
        return money(
            self.cash_balance
            - self.reserved_cash
        )


@dataclass(frozen=True, slots=True)
class PersistedSignal:
    signal_id: str
    account_id: str
    scan_id: str | None

    symbol: str
    generated_at: datetime
    expires_at: datetime

    strategy: str
    recommendation: str
    market_regime: str

    score: float
    confidence: float
    reward_to_risk: float

    entry_low: Decimal
    entry_high: Decimal
    stop_price: Decimal
    targets: tuple[Decimal, ...]

    evidence: tuple[str, ...]
    conflicts: tuple[str, ...]

    threshold_version: str
    app_version: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class PaperOrderRecord:
    order_id: str
    account_id: str
    signal_id: str
    idempotency_key: str

    symbol: str
    side: PositionSide
    quantity: Decimal

    entry_low: Decimal
    entry_high: Decimal
    stop_price: Decimal
    targets: tuple[Decimal, ...]

    estimated_cash_required: Decimal
    reserved_cash: Decimal

    status: OrderStatus
    created_at: datetime
    expires_at: datetime
    filled_at: datetime | None
    closed_at: datetime | None


@dataclass(frozen=True, slots=True)
class PaperFillRecord:
    fill_id: str
    order_id: str

    price: Decimal
    quantity: Decimal
    fees: Decimal
    slippage: Decimal

    filled_at: datetime


@dataclass(frozen=True, slots=True)
class PaperPositionRecord:
    position_id: str
    account_id: str
    order_id: str
    fill_id: str

    symbol: str
    side: PositionSide
    quantity: Decimal

    entry_price: Decimal
    stop_price: Decimal
    targets: tuple[Decimal, ...]

    opened_at: datetime
    expires_at: datetime
    status: PositionStatus
    closed_at: datetime | None


@dataclass(frozen=True, slots=True)
class ClosedPaperTrade:
    trade_id: str
    position_id: str
    account_id: str
    order_id: str
    fill_id: str
    signal_id: str

    symbol: str
    strategy: str
    market_regime: str

    entry_time: datetime
    entry_price: Decimal
    exit_time: datetime
    exit_price: Decimal
    exit_reason: PaperExitReason

    quantity: Decimal
    gross_pnl: Decimal
    fees: Decimal
    slippage: Decimal
    net_pnl: Decimal
    return_pct: float
    holding_seconds: int


@dataclass(frozen=True, slots=True)
class NotificationRecord:
    notification_id: str
    account_id: str
    event_type: str
    reference_type: str
    reference_id: str

    channel: NotificationChannel
    status: NotificationStatus
    payload: Mapping[str, object]

    created_at: datetime
    sent_at: datetime | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class SystemEventRecord:
    event_id: str
    account_id: str | None
    event_type: str
    severity: str

    reference_type: str | None
    reference_id: str | None

    message: str
    metadata: Mapping[str, object]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class AccountReconciliation:
    account_id: str
    stored_cash_balance: Decimal
    ledger_cash_balance: Decimal
    difference: Decimal

    @property
    def reconciled(self) -> bool:
        return self.difference == money(0)
