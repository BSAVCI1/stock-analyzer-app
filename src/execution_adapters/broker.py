"""Provider-neutral broker paper-account models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Mapping


class BrokerOrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class BrokerPositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class BrokerOrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class BrokerPaperConnectionConfig:
    """Paper-only broker connection settings."""

    provider: str
    base_url: str
    account_id: str

    api_key: str = field(
        repr=False,
    )

    api_secret: str | None = field(
        default=None,
        repr=False,
    )

    timeout_seconds: float = 15.0


@dataclass(frozen=True, slots=True)
class BrokerAccountSnapshot:
    provider_account_id: str
    currency: str

    cash: Decimal
    buying_power: Decimal
    equity: Decimal

    captured_at: datetime

    metadata: Mapping[
        str,
        object,
    ] = field(
        default_factory=dict,
    )


@dataclass(frozen=True, slots=True)
class BrokerOrderRequest:
    client_order_id: str
    symbol: str

    side: BrokerOrderSide
    quantity: Decimal

    submitted_at: datetime

    limit_price: Decimal | None = None
    stop_price: Decimal | None = None

    metadata: Mapping[
        str,
        object,
    ] = field(
        default_factory=dict,
    )


@dataclass(frozen=True, slots=True)
class BrokerOrderSnapshot:
    broker_order_id: str
    client_order_id: str
    symbol: str

    side: BrokerOrderSide
    status: BrokerOrderStatus

    quantity: Decimal
    filled_quantity: Decimal

    submitted_at: datetime
    updated_at: datetime

    average_fill_price: Decimal | None = None

    metadata: Mapping[
        str,
        object,
    ] = field(
        default_factory=dict,
    )


@dataclass(frozen=True, slots=True)
class BrokerPositionSnapshot:
    broker_position_id: str
    symbol: str

    side: BrokerPositionSide
    quantity: Decimal

    average_entry_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal

    captured_at: datetime

    metadata: Mapping[
        str,
        object,
    ] = field(
        default_factory=dict,
    )
