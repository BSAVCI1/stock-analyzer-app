"""Execution-adapter identity and safety models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ExecutionAdapterType(str, Enum):
    """High-level execution-adapter implementation type."""

    INTERNAL = "INTERNAL"
    BROKER = "BROKER"


class ExecutionEnvironment(str, Enum):
    """Execution environment permitted by the application."""

    INTERNAL_PAPER = "INTERNAL_PAPER"
    BROKER_PAPER = "BROKER_PAPER"
    LIVE = "LIVE"


@dataclass(frozen=True, slots=True)
class ExecutionAdapterDescriptor:
    """Traceable capabilities and safety state of an adapter."""

    adapter_id: str
    adapter_type: ExecutionAdapterType
    environment: ExecutionEnvironment

    provider: str | None = None
    live_trading_enabled: bool = False

    supports_account_reconciliation: bool = False
    supports_order_reconciliation: bool = False
    supports_position_reconciliation: bool = False

    metadata: Mapping[str, object] | None = None
