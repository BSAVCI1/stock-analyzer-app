"""Provider-neutral broker paper transport contracts."""

from __future__ import annotations

from datetime import datetime
from typing import (
    Protocol,
    runtime_checkable,
)

from .broker import (
    BrokerAccountSnapshot,
    BrokerOrderRequest,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
)
from .models import (
    ExecutionAdapterDescriptor,
)


@runtime_checkable
class BrokerPaperSnapshotTransport(
    Protocol,
):
    """Read-only broker-paper snapshot operations."""

    @property
    def descriptor(
        self,
    ) -> ExecutionAdapterDescriptor:
        ...

    def get_account_snapshot(
        self,
    ) -> BrokerAccountSnapshot:
        ...

    def list_order_snapshots(
        self,
    ) -> tuple[
        BrokerOrderSnapshot,
        ...,
    ]:
        ...

    def list_position_snapshots(
        self,
    ) -> tuple[
        BrokerPositionSnapshot,
        ...,
    ]:
        ...


@runtime_checkable
class BrokerPaperTransport(
    BrokerPaperSnapshotTransport,
    Protocol,
):
    """Full broker-paper transport contract."""

    def submit_order(
        self,
        request: BrokerOrderRequest,
    ) -> BrokerOrderSnapshot:
        ...

    def cancel_order(
        self,
        broker_order_id: str,
        *,
        cancelled_at: datetime,
    ) -> BrokerOrderSnapshot:
        ...
