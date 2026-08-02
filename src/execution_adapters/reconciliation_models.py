"""Persisted broker-paper reconciliation models."""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from datetime import datetime
from enum import Enum
from typing import Mapping


class BrokerReconciliationRunStatus(
    str,
    Enum,
):
    RUNNING = "RUNNING"
    MATCHED = "MATCHED"
    DIFFERENCES = "DIFFERENCES"
    FAILED = "FAILED"


class BrokerReconciliationCategory(
    str,
    Enum,
):
    ACCOUNT = "ACCOUNT"
    ORDER = "ORDER"
    POSITION = "POSITION"


class BrokerReconciliationItemStatus(
    str,
    Enum,
):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    MISSING_INTERNAL = "MISSING_INTERNAL"
    MISSING_BROKER = "MISSING_BROKER"


@dataclass(frozen=True, slots=True)
class BrokerReconciliationRun:
    reconciliation_run_id: str
    account_id: str
    reconciliation_key: str

    provider: str
    broker_account_id: str

    status: BrokerReconciliationRunStatus

    started_at: datetime
    completed_at: datetime | None

    account_item_count: int
    order_item_count: int
    position_item_count: int

    matched_item_count: int
    mismatched_item_count: int
    missing_internal_item_count: int
    missing_broker_item_count: int

    metadata: Mapping[
        str,
        object,
    ] = field(
        default_factory=dict,
    )

    error_message: str | None = None

    @property
    def reconciled(self) -> bool:
        return (
            self.status
            is BrokerReconciliationRunStatus
            .MATCHED
        )

    @property
    def unresolved_item_count(self) -> int:
        return (
            self.mismatched_item_count
            + self.missing_internal_item_count
            + self.missing_broker_item_count
        )


@dataclass(frozen=True, slots=True)
class BrokerReconciliationItem:
    reconciliation_item_id: str
    reconciliation_run_id: str
    account_id: str

    category: BrokerReconciliationCategory
    comparison_key: str

    status: BrokerReconciliationItemStatus

    internal_reference_ids: tuple[
        str,
        ...,
    ]

    broker_reference_ids: tuple[
        str,
        ...,
    ]

    differences: Mapping[
        str,
        object,
    ]

    message: str

    created_at: datetime

    metadata: Mapping[
        str,
        object,
    ] = field(
        default_factory=dict,
    )


@dataclass(frozen=True, slots=True)
class BrokerReconciliationReport:
    run: BrokerReconciliationRun

    items: tuple[
        BrokerReconciliationItem,
        ...,
    ]

    duplicate: bool = False

    @property
    def reconciled(self) -> bool:
        return self.run.reconciled

    @property
    def unresolved_items(
        self,
    ) -> tuple[
        BrokerReconciliationItem,
        ...,
    ]:
        return tuple(
            item
            for item in self.items
            if item.status
            is not
            BrokerReconciliationItemStatus
            .MATCH
        )
