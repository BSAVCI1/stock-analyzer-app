"""Read-only internal-to-broker-paper reconciliation."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Iterable
from uuid import uuid4

from src.paper import PaperRepository

from .broker import BrokerOrderStatus
from .broker_transport import (
    BrokerPaperSnapshotTransport,
)
from .reconciliation_models import (
    BrokerReconciliationCategory,
    BrokerReconciliationItem,
    BrokerReconciliationItemStatus,
    BrokerReconciliationReport,
)
from .reconciliation_repository import (
    BrokerReconciliationRepository,
)


ACTIVE_BROKER_ORDER_STATUSES = {
    BrokerOrderStatus.NEW,
    BrokerOrderStatus.PARTIALLY_FILLED,
}


def _utc(
    value: datetime,
    *,
    field_name: str,
) -> datetime:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} must be timezone-aware."
        )

    return value.astimezone(timezone.utc)


def _value(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)

    return str(value)


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _decimal_text(value: object) -> str:
    return format(_decimal(value), "f")


def _order_side(value: object) -> str:
    normalized = _value(value).strip().upper()

    if normalized in {"LONG", "BUY"}:
        return "BUY"

    if normalized in {"SHORT", "SELL"}:
        return "SELL"

    return normalized


def _position_side(value: object) -> str:
    normalized = _value(value).strip().upper()

    if normalized in {"LONG", "BUY"}:
        return "LONG"

    if normalized in {"SHORT", "SELL"}:
        return "SHORT"

    return normalized


def _differences(
    comparisons: dict[
        str,
        tuple[object, object],
    ],
) -> dict[str, object]:
    result: dict[str, object] = {}

    for field_name, (
        internal_value,
        broker_value,
    ) in comparisons.items():
        if internal_value != broker_value:
            result[field_name] = {
                "internal": internal_value,
                "broker": broker_value,
            }

    return result


class BrokerReconciliationService:
    """Compare persisted state with broker-paper snapshots."""

    def __init__(
        self,
        *,
        paper_repository: PaperRepository,
        reconciliation_repository:
        BrokerReconciliationRepository,
        transport:
        BrokerPaperSnapshotTransport,
    ) -> None:
        if not isinstance(
            transport,
            BrokerPaperSnapshotTransport,
        ):
            raise TypeError(
                "transport must satisfy "
                "BrokerPaperSnapshotTransport."
            )

        self.paper_repository = paper_repository
        self.reconciliation_repository = (
            reconciliation_repository
        )
        self.transport = transport

    @staticmethod
    def _item(
        *,
        run_id: str,
        account_id: str,
        category: BrokerReconciliationCategory,
        comparison_key: str,
        status: BrokerReconciliationItemStatus,
        message: str,
        created_at: datetime,
        internal_reference_ids:
        Iterable[str] = (),
        broker_reference_ids:
        Iterable[str] = (),
        differences:
        dict[str, object] | None = None,
        metadata:
        dict[str, object] | None = None,
    ) -> BrokerReconciliationItem:
        return BrokerReconciliationItem(
            reconciliation_item_id=(
                "BRI-" + uuid4().hex
            ),
            reconciliation_run_id=run_id,
            account_id=account_id,
            category=category,
            comparison_key=comparison_key,
            status=status,
            internal_reference_ids=tuple(
                internal_reference_ids
            ),
            broker_reference_ids=tuple(
                broker_reference_ids
            ),
            differences=dict(
                differences or {}
            ),
            message=message,
            created_at=created_at,
            metadata=dict(metadata or {}),
        )

    def _account_item(
        self,
        *,
        run_id: str,
        account,
        broker_account,
        created_at: datetime,
    ) -> BrokerReconciliationItem:
        differences = _differences(
            {
                "currency": (
                    str(
                        account.base_currency
                    ).upper(),
                    str(
                        broker_account.currency
                    ).upper(),
                ),
                "cash": (
                    _decimal_text(
                        account.cash_balance
                    ),
                    _decimal_text(
                        broker_account.cash
                    ),
                ),
            }
        )

        status = (
            BrokerReconciliationItemStatus.MATCH
            if not differences
            else BrokerReconciliationItemStatus
            .MISMATCH
        )

        return self._item(
            run_id=run_id,
            account_id=account.account_id,
            category=(
                BrokerReconciliationCategory
                .ACCOUNT
            ),
            comparison_key=account.account_id,
            status=status,
            internal_reference_ids=(
                account.account_id,
            ),
            broker_reference_ids=(
                broker_account.provider_account_id,
            ),
            differences=differences,
            message=(
                "Internal and broker-paper "
                "account balances match."
                if not differences
                else
                "Internal and broker-paper "
                "account values differ."
            ),
            created_at=created_at,
            metadata={
                "internal_reserved_cash":
                _decimal_text(
                    account.reserved_cash
                ),
                "broker_buying_power":
                _decimal_text(
                    broker_account.buying_power
                ),
                "broker_equity":
                _decimal_text(
                    broker_account.equity
                ),
            },
        )

    def _order_items(
        self,
        *,
        run_id: str,
        account_id: str,
        internal_orders,
        broker_orders,
        created_at: datetime,
    ) -> tuple[
        BrokerReconciliationItem,
        ...,
    ]:
        internal_by_key = {
            order.idempotency_key: order
            for order in internal_orders
        }

        broker_by_key = {
            order.client_order_id: order
            for order in broker_orders
            if order.status
            in ACTIVE_BROKER_ORDER_STATUSES
        }

        items: list[
            BrokerReconciliationItem
        ] = []

        for key in sorted(
            set(internal_by_key)
            | set(broker_by_key)
        ):
            internal = internal_by_key.get(key)
            broker = broker_by_key.get(key)

            if internal is None:
                items.append(
                    self._item(
                        run_id=run_id,
                        account_id=account_id,
                        category=(
                            BrokerReconciliationCategory
                            .ORDER
                        ),
                        comparison_key=key,
                        status=(
                            BrokerReconciliationItemStatus
                            .MISSING_INTERNAL
                        ),
                        broker_reference_ids=(
                            broker.broker_order_id,
                        ),
                        message=(
                            "Active broker-paper order "
                            "has no matching internal "
                            "pending order."
                        ),
                        created_at=created_at,
                    )
                )
                continue

            if broker is None:
                items.append(
                    self._item(
                        run_id=run_id,
                        account_id=account_id,
                        category=(
                            BrokerReconciliationCategory
                            .ORDER
                        ),
                        comparison_key=key,
                        status=(
                            BrokerReconciliationItemStatus
                            .MISSING_BROKER
                        ),
                        internal_reference_ids=(
                            internal.order_id,
                        ),
                        message=(
                            "Internal pending order "
                            "has no matching active "
                            "broker-paper order."
                        ),
                        created_at=created_at,
                    )
                )
                continue

            differences = _differences(
                {
                    "symbol": (
                        str(
                            internal.symbol
                        ).upper(),
                        str(
                            broker.symbol
                        ).upper(),
                    ),
                    "side": (
                        _order_side(
                            internal.side
                        ),
                        _order_side(
                            broker.side
                        ),
                    ),
                    "quantity": (
                        _decimal_text(
                            internal.quantity
                        ),
                        _decimal_text(
                            broker.quantity
                        ),
                    ),
                }
            )

            items.append(
                self._item(
                    run_id=run_id,
                    account_id=account_id,
                    category=(
                        BrokerReconciliationCategory
                        .ORDER
                    ),
                    comparison_key=key,
                    status=(
                        BrokerReconciliationItemStatus
                        .MATCH
                        if not differences
                        else
                        BrokerReconciliationItemStatus
                        .MISMATCH
                    ),
                    internal_reference_ids=(
                        internal.order_id,
                    ),
                    broker_reference_ids=(
                        broker.broker_order_id,
                    ),
                    differences=differences,
                    message=(
                        "Internal and broker-paper "
                        "active orders match."
                        if not differences
                        else
                        "Internal and broker-paper "
                        "active order values differ."
                    ),
                    created_at=created_at,
                    metadata={
                        "broker_status":
                        broker.status.value,
                        "broker_filled_quantity":
                        _decimal_text(
                            broker.filled_quantity
                        ),
                    },
                )
            )

        return tuple(items)

    @staticmethod
    def _aggregate_positions(
        positions,
        *,
        internal: bool,
    ) -> dict[
        tuple[str, str],
        dict[str, object],
    ]:
        groups = defaultdict(list)

        for position in positions:
            key = (
                str(
                    position.symbol
                ).upper(),
                _position_side(
                    position.side
                ),
            )
            groups[key].append(position)

        aggregates = {}

        for key, members in groups.items():
            quantity = sum(
                (
                    _decimal(
                        member.quantity
                    )
                    for member in members
                ),
                Decimal("0"),
            )

            if internal:
                total_cost = sum(
                    (
                        _decimal(
                            member.quantity
                        )
                        * _decimal(
                            member.entry_price
                        )
                        for member in members
                    ),
                    Decimal("0"),
                )

                record_ids = tuple(
                    member.position_id
                    for member in members
                )
            else:
                total_cost = sum(
                    (
                        _decimal(
                            member.quantity
                        )
                        * _decimal(
                            member
                            .average_entry_price
                        )
                        for member in members
                    ),
                    Decimal("0"),
                )

                record_ids = tuple(
                    member.broker_position_id
                    for member in members
                )

            average_entry = (
                total_cost / quantity
                if quantity
                else Decimal("0")
            )

            aggregates[key] = {
                "quantity": quantity,
                "average_entry_price":
                average_entry,
                "record_ids": record_ids,
            }

        return aggregates

    def _position_items(
        self,
        *,
        run_id: str,
        account_id: str,
        internal_positions,
        broker_positions,
        created_at: datetime,
    ) -> tuple[
        BrokerReconciliationItem,
        ...,
    ]:
        internal_groups = (
            self._aggregate_positions(
                internal_positions,
                internal=True,
            )
        )

        broker_groups = (
            self._aggregate_positions(
                broker_positions,
                internal=False,
            )
        )

        items: list[
            BrokerReconciliationItem
        ] = []

        for key in sorted(
            set(internal_groups)
            | set(broker_groups)
        ):
            symbol, side = key
            comparison_key = (
                f"{symbol}:{side}"
            )

            internal = internal_groups.get(key)
            broker = broker_groups.get(key)

            if internal is None:
                items.append(
                    self._item(
                        run_id=run_id,
                        account_id=account_id,
                        category=(
                            BrokerReconciliationCategory
                            .POSITION
                        ),
                        comparison_key=(
                            comparison_key
                        ),
                        status=(
                            BrokerReconciliationItemStatus
                            .MISSING_INTERNAL
                        ),
                        broker_reference_ids=(
                            broker["record_ids"]
                        ),
                        message=(
                            "Broker-paper position "
                            "has no matching internal "
                            "open position."
                        ),
                        created_at=created_at,
                    )
                )
                continue

            if broker is None:
                items.append(
                    self._item(
                        run_id=run_id,
                        account_id=account_id,
                        category=(
                            BrokerReconciliationCategory
                            .POSITION
                        ),
                        comparison_key=(
                            comparison_key
                        ),
                        status=(
                            BrokerReconciliationItemStatus
                            .MISSING_BROKER
                        ),
                        internal_reference_ids=(
                            internal["record_ids"]
                        ),
                        message=(
                            "Internal open position "
                            "has no matching "
                            "broker-paper position."
                        ),
                        created_at=created_at,
                    )
                )
                continue

            differences = _differences(
                {
                    "quantity": (
                        _decimal_text(
                            internal["quantity"]
                        ),
                        _decimal_text(
                            broker["quantity"]
                        ),
                    ),
                    "average_entry_price": (
                        _decimal_text(
                            internal[
                                "average_entry_price"
                            ]
                        ),
                        _decimal_text(
                            broker[
                                "average_entry_price"
                            ]
                        ),
                    ),
                }
            )

            items.append(
                self._item(
                    run_id=run_id,
                    account_id=account_id,
                    category=(
                        BrokerReconciliationCategory
                        .POSITION
                    ),
                    comparison_key=(
                        comparison_key
                    ),
                    status=(
                        BrokerReconciliationItemStatus
                        .MATCH
                        if not differences
                        else
                        BrokerReconciliationItemStatus
                        .MISMATCH
                    ),
                    internal_reference_ids=(
                        internal["record_ids"]
                    ),
                    broker_reference_ids=(
                        broker["record_ids"]
                    ),
                    differences=differences,
                    message=(
                        "Internal and broker-paper "
                        "positions match."
                        if not differences
                        else
                        "Internal and broker-paper "
                        "position values differ."
                    ),
                    created_at=created_at,
                )
            )

        return tuple(items)

    def reconcile(
        self,
        account_id: str,
        *,
        reconciliation_key: str,
        reconciled_at:
        datetime | None = None,
    ) -> BrokerReconciliationReport:
        at = _utc(
            reconciled_at
            or datetime.now(timezone.utc),
            field_name="reconciled_at",
        )

        provider = str(
            self.transport
            .descriptor
            .provider
            or ""
        ).strip()

        if not provider:
            raise ValueError(
                "Broker transport provider "
                "is required."
            )

        internal_account = (
            self.paper_repository
            .get_account(account_id)
        )

        broker_account = (
            self.transport
            .get_account_snapshot()
        )

        run, created = (
            self.reconciliation_repository
            .start_run(
                account_id=account_id,
                reconciliation_key=(
                    reconciliation_key
                ),
                provider=provider,
                broker_account_id=(
                    broker_account
                    .provider_account_id
                ),
                started_at=at,
                metadata={
                    "transport_adapter_id":
                    self.transport
                    .descriptor
                    .adapter_id,
                    "paper_only": True,
                    "read_only": True,
                },
            )
        )

        if not created:
            return BrokerReconciliationReport(
                run=run,
                items=(
                    self.reconciliation_repository
                    .list_items(
                        run.reconciliation_run_id
                    )
                ),
                duplicate=True,
            )

        try:
            internal_orders = (
                self.paper_repository
                .list_pending_orders(
                    account_id
                )
            )

            internal_positions = (
                self.paper_repository
                .list_open_positions(
                    account_id
                )
            )

            broker_orders = (
                self.transport
                .list_order_snapshots()
            )

            broker_positions = (
                self.transport
                .list_position_snapshots()
            )

            items = (
                self._account_item(
                    run_id=(
                        run.reconciliation_run_id
                    ),
                    account=internal_account,
                    broker_account=(
                        broker_account
                    ),
                    created_at=at,
                ),
                *self._order_items(
                    run_id=(
                        run.reconciliation_run_id
                    ),
                    account_id=account_id,
                    internal_orders=(
                        internal_orders
                    ),
                    broker_orders=(
                        broker_orders
                    ),
                    created_at=at,
                ),
                *self._position_items(
                    run_id=(
                        run.reconciliation_run_id
                    ),
                    account_id=account_id,
                    internal_positions=(
                        internal_positions
                    ),
                    broker_positions=(
                        broker_positions
                    ),
                    created_at=at,
                ),
            )

            completed = (
                self.reconciliation_repository
                .complete_run(
                    run.reconciliation_run_id,
                    items=items,
                    completed_at=at,
                )
            )

        except Exception as exc:
            self.reconciliation_repository.fail_run(
                run.reconciliation_run_id,
                completed_at=at,
                error_message=str(exc),
            )

            self.paper_repository.record_system_event(
                account_id=account_id,
                event_type=(
                    "BROKER_RECONCILIATION_FAILED"
                ),
                severity="ERROR",
                reference_type=(
                    "BROKER_RECONCILIATION_RUN"
                ),
                reference_id=(
                    run.reconciliation_run_id
                ),
                message=(
                    "Broker-paper reconciliation "
                    f"failed: {exc}"
                ),
                metadata={
                    "provider": provider,
                    "paper_only": True,
                },
                created_at=at,
            )

            raise

        return BrokerReconciliationReport(
            run=completed,
            items=tuple(items),
            duplicate=False,
        )
