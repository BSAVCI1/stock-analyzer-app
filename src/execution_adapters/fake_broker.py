"""Deterministic in-memory broker-paper transport."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from decimal import Decimal

from .broker import (
    BrokerAccountSnapshot,
    BrokerOrderRequest,
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerPaperConnectionConfig,
    BrokerPositionSnapshot,
)
from .broker_safety import (
    broker_paper_descriptor,
    validate_broker_paper_config,
)


def _aware(
    value: datetime,
    *,
    field_name: str,
) -> datetime:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            f"{field_name} must be "
            "timezone-aware."
        )

    return value


class InMemoryBrokerPaperTransport:
    """Paper transport with no network or broker dependency."""

    def __init__(
        self,
        *,
        config:
        BrokerPaperConnectionConfig,
        account:
        BrokerAccountSnapshot,
        orders: tuple[
            BrokerOrderSnapshot,
            ...,
        ] = (),
        positions: tuple[
            BrokerPositionSnapshot,
            ...,
        ] = (),
    ) -> None:
        self.config = (
            validate_broker_paper_config(
                config
            )
        )

        self._descriptor = (
            broker_paper_descriptor(
                self.config
            )
        )

        self._account = account

        self._orders = {
            order.broker_order_id: order
            for order in orders
        }

        self._positions = {
            position.broker_position_id:
            position
            for position in positions
        }

    @property
    def descriptor(self):
        return self._descriptor

    def get_account_snapshot(
        self,
    ) -> BrokerAccountSnapshot:
        return self._account

    def list_order_snapshots(
        self,
    ) -> tuple[
        BrokerOrderSnapshot,
        ...,
    ]:
        return tuple(
            sorted(
                self._orders.values(),
                key=lambda order: (
                    order.submitted_at,
                    order.broker_order_id,
                ),
            )
        )

    def list_position_snapshots(
        self,
    ) -> tuple[
        BrokerPositionSnapshot,
        ...,
    ]:
        return tuple(
            sorted(
                self._positions.values(),
                key=lambda position: (
                    position.symbol,
                    position
                    .broker_position_id,
                ),
            )
        )

    def submit_order(
        self,
        request: BrokerOrderRequest,
    ) -> BrokerOrderSnapshot:
        submitted_at = _aware(
            request.submitted_at,
            field_name="submitted_at",
        )

        if request.quantity <= Decimal("0"):
            raise ValueError(
                "Broker order quantity must "
                "be positive."
            )

        for order in self._orders.values():
            if (
                order.client_order_id
                == request.client_order_id
            ):
                return order

        broker_order_id = (
            f"{self.config.provider.upper()}"
            f"-PAPER-"
            f"{len(self._orders) + 1:06d}"
        )

        order = BrokerOrderSnapshot(
            broker_order_id=broker_order_id,
            client_order_id=(
                request.client_order_id
            ),
            symbol=request.symbol,
            side=request.side,
            status=BrokerOrderStatus.NEW,
            quantity=request.quantity,
            filled_quantity=Decimal("0"),
            submitted_at=submitted_at,
            updated_at=submitted_at,
            average_fill_price=None,
            metadata=dict(
                request.metadata
            ),
        )

        self._orders[
            broker_order_id
        ] = order

        return order

    def cancel_order(
        self,
        broker_order_id: str,
        *,
        cancelled_at: datetime,
    ) -> BrokerOrderSnapshot:
        at = _aware(
            cancelled_at,
            field_name="cancelled_at",
        )

        try:
            order = self._orders[
                broker_order_id
            ]
        except KeyError as exc:
            raise KeyError(
                "Unknown broker-paper order: "
                f"{broker_order_id}."
            ) from exc

        if order.status is BrokerOrderStatus.FILLED:
            raise ValueError(
                "A filled broker-paper order "
                "cannot be cancelled."
            )

        if order.status is BrokerOrderStatus.CANCELLED:
            return order

        cancelled = replace(
            order,
            status=(
                BrokerOrderStatus.CANCELLED
            ),
            updated_at=at,
        )

        self._orders[
            broker_order_id
        ] = cancelled

        return cancelled
