"""Deterministic notification rendering."""

from __future__ import annotations

import json
from typing import Mapping

from src.paper import NotificationRecord

from .models import RenderedNotification


def _value(
    payload: Mapping[str, object],
    name: str,
    default: str = "unknown",
) -> str:
    value = payload.get(name)

    if value is None:
        return default

    return str(value)


def render_notification(
    notification: NotificationRecord,
) -> RenderedNotification:
    payload = notification.payload

    explicit_subject = payload.get(
        "subject"
    )

    explicit_text = payload.get("text")

    if explicit_subject and explicit_text:
        return RenderedNotification(
            subject=str(
                explicit_subject
            ).strip(),
            text=str(explicit_text).strip(),
        )

    if (
        notification.event_type
        == "PAPER_BUY_EXECUTED"
    ):
        symbol = _value(
            payload,
            "symbol",
        )

        quantity = _value(
            payload,
            "quantity",
        )

        fill_price = _value(
            payload,
            "fill_price",
        )

        return RenderedNotification(
            subject=(
                f"Paper buy executed: "
                f"{symbol}"
            ),
            text=(
                f"Paper BUY executed\n"
                f"Symbol: {symbol}\n"
                f"Quantity: {quantity}\n"
                f"Fill price: {fill_price}\n"
                f"Stop: "
                f"{_value(payload, 'stop_price')}\n"
                f"Targets: "
                f"{', '.join(str(value) for value in payload.get('targets', []))}"
            ),
        )

    if (
        notification.event_type
        == "PAPER_SELL_EXECUTED"
    ):
        symbol = _value(
            payload,
            "symbol",
        )

        return RenderedNotification(
            subject=(
                f"Paper sell executed: "
                f"{symbol}"
            ),
            text=(
                f"Paper SELL executed\n"
                f"Symbol: {symbol}\n"
                f"Quantity: "
                f"{_value(payload, 'quantity')}\n"
                f"Entry: "
                f"{_value(payload, 'entry_price')}\n"
                f"Exit: "
                f"{_value(payload, 'exit_price')}\n"
                f"Reason: "
                f"{_value(payload, 'exit_reason')}\n"
                f"Net P&L: "
                f"{_value(payload, 'net_pnl')}"
            ),
        )

    readable_event = (
        notification.event_type
        .replace("_", " ")
        .title()
    )

    return RenderedNotification(
        subject=readable_event,
        text=(
            f"{readable_event}\n\n"
            + json.dumps(
                dict(payload),
                indent=2,
                sort_keys=True,
                default=str,
            )
        ),
    )
