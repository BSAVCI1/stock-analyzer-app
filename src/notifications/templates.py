"""Deterministic notification rendering."""

from __future__ import annotations

import json
from typing import Mapping

from src.paper import (
    NotificationChannel,
    NotificationRecord,
)

from .models import RenderedNotification


_SENSITIVE_FRAGMENTS = (
    "password",
    "secret",
    "token",
    "credential",
    "api_key",
    "private_key",
)


def _redact(value):
    if isinstance(value, Mapping):
        result = {}

        for key, nested in value.items():
            normalised = (
                str(key).strip().lower()
            )

            if any(
                fragment in normalised
                for fragment
                in _SENSITIVE_FRAGMENTS
            ):
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = _redact(
                    nested
                )

        return result

    if isinstance(value, (list, tuple)):
        return [
            _redact(item)
            for item in value
        ]

    return value


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
    payload = _redact(
        notification.payload
    )

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

        if (
            notification.channel
            is NotificationChannel.TELEGRAM
        ):
            targets = ", ".join(
                str(value)
                for value
                in payload.get("targets", [])
            )

            return RenderedNotification(
                subject=(
                    f"Paper buy executed: "
                    f"{symbol}"
                ),
                text=(
                    f"[PAPER BUY] {symbol} | "
                    f"Qty {quantity} @ {fill_price} | "
                    f"Stop {_value(payload, 'stop_price')} | "
                    f"Targets {targets or 'unknown'}"
                ),
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

        if (
            notification.channel
            is NotificationChannel.TELEGRAM
        ):
            return RenderedNotification(
                subject=(
                    f"Paper sell executed: "
                    f"{symbol}"
                ),
                text=(
                    f"[PAPER SELL] {symbol} | "
                    f"Qty {_value(payload, 'quantity')} | "
                    f"Exit {_value(payload, 'exit_price')} | "
                    f"{_value(payload, 'exit_reason')} | "
                    f"Net P&L {_value(payload, 'net_pnl')}"
                ),
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

    if (
        notification.channel
        is NotificationChannel.TELEGRAM
    ):
        summary = _value(
            payload,
            "message",
            _value(
                payload,
                "error",
                _value(
                    payload,
                    "reason",
                    "See persisted delivery evidence.",
                ),
            ),
        )

        return RenderedNotification(
            subject=readable_event,
            text=(
                f"[{readable_event}] {summary}"
            ),
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
