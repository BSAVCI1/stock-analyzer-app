"""Deterministic managed-exit policy for paper positions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .models import (
    PaperExitReason,
    money,
    positive_money,
)


@dataclass(frozen=True, slots=True)
class ManagedExitDecision:
    """One auditable instruction to close a paper position."""

    exit_price: Decimal
    reason: PaperExitReason

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exit_price",
            positive_money(
                "exit_price",
                self.exit_price,
            ),
        )

        if not isinstance(
            self.reason,
            PaperExitReason,
        ):
            raise ValueError(
                "reason must be a "
                "PaperExitReason."
            )


def evaluate_managed_long_exit(
    *,
    open_price: object,
    high_price: object,
    low_price: object,
    stop_price: object,
    target_price: object,
    holding_limit_reached: bool = False,
    legacy_expiry_reached: bool = False,
    thesis_invalidated: bool = False,
    regime_invalidated: bool = False,
) -> ManagedExitDecision | None:
    """Apply the conservative P4.6 long-exit precedence."""

    opening = positive_money(
        "open_price",
        open_price,
    )
    high = positive_money(
        "high_price",
        high_price,
    )
    low = positive_money(
        "low_price",
        low_price,
    )
    stop = positive_money(
        "stop_price",
        stop_price,
    )
    target = positive_money(
        "target_price",
        target_price,
    )

    if not low <= opening <= high:
        raise ValueError(
            "open_price must be inside the "
            "session low/high range."
        )

    if stop >= target:
        raise ValueError(
            "stop_price must be below "
            "target_price."
        )

    flags = {
        "holding_limit_reached":
            holding_limit_reached,
        "legacy_expiry_reached":
            legacy_expiry_reached,
        "thesis_invalidated":
            thesis_invalidated,
        "regime_invalidated":
            regime_invalidated,
    }

    for name, value in flags.items():
        if not isinstance(value, bool):
            raise ValueError(
                f"{name} must be boolean."
            )

    # Protective risk exits always win,
    # including a gap below the stop.
    if opening <= stop:
        return ManagedExitDecision(
            exit_price=opening,
            reason=PaperExitReason.STOP_LOSS,
        )

    if low <= stop:
        return ManagedExitDecision(
            exit_price=stop,
            reason=PaperExitReason.STOP_LOSS,
        )

    # Thesis and regime decisions are
    # executed at the next observed open.
    if thesis_invalidated:
        return ManagedExitDecision(
            exit_price=opening,
            reason=(
                PaperExitReason
                .SIGNAL_REVERSAL
            ),
        )

    if regime_invalidated:
        return ManagedExitDecision(
            exit_price=opening,
            reason=(
                PaperExitReason
                .REGIME_INVALIDATION
            ),
        )

    if opening >= target:
        return ManagedExitDecision(
            exit_price=opening,
            reason=PaperExitReason.TARGET,
        )

    if high >= target:
        return ManagedExitDecision(
            exit_price=target,
            reason=PaperExitReason.TARGET,
        )

    if (
        holding_limit_reached
        or legacy_expiry_reached
    ):
        return ManagedExitDecision(
            exit_price=opening,
            reason=PaperExitReason.TIME_EXIT,
        )

    return None
