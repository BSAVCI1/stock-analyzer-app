"""Hard safety boundary for every execution adapter."""

from __future__ import annotations

from .models import (
    ExecutionAdapterDescriptor,
    ExecutionEnvironment,
)


class LiveTradingDisabledError(RuntimeError):
    """Raised whenever an adapter exposes live execution."""


def validate_paper_only_descriptor(
    descriptor: ExecutionAdapterDescriptor,
) -> ExecutionAdapterDescriptor:
    """Reject live environments and live-enabled adapters."""

    if (
        descriptor.environment
        is ExecutionEnvironment.LIVE
    ):
        raise LiveTradingDisabledError(
            "Live execution environments are "
            "disabled by application policy."
        )

    if descriptor.live_trading_enabled:
        raise LiveTradingDisabledError(
            "Execution adapters must not enable "
            "live trading."
        )

    allowed = {
        ExecutionEnvironment.INTERNAL_PAPER,
        ExecutionEnvironment.BROKER_PAPER,
    }

    if descriptor.environment not in allowed:
        raise LiveTradingDisabledError(
            "Only internal-paper and broker-paper "
            "execution environments are allowed."
        )

    return descriptor
