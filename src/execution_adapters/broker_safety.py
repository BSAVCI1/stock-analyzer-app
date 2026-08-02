"""Safety validation for broker paper endpoints."""

from __future__ import annotations

from dataclasses import replace
import re
from urllib.parse import (
    urlsplit,
    urlunsplit,
)

from .broker import (
    BrokerPaperConnectionConfig,
)
from .models import (
    ExecutionAdapterDescriptor,
    ExecutionAdapterType,
    ExecutionEnvironment,
)
from .safety import (
    LiveTradingDisabledError,
    validate_paper_only_descriptor,
)


class BrokerEndpointSafetyError(
    LiveTradingDisabledError
):
    """Raised when an endpoint is not demonstrably paper-only."""


PAPER_MARKERS = {
    "paper",
    "sandbox",
    "demo",
    "sim",
    "simulation",
}

LIVE_MARKERS = {
    "live",
    "prod",
    "production",
}

LOCAL_HOSTS = {
    "localhost",
    "127.0.0.1",
    "::1",
}


def validate_broker_paper_config(
    config: BrokerPaperConnectionConfig,
) -> BrokerPaperConnectionConfig:
    """Validate and normalize a broker-paper connection."""

    provider = config.provider.strip()
    account_id = config.account_id.strip()
    api_key = config.api_key.strip()

    if not provider:
        raise ValueError(
            "Broker provider is required."
        )

    if not account_id:
        raise ValueError(
            "Broker paper account ID is required."
        )

    if not api_key:
        raise ValueError(
            "Broker paper API key is required."
        )

    if config.timeout_seconds <= 0:
        raise ValueError(
            "Broker timeout must be positive."
        )

    parsed = urlsplit(
        config.base_url.strip()
    )

    host = (
        parsed.hostname or ""
    ).lower()

    if not host:
        raise BrokerEndpointSafetyError(
            "Broker base URL must include a host."
        )

    is_local = host in LOCAL_HOSTS

    if (
        parsed.scheme.lower() != "https"
        and not is_local
    ):
        raise BrokerEndpointSafetyError(
            "Remote broker-paper endpoints "
            "must use HTTPS."
        )

    if (
        parsed.username is not None
        or parsed.password is not None
    ):
        raise BrokerEndpointSafetyError(
            "Credentials must not be embedded "
            "in the broker URL."
        )

    if parsed.query or parsed.fragment:
        raise BrokerEndpointSafetyError(
            "Broker base URL must not include "
            "a query string or fragment."
        )

    tokens = {
        token
        for token in re.split(
            r"[.\-_/]+",
            (
                host
                + parsed.path.lower()
            ),
        )
        if token
    }

    if tokens & LIVE_MARKERS:
        raise BrokerEndpointSafetyError(
            "Live or production broker endpoints "
            "are disabled."
        )

    if (
        not is_local
        and not (
            tokens & PAPER_MARKERS
        )
    ):
        raise BrokerEndpointSafetyError(
            "Broker endpoint must be explicitly "
            "identified as paper, sandbox, demo "
            "or simulation."
        )

    path = parsed.path.rstrip("/")

    normalized_url = urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            "",
            "",
        )
    )

    return replace(
        config,
        provider=provider,
        base_url=normalized_url,
        account_id=account_id,
        api_key=api_key,
        api_secret=(
            config.api_secret.strip()
            if config.api_secret
            else None
        ),
    )


def broker_paper_descriptor(
    config: BrokerPaperConnectionConfig,
) -> ExecutionAdapterDescriptor:
    """Build a validated broker-paper descriptor."""

    safe = validate_broker_paper_config(
        config
    )

    descriptor = ExecutionAdapterDescriptor(
        adapter_id=(
            f"{safe.provider.lower()}-paper"
        ),
        adapter_type=(
            ExecutionAdapterType.BROKER
        ),
        environment=(
            ExecutionEnvironment.BROKER_PAPER
        ),
        provider=safe.provider,
        live_trading_enabled=False,
        supports_account_reconciliation=True,
        supports_order_reconciliation=True,
        supports_position_reconciliation=True,
        metadata={
            "base_url": safe.base_url,
            "broker_account_id":
            safe.account_id,
            "paper_only": True,
        },
    )

    return validate_paper_only_descriptor(
        descriptor
    )
