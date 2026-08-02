"""Environment loader for optional broker-paper access."""

from __future__ import annotations

import os
from typing import Mapping

from .broker import (
    BrokerPaperConnectionConfig,
)
from .broker_safety import (
    BrokerEndpointSafetyError,
    validate_broker_paper_config,
)
from .safety import LiveTradingDisabledError


TRUE_VALUES = {
    "1",
    "true",
    "yes",
    "on",
}


def _enabled(
    value: str | None,
) -> bool:
    return (
        str(value or "")
        .strip()
        .lower()
        in TRUE_VALUES
    )


def _required(
    values: Mapping[str, str],
    key: str,
) -> str:
    value = str(
        values.get(key, "")
    ).strip()

    if not value:
        raise ValueError(
            f"{key} is required when "
            "PAPER_BROKER_ENABLED=true."
        )

    return value


def load_broker_paper_config(
    environ: Mapping[
        str,
        str,
    ] | None = None,
) -> BrokerPaperConnectionConfig | None:
    """Load a broker-paper connection only when enabled."""

    values = (
        os.environ
        if environ is None
        else environ
    )

    if not _enabled(
        values.get(
            "PAPER_BROKER_ENABLED"
        )
    ):
        return None

    environment = str(
        values.get(
            "PAPER_BROKER_ENVIRONMENT",
            "paper",
        )
    ).strip().lower()

    if environment != "paper":
        raise LiveTradingDisabledError(
            "PAPER_BROKER_ENVIRONMENT must "
            "be set to paper."
        )

    if _enabled(
        values.get(
            "PAPER_BROKER_LIVE_TRADING"
        )
    ):
        raise LiveTradingDisabledError(
            "PAPER_BROKER_LIVE_TRADING "
            "must remain false."
        )

    timeout_text = str(
        values.get(
            "PAPER_BROKER_TIMEOUT",
            "15",
        )
    ).strip()

    try:
        timeout = float(timeout_text)
    except ValueError as exc:
        raise ValueError(
            "PAPER_BROKER_TIMEOUT must "
            "be numeric."
        ) from exc

    config = BrokerPaperConnectionConfig(
        provider=_required(
            values,
            "PAPER_BROKER_PROVIDER",
        ),
        base_url=_required(
            values,
            "PAPER_BROKER_BASE_URL",
        ),
        account_id=_required(
            values,
            "PAPER_BROKER_ACCOUNT_ID",
        ),
        api_key=_required(
            values,
            "PAPER_BROKER_API_KEY",
        ),
        api_secret=(
            str(
                values.get(
                    "PAPER_BROKER_API_SECRET",
                    "",
                )
            ).strip()
            or None
        ),
        timeout_seconds=timeout,
    )

    try:
        return validate_broker_paper_config(
            config
        )
    except BrokerEndpointSafetyError:
        raise
