from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from decimal import Decimal

import pytest

from src.execution_adapters import (
    BrokerAccountSnapshot,
    BrokerEndpointSafetyError,
    BrokerOrderRequest,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerPaperConnectionConfig,
    BrokerPaperTransport,
    ExecutionEnvironment,
    InMemoryBrokerPaperTransport,
    LiveTradingDisabledError,
    load_broker_paper_config,
    validate_broker_paper_config,
)


T0 = datetime(
    2026,
    8,
    3,
    20,
    0,
    tzinfo=timezone.utc,
)


def config(
    *,
    base_url: str = (
        "https://paper-api.example.com"
    ),
) -> BrokerPaperConnectionConfig:
    return BrokerPaperConnectionConfig(
        provider="Example",
        base_url=base_url,
        account_id="BROKER-PAPER-1",
        api_key="secret-key",
        api_secret="secret-value",
    )


def account() -> BrokerAccountSnapshot:
    return BrokerAccountSnapshot(
        provider_account_id=(
            "BROKER-PAPER-1"
        ),
        currency="USD",
        cash=Decimal("10000"),
        buying_power=Decimal("20000"),
        equity=Decimal("10000"),
        captured_at=T0,
    )


def transport():
    return InMemoryBrokerPaperTransport(
        config=config(),
        account=account(),
    )


def request() -> BrokerOrderRequest:
    return BrokerOrderRequest(
        client_order_id="CLIENT-1",
        symbol="AAPL",
        side=BrokerOrderSide.BUY,
        quantity=Decimal("10"),
        submitted_at=T0,
    )


def test_paper_endpoint_is_normalized() -> None:
    result = validate_broker_paper_config(
        config(
            base_url=(
                "HTTPS://PAPER-API."
                "EXAMPLE.COM/"
            )
        )
    )

    assert result.base_url == (
        "https://paper-api.example.com"
    )


def test_live_endpoint_is_rejected() -> None:
    with pytest.raises(
        BrokerEndpointSafetyError,
        match="Live or production",
    ):
        validate_broker_paper_config(
            config(
                base_url=(
                    "https://live-api."
                    "example.com"
                )
            )
        )


def test_unmarked_endpoint_is_rejected() -> None:
    with pytest.raises(
        BrokerEndpointSafetyError,
        match="explicitly identified",
    ):
        validate_broker_paper_config(
            config(
                base_url=(
                    "https://api.example.com"
                )
            )
        )


def test_remote_http_is_rejected() -> None:
    with pytest.raises(
        BrokerEndpointSafetyError,
        match="must use HTTPS",
    ):
        validate_broker_paper_config(
            config(
                base_url=(
                    "http://paper-api."
                    "example.com"
                )
            )
        )


def test_credentials_are_redacted_from_repr() -> None:
    value = repr(config())

    assert "secret-key" not in value
    assert "secret-value" not in value


def test_disabled_loader_returns_none() -> None:
    assert (
        load_broker_paper_config({})
        is None
    )


def test_loader_rejects_live_environment() -> None:
    with pytest.raises(
        LiveTradingDisabledError,
        match="must be set to paper",
    ):
        load_broker_paper_config(
            {
                "PAPER_BROKER_ENABLED":
                "true",
                "PAPER_BROKER_ENVIRONMENT":
                "live",
            }
        )


def test_loader_builds_safe_config() -> None:
    loaded = load_broker_paper_config(
        {
            "PAPER_BROKER_ENABLED":
            "true",
            "PAPER_BROKER_ENVIRONMENT":
            "paper",
            "PAPER_BROKER_LIVE_TRADING":
            "false",
            "PAPER_BROKER_PROVIDER":
            "Example",
            "PAPER_BROKER_BASE_URL":
            "https://paper-api.example.com",
            "PAPER_BROKER_ACCOUNT_ID":
            "BROKER-PAPER-1",
            "PAPER_BROKER_API_KEY":
            "key",
            "PAPER_BROKER_API_SECRET":
            "secret",
        }
    )

    assert loaded is not None
    assert loaded.provider == "Example"
    assert loaded.account_id == (
        "BROKER-PAPER-1"
    )


def test_fake_transport_satisfies_protocol() -> None:
    value = transport()

    assert isinstance(
        value,
        BrokerPaperTransport,
    )

    assert (
        value.descriptor.environment
        is ExecutionEnvironment.BROKER_PAPER
    )

    assert (
        value.descriptor
        .live_trading_enabled
        is False
    )


def test_fake_transport_reads_account() -> None:
    value = transport()

    assert (
        value.get_account_snapshot()
        == account()
    )

    assert (
        value.list_order_snapshots()
        == ()
    )

    assert (
        value.list_position_snapshots()
        == ()
    )


def test_submission_is_idempotent() -> None:
    value = transport()

    first = value.submit_order(
        request()
    )

    second = value.submit_order(
        request()
    )

    assert first == second

    assert first.status is (
        BrokerOrderStatus.NEW
    )

    assert len(
        value.list_order_snapshots()
    ) == 1


def test_order_can_be_cancelled() -> None:
    value = transport()

    submitted = value.submit_order(
        request()
    )

    cancelled = value.cancel_order(
        submitted.broker_order_id,
        cancelled_at=T0,
    )

    assert cancelled.status is (
        BrokerOrderStatus.CANCELLED
    )

    assert (
        value.list_order_snapshots()[0]
        == cancelled
    )
