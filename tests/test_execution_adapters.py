from __future__ import annotations

from datetime import datetime, timezone
from inspect import getsource
from unittest.mock import MagicMock

import pytest

from src.automation import (
    AutomatedPaperExecutionEngine,
)
from src.execution_adapters import (
    ExecutionAdapter,
    ExecutionAdapterDescriptor,
    ExecutionAdapterType,
    ExecutionEnvironment,
    InternalPaperExecutionAdapter,
    LiveTradingDisabledError,
    validate_paper_only_descriptor,
)
from src.paper import PaperExitReason


T0 = datetime(
    2026,
    8,
    3,
    20,
    0,
    tzinfo=timezone.utc,
)


def make_adapter():
    repository = MagicMock()
    service = MagicMock()

    adapter = InternalPaperExecutionAdapter(
        paper_repository=repository,
        paper_service=service,
    )

    return adapter, repository, service


def test_internal_descriptor_is_paper_only() -> None:
    adapter, _, _ = make_adapter()

    descriptor = adapter.descriptor

    assert (
        descriptor.adapter_type
        is ExecutionAdapterType.INTERNAL
    )

    assert (
        descriptor.environment
        is ExecutionEnvironment.INTERNAL_PAPER
    )

    assert descriptor.live_trading_enabled is False


def test_live_environment_is_rejected() -> None:
    descriptor = ExecutionAdapterDescriptor(
        adapter_id="live-test",
        adapter_type=ExecutionAdapterType.BROKER,
        environment=ExecutionEnvironment.LIVE,
    )

    with pytest.raises(
        LiveTradingDisabledError,
        match="Live execution environments",
    ):
        validate_paper_only_descriptor(
            descriptor
        )


def test_live_trading_flag_is_rejected() -> None:
    descriptor = ExecutionAdapterDescriptor(
        adapter_id="unsafe-paper",
        adapter_type=ExecutionAdapterType.BROKER,
        environment=(
            ExecutionEnvironment.BROKER_PAPER
        ),
        live_trading_enabled=True,
    )

    with pytest.raises(
        LiveTradingDisabledError,
        match="must not enable live trading",
    ):
        validate_paper_only_descriptor(
            descriptor
        )


def test_internal_adapter_satisfies_protocol() -> None:
    adapter, _, _ = make_adapter()

    assert isinstance(
        adapter,
        ExecutionAdapter,
    )


def test_internal_adapter_delegates_fill() -> None:
    adapter, _, service = make_adapter()

    expected = (
        object(),
        object(),
    )

    service.record_automatic_buy_fill.return_value = (
        expected
    )

    result = adapter.record_buy_fill(
        order_id="ORDER-1",
        fill_price="100",
        fees="1",
        slippage="0.5",
        filled_at=T0,
    )

    assert result == expected

    service.record_automatic_buy_fill.assert_called_once_with(
        order_id="ORDER-1",
        fill_price="100",
        fees="1",
        slippage="0.5",
        filled_at=T0,
    )


def test_internal_adapter_delegates_other_lifecycle() -> None:
    adapter, repository, service = (
        make_adapter()
    )

    cancelled = object()
    expired = object()
    closed = object()

    service.cancel_pending_order.return_value = (
        cancelled
    )

    repository.expire_order.return_value = (
        expired
    )

    service.close_automatic_position.return_value = (
        closed
    )

    assert adapter.cancel_order(
        order_id="ORDER-1",
        reason="Risk blocked.",
        cancelled_at=T0,
    ) is cancelled

    assert adapter.expire_order(
        "ORDER-2",
        expired_at=T0,
        reason="Expired.",
    ) is expired

    assert adapter.close_position(
        position_id="POSITION-1",
        exit_price="110",
        exit_reason=PaperExitReason.TARGET,
        exit_fees="1",
        exit_slippage="0.5",
        closed_at=T0,
    ) is closed

    service.cancel_pending_order.assert_called_once_with(
        order_id="ORDER-1",
        reason="Risk blocked.",
        cancelled_at=T0,
    )

    repository.expire_order.assert_called_once_with(
        "ORDER-2",
        expired_at=T0,
        reason="Expired.",
    )

    service.close_automatic_position.assert_called_once_with(
        position_id="POSITION-1",
        exit_price="110",
        exit_reason=PaperExitReason.TARGET,
        exit_fees="1",
        exit_slippage="0.5",
        closed_at=T0,
    )


def test_engine_builds_internal_adapter_by_default() -> None:
    engine = AutomatedPaperExecutionEngine(
        paper_repository=MagicMock(),
        paper_service=MagicMock(),
        scanner_repository=MagicMock(),
        automation_repository=MagicMock(),
    )

    assert isinstance(
        engine.execution_adapter,
        InternalPaperExecutionAdapter,
    )


def test_engine_routes_lifecycle_through_adapter() -> None:
    source = getsource(
        AutomatedPaperExecutionEngine
    )

    required_adapter_calls = (
        "self.execution_adapter.close_position(",
        "self.execution_adapter.cancel_order(",
        "self.execution_adapter.record_buy_fill(",
        "self.execution_adapter.expire_order(",
    )

    for call in required_adapter_calls:
        assert call in source

    forbidden_direct_calls = (
        "self.paper_service.close_automatic_position(",
        "self.paper_service.cancel_pending_order(",
        "self.paper_service.record_automatic_buy_fill(",
        "self.paper_repository.expire_order(",
    )

    for call in forbidden_direct_calls:
        assert call not in source
