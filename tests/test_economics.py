from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

import pytest

from src.backtest import (
    BindingConstraint,
    ClosedTradeRecord,
    ExecutionCostModel,
    ExitReason,
    OrderRecord,
    PositionSide,
    PositionSizingConstraints,
    PositionSizingError,
    apply_entry_slippage,
    apply_exit_slippage,
    apply_position_size,
    calculate_fee,
    calculate_position_size,
    settle_trade,
    validate_order_quantity,
)


T0 = datetime(
    2026,
    7,
    31,
    20,
    0,
    tzinfo=timezone.utc,
)

EXPIRY = T0 + timedelta(days=5)


def make_order(
    *,
    side: PositionSide = PositionSide.LONG,
    quantity: float = 1.0,
) -> OrderRecord:
    if side is PositionSide.LONG:
        stop_price = 95.0
        targets = (110.0, 120.0)
    else:
        stop_price = 105.0
        targets = (90.0, 80.0)

    return OrderRecord(
        order_id="ORD-001",
        signal_id="SIG-001",
        symbol="TEST",
        side=side,
        created_at=T0,
        expires_at=EXPIRY,
        entry_low=99.0,
        entry_high=101.0,
        stop_price=stop_price,
        targets=targets,
        quantity=quantity,
    )


def make_trade(
    *,
    side: PositionSide = PositionSide.LONG,
) -> ClosedTradeRecord:
    if side is PositionSide.LONG:
        stop_price = 95.0
        targets = (110.0, 120.0)
        exit_price = 110.0
    else:
        stop_price = 105.0
        targets = (90.0, 80.0)
        exit_price = 90.0

    return ClosedTradeRecord(
        trade_id="TRADE-001",
        position_id="POS-001",
        order_id="ORD-001",
        fill_id="FILL-001",
        signal_id="SIG-001",
        symbol="TEST",
        side=side,
        opened_at=T0 + timedelta(days=1),
        closed_at=T0 + timedelta(days=2),
        expires_at=EXPIRY,
        entry_price=100.0,
        exit_price=exit_price,
        quantity=10.0,
        stop_price=stop_price,
        targets=targets,
        exit_reason=ExitReason.TARGET,
        target_index=0,
    )


def zero_costs() -> ExecutionCostModel:
    return ExecutionCostModel()


def test_fee_uses_configured_minimum() -> None:
    costs = ExecutionCostModel(
        fixed_fee="0.50",
        commission_bps="5",
        minimum_fee="2.00",
    )

    assert calculate_fee(
        Decimal("100"),
        costs,
    ) == Decimal("2.00000000")


def test_long_slippage_is_adverse() -> None:
    costs = ExecutionCostModel(
        slippage_bps="50",
    )

    assert apply_entry_slippage(
        100,
        PositionSide.LONG,
        costs,
    ) == Decimal("100.50000000")

    assert apply_exit_slippage(
        110,
        PositionSide.LONG,
        costs,
    ) == Decimal("109.45000000")


def test_short_slippage_is_adverse() -> None:
    costs = ExecutionCostModel(
        slippage_bps="50",
    )

    assert apply_entry_slippage(
        100,
        PositionSide.SHORT,
        costs,
    ) == Decimal("99.50000000")

    assert apply_exit_slippage(
        90,
        PositionSide.SHORT,
        costs,
    ) == Decimal("90.45000000")


def test_risk_budget_limits_whole_share_quantity() -> None:
    decision = calculate_position_size(
        make_order(),
        10_000,
        costs=zero_costs(),
        constraints=PositionSizingConstraints(
            risk_fraction="0.01",
            max_allocation_fraction="1",
        ),
    )

    assert decision.quantity == Decimal("16")
    assert (
        decision.binding_constraint
        is BindingConstraint.RISK
    )
    assert (
        decision.estimated_position_risk
        <= decision.risk_budget
    )


def test_allocation_budget_limits_quantity() -> None:
    decision = calculate_position_size(
        make_order(),
        10_000,
        costs=zero_costs(),
        constraints=PositionSizingConstraints(
            risk_fraction="0.50",
            max_allocation_fraction="0.10",
        ),
    )

    assert decision.quantity == Decimal("9")
    assert (
        decision.binding_constraint
        is BindingConstraint.ALLOCATION
    )
    assert (
        decision.capital_required
        <= decision.allocation_budget
    )


def test_fractional_position_sizing_uses_step() -> None:
    decision = calculate_position_size(
        make_order(),
        10_000,
        costs=zero_costs(),
        constraints=PositionSizingConstraints(
            risk_fraction="0.01",
            max_allocation_fraction="1",
            allow_fractional=True,
            quantity_step="0.1",
        ),
    )

    assert decision.quantity == Decimal("16.6")


def test_apply_position_size_returns_new_order() -> None:
    order = make_order(quantity=1)

    decision = calculate_position_size(
        order,
        10_000,
        costs=zero_costs(),
        constraints=PositionSizingConstraints(
            risk_fraction="0.01",
            max_allocation_fraction="1",
        ),
    )

    sized_order = apply_position_size(
        order,
        decision,
    )

    assert order.quantity == 1
    assert sized_order.quantity == 16
    assert sized_order.order_id == order.order_id


def test_excessive_order_quantity_is_rejected() -> None:
    order = make_order(quantity=20)

    with pytest.raises(PositionSizingError):
        validate_order_quantity(
            order,
            10_000,
            costs=zero_costs(),
            constraints=PositionSizingConstraints(
                risk_fraction="0.01",
                max_allocation_fraction="1",
            ),
        )


def test_no_positive_size_is_rejected() -> None:
    with pytest.raises(PositionSizingError):
        calculate_position_size(
            make_order(),
            100,
            costs=zero_costs(),
            constraints=PositionSizingConstraints(
                risk_fraction="0.01",
                max_allocation_fraction="1",
                minimum_cash_reserve="99",
            ),
        )


def test_long_trade_balance_reconciles_exactly() -> None:
    costs = ExecutionCostModel(
        fixed_fee="1",
        commission_bps="10",
        slippage_bps="50",
    )

    settlement = settle_trade(
        make_trade(),
        10_000,
        costs=costs,
    )

    assert (
        settlement.market_gross_pnl
        == Decimal("100.00000000")
    )
    assert (
        settlement.gross_pnl_after_slippage
        == Decimal("89.50000000")
    )
    assert (
        settlement.total_fees
        == Decimal("4.09950000")
    )
    assert (
        settlement.net_pnl
        == Decimal("85.40050000")
    )
    assert (
        settlement.ending_balance
        == Decimal("10085.40050000")
    )
    assert settlement.reconciled is True
    assert (
        settlement.reconciliation_delta
        == Decimal("0.00000000")
    )


def test_short_trade_balance_reconciles_exactly() -> None:
    costs = ExecutionCostModel(
        fixed_fee="1",
        commission_bps="10",
        slippage_bps="50",
    )

    settlement = settle_trade(
        make_trade(
            side=PositionSide.SHORT,
        ),
        10_000,
        costs=costs,
    )

    assert (
        settlement.market_gross_pnl
        == Decimal("100.00000000")
    )
    assert (
        settlement.gross_pnl_after_slippage
        == Decimal("90.50000000")
    )
    assert (
        settlement.total_fees
        == Decimal("3.89950000")
    )
    assert (
        settlement.net_pnl
        == Decimal("86.60050000")
    )
    assert (
        settlement.ending_balance
        == Decimal("10086.60050000")
    )
    assert settlement.reconciled is True


def test_costs_reduce_trade_result() -> None:
    settlement = settle_trade(
        make_trade(),
        10_000,
        costs=ExecutionCostModel(
            fixed_fee="2",
            commission_bps="15",
            transaction_fee_bps="5",
            slippage_bps="25",
        ),
    )

    assert (
        settlement.gross_pnl_after_slippage
        < settlement.market_gross_pnl
    )
    assert (
        settlement.net_pnl
        < settlement.gross_pnl_after_slippage
    )
    assert settlement.total_fees > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"risk_fraction": "0"},
        {"risk_fraction": "1.01"},
        {"max_allocation_fraction": "0"},
        {"max_allocation_fraction": "1.01"},
        {
            "allow_fractional": False,
            "quantity_step": "0.1",
        },
    ],
)
def test_invalid_sizing_constraints_are_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        PositionSizingConstraints(**kwargs)


def test_invalid_cost_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExecutionCostModel(
            fixed_fee="-1",
        )

    with pytest.raises(ValueError):
        ExecutionCostModel(
            slippage_bps="10000",
        )
