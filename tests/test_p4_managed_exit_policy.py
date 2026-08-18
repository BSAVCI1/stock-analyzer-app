from decimal import Decimal

import pytest

from src.paper import (
    PaperExitReason,
    evaluate_managed_long_exit,
)


def evaluate(**overrides):
    values = {
        "open_price": Decimal("100"),
        "high_price": Decimal("105"),
        "low_price": Decimal("95"),
        "stop_price": Decimal("90"),
        "target_price": Decimal("110"),
    }
    values.update(overrides)

    return evaluate_managed_long_exit(
        **values
    )


def test_stop_wins_when_same_bar_hits_stop_and_target():
    decision = evaluate(
        high_price=Decimal("112"),
        low_price=Decimal("88"),
    )

    assert decision is not None
    assert (
        decision.reason
        is PaperExitReason.STOP_LOSS
    )
    assert decision.exit_price == Decimal("90")


def test_gap_through_stop_exits_at_open():
    decision = evaluate(
        open_price=Decimal("85"),
        high_price=Decimal("87"),
        low_price=Decimal("84"),
    )

    assert decision is not None
    assert (
        decision.reason
        is PaperExitReason.STOP_LOSS
    )
    assert decision.exit_price == Decimal("85")


def test_gap_above_target_keeps_favourable_open():
    decision = evaluate(
        open_price=Decimal("115"),
        high_price=Decimal("118"),
        low_price=Decimal("114"),
    )

    assert decision is not None
    assert (
        decision.reason
        is PaperExitReason.TARGET
    )
    assert decision.exit_price == Decimal("115")


def test_thesis_invalidation_precedes_target():
    decision = evaluate(
        high_price=Decimal("112"),
        thesis_invalidated=True,
    )

    assert decision is not None
    assert (
        decision.reason
        is PaperExitReason.SIGNAL_REVERSAL
    )
    assert decision.exit_price == Decimal("100")


def test_regime_invalidation_is_explicit():
    decision = evaluate(
        regime_invalidated=True,
    )

    assert decision is not None
    assert (
        decision.reason
        is PaperExitReason.REGIME_INVALIDATION
    )


def test_holding_limit_exits_at_observed_open():
    decision = evaluate(
        holding_limit_reached=True,
    )

    assert decision is not None
    assert (
        decision.reason
        is PaperExitReason.TIME_EXIT
    )
    assert decision.exit_price == Decimal("100")


def test_no_trigger_keeps_position_open():
    assert evaluate() is None


def test_invalid_bar_is_rejected():
    with pytest.raises(
        ValueError,
        match="inside the session",
    ):
        evaluate(
            open_price=Decimal("106")
        )
