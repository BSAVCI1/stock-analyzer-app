"""P4 chronological, cost-aware trend-pullback replay tests."""

import pandas as pd
import pytest

from src.p4_trade_replay import ReplaySignal, replay_trend_pullback


def _dataset(*, ambiguous_entry_bar: bool = True) -> dict[str, object]:
    index = pd.date_range("2025-01-01", periods=225, freq="D", tz="UTC")
    rows = []
    for position, at in enumerate(index):
        low, high = 99.5, 100.5
        if ambiguous_entry_bar and position == 200:
            low, high = 98.0, 104.0
        rows.append({
            "at": at.isoformat(), "open": 100.0, "high": high,
            "low": low, "close": 100.0, "volume": 1000.0,
        })
    return {
        "schema_version": 2,
        "horizon": "swing",
        "instruments": [{"symbol": "TEST", "rows": rows}],
        "fx": {
            "rates": [
                {"at": at.isoformat(), "rate": 1.0}
                for at in index
            ]
        },
    }


def _one_signal(history, symbol, parameters):
    return ReplaySignal(actionable=len(history) == 200, atr=1.0)


def _one_dollar_fee(quantity, value, side):
    return 1.0


def _accept_economics(quantity, entry, stop, target, fx):
    return True


def test_entry_is_next_session_and_same_bar_ambiguity_uses_stop() -> None:
    dataset = _dataset()
    index = pd.to_datetime(
        [row["at"] for row in dataset["instruments"][0]["rows"]], utc=True
    )
    result = replay_trend_pullback(
        dataset,
        parameters={"buy_score": 75},
        test_start=index[199].to_pydatetime(),
        test_end=index[-1].to_pydatetime(),
        signal_evaluator=_one_signal,
        fee_estimator=_one_dollar_fee,
        economic_evaluator=_accept_economics,
    )
    trade = result["trades"][0]
    assert trade["signal_at"] == index[199].isoformat()
    assert trade["entry_at"] == index[200].isoformat()
    assert trade["exit_at"] == index[200].isoformat()
    assert trade["exit_reason"] == "STOP"
    assert trade["gross_pnl_eur"] == -1.5
    assert trade["execution_costs_eur"] == 2.0
    assert trade["net_pnl_eur"] == -3.5


def test_replay_never_exits_beyond_out_of_sample_end() -> None:
    dataset = _dataset(ambiguous_entry_bar=False)
    index = pd.to_datetime(
        [row["at"] for row in dataset["instruments"][0]["rows"]], utc=True
    )
    result = replay_trend_pullback(
        dataset,
        parameters={"buy_score": 75},
        test_start=index[199].to_pydatetime(),
        test_end=index[205].to_pydatetime(),
        signal_evaluator=_one_signal,
        fee_estimator=_one_dollar_fee,
        economic_evaluator=_accept_economics,
    )
    trade = result["trades"][0]
    assert trade["exit_at"] == index[205].isoformat()
    assert trade["exit_reason"] == "TIME_EXIT"


def test_historical_fx_values_entry_exit_and_fees_separately() -> None:
    dataset = _dataset(ambiguous_entry_bar=False)
    rates = dataset["fx"]["rates"]
    for position, row in enumerate(rates):
        row["rate"] = 0.9 if position <= 200 else 0.8
    index = pd.to_datetime(
        [row["at"] for row in dataset["instruments"][0]["rows"]], utc=True
    )
    result = replay_trend_pullback(
        dataset,
        parameters={"buy_score": 75},
        test_start=index[199].to_pydatetime(),
        test_end=index[205].to_pydatetime(),
        signal_evaluator=_one_signal,
        fee_estimator=_one_dollar_fee,
        economic_evaluator=_accept_economics,
    )
    trade = result["trades"][0]
    assert trade["gross_pnl_eur"] == -10.0
    assert trade["execution_costs_eur"] == pytest.approx(1.7)
    assert trade["net_pnl_eur"] == pytest.approx(-11.7)


def test_signal_evaluator_receives_only_history_available_at_signal() -> None:
    dataset = _dataset()
    seen = []

    def evaluator(history, symbol, parameters):
        seen.append(history.index[-1])
        return ReplaySignal(actionable=len(history) == 200, atr=1.0)

    index = pd.to_datetime(
        [row["at"] for row in dataset["instruments"][0]["rows"]], utc=True
    )
    replay_trend_pullback(
        dataset,
        parameters={"buy_score": 75},
        test_start=index[199].to_pydatetime(),
        test_end=index[-1].to_pydatetime(),
        signal_evaluator=evaluator,
        fee_estimator=_one_dollar_fee,
        economic_evaluator=_accept_economics,
    )
    assert seen[0] == index[199]
    assert all(at <= index[-1] for at in seen)


def test_missing_fx_history_fails_closed() -> None:
    dataset = _dataset()
    dataset["fx"]["rates"] = [{
        "at": "2030-01-01T00:00:00+00:00", "rate": 1.0
    }]
    index = pd.to_datetime(
        [row["at"] for row in dataset["instruments"][0]["rows"]], utc=True
    )
    try:
        replay_trend_pullback(
            dataset,
            parameters={"buy_score": 75},
            test_start=index[199].to_pydatetime(),
            test_end=index[-1].to_pydatetime(),
            signal_evaluator=_one_signal,
            fee_estimator=_one_dollar_fee,
            economic_evaluator=_accept_economics,
        )
    except ValueError as exc:
        assert "No historical FX rate" in str(exc)
    else:
        raise AssertionError("missing historical FX must fail")


def test_economic_gate_rejects_before_fees_or_trade_recording() -> None:
    dataset = _dataset()
    index = pd.to_datetime(
        [row["at"] for row in dataset["instruments"][0]["rows"]], utc=True
    )
    fee_calls = []

    def fee(quantity, value, side):
        fee_calls.append((quantity, value, side))
        return 1.0

    result = replay_trend_pullback(
        dataset,
        parameters={"buy_score": 75},
        test_start=index[199].to_pydatetime(),
        test_end=index[-1].to_pydatetime(),
        signal_evaluator=_one_signal,
        fee_estimator=fee,
        economic_evaluator=lambda quantity, entry, stop, target, fx: False,
    )

    assert result["trade_count"] == 0
    assert result["economic_rejection_count"] == 1
    assert result["risk_geometry_rejection_count"] == 0
    assert fee_calls == []


def test_default_gate_rejects_exact_two_risk_reward_after_costs() -> None:
    dataset = _dataset()
    index = pd.to_datetime(
        [row["at"] for row in dataset["instruments"][0]["rows"]], utc=True
    )
    result = replay_trend_pullback(
        dataset,
        parameters={"buy_score": 75},
        test_start=index[199].to_pydatetime(),
        test_end=index[-1].to_pydatetime(),
        signal_evaluator=_one_signal,
        fee_estimator=_one_dollar_fee,
    )

    assert result["trade_count"] == 0
    assert result["economic_rejection_count"] == 1
