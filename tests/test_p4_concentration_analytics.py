"""P4.10.3 persisted portfolio-concentration evidence tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.paper import PaperRepository, PaperTradingService
from src.portfolio_dashboard import calculate_concentration, concentration_rows


T0 = datetime(2026, 8, 24, 20, 0, tzinfo=timezone.utc)


def _open_position(tmp_path):
    path = tmp_path / "concentration.db"
    paper = PaperRepository(path)
    service = PaperTradingService(
        paper, app_version="test", threshold_version="threshold-test",
    )
    account = service.create_account(name="Concentration Test", created_at=T0)
    signal = service.persist_signal(
        account_id=account.account_id, signal_id="SIG-AAPL", symbol="AAPL",
        generated_at=T0, expires_at=T0 + timedelta(days=7),
        strategy="trend_pullback", recommendation="BUY",
        market_regime="BULLISH", score=80, confidence=.8,
        reward_to_risk=2, entry_low=99, entry_high=101, stop_price=95,
        targets=(110,), evidence=("fixture",), conflicts=(),
    )
    order, _ = service.create_automatic_buy(
        account_id=account.account_id, signal_id=signal.signal_id,
        quantity=10, idempotency_key="POSVAL-AAPL", estimated_fees=1,
        created_at=T0,
    )
    _, position = service.record_automatic_buy_fill(
        order_id=order.order_id, fill_price=100, fees=1, slippage=0,
        filled_at=T0 + timedelta(minutes=1),
    )
    return path, paper, account, position


def test_position_valuation_is_persistent_idempotent_and_immutable(tmp_path) -> None:
    path, paper, account, position = _open_position(tmp_path)
    captured_at = T0 + timedelta(minutes=2)
    first = paper.save_position_valuation_observation(
        account_id=account.account_id, position_id=position.position_id,
        captured_at=captured_at, quote_currency="usd", close_price="110",
        fx_rate="0.9", source="operator-verified-close",
    )
    duplicate = PaperRepository(path).save_position_valuation_observation(
        account_id=account.account_id, position_id=position.position_id,
        captured_at=captured_at, quote_currency="USD", close_price="110",
        fx_rate="0.9", source="operator-verified-close",
    )
    assert duplicate == first
    assert first.quantity == Decimal("10.00000000")
    assert first.market_value_portfolio == Decimal("990.00000000")
    with pytest.raises(ValueError, match="conflicts"):
        paper.save_position_valuation_observation(
            account_id=account.account_id, position_id=position.position_id,
            captured_at=captured_at, quote_currency="USD", close_price="111",
            fx_rate="0.9", source="operator-verified-close",
        )


def test_concentration_requires_complete_aligned_evidence() -> None:
    positions = (
        SimpleNamespace(position_id="P1"),
        SimpleNamespace(position_id="P2"),
        SimpleNamespace(position_id="P3"),
    )
    observations = (
        SimpleNamespace(observation_id="V1", position_id="P1", symbol="AAPL",
                        captured_at=T0, market_value_portfolio=Decimal("500")),
        SimpleNamespace(observation_id="V2", position_id="P2", symbol="MSFT",
                        captured_at=T0, market_value_portfolio=Decimal("300")),
        SimpleNamespace(observation_id="V3", position_id="P3", symbol="AAPL",
                        captured_at=T0, market_value_portfolio=Decimal("200")),
    )
    equity = (SimpleNamespace(
        snapshot_id="EQ1", captured_at=T0, equity=Decimal("2000"),
    ),)
    result = calculate_concentration(positions, observations, equity)
    assert result.sufficient_evidence is True
    assert result.invested_market_value == Decimal("1000")
    assert result.invested_equity_pct == 50.0
    assert result.largest_symbol == "AAPL"
    assert result.largest_symbol_weight_pct == 70.0
    assert result.top_three_weight_pct == 100.0
    assert result.hhi == pytest.approx(0.58)
    assert result.holdings[0].position_ids == ("P1", "P3")
    rows = concentration_rows(SimpleNamespace(concentration=result))
    assert rows[0]["portfolio_weight_pct"] == 70.0

    incomplete = calculate_concentration(positions, observations[:-1], equity)
    assert incomplete.sufficient_evidence is False
    assert "every open position" in incomplete.reason
