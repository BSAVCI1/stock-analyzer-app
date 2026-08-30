"""P4 immutable horizon validation dataset tests."""

from datetime import datetime, timezone
import json

import pandas as pd

from src.data.market_data import MarketSnapshot
from src.jobs.cli import main
from src.p4_validation_dataset import capture_horizon_dataset


def _loader(calls):
    def load(symbol, **kwargs):
        calls.append((symbol, kwargs))
        periods = 252 if kwargs["interval"] == "1d" else 156
        index = pd.date_range("2020-01-01", periods=periods, freq="D", tz="UTC")
        frame = pd.DataFrame({
            "Open": range(1, periods + 1),
            "High": range(2, periods + 2),
            "Low": [value + 0.5 for value in range(periods)],
            "Close": [value + 1.5 for value in range(periods)],
            "Volume": [1000] * periods,
        }, index=index)
        return MarketSnapshot(
            symbol=symbol,
            history=frame,
            metadata={"symbol": symbol},
            fetched_at_utc=datetime(2026, 8, 30, tzinfo=timezone.utc),
            warnings=(),
        )
    return load


def _capture(horizon="swing"):
    calls = []
    result = capture_horizon_dataset(
        horizon=horizon,
        symbols=["MSFT", "AAPL", "AAPL"],
        captured_at=datetime(2026, 8, 30, 8, tzinfo=timezone.utc),
        loader=_loader(calls),
        policy_version="p4.3-1",
        universe_policy_version="p4.4-universe-v1",
    )
    return result, calls


def test_swing_capture_is_sorted_deduplicated_and_fingerprinted() -> None:
    result, calls = _capture()
    assert result["dataset_id"].startswith("sha256:")
    assert result["schema_version"] == 2
    assert result["strategy_version"] == "p4.3-swing-v1"
    assert [item["symbol"] for item in result["instruments"]] == ["AAPL", "MSFT"]
    assert all(item["row_count"] == 252 for item in result["instruments"])
    assert result["fx"]["symbol"] == "USDEUR=X"
    assert result["fx"]["row_count"] == 252
    assert len(calls) == 3
    assert all(call[1] == {"period": "2y", "interval": "1d", "min_rows": 252} for call in calls)


def test_medium_term_capture_uses_independent_weekly_history() -> None:
    result, calls = _capture("medium_term")
    assert result["strategy_version"] == "p4.3-medium-term-v1"
    assert len(calls) == 3
    assert all(call[1] == {"period": "5y", "interval": "1wk", "min_rows": 156} for call in calls)


def test_same_observations_produce_same_dataset_id() -> None:
    first, _ = _capture()
    second, _ = _capture()
    assert first["dataset_id"] == second["dataset_id"]


def test_horizons_never_share_a_dataset_identity() -> None:
    swing, _ = _capture("swing")
    medium, _ = _capture("medium_term")
    assert swing["dataset_id"] != medium["dataset_id"]


def test_invalid_horizon_fails_closed() -> None:
    try:
        _capture("intraday")
    except ValueError as exc:
        assert "swing or medium_term" in str(exc)
    else:
        raise AssertionError("intraday dataset must be rejected")


def test_cli_uses_versioned_policy_and_universe(tmp_path, monkeypatch, capsys) -> None:
    policy = tmp_path / "policy.json"
    universe = tmp_path / "universe.json"
    policy.write_text(json.dumps({
        "policy_version": "p4.3-1", "portfolio": {"currency": "EUR"}
    }))
    universe.write_text(json.dumps({
        "policy_version": "universe-v1",
        "base_symbols": ["AAPL", "MSFT"],
        "include_symbols": ["NVDA"],
        "exclude_symbols": ["MSFT"],
    }))
    calls = []
    monkeypatch.setattr("src.jobs.cli.load_market_snapshot", _loader(calls))
    assert main([
        "p4-capture-validation-dataset",
        "--horizon", "swing",
        "--policy", str(policy),
        "--universe", str(universe),
        "--captured-at", "2026-08-30T08:00:00+00:00",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["symbol"] for item in payload["instruments"]] == ["AAPL", "NVDA"]
    assert payload["policy_version"] == "p4.3-1"
    assert payload["universe_policy_version"] == "universe-v1"
