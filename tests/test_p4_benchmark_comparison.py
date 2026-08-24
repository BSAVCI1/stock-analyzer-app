"""P4.10 persisted benchmark and nominal-cash comparison tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from types import SimpleNamespace

import pytest

from src.jobs.cli import main
from src.paper import PaperRepository
from src.portfolio_dashboard import (
    benchmark_comparison_rows,
    calculate_benchmark_comparisons,
)


T0 = datetime(2026, 8, 3, 20, 30, tzinfo=timezone.utc)


def _account(tmp_path):
    database_path = tmp_path / "benchmark.db"
    paper = PaperRepository(database_path)
    account = paper.create_account(
        name="Benchmark Test",
        base_currency="EUR",
        starting_balance="100",
    )
    return database_path, paper, account


def test_benchmark_observations_are_persistent_and_immutable(tmp_path) -> None:
    database_path, paper, account = _account(tmp_path)
    first = paper.save_benchmark_observation(
        account_id=account.account_id,
        symbol=" vwce.de ",
        captured_at=T0,
        quote_currency="eur",
        close_price="100",
        fx_rate="1",
        source="test-fixture",
    )
    duplicate = PaperRepository(database_path).save_benchmark_observation(
        account_id=account.account_id,
        symbol="VWCE.DE",
        captured_at=T0,
        quote_currency="EUR",
        close_price="100",
        fx_rate="1",
        source="test-fixture",
    )

    assert duplicate == first
    assert first.portfolio_price == Decimal("100.00000000")
    assert PaperRepository(database_path).list_benchmark_observations(
        account.account_id,
        symbol="vwce.de",
    ) == (first,)

    with pytest.raises(ValueError, match="conflicts"):
        paper.save_benchmark_observation(
            account_id=account.account_id,
            symbol="VWCE.DE",
            captured_at=T0,
            quote_currency="EUR",
            close_price="101",
            fx_rate="1",
            source="test-fixture",
        )


def test_comparison_aligns_portfolio_benchmark_and_cash(tmp_path) -> None:
    _, paper, account = _account(tmp_path)
    first = paper.save_benchmark_observation(
        account_id=account.account_id,
        symbol="SPY",
        captured_at=T0,
        quote_currency="USD",
        close_price="100",
        fx_rate="0.9",
        source="test-fixture",
    )
    second = paper.save_benchmark_observation(
        account_id=account.account_id,
        symbol="SPY",
        captured_at=T0 + timedelta(days=1),
        quote_currency="USD",
        close_price="105",
        fx_rate="1",
        source="test-fixture",
    )
    equity = (
        SimpleNamespace(
            snapshot_id="EQ-1",
            captured_at=T0,
            equity=Decimal("100"),
        ),
        SimpleNamespace(
            snapshot_id="EQ-2",
            captured_at=T0 + timedelta(days=1),
            equity=Decimal("110"),
        ),
    )

    comparison = calculate_benchmark_comparisons(
        (first, second),
        equity,
    )[0]

    assert comparison.sufficient_evidence is True
    assert comparison.account_return_pct == 10.0
    assert comparison.benchmark_return_pct == pytest.approx(
        16.6666666667
    )
    assert comparison.cash_return_pct == 0.0
    assert comparison.excess_vs_benchmark_pct == pytest.approx(
        -6.6666666667
    )
    assert comparison.excess_vs_cash_pct == 10.0
    assert comparison.provenance.record_ids == (
        first.observation_id,
        second.observation_id,
        "EQ-1",
        "EQ-2",
    )

    rows = benchmark_comparison_rows(
        SimpleNamespace(
            benchmark_comparisons=(comparison,)
        )
    )
    assert rows[0]["benchmark_return_pct"] == pytest.approx(
        16.6666666667
    )


def test_comparison_reports_insufficient_evidence(tmp_path) -> None:
    _, paper, account = _account(tmp_path)
    mark = paper.save_benchmark_observation(
        account_id=account.account_id,
        symbol="VWCE.DE",
        captured_at=T0,
        quote_currency="EUR",
        close_price="100",
        fx_rate="1",
        source="test-fixture",
    )

    comparison = calculate_benchmark_comparisons(
        (mark,),
        (),
    )[0]
    assert comparison.sufficient_evidence is False
    assert comparison.account_return_pct is None
    assert comparison.benchmark_return_pct is None
    assert comparison.reason == (
        "At least two benchmark observations are required."
    )


def test_benchmark_cli_records_and_lists_evidence(tmp_path, capsys) -> None:
    database_path, _, account = _account(tmp_path)
    common = [
        "--database",
        str(database_path),
        "--account-id",
        account.account_id,
    ]
    assert main(
        [
            "benchmark",
            *common,
            "record",
            "VWCE.DE",
            "--captured-at",
            T0.isoformat(),
            "--quote-currency",
            "EUR",
            "--close-price",
            "100",
            "--fx-rate",
            "1",
            "--source",
            "operator-verified-close",
        ]
    ) == 0
    recorded = json.loads(capsys.readouterr().out)
    assert recorded["portfolio_price"] == "100.00000000"

    assert main(
        [
            "benchmark",
            *common,
            "list",
            "VWCE.DE",
        ]
    ) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["total"] == 1
    assert listed["observations"][0]["symbol"] == "VWCE.DE"
