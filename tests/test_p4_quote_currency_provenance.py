from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

import pytest

import src.paper.migrations as migrations
from src.paper import (
    PaperRepository,
    PaperTradingService,
    initialize_database,
)


T0 = datetime(
    2026,
    8,
    8,
    11,
    0,
    tzinfo=timezone.utc,
)


def make_service(database):
    repository = PaperRepository(
        database
    )

    account = repository.create_account(
        account_id="ACC-P4-CURRENCY",
        name="P4 Currency",
        base_currency="EUR",
        starting_balance="2000",
        created_at=T0,
    )

    service = PaperTradingService(
        repository,
        app_version="test",
        threshold_version="test",
    )

    return (
        repository,
        service,
        account,
    )


def persist_signal(
    service,
    account_id,
    *,
    quote_currency=None,
    signal_id="SIG-P4-CURRENCY",
):
    return service.persist_signal(
        account_id=account_id,
        signal_id=signal_id,
        symbol="AAPL",
        quote_currency=quote_currency,
        generated_at=T0,
        expires_at=T0.replace(
            day=15
        ),
        strategy="trend_pullback",
        recommendation="BUY",
        market_regime="BULLISH",
        score=80,
        confidence=0.9,
        reward_to_risk=2.5,
        entry_low=99,
        entry_high=101,
        stop_price=95,
        targets=(110,),
        evidence=("test",),
    )


def test_latest_schema_is_seven(
    tmp_path,
) -> None:
    database = tmp_path / "latest.db"

    initialize_database(
        database
    )

    connection = sqlite3.connect(
        database
    )

    try:
        version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        columns = {
            row[1]
            for row in connection.execute(
                """
                PRAGMA table_info(
                    paper_signals
                )
                """
            )
        }
    finally:
        connection.close()

    assert version == 16
    assert "quote_currency" in columns


def test_genuine_v6_upgrades_to_v7(
    tmp_path,
) -> None:
    database = tmp_path / "v6.db"

    connection = sqlite3.connect(
        database
    )

    try:
        for script in (
            migrations._SCHEMA_V1,
            migrations._SCHEMA_V2,
            migrations._SCHEMA_V3,
            migrations._SCHEMA_V4,
            migrations._SCHEMA_V5,
            migrations._SCHEMA_V6,
        ):
            connection.executescript(
                script
            )

        before = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        before_columns = {
            row[1]
            for row in connection.execute(
                """
                PRAGMA table_info(
                    paper_signals
                )
                """
            )
        }
    finally:
        connection.close()

    assert before == 6

    assert (
        "quote_currency"
        not in before_columns
    )

    initialize_database(
        database
    )

    connection = sqlite3.connect(
        database
    )

    try:
        after = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        after_columns = {
            row[1]
            for row in connection.execute(
                """
                PRAGMA table_info(
                    paper_signals
                )
                """
            )
        }
    finally:
        connection.close()

    assert after == 16
    assert "quote_currency" in after_columns


def test_quote_currency_round_trip(
    tmp_path,
) -> None:
    (
        repository,
        service,
        account,
    ) = make_service(
        tmp_path / "roundtrip.db"
    )

    signal = persist_signal(
        service,
        account.account_id,
        quote_currency="usd",
    )

    assert signal.quote_currency == "USD"

    assert (
        repository
        .get_signal(signal.signal_id)
        .quote_currency
        == "USD"
    )


def test_legacy_caller_can_store_null_currency(
    tmp_path,
) -> None:
    (
        _,
        service,
        account,
    ) = make_service(
        tmp_path / "legacy.db"
    )

    signal = persist_signal(
        service,
        account.account_id,
        quote_currency=None,
        signal_id="SIG-LEGACY",
    )

    assert signal.quote_currency is None


def test_invalid_currency_is_rejected(
    tmp_path,
) -> None:
    (
        _,
        service,
        account,
    ) = make_service(
        tmp_path / "invalid.db"
    )

    with pytest.raises(
        ValueError,
        match="three-letter currency code",
    ):
        persist_signal(
            service,
            account.account_id,
            quote_currency="US",
        )
