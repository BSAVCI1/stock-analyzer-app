"""Guarded bootstrap for a dedicated local paper account."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import os
from pathlib import Path
from typing import Mapping

from src.paper import PaperRepository


_TRUE_VALUES = {"1", "true", "yes"}


def ensure_local_paper_account(
    environ: Mapping[str, str] | None = None,
):
    """Create an isolated local paper account once."""
    values = os.environ if environ is None else environ
    enabled = (
        values.get("BSAVCI_LOCAL_PAPER_BOOTSTRAP", "")
        .strip()
        .lower()
        in _TRUE_VALUES
    )

    if not enabled:
        raise RuntimeError("Local paper bootstrap is disabled.")

    if (
        values.get("PAPER_BROKER_ENABLED", "").strip().lower()
        in _TRUE_VALUES
        or values.get(
            "PAPER_BROKER_LIVE_TRADING", ""
        ).strip().lower()
        in _TRUE_VALUES
    ):
        raise RuntimeError(
            "Local bootstrap prohibits broker "
            "and live-trading settings."
        )

    account_id = values.get("PAPER_ACCOUNT_ID", "").strip()
    if not account_id:
        raise ValueError("PAPER_ACCOUNT_ID is required.")

    database_path = Path(
        values.get(
            "BSAVCI_DATABASE_PATH",
            values.get(
                "PAPER_DATABASE_PATH",
                "data/paper_trading.db",
            ),
        )
    )
    repository = PaperRepository(database_path)

    try:
        return repository.get_account(account_id)
    except ValueError:
        return repository.create_account(
            account_id=account_id,
            name=values.get(
                "BSAVCI_LOCAL_PAPER_ACCOUNT_NAME",
                "Local Device Paper Account",
            ),
            base_currency=values.get(
                "BSAVCI_LOCAL_PAPER_BASE_CURRENCY",
                "EUR",
            ),
            starting_balance=Decimal(
                values.get(
                    "BSAVCI_LOCAL_PAPER_STARTING_BALANCE",
                    "100000",
                )
            ),
            created_at=datetime.now(timezone.utc),
        )


def main() -> None:
    account = ensure_local_paper_account()
    print(account.account_id)


if __name__ == "__main__":
    main()
