"""Configured stock-universe loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .models import StockUniverse


DEFAULT_UNIVERSE_PATH = Path(
    "config/stock_universe.json"
)


def load_stock_universe(
    path: str | Path = DEFAULT_UNIVERSE_PATH,
) -> StockUniverse:
    universe_path = Path(path)

    if not universe_path.is_file():
        raise ValueError(
            f"Stock-universe file does not exist: "
            f"{universe_path}."
        )

    try:
        payload = json.loads(
            universe_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            "Stock-universe file could not be read."
        ) from exc

    if not isinstance(payload, Mapping):
        raise ValueError(
            "Stock-universe file must contain "
            "a JSON object."
        )

    if payload.get("schema_version") != 1:
        raise ValueError(
            "Unsupported stock-universe schema."
        )

    symbols = payload.get("symbols")

    if not isinstance(symbols, list):
        raise ValueError(
            "Stock-universe symbols must be a list."
        )

    return StockUniverse(
        name=payload.get("name"),
        description=payload.get(
            "description",
            "",
        ),
        symbols=tuple(symbols),
    )
