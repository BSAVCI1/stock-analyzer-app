"""Configured, versioned stock-universe loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .models import (
    StockUniverse,
    normalise_symbol,
)


DEFAULT_UNIVERSE_PATH = Path(
    "config/stock_universe.json"
)

UNIVERSE_POLICY_VERSION = (
    "p4.4-universe-v1"
)


def _symbol_list(
    payload: Mapping[str, object],
    key: str,
) -> tuple[str, ...]:
    value = payload.get(key)

    if not isinstance(value, list):
        raise ValueError(
            f"Stock-universe {key} must "
            "be a list."
        )

    result: list[str] = []
    seen: set[str] = set()

    for raw_symbol in value:
        symbol = normalise_symbol(
            raw_symbol
        )

        if symbol in seen:
            raise ValueError(
                f"Stock-universe {key} "
                f"contains duplicate {symbol}."
            )

        seen.add(symbol)
        result.append(symbol)

    return tuple(result)


def _legacy_universe(
    payload: Mapping[str, object],
) -> StockUniverse:
    expected = {
        "schema_version",
        "name",
        "description",
        "symbols",
    }

    if set(payload) != expected:
        raise ValueError(
            "Legacy stock-universe schema "
            "contains unexpected or missing keys."
        )

    return StockUniverse(
        name=payload.get("name"),
        description=payload.get(
            "description",
            "",
        ),
        symbols=_symbol_list(
            payload,
            "symbols",
        ),
    )


def _versioned_universe(
    payload: Mapping[str, object],
) -> StockUniverse:
    expected = {
        "schema_version",
        "policy_version",
        "name",
        "description",
        "base_symbols",
        "include_symbols",
        "exclude_symbols",
    }

    if set(payload) != expected:
        raise ValueError(
            "Versioned stock-universe schema "
            "contains unexpected or missing keys."
        )

    if (
        payload.get("policy_version")
        != UNIVERSE_POLICY_VERSION
    ):
        raise ValueError(
            "Unsupported stock-universe "
            "policy_version."
        )

    base = _symbol_list(
        payload,
        "base_symbols",
    )
    included = _symbol_list(
        payload,
        "include_symbols",
    )
    excluded = _symbol_list(
        payload,
        "exclude_symbols",
    )

    overlap = (
        set(included)
        & set(excluded)
    )

    if overlap:
        raise ValueError(
            "Stock-universe include_symbols "
            "and exclude_symbols must be "
            "disjoint."
        )

    excluded_set = set(excluded)
    effective = tuple(
        symbol
        for symbol in (
            *base,
            *included,
        )
        if symbol not in excluded_set
    )

    effective = tuple(
        dict.fromkeys(effective)
    )

    return StockUniverse(
        name=payload.get("name"),
        description=payload.get(
            "description",
            "",
        ),
        policy_version=(
            UNIVERSE_POLICY_VERSION
        ),
        symbols=effective,
        included_symbols=included,
        excluded_symbols=excluded,
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

    schema_version = payload.get(
        "schema_version"
    )

    if schema_version == 1:
        return _legacy_universe(
            payload
        )

    if schema_version == 2:
        return _versioned_universe(
            payload
        )

    raise ValueError(
        "Unsupported stock-universe schema."
    )
