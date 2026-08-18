"""Secret resolution without accidental value disclosure."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Mapping


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedSecret:
    name: str
    source: str
    _value: str

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return (
            "ResolvedSecret("
            f"name={self.name!r}, "
            f"source={self.source!r}, "
            "value=<redacted>)"
        )

    def __str__(self) -> str:
        return "<redacted>"


def resolve_secret(
    name: str,
    *,
    environ: Mapping[str, str] | None = None,
    require_file: bool = False,
) -> ResolvedSecret | None:
    """Resolve NAME or NAME_FILE, never both."""

    values = (
        environ
        if environ is not None
        else os.environ
    )
    direct = values.get(name)
    file_value = values.get(
        f"{name}_FILE"
    )

    if direct and file_value:
        raise ValueError(
            f"{name} and {name}_FILE "
            "cannot both be configured."
        )

    if direct:
        if require_file:
            raise ValueError(
                f"{name} must be provided "
                "through a secret file."
            )

        if not direct.strip():
            raise ValueError(
                f"{name} cannot be empty."
            )

        return ResolvedSecret(
            name=name,
            source="environment",
            _value=direct,
        )

    if not file_value:
        return None

    path = Path(file_value)

    if not path.is_file():
        raise ValueError(
            f"{name}_FILE does not point "
            "to a readable file."
        )

    mode = stat.S_IMODE(
        path.stat().st_mode
    )

    if mode & 0o022:
        raise ValueError(
            f"{name}_FILE must not be "
            "group- or world-writable."
        )

    try:
        value = path.read_text(
            encoding="utf-8"
        ).rstrip("\r\n")
    except OSError as exc:
        raise ValueError(
            f"{name}_FILE could not be read."
        ) from exc

    if not value:
        raise ValueError(
            f"{name}_FILE contains an "
            "empty secret."
        )

    return ResolvedSecret(
        name=name,
        source="file",
        _value=value,
    )
