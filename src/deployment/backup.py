"""Verified SQLite backup and atomic restore operations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4


_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(
            "Backup time must be timezone-aware."
        )

    return value.astimezone(
        timezone.utc
    ).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _verify_sqlite(path: Path) -> None:
    connection = sqlite3.connect(
        str(path),
        timeout=30,
    )

    try:
        rows = connection.execute(
            "PRAGMA integrity_check"
        ).fetchall()
    finally:
        connection.close()

    messages = [
        str(row[0])
        for row in rows
    ]

    if messages != ["ok"]:
        raise RuntimeError(
            "SQLite integrity check failed."
        )


@dataclass(frozen=True, slots=True)
class BackupArtifact:
    database_path: Path
    manifest_path: Path
    sha256: str
    size_bytes: int
    created_at: datetime


class DatabaseBackupService:
    """Create verified online backups and controlled restores."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        backup_directory: str | Path,
    ) -> None:
        self.database_path = Path(database_path)
        self.backup_directory = Path(
            backup_directory
        )

    def create_backup(
        self,
        *,
        label: str = "scheduled",
        created_at: datetime | None = None,
    ) -> BackupArtifact:
        normalized_label = label.strip()

        if not _LABEL.fullmatch(
            normalized_label
        ):
            raise ValueError(
                "Backup label contains "
                "unsupported characters."
            )

        if not self.database_path.is_file():
            raise FileNotFoundError(
                "Source database does not exist."
            )

        at = created_at or _utc_now()
        stamp = (
            at.astimezone(timezone.utc)
            .strftime("%Y%m%dT%H%M%SZ")
        )
        name = (
            f"paper-{stamp}-"
            f"{normalized_label}.sqlite3"
        )
        destination = (
            self.backup_directory / name
        )
        manifest = destination.with_suffix(
            ".manifest.json"
        )

        self.backup_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            destination.exists()
            or manifest.exists()
        ):
            raise FileExistsError(
                "Backup artifact already exists."
            )

        temporary = (
            self.backup_directory
            / f".{name}.{uuid4().hex}.tmp"
        )

        source_connection = sqlite3.connect(
            str(self.database_path),
            timeout=30,
        )
        target_connection = sqlite3.connect(
            str(temporary),
            timeout=30,
        )

        try:
            source_connection.backup(
                target_connection
            )
        finally:
            target_connection.close()
            source_connection.close()

        try:
            _verify_sqlite(temporary)
            checksum = _sha256(temporary)
            size = temporary.stat().st_size
            temporary.replace(destination)

            manifest_temporary = (
                manifest.with_suffix(
                    manifest.suffix + ".tmp"
                )
            )
            manifest_temporary.write_text(
                json.dumps(
                    {
                        "created_at":
                            _timestamp(at),
                        "database_file":
                            destination.name,
                        "sha256": checksum,
                        "size_bytes": size,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_temporary.replace(manifest)
        except Exception:
            temporary.unlink(
                missing_ok=True
            )
            destination.unlink(
                missing_ok=True
            )
            manifest.unlink(
                missing_ok=True
            )
            raise

        return BackupArtifact(
            database_path=destination,
            manifest_path=manifest,
            sha256=checksum,
            size_bytes=size,
            created_at=at.astimezone(
                timezone.utc
            ),
        )

    @staticmethod
    def restore_backup(
        *,
        backup_path: str | Path,
        target_path: str | Path,
        replace_existing: bool = False,
    ) -> Path:
        source = Path(backup_path)
        target = Path(target_path)
        manifest = source.with_suffix(
            ".manifest.json"
        )

        if not source.is_file():
            raise FileNotFoundError(
                "Backup database does not exist."
            )

        if not manifest.is_file():
            raise FileNotFoundError(
                "Backup manifest does not exist."
            )

        if (
            target.exists()
            and not replace_existing
        ):
            raise FileExistsError(
                "Restore target already exists."
            )

        sidecars = (
            Path(str(target) + "-wal"),
            Path(str(target) + "-shm"),
        )

        if any(
            path.exists()
            for path in sidecars
        ):
            raise RuntimeError(
                "Restore target has active SQLite "
                "sidecar files; stop the service "
                "before restoring."
            )

        metadata = json.loads(
            manifest.read_text(
                encoding="utf-8"
            )
        )

        if (
            metadata.get("database_file")
            != source.name
        ):
            raise RuntimeError(
                "Backup manifest does not match "
                "the database file."
            )

        actual_size = source.stat().st_size
        actual_checksum = _sha256(source)

        if (
            metadata.get("size_bytes")
            != actual_size
            or metadata.get("sha256")
            != actual_checksum
        ):
            raise RuntimeError(
                "Backup checksum verification "
                "failed."
            )

        _verify_sqlite(source)
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        temporary = target.with_name(
            f".{target.name}."
            f"{uuid4().hex}.restore"
        )

        source_connection = sqlite3.connect(
            str(source),
            timeout=30,
        )
        target_connection = sqlite3.connect(
            str(temporary),
            timeout=30,
        )

        try:
            source_connection.backup(
                target_connection
            )
        finally:
            target_connection.close()
            source_connection.close()

        try:
            _verify_sqlite(temporary)
            temporary.replace(target)
        except Exception:
            temporary.unlink(
                missing_ok=True
            )
            raise

        return target


def serve() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create or restore verified "
            "SQLite backups."
        )
    )
    subcommands = parser.add_subparsers(
        dest="command",
        required=True,
    )
    backup = subcommands.add_parser(
        "backup"
    )
    backup.add_argument(
        "--label",
        default="manual",
    )
    restore = subcommands.add_parser(
        "restore"
    )
    restore.add_argument(
        "backup_path",
    )
    restore.add_argument(
        "--target",
        required=True,
    )
    restore.add_argument(
        "--replace",
        action="store_true",
    )
    arguments = parser.parse_args()
    database_path = os.environ.get(
        "BSAVCI_DATABASE_PATH",
        "data/paper_trading.db",
    )
    backup_directory = os.environ.get(
        "BSAVCI_BACKUP_DIRECTORY",
        "data/backups",
    )
    service = DatabaseBackupService(
        database_path=database_path,
        backup_directory=backup_directory,
    )

    if arguments.command == "backup":
        artifact = service.create_backup(
            label=arguments.label
        )
        print(artifact.manifest_path)
        return

    restored = service.restore_backup(
        backup_path=arguments.backup_path,
        target_path=arguments.target,
        replace_existing=arguments.replace,
    )
    print(restored)


if __name__ == "__main__":
    serve()
