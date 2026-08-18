from datetime import (
    datetime,
    timezone,
)
import json
import sqlite3

import pytest

from src.deployment import (
    DatabaseBackupService,
)


NOW = datetime(
    2026,
    8,
    18,
    14,
    0,
    tzinfo=timezone.utc,
)


def _create_database(path):
    connection = sqlite3.connect(path)
    connection.execute(
        "PRAGMA journal_mode = WAL"
    )
    connection.execute(
        """
        CREATE TABLE evidence(
            id INTEGER PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO evidence(value)
        VALUES (?)
        """,
        ("persisted-value",),
    )
    connection.commit()
    return connection


def test_online_backup_includes_committed_wal_data(
    tmp_path,
):
    source = tmp_path / "live.db"
    open_connection = _create_database(
        source
    )
    service = DatabaseBackupService(
        database_path=source,
        backup_directory=(
            tmp_path / "backups"
        ),
    )

    try:
        artifact = service.create_backup(
            label="daily",
            created_at=NOW,
        )
    finally:
        open_connection.close()

    backup = sqlite3.connect(
        artifact.database_path
    )

    try:
        value = backup.execute(
            "SELECT value FROM evidence"
        ).fetchone()[0]
    finally:
        backup.close()

    assert value == "persisted-value"
    assert artifact.database_path.name == (
        "paper-20260818T140000Z-"
        "daily.sqlite3"
    )
    assert artifact.size_bytes > 0
    assert len(artifact.sha256) == 64


def test_backup_manifest_matches_artifact(
    tmp_path,
):
    source = tmp_path / "live.db"
    _create_database(source).close()
    artifact = DatabaseBackupService(
        database_path=source,
        backup_directory=tmp_path / "backup",
    ).create_backup(
        created_at=NOW
    )

    manifest = json.loads(
        artifact.manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert manifest == {
        "created_at": NOW.isoformat(),
        "database_file":
            artifact.database_path.name,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
    }


def test_restore_recreates_verified_database(
    tmp_path,
):
    source = tmp_path / "live.db"
    _create_database(source).close()
    service = DatabaseBackupService(
        database_path=source,
        backup_directory=tmp_path / "backup",
    )
    artifact = service.create_backup(
        created_at=NOW
    )
    restored = tmp_path / "restore" / "paper.db"

    service.restore_backup(
        backup_path=artifact.database_path,
        target_path=restored,
    )

    connection = sqlite3.connect(restored)

    try:
        value = connection.execute(
            "SELECT value FROM evidence"
        ).fetchone()[0]
    finally:
        connection.close()

    assert value == "persisted-value"


def test_restore_refuses_implicit_overwrite(
    tmp_path,
):
    source = tmp_path / "live.db"
    _create_database(source).close()
    artifact = DatabaseBackupService(
        database_path=source,
        backup_directory=tmp_path / "backup",
    ).create_backup(
        created_at=NOW
    )
    target = tmp_path / "target.db"
    target.write_bytes(b"keep-me")

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        DatabaseBackupService.restore_backup(
            backup_path=artifact.database_path,
            target_path=target,
        )

    assert target.read_bytes() == b"keep-me"


def test_restore_rejects_tampered_backup(
    tmp_path,
):
    source = tmp_path / "live.db"
    _create_database(source).close()
    artifact = DatabaseBackupService(
        database_path=source,
        backup_directory=tmp_path / "backup",
    ).create_backup(
        created_at=NOW
    )
    artifact.database_path.write_bytes(
        artifact.database_path.read_bytes()
        + b"tampered"
    )

    with pytest.raises(
        RuntimeError,
        match="checksum",
    ):
        DatabaseBackupService.restore_backup(
            backup_path=artifact.database_path,
            target_path=tmp_path / "target.db",
        )


def test_restore_refuses_active_wal_sidecar(
    tmp_path,
):
    source = tmp_path / "live.db"
    _create_database(source).close()
    artifact = DatabaseBackupService(
        database_path=source,
        backup_directory=tmp_path / "backup",
    ).create_backup(
        created_at=NOW
    )
    target = tmp_path / "target.db"
    target.write_bytes(b"old")
    target.with_name(
        target.name + "-wal"
    ).write_bytes(b"active")

    with pytest.raises(
        RuntimeError,
        match="stop the service",
    ):
        DatabaseBackupService.restore_backup(
            backup_path=artifact.database_path,
            target_path=target,
            replace_existing=True,
        )


def test_backup_rejects_unsafe_label(
    tmp_path,
):
    source = tmp_path / "live.db"
    _create_database(source).close()
    service = DatabaseBackupService(
        database_path=source,
        backup_directory=tmp_path / "backup",
    )

    with pytest.raises(
        ValueError,
        match="unsupported",
    ):
        service.create_backup(
            label="../../escape",
            created_at=NOW,
        )
