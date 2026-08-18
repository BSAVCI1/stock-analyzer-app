from datetime import (
    datetime,
    timezone,
)
import sqlite3

import pytest

from src.deployment import (
    ControlledRelease,
    DatabaseBackupService,
    ReleaseStatus,
)


NOW = datetime(
    2026,
    8,
    18,
    16,
    0,
    tzinfo=timezone.utc,
)


class Runtime:
    def __init__(
        self,
        *,
        health_by_image,
        current="image:previous",
        fail_images=(),
    ):
        self.current = current
        self.health_by_image = dict(
            health_by_image
        )
        self.fail_images = set(fail_images)
        self.deployments = []

    def current_image(self):
        return self.current

    def deploy(self, image):
        self.deployments.append(image)

        if image in self.fail_images:
            raise RuntimeError(
                "provider error with secret"
            )

        self.current = image

    def healthy(self):
        return self.health_by_image.get(
            self.current,
            False,
        )


def _backup_service(tmp_path):
    database = tmp_path / "paper.db"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE evidence(
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO evidence(value)
        VALUES ('before-deploy')
        """
    )
    connection.commit()
    connection.close()

    return DatabaseBackupService(
        database_path=database,
        backup_directory=(
            tmp_path / "backups"
        ),
    )


def test_healthy_candidate_is_promoted(
    tmp_path,
):
    runtime = Runtime(
        health_by_image={
            "image:candidate": True,
        }
    )
    release = ControlledRelease(
        runtime=runtime,
        backup_service=_backup_service(
            tmp_path
        ),
    )

    result = release.deploy(
        "image:candidate",
        deployed_at=NOW,
    )

    assert result.status is (
        ReleaseStatus.PROMOTED
    )
    assert result.active_image == (
        "image:candidate"
    )
    assert runtime.deployments == [
        "image:candidate"
    ]
    assert (
        result.backup.database_path.exists()
    )


def test_unhealthy_candidate_rolls_back(
    tmp_path,
):
    runtime = Runtime(
        health_by_image={
            "image:candidate": False,
            "image:previous": True,
        }
    )
    release = ControlledRelease(
        runtime=runtime,
        backup_service=_backup_service(
            tmp_path
        ),
    )

    result = release.deploy(
        "image:candidate",
        deployed_at=NOW,
    )

    assert result.status is (
        ReleaseStatus.ROLLED_BACK
    )
    assert result.active_image == (
        "image:previous"
    )
    assert runtime.deployments == [
        "image:candidate",
        "image:previous",
    ]


def test_candidate_start_failure_rolls_back(
    tmp_path,
):
    runtime = Runtime(
        health_by_image={
            "image:previous": True,
        },
        fail_images={
            "image:candidate",
        },
    )
    release = ControlledRelease(
        runtime=runtime,
        backup_service=_backup_service(
            tmp_path
        ),
    )

    result = release.deploy(
        "image:candidate",
        deployed_at=NOW,
    )

    assert result.status is (
        ReleaseStatus.ROLLED_BACK
    )
    assert runtime.deployments == [
        "image:candidate",
        "image:previous",
    ]


def test_failed_rollback_is_blocking(
    tmp_path,
):
    runtime = Runtime(
        health_by_image={
            "image:candidate": False,
            "image:previous": False,
        }
    )
    release = ControlledRelease(
        runtime=runtime,
        backup_service=_backup_service(
            tmp_path
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Rollback health",
    ):
        release.deploy(
            "image:candidate",
            deployed_at=NOW,
        )


def test_backup_failure_prevents_deployment(
    tmp_path,
):
    runtime = Runtime(
        health_by_image={
            "image:candidate": True,
        }
    )
    release = ControlledRelease(
        runtime=runtime,
        backup_service=DatabaseBackupService(
            database_path=(
                tmp_path / "missing.db"
            ),
            backup_directory=(
                tmp_path / "backups"
            ),
        ),
    )

    with pytest.raises(FileNotFoundError):
        release.deploy(
            "image:candidate",
            deployed_at=NOW,
        )

    assert runtime.deployments == []


def test_unknown_previous_image_blocks_release(
    tmp_path,
):
    runtime = Runtime(
        health_by_image={},
        current="",
    )
    release = ControlledRelease(
        runtime=runtime,
        backup_service=_backup_service(
            tmp_path
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="rollback is unavailable",
    ):
        release.deploy(
            "image:candidate",
            deployed_at=NOW,
        )
