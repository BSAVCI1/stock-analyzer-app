"""Health-gated deployment promotion and rollback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol

from .backup import (
    BackupArtifact,
    DatabaseBackupService,
)


class ReleaseRuntime(Protocol):
    def current_image(self) -> str:
        ...

    def deploy(self, image: str) -> None:
        ...

    def healthy(self) -> bool:
        ...


class ReleaseStatus(str, Enum):
    PROMOTED = "PROMOTED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True, slots=True)
class ReleaseResult:
    status: ReleaseStatus
    candidate_image: str
    active_image: str
    previous_image: str
    backup: BackupArtifact
    completed_at: datetime


class ControlledRelease:
    """Promote only healthy images; otherwise roll back."""

    def __init__(
        self,
        *,
        runtime: ReleaseRuntime,
        backup_service: DatabaseBackupService,
    ) -> None:
        self.runtime = runtime
        self.backup_service = backup_service

    def deploy(
        self,
        candidate_image: str,
        *,
        deployed_at: datetime | None = None,
    ) -> ReleaseResult:
        candidate = candidate_image.strip()

        if not candidate:
            raise ValueError(
                "candidate_image is required."
            )

        at = (
            deployed_at
            or datetime.now(timezone.utc)
        )

        if (
            at.tzinfo is None
            or at.utcoffset() is None
        ):
            raise ValueError(
                "deployed_at must be "
                "timezone-aware."
            )

        previous = (
            self.runtime.current_image()
            .strip()
        )

        if not previous:
            raise RuntimeError(
                "Current image is unknown; "
                "controlled rollback is unavailable."
            )

        if previous == candidate:
            raise ValueError(
                "Candidate image is already active."
            )

        backup = (
            self.backup_service
            .create_backup(
                label="predeploy",
                created_at=at,
            )
        )

        candidate_healthy = False

        try:
            self.runtime.deploy(candidate)
            candidate_healthy = (
                self.runtime.healthy()
            )
        except Exception:
            candidate_healthy = False

        if candidate_healthy:
            return ReleaseResult(
                status=ReleaseStatus.PROMOTED,
                candidate_image=candidate,
                active_image=candidate,
                previous_image=previous,
                backup=backup,
                completed_at=at.astimezone(
                    timezone.utc
                ),
            )

        try:
            self.runtime.deploy(previous)
            rollback_healthy = (
                self.runtime.healthy()
            )
        except Exception as exc:
            raise RuntimeError(
                "Rollback deployment failed."
            ) from exc

        if not rollback_healthy:
            raise RuntimeError(
                "Rollback health verification "
                "failed."
            )

        return ReleaseResult(
            status=ReleaseStatus.ROLLED_BACK,
            candidate_image=candidate,
            active_image=previous,
            previous_image=previous,
            backup=backup,
            completed_at=at.astimezone(
                timezone.utc
            ),
        )
