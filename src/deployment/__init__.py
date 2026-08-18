"""Container deployment health, worker and backup contracts."""

from typing import Any

from .backup import (
    BackupArtifact,
    DatabaseBackupService,
)
from .release import (
    ControlledRelease,
    ReleaseResult,
    ReleaseRuntime,
    ReleaseStatus,
)
from .health import (
    HeartbeatRecord,
    HeartbeatStore,
    HealthEvaluator,
    HealthResult,
    create_health_server,
)
from .worker import (
    ScheduledCycle,
    ScheduledWorker,
    ScheduledWorkerConfig,
    load_scheduled_cycle,
    scheduled_run_key,
)


def __getattr__(name: str) -> Any:
    """Load combined-runtime exports without preloading its CLI module."""
    if name in {
        "PortableRuntime",
        "build_portable_runtime",
    }:
        from .combined import (
            PortableRuntime,
            build_portable_runtime,
        )

        exports = {
            "PortableRuntime": PortableRuntime,
            "build_portable_runtime": build_portable_runtime,
        }
        return exports[name]

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    "PortableRuntime",
    "build_portable_runtime",
    "BackupArtifact",
    "DatabaseBackupService",
    "ControlledRelease",
    "ReleaseResult",
    "ReleaseRuntime",
    "ReleaseStatus",
    "HeartbeatRecord",
    "HeartbeatStore",
    "HealthEvaluator",
    "HealthResult",
    "create_health_server",
    "ScheduledCycle",
    "ScheduledWorker",
    "ScheduledWorkerConfig",
    "load_scheduled_cycle",
    "scheduled_run_key",
]
