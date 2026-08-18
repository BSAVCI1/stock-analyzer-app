"""Persistent bounded-recovery orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import (
    datetime,
    timedelta,
)

from .orchestration_repository import (
    OrchestrationRepository,
)
from .orchestrator import (
    InvocationExecutor,
    OrchestrationCycleReport,
    due_invocations,
    run_orchestration_cycle,
)
from .schedule import (
    AUTONOMOUS_SCHEDULE_VERSION,
    AutonomousSchedulePolicy,
)


@dataclass(frozen=True, slots=True)
class PersistentCycleReport:
    cycle: OrchestrationCycleReport
    previous_checkpoint: datetime | None
    stored_checkpoint: datetime
    missed_count: int


class PersistentOrchestrationService:
    """Run one durable cycle with bounded recovery."""

    def __init__(
        self,
        *,
        account_id: str,
        repository:
        OrchestrationRepository,
        executor: InvocationExecutor,
        policy: (
            AutonomousSchedulePolicy | None
        ) = None,
        initial_lookback: timedelta = (
            timedelta(minutes=5)
        ),
        maximum_recovery: timedelta = (
            timedelta(hours=6)
        ),
    ) -> None:
        value = str(account_id).strip()

        if not value:
            raise ValueError(
                "account_id is required."
            )

        if initial_lookback <= timedelta(0):
            raise ValueError(
                "initial_lookback must be "
                "positive."
            )

        if maximum_recovery <= timedelta(0):
            raise ValueError(
                "maximum_recovery must be "
                "positive."
            )

        self.account_id = value
        self.repository = repository
        self.executor = executor
        self.policy = (
            policy
            or AutonomousSchedulePolicy()
        )
        self.initial_lookback = (
            initial_lookback
        )
        self.maximum_recovery = (
            maximum_recovery
        )

    @property
    def policy_version(self) -> str:
        return self.policy.policy_version

    def run(
        self,
        *,
        now: datetime,
    ) -> PersistentCycleReport:
        if (
            now.tzinfo is None
            or now.utcoffset() is None
        ):
            raise ValueError(
                "now must be timezone-aware."
            )

        previous = (
            self.repository.get_checkpoint(
                account_id=self.account_id,
                policy_version=(
                    self.policy_version
                ),
            )
        )

        if (
            previous is not None
            and previous > now
        ):
            raise RuntimeError(
                "Persisted orchestration "
                "checkpoint is in the future."
            )

        desired_start = (
            previous
            if previous is not None
            else now - self.initial_lookback
        )
        recovery_start = max(
            desired_start,
            now - self.maximum_recovery,
        )
        missed_count = 0

        if desired_start < recovery_start:
            missed = due_invocations(
                window_started_at=(
                    desired_start
                ),
                window_ended_at=(
                    recovery_start
                ),
                policy=self.policy,
            )
            missed_count = (
                self.repository.record_missed(
                    account_id=self.account_id,
                    policy_version=(
                        self.policy_version
                    ),
                    invocations=missed,
                    detected_at=now,
                )
            )

        completed = (
            self.repository.completed_keys(
                account_id=self.account_id,
                policy_version=(
                    self.policy_version
                ),
            )
        )
        cycle = run_orchestration_cycle(
            window_started_at=recovery_start,
            window_ended_at=now,
            executor=self.executor,
            policy=self.policy,
            completed_keys=completed,
        )
        failures = tuple(
            result
            for result in cycle.results
            if result.status.value == "FAILED"
        )

        stored_checkpoint = now

        if failures:
            earliest = min(
                result.invocation.scheduled_for
                for result in failures
            )
            stored_checkpoint = (
                earliest
                - timedelta(microseconds=1)
            )

        self.repository.record_cycle(
            account_id=self.account_id,
            policy_version=(
                self.policy_version
            ),
            report=cycle,
            recorded_at=now,
            checkpoint_at=stored_checkpoint,
        )

        return PersistentCycleReport(
            cycle=cycle,
            previous_checkpoint=previous,
            stored_checkpoint=(
                stored_checkpoint
            ),
            missed_count=missed_count,
        )
