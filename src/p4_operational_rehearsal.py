"""Read-only operational rehearsal for the P4 release gate."""

from __future__ import annotations

from dataclasses import dataclass

from src.p4_release_gate import P4ReleaseGateReport


_CHECK_ACTIONS = {
    "paper_only_invariants": (
        "Restore the approved paper-only product policy and regenerate policy evidence."
    ),
    "eur_portfolio_policy": (
        "Restore the approved EUR portfolio policy and regenerate policy evidence."
    ),
    "scheduler_deployment": (
        "Capture genuine health, completed-cycle, restart, and persistent-storage evidence "
        "from the target paper runtime."
    ),
    "email_delivery": (
        "Send a genuine application email and capture its persisted SENT record."
    ),
    "telegram_delivery": (
        "Send a genuine application Telegram notification and capture its persisted SENT record."
    ),
    "recovery_controls": (
        "Complete the paper-runtime restart, replay, circuit-breaker, outage, and incident drills."
    ),
    "kill_switch": (
        "Complete the global kill-switch activation, order-blocking, audit, and recovery drill."
    ),
    "strategy_horizon_acceptance": (
        "Supply independent accepted swing and medium-term validation reports."
    ),
}


@dataclass(frozen=True, slots=True)
class P4OperationalRehearsalReport:
    """A safe gap plan derived exclusively from an evaluated gate report."""

    release_id: str
    status: str
    safe_to_start_p5: bool
    blocking_checks: tuple[str, ...]
    reasons: tuple[str, ...]
    next_actions: tuple[str, ...]


def build_p4_operational_rehearsal(
    report: P4ReleaseGateReport,
) -> P4OperationalRehearsalReport:
    """Translate release-gate blockers into deterministic operator actions."""
    if not isinstance(report, P4ReleaseGateReport):
        raise ValueError("report must be a P4ReleaseGateReport.")

    evidence = report.evidence
    actions: list[str] = []
    if not evidence.regression.passed:
        actions.append(
            "Replace regression evidence with a successful, traceable P0-P4 GitHub Actions run."
        )
    missing_phases = tuple(
        phase for phase in ("P0", "P1", "P2", "P3", "P4")
        if phase not in set(evidence.regression.covered_phases)
    )
    if missing_phases:
        actions.append(
            "Add regression coverage for phases: " + ", ".join(missing_phases) + "."
        )
    actions.extend(_CHECK_ACTIONS[name] for name in report.blocking_checks)
    if evidence.execution_mode != "PAPER":
        actions.append("Restore execution_mode to PAPER before collecting release evidence.")
    if evidence.live_execution_enabled:
        actions.append("Disable live execution capability before release acceptance.")
    if evidence.unresolved_operational_failures:
        actions.append(
            "Resolve and close all operational failures before release acceptance."
        )

    return P4OperationalRehearsalReport(
        release_id=evidence.release_id,
        status=report.status.value,
        safe_to_start_p5=report.release_ready,
        blocking_checks=report.blocking_checks,
        reasons=report.reasons,
        next_actions=tuple(actions),
    )


__all__ = ["P4OperationalRehearsalReport", "build_p4_operational_rehearsal"]
