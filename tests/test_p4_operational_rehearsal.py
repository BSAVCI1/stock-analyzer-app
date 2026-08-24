"""P4.11.8 operational acceptance rehearsal tests."""

from __future__ import annotations

import json

from src.jobs.cli import main
from src.p4_operational_rehearsal import build_p4_operational_rehearsal
from src.p4_release_gate import evaluate_p4_release_gate, p4_evidence_from_mapping


def _manifest() -> dict[str, object]:
    with open("config/p4_release_evidence.example.json", encoding="utf-8") as handle:
        return json.load(handle)


def test_blocked_gate_produces_specific_read_only_action_plan() -> None:
    gate = evaluate_p4_release_gate(p4_evidence_from_mapping(_manifest()))
    report = build_p4_operational_rehearsal(gate)
    assert report.safe_to_start_p5 is False
    assert report.status == "BLOCKED"
    assert any("GitHub Actions" in action for action in report.next_actions)
    assert any("Telegram" in action for action in report.next_actions)
    assert any("kill-switch" in action for action in report.next_actions)


def test_ready_gate_has_no_next_actions() -> None:
    payload = _manifest()
    payload["regression"] = {
        "passed": True, "test_count": 715,
        "covered_phases": ["P0", "P1", "P2", "P3", "P4"],
        "workflow": "GitHub Actions run 1",
    }
    for check in payload["checks"]:
        check["status"] = "PASS"
        check["evidence_ids"] = ["GENUINE-1"]
    report = build_p4_operational_rehearsal(
        evaluate_p4_release_gate(p4_evidence_from_mapping(payload))
    )
    assert report.safe_to_start_p5 is True
    assert report.next_actions == ()


def test_nonpaper_live_and_failures_are_explicit_actions() -> None:
    payload = _manifest()
    payload["execution_mode"] = "LIVE"
    payload["live_execution_enabled"] = True
    payload["unresolved_operational_failures"] = 2
    report = build_p4_operational_rehearsal(
        evaluate_p4_release_gate(p4_evidence_from_mapping(payload))
    )
    assert any("execution_mode" in action for action in report.next_actions)
    assert any("Disable live" in action for action in report.next_actions)
    assert any("operational failures" in action for action in report.next_actions)


def test_cli_examples_remain_blocked_and_read_only(capsys) -> None:
    result = main([
        "p4-release-rehearsal",
        "--manifest", "config/p4_release_evidence.example.json",
    ])
    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "BLOCKED"
    assert payload["safe_to_start_p5"] is False
    assert payload["read_only"] is True
    assert payload["next_actions"]
