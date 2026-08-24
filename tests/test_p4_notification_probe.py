"""P4.11.10 persisted operational notification probe tests."""

from __future__ import annotations

import json

from src.jobs.cli import main
from src.notifications import DeliveryResult
from src.paper import NotificationChannel, NotificationStatus, PaperRepository


class _Sender:
    def send(self, notification):
        assert "operational delivery probe" in notification.text
        return DeliveryResult(
            provider_message_id="provider-42",
            metadata={"provider": "test"},
        )


def _environment(monkeypatch, tmp_path) -> tuple[PaperRepository, str]:
    path = tmp_path / "paper.db"
    repository = PaperRepository(path)
    account = repository.create_account(
        account_id="ACC-P4-EUR-2000",
        name="P4 Operational",
        base_currency="EUR",
        starting_balance="2000",
    )
    monkeypatch.setenv("PAPER_DATABASE_PATH", str(path))
    monkeypatch.setenv("PAPER_ACCOUNT_ID", account.account_id)
    return repository, account.account_id


def test_probe_persists_sent_provider_evidence(
    monkeypatch, tmp_path, capsys,
) -> None:
    repository, account_id = _environment(monkeypatch, tmp_path)
    monkeypatch.setenv("PAPER_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("PAPER_TELEGRAM_CHAT_ID", "42")

    from src.jobs import cli as cli_module
    original = cli_module.build_runtime

    def configured(settings):
        runtime = original(settings)
        runtime.notification_service.senders[NotificationChannel.TELEGRAM] = _Sender()
        return runtime

    monkeypatch.setattr(cli_module, "build_runtime", configured)
    assert main([
        "notification-probe", "--channel", "TELEGRAM",
        "--operator", "Salih AVCI", "--reason", "P4 acceptance",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["account_id"] == account_id
    assert payload["status"] == "SENT"
    assert payload["attempt_count"] == 1
    assert payload["provider_message_id"] == "provider-42"
    persisted = repository.get_notification(payload["notification_id"])
    assert persisted.status is NotificationStatus.SENT
    assert persisted.reference_type == "OPERATIONAL_PROBE"


def test_probe_fails_closed_without_sender(monkeypatch, tmp_path, capsys) -> None:
    _environment(monkeypatch, tmp_path)
    assert main([
        "notification-probe", "--channel", "TELEGRAM",
        "--operator", "Salih AVCI", "--reason", "P4 acceptance",
    ]) == 2
    assert "sender is not configured" in capsys.readouterr().err
