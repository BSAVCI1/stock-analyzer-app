import os

import pytest

from src.notifications import (
    load_email_config,
    load_telegram_config,
)
from src.secrets import (
    resolve_secret,
)


def test_secret_file_is_resolved_without_newline(
    tmp_path,
):
    path = tmp_path / "telegram-token"
    path.write_text(
        "token-value\n",
        encoding="utf-8",
    )
    path.chmod(0o440)

    secret = resolve_secret(
        "PAPER_TELEGRAM_BOT_TOKEN",
        environ={
            "PAPER_TELEGRAM_BOT_TOKEN_FILE":
                str(path),
        },
        require_file=True,
    )

    assert secret.reveal() == "token-value"
    assert secret.source == "file"
    assert "token-value" not in repr(secret)
    assert str(secret) == "<redacted>"


def test_secret_rejects_ambiguous_sources(
    tmp_path,
):
    path = tmp_path / "secret"
    path.write_text(
        "file-value",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="cannot both",
    ):
        resolve_secret(
            "PAPER_SMTP_PASSWORD",
            environ={
                "PAPER_SMTP_PASSWORD":
                    "environment-value",
                "PAPER_SMTP_PASSWORD_FILE":
                    str(path),
            },
        )


def test_deployment_mode_rejects_direct_secret():
    with pytest.raises(
        ValueError,
        match="secret file",
    ):
        resolve_secret(
            "PAPER_SMTP_PASSWORD",
            environ={
                "PAPER_SMTP_PASSWORD":
                    "environment-value",
            },
            require_file=True,
        )


def test_secret_rejects_writable_file(
    tmp_path,
):
    path = tmp_path / "secret"
    path.write_text(
        "secret-value",
        encoding="utf-8",
    )
    path.chmod(0o666)

    with pytest.raises(
        ValueError,
        match="world-writable",
    ):
        resolve_secret(
            "PAPER_SMTP_PASSWORD",
            environ={
                "PAPER_SMTP_PASSWORD_FILE":
                    str(path),
            },
        )


def test_telegram_config_reads_mounted_secret(
    tmp_path,
):
    path = tmp_path / "telegram-token"
    path.write_text(
        "telegram-secret",
        encoding="utf-8",
    )
    path.chmod(0o400)

    config = load_telegram_config(
        {
            "BSAVCI_REQUIRE_SECRET_FILES": "true",
            "PAPER_TELEGRAM_BOT_TOKEN_FILE":
                str(path),
            "PAPER_TELEGRAM_CHAT_ID": "12345",
        }
    )

    assert config.bot_token == (
        "telegram-secret"
    )
    assert config.chat_id == "12345"


def test_email_config_reads_mounted_secret(
    tmp_path,
):
    path = tmp_path / "smtp-password"
    path.write_text(
        "smtp-secret\n",
        encoding="utf-8",
    )
    path.chmod(0o440)

    config = load_email_config(
        {
            "BSAVCI_REQUIRE_SECRET_FILES": "1",
            "PAPER_SMTP_HOST":
                "smtp.example.test",
            "PAPER_EMAIL_FROM":
                "bot@example.test",
            "PAPER_EMAIL_TO":
                "owner@example.test",
            "PAPER_SMTP_USERNAME": "bot",
            "PAPER_SMTP_PASSWORD_FILE":
                str(path),
        }
    )

    assert config.password == "smtp-secret"


def test_local_mode_preserves_direct_secret_support():
    secret = resolve_secret(
        "PAPER_SMTP_PASSWORD",
        environ={
            "PAPER_SMTP_PASSWORD":
                "local-only",
        },
    )

    assert secret.reveal() == "local-only"
    assert secret.source == "environment"
