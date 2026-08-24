"""Fail-closed P4 email and Telegram delivery evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime

from src.p4_release_gate import P4CheckStatus, P4GateCheck


_CHANNEL_CHECKS = {
    "EMAIL": "email_delivery",
    "TELEGRAM": "telegram_delivery",
}


def _fingerprint(value: Mapping[str, object]) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _aware_timestamp(value: object) -> bool:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _channel_check(
    channel: str,
    evidence: Mapping[str, object],
    *,
    release_id: str,
    context_fingerprint: str,
) -> P4GateCheck:
    failures: list[str] = []
    for key in ("persisted", "deduplicated", "retryable"):
        if evidence.get(key) is not True:
            failures.append(f"{channel}.{key} must be true.")

    for key in ("pending_count", "failed_count"):
        value = evidence.get(key)
        if type(value) is not int or value != 0:
            failures.append(f"{channel}.{key} must be integer 0.")

    raw_records = evidence.get("sent_records")
    records = raw_records if isinstance(raw_records, list) else []
    if not records:
        failures.append(f"{channel}.sent_records must contain sent evidence.")

    notification_ids: list[str] = []
    evidence_ids: list[str] = []
    for index, raw in enumerate(records):
        path = f"{channel}.sent_records[{index}]"
        if not isinstance(raw, Mapping):
            failures.append(f"{path} must be an object.")
            continue
        notification_id = raw.get("notification_id")
        reference_type = raw.get("reference_type")
        reference_id = raw.get("reference_id")
        attempt_count = raw.get("attempt_count")
        if not isinstance(notification_id, str) or not notification_id.strip():
            failures.append(f"{path}.notification_id is required.")
        else:
            notification_ids.append(notification_id.strip())
            evidence_ids.append(f"NOTIFICATION:{notification_id.strip()}")
        if str(raw.get("channel", "")).strip().upper() != channel:
            failures.append(f"{path}.channel must be {channel}.")
        if str(raw.get("status", "")).strip().upper() != "SENT":
            failures.append(f"{path}.status must be SENT.")
        if not isinstance(reference_type, str) or not reference_type.strip():
            failures.append(f"{path}.reference_type is required.")
        if not isinstance(reference_id, str) or not reference_id.strip():
            failures.append(f"{path}.reference_id is required.")
        if not _aware_timestamp(raw.get("sent_at")):
            failures.append(f"{path}.sent_at must be timezone-aware.")
        if type(attempt_count) is not int or attempt_count < 1:
            failures.append(f"{path}.attempt_count must be at least 1.")
        provider_id = raw.get("provider_message_id")
        if provider_id is not None:
            if not isinstance(provider_id, str) or not provider_id.strip():
                failures.append(f"{path}.provider_message_id cannot be blank.")
            else:
                evidence_ids.append(f"PROVIDER-MESSAGE:{provider_id.strip()}")

    if len(notification_ids) != len(set(notification_ids)):
        failures.append(f"{channel}.notification_id values must be unique.")

    check_name = _CHANNEL_CHECKS[channel]
    if failures:
        return P4GateCheck(
            name=check_name,
            status=P4CheckStatus.FAIL,
            evidence_ids=(),
            details=tuple(failures),
        )

    return P4GateCheck(
        name=check_name,
        status=P4CheckStatus.PASS,
        evidence_ids=(
            f"DELIVERY:{release_id}:{channel}:sha256:{context_fingerprint}",
            *tuple(dict.fromkeys(evidence_ids)),
        ),
        details=(
            f"Persisted application-level {channel} sent evidence verified; "
            "no pending or failed delivery remains.",
        ),
    )


def build_notification_delivery_checks(
    evidence: Mapping[str, object],
) -> tuple[P4GateCheck, P4GateCheck]:
    """Build independent email and Telegram release checks."""
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must be an object.")
    common_failures: list[str] = []
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != 1
    ):
        common_failures.append("schema_version must be 1.")
    release_id = str(evidence.get("release_id", "")).strip()
    if not release_id:
        common_failures.append("release_id is required.")
    account_id = str(evidence.get("account_id", "")).strip()
    if not account_id:
        common_failures.append("account_id is required.")
    if not _aware_timestamp(evidence.get("observed_at")):
        common_failures.append("observed_at must be timezone-aware.")

    checks: list[P4GateCheck] = []
    for channel in ("EMAIL", "TELEGRAM"):
        raw = evidence.get(channel.lower())
        channel_evidence = raw if isinstance(raw, Mapping) else {}
        context_fingerprint = _fingerprint({
            "schema_version": evidence.get("schema_version"),
            "release_id": evidence.get("release_id"),
            "account_id": evidence.get("account_id"),
            "observed_at": evidence.get("observed_at"),
            "channel": channel,
            "delivery": channel_evidence,
        })
        check = _channel_check(
            channel,
            channel_evidence,
            release_id=release_id,
            context_fingerprint=context_fingerprint,
        )
        if common_failures:
            check = P4GateCheck(
                name=_CHANNEL_CHECKS[channel],
                status=P4CheckStatus.FAIL,
                evidence_ids=(),
                details=tuple(common_failures) + check.details,
            )
        checks.append(check)
    return checks[0], checks[1]


__all__ = ["build_notification_delivery_checks"]
