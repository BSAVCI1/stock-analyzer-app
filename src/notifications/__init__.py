"""Persistent notification delivery."""

from .config import (
    load_email_config,
    load_telegram_config,
)
from .health import (
    ChannelHealth,
    ChannelHealthStatus,
    notification_channel_health,
)
from .models import (
    DeliveryResult,
    DispatchReport,
    EmailConfig,
    RenderedNotification,
    TelegramConfig,
)
from .retry import (
    NOTIFICATION_RETRY_POLICY_VERSION,
    NotificationRetryPolicy,
    RetryDecision,
    RetryEligibility,
    evaluate_notification_retry,
)
from .routing import (
    NOTIFICATION_ROUTING_POLICY_VERSION,
    NotificationPurpose,
    NotificationRoute,
    NotificationSeverity,
    route_notification_event,
)
from .senders import (
    EmailNotificationSender,
    InternalNotificationSender,
    NotificationSender,
    TelegramNotificationSender,
)
from .service import NotificationService
from .templates import render_notification

__all__ = [
    "ChannelHealth",
    "ChannelHealthStatus",
    "DeliveryResult",
    "DispatchReport",
    "EmailConfig",
    "EmailNotificationSender",
    "InternalNotificationSender",
    "NOTIFICATION_RETRY_POLICY_VERSION",
    "NOTIFICATION_ROUTING_POLICY_VERSION",
    "NotificationPurpose",
    "NotificationRetryPolicy",
    "NotificationRoute",
    "NotificationSender",
    "NotificationService",
    "NotificationSeverity",
    "RenderedNotification",
    "RetryDecision",
    "RetryEligibility",
    "TelegramConfig",
    "TelegramNotificationSender",
    "load_email_config",
    "notification_channel_health",
    "load_telegram_config",
    "evaluate_notification_retry",
    "render_notification",
    "route_notification_event",
]
