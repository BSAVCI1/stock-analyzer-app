"""Persistent notification delivery."""

from .config import (
    load_email_config,
    load_telegram_config,
)
from .models import (
    DeliveryResult,
    DispatchReport,
    EmailConfig,
    RenderedNotification,
    TelegramConfig,
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
    "DeliveryResult",
    "DispatchReport",
    "EmailConfig",
    "EmailNotificationSender",
    "InternalNotificationSender",
    "NOTIFICATION_ROUTING_POLICY_VERSION",
    "NotificationPurpose",
    "NotificationRoute",
    "NotificationSender",
    "NotificationService",
    "NotificationSeverity",
    "RenderedNotification",
    "TelegramConfig",
    "TelegramNotificationSender",
    "load_email_config",
    "load_telegram_config",
    "render_notification",
    "route_notification_event",
]
