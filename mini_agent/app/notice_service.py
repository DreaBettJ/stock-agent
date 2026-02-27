"""Notice delivery service (SMTP channel)."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

from mini_agent.config import SmtpNoticeConfig


def send_smtp_notice(*, cfg: SmtpNoticeConfig, subject: str, body: str) -> None:
    """Send one email via SMTP.

    Raises:
        ValueError: if required smtp fields are missing.
        smtplib.SMTPException: on smtp failures.
    """
    if not cfg.enabled:
        return
    if not cfg.host:
        raise ValueError("smtp host is required")
    if not cfg.from_addr:
        raise ValueError("smtp from_addr is required")
    if not cfg.to_addrs:
        raise ValueError("smtp to_addrs is required")

    msg = EmailMessage()
    msg["From"] = cfg.from_addr
    msg["To"] = ", ".join(cfg.to_addrs)
    msg["Subject"] = subject
    msg.set_content(body)

    timeout = float(cfg.timeout_seconds or 10.0)
    if cfg.use_ssl:
        with smtplib.SMTP_SSL(cfg.host, int(cfg.port), timeout=timeout) as smtp:
            if cfg.username:
                smtp.login(cfg.username, cfg.password)
            smtp.send_message(msg)
        return

    with smtplib.SMTP(cfg.host, int(cfg.port), timeout=timeout) as smtp:
        if cfg.use_starttls:
            smtp.starttls()
        if cfg.username:
            smtp.login(cfg.username, cfg.password)
        smtp.send_message(msg)

