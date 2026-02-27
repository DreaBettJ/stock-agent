"""Tests for SMTP notice service."""

from __future__ import annotations

from mini_agent.app.notice_service import send_smtp_notice
from mini_agent.config import SmtpNoticeConfig


class _FakeSMTP:
    def __init__(self, host, port, timeout=10):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.logged_in = None
        self.sent = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def login(self, username, password):
        self.logged_in = (username, password)

    def send_message(self, msg):
        self.sent = True
        assert msg["From"] == "bot@example.com"
        assert "you@example.com" in str(msg["To"])


def test_send_smtp_notice_ssl(monkeypatch):
    created: list[_FakeSMTP] = []

    def _factory(host, port, timeout=10):
        inst = _FakeSMTP(host, port, timeout=timeout)
        created.append(inst)
        return inst

    monkeypatch.setattr("mini_agent.app.notice_service.smtplib.SMTP_SSL", _factory)
    cfg = SmtpNoticeConfig(
        enabled=True,
        host="smtp.example.com",
        port=465,
        username="bot",
        password="pwd",
        use_ssl=True,
        from_addr="bot@example.com",
        to_addrs=["you@example.com"],
    )
    send_smtp_notice(cfg=cfg, subject="s", body="b")
    assert len(created) == 1
    assert created[0].sent is True
    assert created[0].logged_in == ("bot", "pwd")


def test_send_smtp_notice_missing_required_fields():
    cfg = SmtpNoticeConfig(enabled=True, host="", from_addr="", to_addrs=[])
    try:
        send_smtp_notice(cfg=cfg, subject="s", body="b")
        assert False, "expected ValueError"
    except ValueError:
        assert True

