"""Behavior tests for CLI event trigger command."""

from __future__ import annotations

import argparse
from pathlib import Path

from mini_agent.cli import handle_event_command
from mini_agent.session import SessionManager


def _make_args(
    *,
    event_type: str = "daily_review",
    all_sessions: bool = False,
    session_id: int | None = None,
    payload: str | None = None,
    debug: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        event_command="trigger",
        event_type=event_type,
        all_sessions=all_sessions,
        session_id=session_id,
        payload=payload,
        debug=debug,
    )


def test_event_trigger_single_session_requires_online_runtime(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "event_cli_offline.db"
    workspace_dir = tmp_path
    manager = SessionManager(db_path=str(db_path))
    session_id = manager.create_session(name="s1", system_prompt="p", mode="simulation")

    monkeypatch.setattr("mini_agent.cli.get_memory_db_path", lambda _workspace: db_path)
    args = _make_args(session_id=session_id, all_sessions=False)
    handle_event_command(args, workspace_dir)

    output = capsys.readouterr().out
    assert "skipped" in output
    assert "offline" in output


def test_event_trigger_single_session_enqueues_when_online(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "event_cli_online.db"
    workspace_dir = tmp_path
    manager = SessionManager(db_path=str(db_path))
    session_id = manager.create_session(name="s1", system_prompt="p", mode="simulation")
    manager.register_session_runtime(session_id, runtime_type="cli", pid=1001)

    monkeypatch.setattr("mini_agent.cli.get_memory_db_path", lambda _workspace: db_path)
    args = _make_args(session_id=session_id, all_sessions=False, payload='{"date":"2024-01-03"}')
    handle_event_command(args, workspace_dir)

    output = capsys.readouterr().out
    assert "queued" in output
    inbox = manager.list_event_inbox(session_id=session_id, limit=10)
    assert len(inbox) == 1
    assert inbox[0]["status"] == "pending"
    assert inbox[0]["event_type"] == "daily_review"
    assert inbox[0]["event_payload"]["date"] == "2024-01-03"


def test_event_trigger_all_targets_online_sessions_only(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "event_cli_all.db"
    workspace_dir = tmp_path
    manager = SessionManager(db_path=str(db_path))

    sid_online = manager.create_session(name="online", system_prompt="p", mode="simulation")
    sid_offline = manager.create_session(name="offline", system_prompt="p", mode="simulation")
    manager.register_session_runtime(sid_online, runtime_type="cli", pid=2001)

    monkeypatch.setattr("mini_agent.cli.get_memory_db_path", lambda _workspace: db_path)
    args = _make_args(all_sessions=True, payload='{"date":"2024-01-04"}')
    handle_event_command(args, workspace_dir)

    output = capsys.readouterr().out
    assert "target_sessions: 1" in output
    inbox_online = manager.list_event_inbox(session_id=sid_online, limit=10)
    inbox_offline = manager.list_event_inbox(session_id=sid_offline, limit=10)
    assert len(inbox_online) == 1
    assert len(inbox_offline) == 0


def test_event_trigger_idempotent_by_session_and_event_id(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "event_cli_idempotent.db"
    workspace_dir = tmp_path
    manager = SessionManager(db_path=str(db_path))
    sid = manager.create_session(name="idempotent", system_prompt="p", mode="simulation")
    manager.register_session_runtime(sid, runtime_type="cli", pid=3001)

    monkeypatch.setattr("mini_agent.cli.get_memory_db_path", lambda _workspace: db_path)
    args = _make_args(all_sessions=True, payload='{"date":"2024-01-05","event_id":"ev-2024-01-05"}')

    handle_event_command(args, workspace_dir)
    handle_event_command(args, workspace_dir)
    output = capsys.readouterr().out

    assert "duplicate event_id=ev-2024-01-05" in output
    inbox = manager.list_event_inbox(session_id=sid, limit=10)
    assert len(inbox) == 1
    assert inbox[0]["event_id"] == "ev-2024-01-05"


def test_event_trigger_debug_mode_bypasses_duplicate_check(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "event_cli_debug.db"
    workspace_dir = tmp_path
    manager = SessionManager(db_path=str(db_path))
    sid = manager.create_session(name="debug", system_prompt="p", mode="simulation")
    manager.register_session_runtime(sid, runtime_type="cli", pid=4001)

    monkeypatch.setattr("mini_agent.cli.get_memory_db_path", lambda _workspace: db_path)
    args = _make_args(all_sessions=True, payload='{"date":"2024-01-05","event_id":"ev-2024-01-05"}', debug=True)

    handle_event_command(args, workspace_dir)
    handle_event_command(args, workspace_dir)
    output = capsys.readouterr().out

    assert "debug mode enabled" in output
    inbox = manager.list_event_inbox(session_id=sid, limit=20)
    assert len(inbox) == 2
    assert all("ev-2024-01-05:debug:" in str(row["event_id"]) for row in inbox)


def test_critical_memory_idempotent_by_session_event_and_operation(tmp_path: Path):
    db_path = tmp_path / "critical_memory_idempotent.db"
    manager = SessionManager(db_path=str(db_path))
    sid = manager.create_session(name="critical", system_prompt="p", mode="simulation")

    mid1 = manager.record_critical_memory(
        sid,
        event_id="ev-1",
        event_type="daily_review",
        operation="buy",
        reason="signal matched",
        content='{"ticker":"600519","quantity":100}',
    )
    mid2 = manager.record_critical_memory(
        sid,
        event_id="ev-1",
        event_type="daily_review",
        operation="buy",
        reason="signal matched again",
        content='{"ticker":"600519","quantity":100}',
    )

    assert mid1 == mid2
    rows = manager.list_critical_memories(sid, limit=10)
    assert len(rows) == 1
