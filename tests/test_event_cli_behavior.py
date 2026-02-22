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
) -> argparse.Namespace:
    return argparse.Namespace(
        event_command="trigger",
        event_type=event_type,
        all_sessions=all_sessions,
        session_id=session_id,
        payload=payload,
    )


def test_event_trigger_single_session_respects_listening_flag(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "event_cli.db"
    workspace_dir = tmp_path
    manager = SessionManager(db_path=str(db_path))
    session_id = manager.create_session(name="s1", system_prompt="p", mode="simulation")
    # Not started: is_listening=False

    monkeypatch.setattr("mini_agent.cli.get_memory_db_path", lambda _workspace: db_path)
    args = _make_args(session_id=session_id, all_sessions=False)
    handle_event_command(args, workspace_dir)

    output = capsys.readouterr().out
    assert "skipped" in output
    assert "not listening" in output


def test_event_trigger_single_session_respects_event_filter(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "event_cli_filter.db"
    workspace_dir = tmp_path
    manager = SessionManager(db_path=str(db_path))
    session_id = manager.create_session(
        name="s1",
        system_prompt="p",
        mode="simulation",
        event_filter=["top_gainers"],
    )
    manager.start_session(session_id)

    monkeypatch.setattr("mini_agent.cli.get_memory_db_path", lambda _workspace: db_path)
    args = _make_args(session_id=session_id, all_sessions=False, event_type="daily_review")
    handle_event_command(args, workspace_dir)

    output = capsys.readouterr().out
    assert "skipped" in output
    assert "event_filter" in output


def test_event_trigger_all_uses_filtered_sessions_and_payload_date(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "event_cli_auto.db"
    workspace_dir = tmp_path
    manager = SessionManager(db_path=str(db_path))

    sid_match = manager.create_session(
        name="match",
        system_prompt="p",
        mode="simulation",
        event_filter=["daily_review"],
    )
    manager.start_session(sid_match)

    sid_mismatch = manager.create_session(
        name="mismatch",
        system_prompt="p",
        mode="simulation",
        event_filter=["top_gainers"],
    )
    manager.start_session(sid_mismatch)

    monkeypatch.setattr("mini_agent.cli.get_memory_db_path", lambda _workspace: db_path)
    monkeypatch.setattr("mini_agent.cli.build_decision_runtime", lambda *args, **kwargs: object())

    calls: list[tuple[int, str]] = []

    async def _fake_run_llm_decision(runtime, session_id: int, trading_date: str):
        calls.append((session_id, trading_date))
        return {
            "session_id": session_id,
            "agent_analysis": "ok",
            "trade_signal": None,
            "execution": None,
            "execution_error": None,
        }

    monkeypatch.setattr("mini_agent.cli.run_llm_decision", _fake_run_llm_decision)

    args = _make_args(
        all_sessions=True,
        payload='{"date":"2024-01-03"}',
    )
    handle_event_command(args, workspace_dir)

    output = capsys.readouterr().out
    assert "matched_sessions: 1" in output
    assert calls == [(sid_match, "2024-01-03")]
    assert sid_mismatch not in [sid for sid, _ in calls]

    logs = manager.list_event_logs(session_id=sid_match, limit=10)
    assert len(logs) == 1
    assert logs[0]["event_type"] == "daily_review"
    assert logs[0]["success"] is True
