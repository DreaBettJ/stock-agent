"""Tests for low-coupling evolution use-case prompt service."""

from __future__ import annotations

import json
from pathlib import Path

from mini_agent.app.evolution_service import EvolutionUseCaseService
from mini_agent.session import SessionManager


def _seed_failed_event(manager: SessionManager, session_id: int, event_id: str, error: str) -> None:
    event = {"type": "daily_review", "event_id": event_id, "date": "2026-02-22"}
    token = manager.begin_event_processing(session_id, event)
    manager.finalize_event_processing(
        session_id,
        event,
        status="failed",
        success=False,
        error=error,
        started_at=token.get("started_at"),
    )


def test_generate_use_cases_and_render_prompt(tmp_path):
    db_path = tmp_path / "evo_uc.db"
    manager = SessionManager(db_path=str(db_path))
    sid = manager.create_session(name="evo", system_prompt="p", mode="simulation")
    _seed_failed_event(manager, sid, "ev-1", "simulate trade failed: session_id must be integer, got: x")
    _seed_failed_event(manager, sid, "ev-2", "simulate trade failed: insufficient cash")

    svc = EvolutionUseCaseService(db_path=str(db_path))
    created = svc.generate_use_cases_with_rules(limit=100)
    assert len(created) >= 2

    rows = svc.list_use_cases(enabled=1, limit=20)
    assert len(rows) >= 2

    block = svc.render_prompt_block(limit=20)
    assert "session_binding_error" in block or "insufficient_cash_guard" in block


def test_enable_disable_use_case(tmp_path):
    db_path = tmp_path / "evo_uc_toggle.db"
    svc = EvolutionUseCaseService(db_path=str(db_path))
    ok, uid = svc.create_use_case(
        use_case_id="uc-test-1",
        issue_type="execution_failure_general",
        title="test",
        trigger_pattern="failure",
        prompt_snippet="fallback once",
    )
    assert ok is True

    svc.set_use_case_enabled(uid, False)
    rows_off = svc.list_use_cases(enabled=0, limit=10)
    assert any(r["use_case_id"] == uid for r in rows_off)

    svc.set_use_case_enabled(uid, True)
    rows_on = svc.list_use_cases(enabled=1, limit=10)
    assert any(r["use_case_id"] == uid for r in rows_on)


def test_ingest_intercept_log_to_structured_trace(tmp_path):
    db_path = tmp_path / "evo_trace.db"
    svc = EvolutionUseCaseService(db_path=str(db_path))

    log_path = tmp_path / "agent_intercept_s9.jsonl"
    lines = [
        {
            "timestamp": "2026-02-22T22:00:00.000+08:00",
            "event": "before_send",
            "step": 1,
            "tool_count": 2,
        },
        {
            "timestamp": "2026-02-22T22:00:01.000+08:00",
            "event": "after_response",
            "step": 1,
            "finish_reason": "tool_use",
            "tool_call_count": 1,
            "tool_calls": ["simulate_trade"],
            "content_chars": 20,
            "thinking_chars": 100,
        },
        {
            "timestamp": "2026-02-22T22:00:02.000+08:00",
            "event": "after_tool",
            "step": 1,
            "tool_name": "simulate_trade",
            "tool_call_id": "call_1",
            "success": False,
            "result_chars": 0,
            "error_chars": 42,
        },
    ]
    log_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines), encoding="utf-8")

    r1 = svc.ingest_intercept_log(session_id=9, log_path=Path(log_path))
    assert r1["inserted"] == 3
    r2 = svc.ingest_intercept_log(session_id=9, log_path=Path(log_path))
    assert r2["inserted"] == 0

    summary = svc.trace_summary(session_id=9, limit=10)
    assert summary["total_rows"] == 3
    assert any(row["event"] == "after_tool" for row in summary["events"])
    assert any((row.get("tool_name") or "") == "simulate_trade" for row in summary["tool_failures"])
