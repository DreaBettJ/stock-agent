"""Tests for runtime session factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_agent.app.runtime_factory import build_runtime_session_context
from mini_agent.session import SessionManager


def test_build_runtime_session_context_create_and_close(tmp_path: Path):
    db_path = tmp_path / "runtime_factory_create.db"

    ctx = build_runtime_session_context(
        memory_db_path=db_path,
        system_prompt="test-prompt",
        runtime_type="daemon",
        session_name_prefix="worker",
        mode="simulation",
    )

    assert ctx.session_id > 0
    assert ctx.attached is False
    assert ctx.session_name.startswith("worker-")
    assert ctx.effective_system_prompt == "test-prompt"

    manager = SessionManager(db_path=str(db_path))
    session = manager.get_session(ctx.session_id)
    assert session.status == "running"
    assert manager.is_session_online(ctx.session_id, heartbeat_ttl_seconds=3600) is True

    ctx.close(stop_session=True)

    session_after = manager.get_session(ctx.session_id)
    assert session_after.status == "stopped"
    assert manager.is_session_online(ctx.session_id, heartbeat_ttl_seconds=3600) is False


def test_build_runtime_session_context_attach_uses_existing_prompt(tmp_path: Path):
    db_path = tmp_path / "runtime_factory_attach.db"
    manager = SessionManager(db_path=str(db_path))
    sid = manager.create_session(
        name="seed",
        system_prompt="existing-prompt",
        mode="simulation",
    )

    ctx = build_runtime_session_context(
        memory_db_path=db_path,
        system_prompt="new-prompt",
        attach_session_id=sid,
        runtime_type="service",
        mode="simulation",
    )
    assert ctx.attached is True
    assert ctx.session_id == sid
    assert ctx.effective_system_prompt == "existing-prompt"
    ctx.close(stop_session=True)


def test_build_runtime_session_context_attach_mode_mismatch(tmp_path: Path):
    db_path = tmp_path / "runtime_factory_mode.db"
    manager = SessionManager(db_path=str(db_path))
    sid = manager.create_session(
        name="seed-backtest",
        system_prompt="seed",
        mode="backtest",
    )

    with pytest.raises(ValueError, match="session mode mismatch"):
        build_runtime_session_context(
            memory_db_path=db_path,
            system_prompt="x",
            attach_session_id=sid,
            mode="simulation",
        )


def test_build_runtime_session_context_attach_without_mode_check(tmp_path: Path):
    db_path = tmp_path / "runtime_factory_mode_relaxed.db"
    manager = SessionManager(db_path=str(db_path))
    sid = manager.create_session(
        name="seed-backtest",
        system_prompt="seed",
        mode="backtest",
    )

    ctx = build_runtime_session_context(
        memory_db_path=db_path,
        system_prompt="x",
        attach_session_id=sid,
        mode=None,
    )
    assert ctx.session_id == sid
    assert ctx.attached is True
    ctx.close(stop_session=True)


def test_build_runtime_session_context_create_with_profile_kwargs(tmp_path: Path):
    db_path = tmp_path / "runtime_factory_profile.db"
    ctx = build_runtime_session_context(
        memory_db_path=db_path,
        system_prompt="profile",
        session_name="profile-session",
        mode="simulation",
        initial_capital=500000.0,
        create_session_kwargs={
            "risk_preference": "high",
            "stop_loss_pct": 6.0,
            "take_profit_pct": 18.0,
            "single_position_cap_pct": 30.0,
            "max_single_loss_pct": 3.0,
            "investment_horizon": "长线",
        },
    )
    manager = SessionManager(db_path=str(db_path))
    session = manager.get_session(ctx.session_id)
    assert session.name == "profile-session"
    assert session.initial_capital == pytest.approx(500000.0)
    assert session.risk_preference == "high"
    assert session.stop_loss_pct == pytest.approx(6.0)
    assert session.take_profit_pct == pytest.approx(18.0)
    ctx.close(stop_session=True)
