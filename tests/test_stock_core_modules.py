"""Tests for stock experiment core modules."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mini_agent.event_broadcaster import EventBroadcaster
from mini_agent.session import SessionManager
from mini_agent.tools.kline_db_tool import KLineDB, KLineQueryTool
from mini_agent.tools.sim_trade_tool import SimulateTradeTool


def test_session_manager_lifecycle(tmp_path: Path):
    db_path = tmp_path / "core.db"
    manager = SessionManager(db_path=str(db_path))

    sid = manager.create_session(
        name="ma-v1",
        system_prompt="test prompt",
        mode="simulation",
        risk_preference="high",
        stop_loss_pct=6.0,
        event_filter=["daily_review"],
    )

    session = manager.get_session(sid)
    assert session.name == "ma-v1"
    assert session.status == "stopped"
    assert session.is_listening is False
    assert session.event_filter == ["daily_review"]
    assert session.risk_preference == "high"
    assert session.stop_loss_pct == pytest.approx(6.0)

    manager.start_session(sid)
    running = manager.get_session(sid)
    assert running.status == "running"
    assert running.is_listening is True

    manager.stop_session(sid)
    stopped = manager.get_session(sid)
    assert stopped.status == "stopped"
    assert stopped.is_listening is False

    manager.delete_session(sid)
    with pytest.raises(KeyError):
        manager.get_session(sid)


def test_kline_query_tool_and_db(tmp_path: Path):
    db_path = tmp_path / "market.db"
    db = KLineDB(db_path=str(db_path))
    db.upsert_daily_kline("600519.SH", "2024-01-02", 10.0, 11.0, 9.8, 10.6, 1000, 10000)
    db.upsert_daily_kline("600519.SH", "2024-01-03", 10.7, 11.2, 10.5, 11.0, 1200, 12000)

    assert db.get_latest_price("600519") == 11.0
    assert db.get_next_open_price("600519.SH", "2024-01-02") == 10.7
    assert db.get_trading_days("2024-01-01", "2024-01-03") == ["2024-01-02", "2024-01-03"]

    tool = KLineQueryTool(db_path=str(db_path))
    result = asyncio.run(tool.execute(ticker="600519", start="2024-01-01", end="2024-01-03"))
    assert result.success
    assert "2024-01-02" in result.content
    assert "2024-01-03" in result.content


@pytest.mark.asyncio
async def test_simulate_trade_tool_buy_sell(tmp_path: Path):
    db_path = tmp_path / "sim.db"
    manager = SessionManager(db_path=str(db_path))
    session_id = manager.create_session(name="sim-1", system_prompt="p", mode="simulation", initial_capital=100000)

    kline = KLineDB(db_path=str(db_path))
    kline.upsert_daily_kline("600519", "2024-01-02", 10.0, 10.8, 9.9, 10.5, 1000, 10000)
    kline.upsert_daily_kline("600519", "2024-01-03", 11.0, 11.2, 10.8, 11.1, 1000, 11000)
    kline.upsert_daily_kline("600519", "2024-01-04", 12.0, 12.3, 11.8, 12.1, 1000, 12000)

    tool = SimulateTradeTool(db_path=str(db_path))

    buy_result = await tool.execute(
        session_id=session_id,
        action="buy",
        ticker="600519.SH",
        quantity=1000,
        trade_date="2024-01-02",
    )
    assert buy_result.success, buy_result.error

    after_buy = manager.get_session(session_id)
    assert after_buy.current_cash == pytest.approx(100000 - (11.0 * 1000 * 1.0003), rel=1e-6)

    sell_result = await tool.execute(
        session_id=session_id,
        action="sell",
        ticker="600519",
        quantity=600,
        trade_date="2024-01-03",
    )
    assert sell_result.success, sell_result.error

    bad_sell = await tool.execute(
        session_id=session_id,
        action="sell",
        ticker="600519",
        quantity=10000,
        trade_date="2024-01-03",
    )
    assert bad_sell.success is False
    assert bad_sell.error == "insufficient position"

    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        buy_row = conn.execute(
            "SELECT profit, reason, price_source FROM sim_trades WHERE session_id = ? AND action = 'buy' ORDER BY id ASC LIMIT 1",
            (str(session_id),),
        ).fetchone()
    assert buy_row is not None
    assert float(buy_row["profit"]) == pytest.approx(0.0)
    assert str(buy_row["reason"]).startswith("auto:buy:")
    assert str(buy_row["price_source"]) in {"next_open", "asof_close"}


@pytest.mark.asyncio
async def test_simulate_trade_tool_fallback_to_asof_close_when_next_open_missing(tmp_path: Path):
    db_path = tmp_path / "sim_fallback.db"
    manager = SessionManager(db_path=str(db_path))
    session_id = manager.create_session(name="sim-fallback", system_prompt="p", mode="simulation", initial_capital=100000)

    kline = KLineDB(db_path=str(db_path))
    # Only one day exists, so next_open is unavailable for 2024-01-02.
    kline.upsert_daily_kline("600519", "2024-01-02", 10.0, 10.8, 9.9, 10.5, 1000, 10000)

    tool = SimulateTradeTool(db_path=str(db_path))
    buy_result = await tool.execute(
        session_id=session_id,
        action="buy",
        ticker="600519.SH",
        quantity=1000,
        trade_date="2024-01-03",
    )
    assert buy_result.success, buy_result.error
    assert "price_source=asof_close" in buy_result.content

    after_buy = manager.get_session(session_id)
    assert after_buy.current_cash == pytest.approx(100000 - (10.5 * 1000 * 1.0003), rel=1e-6)


@pytest.mark.asyncio
async def test_simulate_trade_tool_accepts_backtest_session_mode(tmp_path: Path):
    db_path = tmp_path / "sim_backtest_mode.db"
    manager = SessionManager(db_path=str(db_path))
    session_id = manager.create_session(name="bt-sim", system_prompt="p", mode="backtest", initial_capital=100000)

    kline = KLineDB(db_path=str(db_path))
    kline.upsert_daily_kline("600519", "2024-01-02", 10.0, 10.8, 9.9, 10.5, 1000, 10000)
    kline.upsert_daily_kline("600519", "2024-01-03", 11.0, 11.2, 10.8, 11.1, 1000, 11000)

    tool = SimulateTradeTool(db_path=str(db_path))
    buy_result = await tool.execute(
        session_id=session_id,
        action="buy",
        ticker="600519",
        quantity=100,
        trade_date="2024-01-02",
    )
    assert buy_result.success, buy_result.error
    assert "SIM_TRADE_OK" in buy_result.content


@pytest.mark.asyncio
async def test_simulate_trade_tool_blocks_kechuang_buy_without_permission(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "sim_kechuang_block.db"
    manager = SessionManager(db_path=str(db_path))
    session_id = manager.create_session(name="sim-kc-block", system_prompt="p", mode="simulation", initial_capital=100000)
    kline = KLineDB(db_path=str(db_path))
    kline.upsert_daily_kline("688521", "2024-01-02", 240.0, 248.0, 239.0, 245.0, 1000, 10000)
    kline.upsert_daily_kline("688521", "2024-01-03", 247.0, 252.0, 246.0, 250.0, 1000, 11000)
    monkeypatch.delenv("MINI_AGENT_ENABLE_KECHUANG", raising=False)

    tool = SimulateTradeTool(db_path=str(db_path))
    result = await tool.execute(
        session_id=session_id,
        action="buy",
        ticker="688521",
        quantity=100,
        trade_date="2024-01-02",
    )
    assert result.success is False
    assert "kechuang buy requires" in str(result.error)


@pytest.mark.asyncio
async def test_simulate_trade_tool_allows_kechuang_buy_when_permission_enabled(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "sim_kechuang_allow.db"
    manager = SessionManager(db_path=str(db_path))
    session_id = manager.create_session(name="sim-kc-allow", system_prompt="p", mode="simulation", initial_capital=100000)
    kline = KLineDB(db_path=str(db_path))
    kline.upsert_daily_kline("688521", "2024-01-02", 240.0, 248.0, 239.0, 245.0, 1000, 10000)
    kline.upsert_daily_kline("688521", "2024-01-03", 247.0, 252.0, 246.0, 250.0, 1000, 11000)
    monkeypatch.setenv("MINI_AGENT_ENABLE_KECHUANG", "1")

    tool = SimulateTradeTool(db_path=str(db_path))
    result = await tool.execute(
        session_id=session_id,
        action="buy",
        ticker="688521",
        quantity=100,
        trade_date="2024-01-02",
    )
    assert result.success is True
    assert "SIM_TRADE_OK" in result.content


@pytest.mark.asyncio
async def test_event_broadcaster_filters_sessions(tmp_path: Path):
    db_path = tmp_path / "event.db"
    manager = SessionManager(db_path=str(db_path))

    sid_a = manager.create_session(
        name="a",
        system_prompt="p",
        mode="simulation",
        event_filter=["daily_review"],
    )
    sid_b = manager.create_session(
        name="b",
        system_prompt="p",
        mode="simulation",
        event_filter=["top_gainers"],
    )
    sid_c = manager.create_session(name="c", system_prompt="p", mode="simulation")

    manager.start_session(sid_a)
    manager.start_session(sid_b)
    # sid_c remains not listening

    called: list[str] = []

    async def trigger(session, event):
        called.append(session.session_id)
        return f"ok:{event['type']}"

    broadcaster = EventBroadcaster(session_manager=manager, trigger=trigger)
    results = await broadcaster.broadcast({"type": "daily_review", "date": "2024-01-02"})

    assert len(results) == 1
    assert called == [sid_a]
    assert results[0]["success"] is True
    assert results[0]["event_type"] == "daily_review"

    logs = manager.list_event_logs(session_id=sid_a, limit=10)
    assert len(logs) == 1
    assert logs[0]["event_type"] == "daily_review"
    assert logs[0]["success"] is True
    assert logs[0]["status"] == "succeeded"
    assert logs[0]["retry_count"] == 0

    # Trigger same event again: should be idempotent and not execute twice.
    results2 = await broadcaster.broadcast({"type": "daily_review", "date": "2024-01-02", "event_id": "daily_review:2024-01-02"})
    assert len(results2) == 1
    assert results2[0]["success"] is False
    assert "idempotent_skip" in results2[0]["error"]
    assert called == [sid_a]


def test_event_processing_retry_count(tmp_path: Path):
    db_path = tmp_path / "event_retry.db"
    manager = SessionManager(db_path=str(db_path))
    sid = manager.create_session(name="r1", system_prompt="p", mode="simulation")
    event = {"type": "daily_review", "date": "2024-01-03", "event_id": "daily_review:2024-01-03"}

    first = manager.begin_event_processing(sid, event)
    assert first["process"] is True
    manager.finalize_event_processing(
        sid,
        event,
        status="failed",
        success=False,
        error="network timeout",
        started_at=first["started_at"],
    )

    second = manager.begin_event_processing(sid, event)
    assert second["process"] is True
    assert second["retry_count"] == 1


def test_session_update_risk_profile(tmp_path: Path):
    db_path = tmp_path / "risk_profile.db"
    manager = SessionManager(db_path=str(db_path))
    sid = manager.create_session(name="risk", system_prompt="p", mode="simulation")

    manager.update_risk_profile(
        sid,
        risk_preference="low",
        max_single_loss_pct=1.5,
        single_position_cap_pct=18.0,
        stop_loss_pct=5.5,
        take_profit_pct=12.0,
        investment_horizon="短线",
    )
    manager.update_system_prompt(sid, "new prompt")

    session = manager.get_session(sid)
    assert session.system_prompt == "new prompt"
    assert session.risk_preference == "low"
    assert session.max_single_loss_pct == pytest.approx(1.5)
    assert session.single_position_cap_pct == pytest.approx(18.0)
    assert session.stop_loss_pct == pytest.approx(5.5)
    assert session.take_profit_pct == pytest.approx(12.0)
    assert session.investment_horizon == "短线"
    assert session.trade_notice_enabled is False

    manager.update_risk_profile(sid, trade_notice_enabled=True)
    session2 = manager.get_session(sid)
    assert session2.trade_notice_enabled is True


def test_trade_action_logs_roundtrip(tmp_path: Path):
    db_path = tmp_path / "trade_action.db"
    manager = SessionManager(db_path=str(db_path))
    sid = manager.create_session(name="ta", system_prompt="p", mode="simulation")
    manager.record_trade_action_log(
        session_id=sid,
        event_id="daily_review:2024-01-03",
        event_type="daily_review",
        action="buy",
        ticker="600519",
        quantity=100,
        trade_date="2024-01-03",
        status="succeeded",
        reason="unit_test",
        request_payload={"action": "buy"},
        response_payload={"ok": True},
    )
    rows = manager.list_trade_action_logs(sid, limit=10)
    assert len(rows) == 1
    assert rows[0]["action"] == "buy"
    assert rows[0]["ticker"] == "600519"
    assert rows[0]["request_payload"]["action"] == "buy"
