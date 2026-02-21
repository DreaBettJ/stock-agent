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
        event_filter=["daily_review"],
    )

    session = manager.get_session(sid)
    assert session.name == "ma-v1"
    assert session.status == "stopped"
    assert session.is_listening is False
    assert session.event_filter == ["daily_review"]

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
