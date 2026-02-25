from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from mini_agent.app.decision_service import DecisionRuntime, _ensure_market_data_fresh, _has_market_data_on_date
from mini_agent.config import AgentConfig, Config, LLMConfig, ToolsConfig
from mini_agent.tools.kline_db_tool import KLineDB


def test_has_market_data_on_date(tmp_path: Path):
    db_path = tmp_path / "k.db"
    kdb = KLineDB(db_path=str(db_path))
    kdb.upsert_daily_kline("600519", "2026-02-24", 10, 11, 9, 10.5, 1000, 10000)
    assert _has_market_data_on_date(kdb, "2026-02-24") is True
    assert _has_market_data_on_date(kdb, "2026-02-25") is False


def test_ensure_market_data_fresh_calls_sync_when_missing(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "k2.db"
    kdb = KLineDB(db_path=str(db_path))
    runtime = DecisionRuntime(
        session_manager=MagicMock(),
        kline_db=kdb,
        workflow=MagicMock(),
        trade_tool=MagicMock(),
    )

    cfg = Config(
        llm=LLMConfig(api_key="x"),
        agent=AgentConfig(),
        tools=ToolsConfig(tushare_token="t"),
    )
    monkeypatch.setattr("mini_agent.app.decision_service.Config.from_yaml", lambda _: cfg)
    monkeypatch.setattr("mini_agent.app.decision_service.Config.get_default_config_path", lambda: Path("/tmp/fake.yaml"))

    called = {"n": 0}

    def _fake_sync(kline_db, trading_date, token):
        called["n"] += 1
        kline_db.upsert_daily_kline("000001", trading_date, 10, 10, 10, 10, 1, 1)
        return (1, 1)

    monkeypatch.setattr("mini_agent.app.decision_service.sync_all_for_trade_date_with_tushare", _fake_sync)
    _ensure_market_data_fresh(runtime=runtime, trading_date="2026-02-25")
    assert called["n"] == 1
    assert _has_market_data_on_date(kdb, "2026-02-25") is True
