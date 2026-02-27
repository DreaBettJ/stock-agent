"""Tests for auto_trading module."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.auto_trading import AutoTradingWorkflow
from mini_agent.session import SessionManager
from mini_agent.tools.kline_db_tool import KLineDB


class TestAutoTradingWorkflow:
    """Test AutoTradingWorkflow class."""

    @pytest.fixture
    def setup_workflow(self, tmp_path):
        """Set up workflow with test data."""
        # Create session manager
        session_db = tmp_path / "session.db"
        manager = SessionManager(db_path=str(session_db))
        
        # Create session
        session_id = manager.create_session(
            name="test-trading",
            system_prompt="Test strategy",
            mode="simulation",
            initial_capital=100000,
        )
        
        # Create KLine DB
        kline_db_path = tmp_path / "kline.db"
        kline_db = KLineDB(db_path=str(kline_db_path))
        
        # Insert test K-line data
        for date in ["2024-01-02", "2024-01-03", "2024-01-04"]:
            kline_db.upsert_daily_kline("600519", date, 10.0, 11.0, 9.8, 10.5, 1000, 10000)
        
        # Create workflow
        workflow = AutoTradingWorkflow(
            session_manager=manager,
            kline_db=kline_db,
            llm_provider="anthropic",
            api_key="test-key",
            model="test-model",
        )
        
        return {
            "workflow": workflow,
            "manager": manager,
            "session_id": session_id,
            "kline_db": kline_db,
        }

    def test_init(self, setup_workflow):
        """Test workflow initialization."""
        workflow = setup_workflow["workflow"]
        
        assert workflow.session_manager is not None
        assert workflow.kline_db is not None
        assert workflow.api_key == "test-key"
        assert workflow.model == "test-model"

    def test_prepare_market_data(self, setup_workflow):
        """Test preparing market data."""
        workflow = setup_workflow["workflow"]
        
        data = workflow._prepare_market_data("2024-01-03")
        
        assert "date" in data
        assert "top_gainers" in data
        assert "total_stocks" in data

    def test_get_top_gainers(self, setup_workflow):
        """Test getting top gainers."""
        workflow = setup_workflow["workflow"]
        
        gainers = workflow._get_top_gainers("2024-01-03")
        
        # May be empty if no data for that date
        assert isinstance(gainers, list)

    def test_get_positions_empty(self, setup_workflow):
        """Test getting positions when none exist."""
        workflow = setup_workflow["workflow"]
        session_id = setup_workflow["session_id"]
        
        positions = workflow._get_positions(session_id)
        
        assert positions == []

    def test_build_review_prompt(self, setup_workflow):
        """Test building review prompt."""
        workflow = setup_workflow["workflow"]
        session = setup_workflow["manager"].get_session(setup_workflow["session_id"])
        
        prompt = workflow._build_review_prompt(
            session=session,
            market_data={"date": "2024-01-03", "top_gainers": [], "technical_signals": []},
            positions=[],
            date="2024-01-03",
            pipeline={"mainlines": [], "leaders": [], "position_diagnostics": {}, "rebalance_proposal": {}},
        )
        
        assert "2024-01-03" in prompt
        assert "每日复盘" in prompt
        assert "技术信号摘要" in prompt
        assert "【你的策略】" not in prompt
        assert "## 一、账户信息" not in prompt
        assert "全市场扫描快照（全A样本）" in prompt

    def test_build_review_prompt_with_positions(self, setup_workflow):
        """Test building prompt with positions."""
        workflow = setup_workflow["workflow"]
        session = setup_workflow["manager"].get_session(setup_workflow["session_id"])
        
        positions = [
            {
                "ticker": "600519",
                "quantity": 100,
                "avg_cost": 100.0,
                "current_price": 110.0,
                "profit": 1000.0,
                "profit_rate": 10.0,
                "klines": [],
            }
        ]
        
        prompt = workflow._build_review_prompt(
            session=session,
            market_data={"date": "2024-01-03", "top_gainers": [], "technical_signals": []},
            positions=positions,
            date="2024-01-03",
            pipeline={"mainlines": [], "leaders": [], "position_diagnostics": {}, "rebalance_proposal": {}},
        )
        
        assert "600519" in prompt
        assert "100" in prompt  # quantity
        assert "110" in prompt  # current_price
        assert "市场分层结果" in prompt

    def test_build_daily_review_system_prompt(self, setup_workflow):
        """System prompt should contain strategy and optional execution constraints."""
        workflow = setup_workflow["workflow"]
        strategy = "Test strategy block"

        system_prompt = workflow._build_daily_review_system_prompt(
            strategy_prompt=strategy,
            allow_tool_execution=True,
        )

        assert "Test strategy block" in system_prompt
        assert "【执行约束】" in system_prompt
        assert "严禁调用任何交易执行工具" in system_prompt
        assert "只能输出结构化决策 JSON" in system_prompt

    @pytest.mark.asyncio
    async def test_build_technical_signals_for_market(self, setup_workflow):
        """Test technical signal summary building with local K-line data."""
        workflow = setup_workflow["workflow"]
        # Seed enough k-line rows so signal can be calculated from local db.
        for idx in range(1, 60):
            day = f"2024-02-{idx:02d}" if idx <= 29 else f"2024-03-{(idx-29):02d}"
            base = 10.0 + idx * 0.05
            setup_workflow["kline_db"].upsert_daily_kline("600519", day, base, base + 0.3, base - 0.2, base + 0.1, 1000, 10000)
        market_data = {"top_gainers": [{"ticker": "600519", "change_pct": 3.2}]}
        items = await workflow._build_technical_signals_for_market(market_data, "2024-03-31")
        assert len(items) == 1
        assert items[0]["ticker"] == "600519"
        assert isinstance(items[0]["score"], int)

    def test_build_full_market_scan_snapshot(self, setup_workflow):
        workflow = setup_workflow["workflow"]
        snapshot = workflow._build_full_market_scan_snapshot("2024-01-03")
        assert snapshot["status"] == "ok"
        assert snapshot["total_stocks"] >= 1
        assert isinstance(snapshot.get("top_up"), list)

    def test_rule_engine_uses_strategy_dynamic_thresholds(self, setup_workflow):
        workflow = setup_workflow["workflow"]
        manager = setup_workflow["manager"]
        sid = setup_workflow["session_id"]
        manager.update_system_prompt(sid, "当前策略模板：激进策略")

        market_data = {
            "top_gainers": [{"ticker": "600519", "close": 10.0, "change_pct": 12.0}],
            "technical_signals": [{"ticker": "600519", "trend": "up", "score": 52}],
        }
        snap = workflow._build_rule_engine_snapshot(
            session_id=sid,
            market_data=market_data,
            position_diagnostics={"summary": "no_positions"},
            trading_date="2024-01-03",
        )
        assert snap["status"] == "ok"
        assert snap["policy"]["chase_limit_pct"] == 18.0
        assert snap["policy"]["min_score"] == 50
        assert snap["actions"]
        assert not any("chase_limit" in x for x in snap["veto_reasons"])

    def test_parse_trade_decision_no_operation(self, setup_workflow):
        """Test parsing no-operation signal."""
        workflow = setup_workflow["workflow"]
        
        # Test various "no operation" signals
        for response in ["不操作", "无操作", "观望"]:
            decision = workflow._parse_trade_decision(response, "2024-01-03")
            assert decision["decision"] == "hold"
            assert decision["actions"] == []

    def test_parse_trade_decision_buy_legacy(self, setup_workflow):
        """Test parsing buy signal."""
        workflow = setup_workflow["workflow"]
        
        decision = workflow._parse_trade_decision(
            "分析完成，买入:600519,100",
            "2024-01-03",
        )
        
        assert decision["decision"] == "trade"
        assert decision["actions"][0]["action"] == "buy"
        assert decision["actions"][0]["ticker"] == "600519"
        assert decision["actions"][0]["quantity"] == 100

    def test_parse_trade_decision_sell_legacy(self, setup_workflow):
        """Test parsing sell signal."""
        workflow = setup_workflow["workflow"]
        
        decision = workflow._parse_trade_decision(
            "需要卖出:600519,50",
            "2024-01-03",
        )
        
        assert decision["decision"] == "trade"
        assert decision["actions"][0]["action"] == "sell"
        assert decision["actions"][0]["ticker"] == "600519"
        assert decision["actions"][0]["quantity"] == 50

    def test_parse_trade_decision_json(self, setup_workflow):
        workflow = setup_workflow["workflow"]
        response = """
```json
{"decision":"trade","summary":"test","risk_warnings":["r1"],"actions":[{"action":"buy","ticker":"600519.SH","quantity":100}]}
```
"""
        decision = workflow._parse_trade_decision(response, "2024-01-03")
        assert decision["decision"] == "trade"
        assert len(decision["actions"]) == 1
        assert decision["actions"][0]["ticker"] == "600519.SH"
        assert decision["actions"][0]["trade_date"] == "2024-01-03"

    def test_parse_trade_decision_invalid(self, setup_workflow):
        """Test parsing invalid signal."""
        workflow = setup_workflow["workflow"]
        
        decision = workflow._parse_trade_decision(
            "我会考虑看看",
            "2024-01-03",
        )
        
        assert decision["decision"] == "hold"
        assert decision["actions"] == []

    @pytest.mark.asyncio
    async def test_trigger_daily_review_structure(self, setup_workflow):
        """Test trigger daily review returns correct structure."""
        workflow = setup_workflow["workflow"]
        
        # Mock the LLM and Agent to avoid actual API calls
        with patch("mini_agent.auto_trading.LLMClient") as mock_llm:
            with patch("mini_agent.auto_trading.Agent") as mock_agent:
                mock_agent_instance = MagicMock()
                mock_agent_instance.run = AsyncMock(return_value="不操作")
                mock_agent.return_value = mock_agent_instance
                
                result = await workflow.trigger_daily_review(setup_workflow["session_id"])
                
                assert "session_id" in result
                assert "date" in result
                assert "requested_date" in result
                assert "agent_analysis" in result
                assert "trade_signal" in result

    @pytest.mark.asyncio
    async def test_trigger_daily_review_fallback_to_previous_market_date(self, setup_workflow):
        """If requested date has no market data, fallback to latest available date."""
        workflow = setup_workflow["workflow"]

        with patch("mini_agent.auto_trading.LLMClient"):
            with patch("mini_agent.auto_trading.Agent") as mock_agent:
                mock_agent_instance = MagicMock()
                mock_agent_instance.run = AsyncMock(return_value="不操作")
                mock_agent.return_value = mock_agent_instance

                result = await workflow.trigger_daily_review(
                    setup_workflow["session_id"],
                    trading_date="2024-01-10",
                )

                assert result["requested_date"] == "2024-01-10"
                assert result["date"] == "2024-01-04"

    @pytest.mark.asyncio
    async def test_trigger_daily_review_records_trade_action_logs(self, setup_workflow):
        workflow = setup_workflow["workflow"]
        manager = setup_workflow["manager"]
        sid = setup_workflow["session_id"]

        class _ExecResult:
            success = True
            content = "SIM_TRADE_OK id=1 action=buy ticker=600519 qty=100"
            error = None

        trade_executor = MagicMock()
        trade_executor.execute = AsyncMock(return_value=_ExecResult())

        with patch("mini_agent.auto_trading.LLMClient"):
            with patch("mini_agent.auto_trading.Agent") as mock_agent:
                mock_agent_instance = MagicMock()
                mock_agent_instance.run = AsyncMock(
                    return_value='{"decision":"trade","actions":[{"action":"buy","ticker":"600519","quantity":100,"reason":"test"}]}'
                )
                mock_agent_instance.messages = []
                mock_agent.return_value = mock_agent_instance

                result = await workflow.trigger_daily_review(
                    sid,
                    trading_date="2024-01-03",
                    event_id="daily_review:2024-01-03",
                    auto_execute=True,
                    trade_executor=trade_executor,
                )
                assert len(result.get("execution_results") or []) == 1

        logs = manager.list_trade_action_logs(sid, limit=10)
        assert len(logs) >= 1
        assert logs[0]["event_id"] == "daily_review:2024-01-03"
        assert logs[0]["action"] == "buy"

    @pytest.mark.asyncio
    async def test_trigger_daily_review_ignores_agent_tool_messages(self, setup_workflow):
        workflow = setup_workflow["workflow"]
        manager = setup_workflow["manager"]
        sid = setup_workflow["session_id"]

        class _ExecResult:
            success = True
            content = "SIM_TRADE_OK id=2 action=buy ticker=600519 qty=100"
            error = None

        trade_executor = MagicMock()
        trade_executor.execute = AsyncMock(return_value=_ExecResult())

        with patch("mini_agent.auto_trading.LLMClient"):
            with patch("mini_agent.auto_trading.Agent") as mock_agent:
                mock_agent_instance = MagicMock()
                mock_agent_instance.run = AsyncMock(
                    return_value='{"decision":"trade","actions":[{"action":"buy","ticker":"600519","quantity":100,"reason":"from_json"}]}'
                )
                mock_agent_instance.messages = [
                    MagicMock(role="tool", name="simulate_trade", content="Error: quantity must be > 0")
                ]
                mock_agent.return_value = mock_agent_instance

                result = await workflow.trigger_daily_review(
                    sid,
                    trading_date="2024-01-03",
                    event_id="daily_review:2024-01-03",
                    auto_execute=True,
                    trade_executor=trade_executor,
                )
                assert len(result.get("tool_executions") or []) == 0
                assert len(result.get("execution_results") or []) == 1
                assert result["execution_results"][0]["success"] is True

        logs = manager.list_trade_action_logs(sid, limit=10)
        assert len(logs) >= 1
        assert logs[0]["status"] == "succeeded"
        assert logs[0]["reason"] == "from_json"

    @pytest.mark.asyncio
    async def test_trigger_daily_review_does_not_recover_actions_from_tool_message(self, setup_workflow):
        workflow = setup_workflow["workflow"]
        sid = setup_workflow["session_id"]

        with patch("mini_agent.auto_trading.LLMClient"):
            with patch("mini_agent.auto_trading.Agent") as mock_agent:
                mock_agent_instance = MagicMock()
                mock_agent_instance.run = AsyncMock(return_value="Task couldn't be completed after 3 steps.")
                mock_agent_instance.messages = [
                    MagicMock(
                        role="tool",
                        name="simulate_trade",
                        content="SIM_TRADE_OK id=1 session=1 action=buy ticker=688521 qty=100 price=247.000",
                    )
                ]
                mock_agent.return_value = mock_agent_instance

                result = await workflow.trigger_daily_review(
                    sid,
                    trading_date="2024-01-03",
                    event_id="daily_review:2024-01-03",
                    auto_execute=True,
                    trade_executor=MagicMock(),
                )
                assert not result.get("execution")
                assert not result.get("execution_error")
                assert len(result.get("trade_actions") or []) == 0
                assert len(result.get("execution_results") or []) == 0


class TestAutoTradingWorkflowConfig:
    """Test AutoTradingWorkflow configuration loading."""

    def test_load_api_key_from_config(self, tmp_path):
        """Test loading API key from config file."""
        # Create a temporary config file
        config_content = """
api_key: "config-api-key"
api_base: "https://api.test.com"
model: "test-model"
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)
        
        # Mock the config path search
        with patch.object(Path, "exists", return_value=False):
            with patch("pathlib.Path.__truediv__", side_effect=[config_path, config_path]):
                # Should still work but will fall back
                pass

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        with tempfile.TemporaryDirectory():
            # Force config fallback path to fail, then api_key should stay None.
            with patch("mini_agent.auto_trading.Config.from_yaml", side_effect=RuntimeError("no config")):
                workflow = AutoTradingWorkflow(
                    session_manager=MagicMock(),
                    kline_db=MagicMock(),
                    api_key=None,
                )
                assert workflow.api_key is None


class TestAutoTradingWorkflowIntegration:
    """Integration tests for AutoTradingWorkflow."""

    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self, tmp_path):
        """Test full workflow in simulation mode."""
        # Setup
        session_db = tmp_path / "session.db"
        manager = SessionManager(db_path=str(session_db))
        
        session_id = manager.create_session(
            name="sim-test",
            system_prompt="Test",
            mode="simulation",
            initial_capital=100000,
        )
        
        kline_db_path = tmp_path / "kline.db"
        kline_db = KLineDB(db_path=str(kline_db_path))
        
        # Add some K-line data
        kline_db.upsert_daily_kline("600519", "2024-01-02", 10.0, 11.0, 9.8, 10.5, 1000, 10000)
        
        workflow = AutoTradingWorkflow(
            session_manager=manager,
            kline_db=kline_db,
            api_key="test",
        )
        
        # Test market data preparation
        data = workflow._prepare_market_data("2024-01-03")
        assert data is not None
        
        # Test positions
        positions = workflow._get_positions(session_id)
        assert positions == []
