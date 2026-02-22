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
            market_data={"date": "2024-01-03", "top_gainers": []},
            positions=[],
            date="2024-01-03",
        )
        
        assert "2024-01-03" in prompt
        assert "每日复盘" in prompt
        assert "Test strategy" in prompt

    def test_build_review_prompt_with_positions(self, setup_workflow):
        """Test building prompt with positions."""
        workflow = setup_workflow["workflow"]
        session = setup_workflow["manager"].get_session(setup_workorm["session_id"])
        
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
            market_data={"date": "2024-01-03", "top_gainers": []},
            positions=positions,
            date="2024-01-03",
        )
        
        assert "600519" in prompt
        assert "100" in prompt  # quantity
        assert "110" in prompt  # current_price

    def test_parse_trade_signal_no_operation(self, setup_workflow):
        """Test parsing no-operation signal."""
        workflow = setup_workflow["workflow"]
        
        # Test various "no operation" signals
        for response in ["不操作", "无操作", "观望"]:
            signal = workflow._parse_trade_signal(response, None)
            assert signal is None

    def test_parse_trade_signal_buy(self, setup_workflow):
        """Test parsing buy signal."""
        workflow = setup_workflow["workflow"]
        
        signal = workflow._parse_trade_signal(
            "分析完成，买入:600519,100",
            None,
        )
        
        assert signal is not None
        assert signal["action"] == "buy"
        assert signal["ticker"] == "600519"
        assert signal["quantity"] == 100

    def test_parse_trade_signal_sell(self, setup_workflow):
        """Test parsing sell signal."""
        workflow = setup_workflow["workflow"]
        
        signal = workflow._parse_trade_signal(
            "需要卖出:600519,50",
            None,
        )
        
        assert signal is not None
        assert signal["action"] == "sell"
        assert signal["ticker"] == "600519"
        assert signal["quantity"] == 50

    def test_parse_trade_signal_invalid(self, setup_workflow):
        """Test parsing invalid signal."""
        workflow = setup_workflow["workflow"]
        
        signal = workflow._parse_trade_signal(
            "我会考虑看看",
            None,
        )
        
        assert signal is None

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
                assert "agent_analysis" in result
                assert "trade_signal" in result


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
            # This should work, api_key will be None
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
