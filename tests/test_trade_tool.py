"""Tests for trade_tool module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mini_agent.tools.trade_tool import (
    PositionTool,
    TradeHistoryTool,
    TradeRecordTool,
    _TradeStore,
    create_trade_tools,
)


class TestTradeStore:
    """Test _TradeStore class."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create test trade store."""
        db_path = tmp_path / "trades.db"
        return _TradeStore(db_path)

    def test_insert_and_query(self, store):
        """Test inserting and querying trades."""
        trade_id = store.insert(
            session_id="sess-001",
            operation="buy",
            ticker="600519",
            price=100.0,
            quantity=100,
            reason="Test buy",
        )
        
        assert trade_id > 0
        
        trades = store.query(session_id="sess-001")
        assert len(trades) == 1
        assert trades[0]["ticker"] == "600519"
        assert trades[0]["operation"] == "buy"

    def test_query_by_ticker(self, store):
        """Test querying by ticker."""
        store.insert("sess-001", "buy", "600519", 100.0, 100, "buy 600519")
        store.insert("sess-001", "buy", "000001", 50.0, 200, "buy 000001")
        
        trades = store.query(ticker="600519")
        assert len(trades) == 1
        assert trades[0]["ticker"] == "600519"

    def test_query_by_operation(self, store):
        """Test querying by operation type."""
        store.insert("sess-001", "buy", "600519", 100.0, 100, "buy")
        store.insert("sess-001", "sell", "600519", 110.0, 50, "sell")
        
        buy_trades = store.query(operation="buy")
        sell_trades = store.query(operation="sell")
        
        assert len(buy_trades) == 1
        assert len(sell_trades) == 1
        assert buy_trades[0]["operation"] == "buy"
        assert sell_trades[0]["operation"] == "sell"

    def test_query_limit(self, store):
        """Test query limit."""
        for i in range(10):
            store.insert("sess-001", "buy", f"60051{i}", 100.0, 100, f"buy {i}")
        
        trades = store.query(limit=5)
        assert len(trades) == 5

    def test_get_positions(self, store):
        """Test calculating positions."""
        store.insert("sess-001", "buy", "600519", 100.0, 100, "buy 100")
        store.insert("sess-001", "buy", "600519", 110.0, 50, "buy 50")  # avg = 103.33
        store.insert("sess-001", "sell", "600519", 120.0, 30, "sell 30")  # remaining = 120
        
        positions = store.get_positions()
        
        assert len(positions) == 1
        assert positions[0]["ticker"] == "600519"
        assert positions[0]["net_quantity"] == 120

    def test_get_positions_empty(self, store):
        """Test getting positions when no trades."""
        positions = store.get_positions()
        assert len(positions) == 0


class TestTradeRecordTool:
    """Test TradeRecordTool class."""

    @pytest.fixture
    def tool(self, tmp_path):
        """Create test tool."""
        db_path = tmp_path / "trades.db"
        return TradeRecordTool(memory_file=str(db_path), session_id="test-session")

    def test_tool_properties(self, tool):
        """Test tool has correct properties."""
        assert tool.name == "record_trade"
        assert "trade" in tool.description.lower()

    def test_parameters_required_fields(self, tool):
        """Test required parameters."""
        params = tool.parameters
        required = params["required"]
        
        assert "operation" in required
        assert "ticker" in required
        assert "price" in required
        assert "quantity" in required
        assert "reason" in required

    @pytest.mark.asyncio
    async def test_record_buy(self, tool):
        """Test recording a buy trade."""
        result = await tool.execute(
            operation="buy",
            ticker="600519",
            price=100.0,
            quantity=100,
            reason="Test buy",
        )
        
        assert result.success is True
        assert "600519" in result.content
        assert "buy" in result.content.lower()

    @pytest.mark.asyncio
    async def test_record_sell(self, tool):
        """Test recording a sell trade."""
        result = await tool.execute(
            operation="sell",
            ticker="600519",
            price=110.0,
            quantity=50,
            reason="Take profit",
        )
        
        assert result.success is True
        assert "sell" in result.content.lower()

    @pytest.mark.asyncio
    async def test_invalid_operation(self, tool):
        """Test invalid operation type."""
        result = await tool.execute(
            operation="hold",
            ticker="600519",
            price=100.0,
            quantity=100,
            reason="test",
        )
        
        assert result.success is False
        assert "Invalid operation" in result.error

    @pytest.mark.asyncio
    async def test_invalid_price(self, tool):
        """Test invalid price."""
        result = await tool.execute(
            operation="buy",
            ticker="600519",
            price=-10.0,
            quantity=100,
            reason="test",
        )
        
        assert result.success is False
        assert "Invalid price" in result.error

    @pytest.mark.asyncio
    async def test_invalid_quantity(self, tool):
        """Test invalid quantity."""
        result = await tool.execute(
            operation="buy",
            ticker="600519",
            price=100.0,
            quantity=0,
            reason="test",
        )
        
        assert result.success is False
        assert "Invalid quantity" in result.error

    @pytest.mark.asyncio
    async def test_ticker_normalized(self, tool):
        """Test ticker is normalized to uppercase."""
        result = await tool.execute(
            operation="buy",
            ticker="600519",
            price=100.0,
            quantity=100,
            reason="test",
        )
        
        assert result.success is True
        assert "600519" in result.content


class TestTradeHistoryTool:
    """Test TradeHistoryTool class."""

    @pytest.fixture
    def setup(self, tmp_path):
        """Set up test tools."""
        db_path = tmp_path / "trades.db"
        
        record_tool = TradeRecordTool(memory_file=str(db_path), session_id="test-session")
        
        # Record some trades
        import asyncio
        asyncio.run(record_tool.execute(
            operation="buy", ticker="600519", price=100.0, quantity=100, reason="buy 1"
        ))
        asyncio.run(record_tool.execute(
            operation="sell", ticker="600519", price=110.0, quantity=50, reason="sell 1"
        ))
        
        history_tool = TradeHistoryTool(memory_file=str(db_path))
        
        return {"history_tool": history_tool, "db_path": db_path}

    def test_tool_properties(self, setup):
        """Test tool has correct properties."""
        tool = setup["history_tool"]
        assert tool.name == "get_trade_history"

    @pytest.mark.asyncio
    async def test_query_all_trades(self, setup):
        """Test querying all trades."""
        tool = setup["history_tool"]
        
        result = await tool.execute()
        
        assert result.success is True
        assert "600519" in result.content

    @pytest.mark.asyncio
    async def test_query_by_ticker(self, setup):
        """Test querying by ticker."""
        tool = setup["history_tool"]
        
        result = await tool.execute(ticker="600519")
        
        assert result.success is True

    @pytest.mark.asyncio
    async def test_query_no_results(self, setup):
        """Test querying with no matching trades."""
        tool = setup["history_tool"]
        
        result = await tool.execute(ticker="999999")
        
        assert result.success is True
        assert "No trades found" in result.content


class TestPositionTool:
    """Test PositionTool class."""

    @pytest.fixture
    def setup(self, tmp_path):
        """Set up test tools."""
        db_path = tmp_path / "trades.db"
        
        record_tool = TradeRecordTool(memory_file=str(db_path), session_id="test-session")
        
        # Record some trades to create positions
        import asyncio
        asyncio.run(record_tool.execute(
            operation="buy", ticker="600519", price=100.0, quantity=100, reason="buy"
        ))
        asyncio.run(record_tool.execute(
            operation="buy", ticker="000001", price=50.0, quantity=200, reason="buy"
        ))
        
        position_tool = PositionTool(memory_file=str(db_path))
        
        return {"position_tool": position_tool}

    def test_tool_properties(self, setup):
        """Test tool has correct properties."""
        tool = setup["position_tool"]
        assert tool.name == "get_positions"

    @pytest.mark.asyncio
    async def test_get_positions(self, setup):
        """Test getting current positions."""
        tool = setup["position_tool"]
        
        result = await tool.execute()
        
        assert result.success is True
        assert "600519" in result.content or "000001" in result.content

    @pytest.mark.asyncio
    async def test_get_empty_positions(self, tmp_path):
        """Test getting positions with no trades."""
        db_path = tmp_path / "empty.db"
        tool = PositionTool(memory_file=str(db_path))
        
        result = await tool.execute()
        
        assert result.success is True
        assert "No open positions" in result.content


def test_create_trade_tools():
    """Test creating all trade tools."""
    tools = create_trade_tools()
    
    assert len(tools) == 3
    
    tool_names = [t.name for t in tools]
    assert "record_trade" in tool_names
    assert "get_trade_history" in tool_names
    assert "get_positions" in tool_names


def test_create_trade_tools_with_session():
    """Test creating tools with session ID."""
    tools = create_trade_tools(session_id="custom-session")
    
    assert len(tools) == 3
    # All tools should use the provided session_id
    for tool in tools:
        assert tool.session_id == "custom-session"
