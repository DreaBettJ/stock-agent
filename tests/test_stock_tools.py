"""Tests for stock_tools module."""

from __future__ import annotations

import pytest

from mini_agent.tools.stock_tools import (
    AShareFundamentalsTool,
    AShareNewsTool,
    AShareQuoteTool,
    AShareScreenTool,
    ConservativeTradePlanTool,
    normalize_timestamp,
    PlaceholderAShareDataBackend,
)


class TestNormalizeTimestamp:
    """Test normalize_timestamp function."""

    def test_none_returns_current_time(self):
        """Test that None returns current timestamp."""
        result = normalize_timestamp(None)
        assert "T" in result  # ISO format
        assert "+08:00" in result

    def test_empty_string_returns_current_time(self):
        """Test that empty string returns current timestamp."""
        result = normalize_timestamp("")
        assert "T" in result
        assert "+08:00" in result

    def test_date_only_format(self):
        """Test date-only format YYYY-MM-DD."""
        result = normalize_timestamp("2024-01-15")
        assert result == "2024-01-15T00:00:00+08:00"

    def test_full_datetime_format(self):
        """Test full datetime format."""
        result = normalize_timestamp("2024-01-15T10:30:00")
        assert result == "2024-01-15T10:30:00+08:00"

    def test_datetime_with_timezone(self):
        """Test datetime with timezone."""
        result = normalize_timestamp("2024-01-15T10:30:00+08:00")
        assert result == "2024-01-15T10:30:00+08:00"

    def test_invalid_format_raises(self):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError):
            normalize_timestamp("invalid-date")


class TestPlaceholderAShareDataBackend:
    """Test PlaceholderAShareDataBackend class."""

    def test_raises_not_implemented(self):
        """Test that methods raise NotImplementedError."""
        import asyncio
        backend = PlaceholderAShareDataBackend()
        
        with pytest.raises(NotImplementedError):
            asyncio.run(backend.get_quote("600519"))
        
        with pytest.raises(NotImplementedError):
            asyncio.run(backend.get_fundamentals("600519"))
        
        with pytest.raises(NotImplementedError):
            asyncio.run(backend.get_news())
        
        with pytest.raises(NotImplementedError):
            asyncio.run(backend.screen_stocks("quality"))


class TestAShareQuoteTool:
    """Test AShareQuoteTool class."""

    def test_tool_properties(self):
        """Test tool has correct properties."""
        backend = PlaceholderAShareDataBackend()
        tool = AShareQuoteTool(backend)
        
        assert tool.name == "get_a_share_quote"
        assert "A-share" in tool.description
        assert isinstance(tool.parameters, dict)

    def test_parameters_structure(self):
        """Test parameters have correct structure."""
        backend = PlaceholderAShareDataBackend()
        tool = AShareQuoteTool(backend)
        
        params = tool.parameters
        assert params["type"] == "object"
        assert "ticker" in params["properties"]
        assert "timestamp" in params["properties"]
        assert params["required"] == ["ticker"]

    @pytest.mark.asyncio
    async def test_execute_returns_error(self):
        """Test execute returns error for placeholder backend."""
        backend = PlaceholderAShareDataBackend()
        tool = AShareQuoteTool(backend)
        
        result = await tool.execute(ticker="600519")
        
        assert result.success is False
        assert "TODO" in result.error or "NotImplementedError" in result.error


class TestAShareFundamentalsTool:
    """Test AShareFundamentalsTool class."""

    def test_tool_properties(self):
        """Test tool has correct properties."""
        backend = PlaceholderAShareDataBackend()
        tool = AShareFundamentalsTool(backend)
        
        assert tool.name == "get_a_share_fundamentals"
        assert "ticker" in tool.parameters["properties"]

    @pytest.mark.asyncio
    async def test_execute_returns_error(self):
        """Test execute returns error for placeholder backend."""
        backend = PlaceholderAShareDataBackend()
        tool = AShareFundamentalsTool(backend)
        
        result = await tool.execute(ticker="600519")
        
        assert result.success is False


class TestAShareNewsTool:
    """Test AShareNewsTool class."""

    def test_tool_properties(self):
        """Test tool has correct properties."""
        backend = PlaceholderAShareDataBackend()
        tool = AShareNewsTool(backend)
        
        assert tool.name == "get_a_share_news"

    @pytest.mark.asyncio
    async def test_execute_returns_error(self):
        """Test execute returns error for placeholder backend."""
        backend = PlaceholderAShareDataBackend()
        tool = AShareNewsTool(backend)
        
        result = await tool.execute()
        
        assert result.success is False


class TestAShareScreenTool:
    """Test AShareScreenTool class."""

    def test_tool_properties(self):
        """Test tool has correct properties."""
        backend = PlaceholderAShareDataBackend()
        tool = AShareScreenTool(backend)
        
        assert tool.name == "screen_a_share_stocks"

    def test_parameters_enum(self):
        """Test strategy parameter has correct enum values."""
        backend = PlaceholderAShareDataBackend()
        tool = AShareScreenTool(backend)
        
        strategy_param = tool.parameters["properties"]["strategy"]
        assert strategy_param["enum"] == ["quality", "dividend", "low_vol", "value"]

    @pytest.mark.asyncio
    async def test_execute_returns_error(self):
        """Test execute returns error for placeholder backend."""
        backend = PlaceholderAShareDataBackend()
        tool = AShareScreenTool(backend)
        
        result = await tool.execute(strategy="quality")
        
        assert result.success is False


class TestConservativeTradePlanTool:
    """Test ConservativeTradePlanTool class."""

    def test_tool_properties(self):
        """Test tool has correct properties."""
        tool = ConservativeTradePlanTool()
        
        assert tool.name == "build_conservative_trade_plan"

    def test_parameters_structure(self):
        """Test parameters have correct structure."""
        tool = ConservativeTradePlanTool()
        
        params = tool.parameters
        required = params["required"]
        
        assert "ticker" in required
        assert "current_price" in required
        assert "support_price" in required

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful trade plan generation."""
        tool = ConservativeTradePlanTool()
        
        result = await tool.execute(
            ticker="600519",
            current_price=100.0,
            support_price=95.0,
        )
        
        assert result.success is True
        assert "plan" in result.content.lower() or "600519" in result.content

    @pytest.mark.asyncio
    async def test_execute_with_resistance(self):
        """Test with resistance price."""
        tool = ConservativeTradePlanTool()
        
        result = await tool.execute(
            ticker="600519",
            current_price=100.0,
            support_price=95.0,
            resistance_price=110.0,
        )
        
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_account_size(self):
        """Test with account size for position sizing."""
        tool = ConservativeTradePlanTool()
        
        result = await tool.execute(
            ticker="600519",
            current_price=100.0,
            support_price=95.0,
            account_size=100000,
            risk_level="low",
        )
        
        assert result.success is True
        assert "risk" in result.content.lower() or "position" in result.content.lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_risk_level(self):
        """Test with invalid risk level defaults to low."""
        tool = ConservativeTradePlanTool()
        
        result = await tool.execute(
            ticker="600519",
            current_price=100.0,
            support_price=95.0,
            risk_level="invalid",
        )
        
        assert result.success is True


def test_create_a_share_tools():
    """Test creating all A-share tools."""
    from mini_agent.tools.stock_tools import create_a_share_tools
    
    tools = create_a_share_tools()
    
    assert len(tools) == 5
    
    tool_names = [t.name for t in tools]
    assert "screen_a_share_stocks" in tool_names
    assert "get_a_share_quote" in tool_names
    assert "get_a_share_fundamentals" in tool_names
    assert "get_a_share_news" in tool_names
    assert "build_conservative_trade_plan" in tool_names


def test_create_a_share_tools_with_backend():
    """Test creating tools with custom backend."""
    from mini_agent.tools.stock_tools import create_a_share_tools
    
    backend = PlaceholderAShareDataBackend()
    tools = create_a_share_tools(backend)
    
    assert len(tools) == 5
