"""Tests for stock_tools module."""

from __future__ import annotations

import pytest

from mini_agent.tools.kline_db_tool import KLineDB
from mini_agent.tools.stock_tools import (
    AShareFundamentalsTool,
    AShareNewsTool,
    AShareQuoteTool,
    AShareScreenTool,
    AShareTechnicalSignalsTool,
    ConservativeTradePlanTool,
    LocalAShareDataBackend,
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

        with pytest.raises(NotImplementedError):
            asyncio.run(backend.get_technical_signals("600519"))


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


@pytest.mark.asyncio
async def test_local_screen_stocks_strategy_diff_and_scores(tmp_path):
    db_path = tmp_path / "local_screen.db"
    kdb = KLineDB(db_path=str(db_path))
    date = "2026-02-25"
    # hot momentum sample (should be filtered by value/low_vol)
    kdb.upsert_daily_kline("300001", date, 10.0, 12.5, 9.8, 12.2, 1000, 8e8)
    # balanced sample
    kdb.upsert_daily_kline("600001", date, 10.0, 10.6, 9.9, 10.2, 1000, 6e8)
    # low volatility sample
    kdb.upsert_daily_kline("600002", date, 10.0, 10.15, 9.95, 10.01, 1000, 4e8)
    # pullback sample
    kdb.upsert_daily_kline("000001", date, 10.0, 10.2, 9.6, 9.7, 1000, 5e8)

    backend = LocalAShareDataBackend(db_path=str(db_path))
    value_rows = await backend.screen_stocks(strategy="value", timestamp=f"{date}T15:00:00+08:00", max_results=10)
    low_vol_rows = await backend.screen_stocks(strategy="low_vol", timestamp=f"{date}T15:00:00+08:00", max_results=10)

    assert value_rows
    assert low_vol_rows
    assert any(float(x.get("涨跌幅") or 0.0) < 0 for x in value_rows)
    assert any(float(x.get("振幅") or 0.0) <= 5.0 for x in low_vol_rows)
    assert all(x.get("total_score") is not None for x in value_rows)
    assert all(isinstance(x.get("factor_scores"), dict) and x.get("factor_scores") for x in value_rows)


class TestAShareTechnicalSignalsTool:
    """Test AShareTechnicalSignalsTool class."""

    def test_tool_properties(self):
        """Test tool has correct properties."""
        backend = PlaceholderAShareDataBackend()
        tool = AShareTechnicalSignalsTool(backend)

        assert tool.name == "get_a_share_technical_signals"
        assert "技术信号" in tool.description

    def test_parameters_structure(self):
        """Test technical signal parameters."""
        backend = PlaceholderAShareDataBackend()
        tool = AShareTechnicalSignalsTool(backend)

        params = tool.parameters
        assert "ticker" in params["properties"]
        assert "window" in params["properties"]
        assert params["required"] == ["ticker"]

    @pytest.mark.asyncio
    async def test_execute_returns_error(self):
        """Test execute returns error for placeholder backend."""
        backend = PlaceholderAShareDataBackend()
        tool = AShareTechnicalSignalsTool(backend)

        result = await tool.execute(ticker="600519")
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
    
    assert len(tools) == 6
    
    tool_names = [t.name for t in tools]
    assert "screen_a_share_stocks" in tool_names
    assert "get_a_share_technical_signals" in tool_names
    assert "get_a_share_quote" in tool_names
    assert "get_a_share_fundamentals" in tool_names
    assert "get_a_share_news" in tool_names
    assert "build_conservative_trade_plan" in tool_names


def test_create_a_share_tools_with_backend():
    """Test creating tools with custom backend."""
    from mini_agent.tools.stock_tools import create_a_share_tools
    
    backend = PlaceholderAShareDataBackend()
    tools = create_a_share_tools(backend)
    
    assert len(tools) == 6


def test_create_a_share_tools_prefers_tushare_chain(monkeypatch):
    import mini_agent.tools.stock_tools as stock_tools
    from mini_agent.tools.stock_tools import create_a_share_tools

    class DummyAk(stock_tools.AShareDataBackend):
        async def get_quote(self, ticker: str, timestamp: str | None = None):
            return {}

        async def get_fundamentals(self, ticker: str, timestamp: str | None = None, period: str = "ttm"):
            return {}

        async def get_news(self, timestamp: str | None = None, ticker: str | None = None, limit: int = 10):
            return []

        async def screen_stocks(self, strategy: str, timestamp: str | None = None, max_results: int = 20):
            return []

        async def get_technical_signals(self, ticker: str, timestamp: str | None = None, window: int = 120):
            return {}

    class DummyTs(DummyAk):
        pass

    monkeypatch.setattr(stock_tools, "AkShareDataBackend", DummyAk)
    monkeypatch.setattr(stock_tools, "TuShareDataBackend", DummyTs)

    tools = create_a_share_tools()
    assert len(tools) == 6
    assert isinstance(tools[0].backend, stock_tools.PreferredAShareDataBackend)


def test_create_a_share_tools_falls_back_to_akshare_when_tushare_unavailable(monkeypatch):
    import mini_agent.tools.stock_tools as stock_tools
    from mini_agent.tools.stock_tools import create_a_share_tools

    class DummyAk(stock_tools.AShareDataBackend):
        async def get_quote(self, ticker: str, timestamp: str | None = None):
            return {}

        async def get_fundamentals(self, ticker: str, timestamp: str | None = None, period: str = "ttm"):
            return {}

        async def get_news(self, timestamp: str | None = None, ticker: str | None = None, limit: int = 10):
            return []

        async def screen_stocks(self, strategy: str, timestamp: str | None = None, max_results: int = 20):
            return []

        async def get_technical_signals(self, ticker: str, timestamp: str | None = None, window: int = 120):
            return {}

    class DummyTsFailure:
        def __init__(self, *args, **kwargs):
            raise ValueError("missing token")

    monkeypatch.setattr(stock_tools, "AkShareDataBackend", DummyAk)
    monkeypatch.setattr(stock_tools, "TuShareDataBackend", DummyTsFailure)

    tools = create_a_share_tools()
    assert len(tools) == 6
    # Local backend is wrapped as primary; remote AkShare is fallback.
    assert isinstance(tools[0].backend, stock_tools.PreferredAShareDataBackend)


@pytest.mark.asyncio
async def test_preferred_backend_fallback_on_empty_quote():
    from mini_agent.tools.stock_tools import AShareDataBackend, PreferredAShareDataBackend

    class EmptyPrimary(AShareDataBackend):
        async def get_quote(self, ticker: str, timestamp: str | None = None):
            return {"ticker": ticker, "quote": None}

        async def get_fundamentals(self, ticker: str, timestamp: str | None = None, period: str = "ttm"):
            return {"financial_indicator": None, "valuation_indicator": None}

        async def get_news(self, timestamp: str | None = None, ticker: str | None = None, limit: int = 10):
            return []

        async def screen_stocks(self, strategy: str, timestamp: str | None = None, max_results: int = 20):
            return []

        async def get_technical_signals(self, ticker: str, timestamp: str | None = None, window: int = 120):
            return {"signal_status": "no_data"}

    class Secondary(AShareDataBackend):
        async def get_quote(self, ticker: str, timestamp: str | None = None):
            return {"ticker": ticker, "quote": {"close": 10.0}, "source": "fallback"}

        async def get_fundamentals(self, ticker: str, timestamp: str | None = None, period: str = "ttm"):
            return {"financial_indicator": {"roe": 0.1}, "valuation_indicator": {"pe": 10}}

        async def get_news(self, timestamp: str | None = None, ticker: str | None = None, limit: int = 10):
            return [{"title": "ok"}]

        async def screen_stocks(self, strategy: str, timestamp: str | None = None, max_results: int = 20):
            return [{"代码": "600519"}]

        async def get_technical_signals(self, ticker: str, timestamp: str | None = None, window: int = 120):
            return {"signal_status": "ok"}

    backend = PreferredAShareDataBackend(primary=EmptyPrimary(), fallback=Secondary())
    quote = await backend.get_quote("600519.SH")
    assert quote.get("source") == "fallback"
    assert quote.get("quote", {}).get("close") == 10.0
