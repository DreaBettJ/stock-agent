"""Tests for backtest module."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_agent.backtest import (
    BacktestEngine,
    HistoricalEventGenerator,
    PerformanceAnalyzer,
)
from mini_agent.session import SessionManager
from mini_agent.tools.kline_db_tool import KLineDB


class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer class."""

    def test_calculate_empty_equity_curve(self):
        """Test calculation with empty equity curve."""
        result = PerformanceAnalyzer.calculate([], [])
        assert "error" in result

    def test_calculate_basic_metrics(self):
        """Test basic performance metrics calculation."""
        trades = []
        equity_curve = [
            {"date": "2024-01-01", "value": 100000},
            {"date": "2024-01-02", "value": 105000},
            {"date": "2024-01-03", "value": 103000},
            {"date": "2024-01-04", "value": 110000},
        ]
        
        result = PerformanceAnalyzer.calculate(trades, equity_curve)
        
        assert result["initial_capital"] == 100000
        assert result["final_value"] == 110000
        assert result["total_return"] == 0.1  # 10%
        assert "annual_return" in result
        assert "max_drawdown" in result

    def test_calculate_with_trades(self):
        """Test calculation with trades."""
        trades = [
            {"action": "sell", "profit": 1000},
            {"action": "sell", "profit": -500},
            {"action": "sell", "profit": 800},
        ]
        equity_curve = [
            {"date": "2024-01-01", "value": 100000},
            {"date": "2024-01-02", "value": 101000},
            {"date": "2024-01-03", "value": 101500},
            {"date": "2024-01-04", "value": 103000},
        ]
        
        result = PerformanceAnalyzer.calculate(trades, equity_curve)
        
        assert result["total_trades"] == 3
        assert result["winning_trades"] == 2
        assert result["losing_trades"] == 1
        assert result["win_rate"] == pytest.approx(2/3)

    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation."""
        trades = []
        equity_curve = [
            {"date": "2024-01-01", "value": 100000},
            {"date": "2024-01-02", "value": 110000},  # peak
            {"date": "2024-01-03", "value": 100000},  # drawdown
            {"date": "2024-01-04", "value": 105000},
        ]
        
        result = PerformanceAnalyzer.calculate(trades, equity_curve)
        
        # Max drawdown from 110000 to 100000 = ~9.09%
        assert result["max_drawdown"] < 0

    def test_calculate_profit_factor(self):
        """Test profit factor calculation."""
        trades = [
            {"action": "sell", "profit": 1000},
            {"action": "sell", "profit": -200},
            {"action": "sell", "profit": 500},
        ]
        equity_curve = [
            {"date": "2024-01-01", "value": 100000},
            {"date": "2024-01-02", "value": 101300},
            {"date": "2024-01-04", "value": 103000},
        ]
        
        result = PerformanceAnalyzer.calculate(trades, equity_curve)
        
        # Profit factor = (1000 + 500) / 200 = 7.5
        assert result["profit_factor"] == pytest.approx(7.5)


class TestHistoricalEventGenerator:
    """Test HistoricalEventGenerator class."""

    @pytest.fixture
    def kline_db(self, tmp_path):
        """Create test KLineDB."""
        db_path = tmp_path / "test_kline.db"
        db = KLineDB(db_path=str(db_path))
        
        # Insert test data
        db.upsert_daily_kline("600519", "2024-01-02", 10.0, 11.0, 9.8, 10.5, 1000, 10000)
        db.upsert_daily_kline("600519", "2024-01-03", 10.6, 11.2, 10.4, 10.8, 1100, 11000)
        db.upsert_daily_kline("000001", "2024-01-02", 20.0, 21.0, 19.8, 20.5, 2000, 20000)
        
        return db

    def test_generate_daily_review_event(self, kline_db):
        """Test generating daily review event."""
        generator = HistoricalEventGenerator(kline_db)
        
        event = generator.generate_daily_review_event("2024-01-03")
        
        assert event["type"] == "daily_review"
        assert event["date"] == "2024-01-03"
        assert "top_gainers" in event

    def test_generate_daily_review_no_previous_day(self, kline_db):
        """Test generating event with no previous trading day."""
        generator = HistoricalEventGenerator(kline_db)
        
        event = generator.generate_daily_review_event("2024-01-02")
        
        assert event["type"] == "daily_review"
        # May have error if no previous day

    def test_get_previous_trading_day(self, kline_db):
        """Test getting previous trading day."""
        generator = HistoricalEventGenerator(kline_db)
        
        prev = generator._get_previous_trading_day("2024-01-03")
        assert prev == "2024-01-02"

    def test_get_top_gainers(self, kline_db):
        """Test getting top gainers."""
        generator = HistoricalEventGenerator(kline_db)
        
        gainers = generator._get_top_gainers("2024-01-02")
        
        assert len(gainers) > 0
        # Check structure
        for g in gainers:
            assert "ticker" in g
            assert "close" in g
            assert "change_pct" in g


class TestBacktestEngine:
    """Test BacktestEngine class."""

    @pytest.fixture
    def setup_engine(self, tmp_path):
        """Set up backtest engine with test data."""
        # Create session manager
        session_db = tmp_path / "session.db"
        manager = SessionManager(db_path=str(session_db))
        
        # Create session
        session_id = manager.create_session(
            name="test-backtest",
            system_prompt="Test strategy",
            mode="backtest",
            initial_capital=100000,
        )
        
        # Create KLine DB
        kline_db_path = tmp_path / "kline.db"
        kline_db = KLineDB(db_path=str(kline_db_path))
        
        # Insert test K-line data
        for date in ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]:
            kline_db.upsert_daily_kline("600519", date, 10.0, 11.0, 9.8, 10.5, 1000, 10000)
            kline_db.upsert_daily_kline("000001", date, 20.0, 21.0, 19.5, 20.8, 2000, 40000)
        
        # Create mock event broadcaster
        mock_broadcaster = MagicMock()
        mock_broadcaster.broadcast = AsyncMock(return_value=[])
        
        engine = BacktestEngine(
            session_manager=manager,
            kline_db=kline_db,
            event_broadcaster=mock_broadcaster,
        )
        
        return {
            "engine": engine,
            "manager": manager,
            "session_id": session_id,
            "kline_db": kline_db,
        }

    @pytest.mark.asyncio
    async def test_run_backtest(self, setup_engine):
        """Test running backtest."""
        setup = setup_engine
        
        result = await setup["engine"].run(
            session_id=setup["session_id"],
            start_date="2024-01-02",
            end_date="2024-01-05",
        )
        
        assert "session_id" in result
        assert "start_date" in result
        assert "end_date" in result
        assert "performance" in result
        assert "equity_curve" in result
        assert "trades" in result

    @pytest.mark.asyncio
    async def test_run_backtest_no_trading_days(self, setup_engine):
        """Test backtest with no trading days."""
        setup = setup_engine
        
        result = await setup["engine"].run(
            session_id=setup["session_id"],
            start_date="2025-01-01",
            end_date="2025-01-02",
        )
        
        assert "error" in result

    @pytest.mark.asyncio
    async def test_calculate_position_value(self, setup_engine):
        """Test calculating position value."""
        setup = setup_engine
        
        value = setup["engine"]._calculate_position_value(setup["session_id"])
        
        assert value >= 0

    @pytest.mark.asyncio
    async def test_get_session_trades(self, setup_engine):
        """Test getting session trades."""
        setup = setup_engine
        
        trades = setup["engine"]._get_session_trades(setup["session_id"])
        
        assert isinstance(trades, list)

    @pytest.mark.asyncio
    async def test_backtest_generates_run_scoped_event_ids(self, setup_engine):
        setup = setup_engine
        captured_events: list[dict] = []

        class _Broadcaster:
            async def trigger_session(self, session, event):
                captured_events.append(dict(event))
                return {"ok": True}

        setup["engine"].event_broadcaster = _Broadcaster()
        await setup["engine"].run(
            session_id=setup["session_id"],
            start_date="2024-01-02",
            end_date="2024-01-03",
        )

        assert captured_events
        assert all(str(e.get("event_id") or "").startswith(f"backtest:{setup['session_id']}:") for e in captured_events)
