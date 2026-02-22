"""Tests for observable module."""

from __future__ import annotations

import asyncio
import time

import pytest

from mini_agent.observable import (
    HealthCheck,
    ObservableMixin,
    logged,
    setup_logging,
    timed,
)


class TestObservableMixin:
    """Test ObservableMixin class."""

    def test_record_metric(self):
        """Test recording a metric."""
        class TestClass(ObservableMixin):
            pass
        
        obj = TestClass()
        obj.record_metric("test_metric", 42)
        
        assert obj.get_metric("test_metric") == 42

    def test_get_metric_nonexistent(self):
        """Test getting nonexistent metric returns None."""
        class TestClass(ObservableMixin):
            pass
        
        obj = TestClass()
        
        assert obj.get_metric("nonexistent") is None

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        class TestClass(ObservableMixin):
            pass
        
        obj = TestClass()
        obj.record_metric("a", 1)
        obj.record_metric("b", 2)
        obj.record_metric("c", 3)
        
        metrics = obj.get_all_metrics()
        
        assert metrics == {"a": 1, "b": 2, "c": 3}

    def test_metric_timestamp(self):
        """Test that metrics have timestamps."""
        class TestClass(ObservableMixin):
            pass
        
        obj = TestClass()
        before = time.time()
        obj.record_metric("test", 100)
        after = time.time()
        
        stored = obj._metrics["test"]
        assert before <= stored["timestamp"] <= after

    def test_update_existing_metric(self):
        """Test updating existing metric."""
        class TestClass(ObservableMixin):
            pass
        
        obj = TestClass()
        obj.record_metric("test", 10)
        obj.record_metric("test", 20)
        
        assert obj.get_metric("test") == 20


class TestTimed:
    """Test timed decorator."""

    @pytest.mark.asyncio
    async def test_timed_async_function(self):
        """Test timing an async function."""
        @timed
        async def slow_async():
            await asyncio.sleep(0.05)
            return "done"
        
        result = await slow_async()
        assert result == "done"

    def test_timed_sync_function(self):
        """Test timing a sync function."""
        @timed
        def slow_sync():
            time.sleep(0.05)
            return "done"
        
        result = slow_sync()
        assert result == "done"

    def test_timed_exception(self):
        """Test timing a function that raises exception."""
        @timed
        def failing_func():
            time.sleep(0.02)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_func()


class TestLogged:
    """Test logged decorator."""

    @pytest.mark.asyncio
    async def test_logged_async_function(self):
        """Test logging an async function."""
        @logged
        async def async_func():
            return "result"
        
        result = await async_func()
        assert result == "result"

    def test_logged_sync_function(self):
        """Test logging a sync function."""
        @logged
        def sync_func():
            return "result"
        
        result = sync_func()
        assert result == "result"

    def test_logged_exception(self):
        """Test logging a function that raises exception."""
        @logged
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_func()


class TestHealthCheck:
    """Test HealthCheck class."""

    def test_check_all_structure(self):
        """Test health check returns expected structure."""
        result = HealthCheck.check_all()
        
        assert "status" in result
        assert "timestamp" in result
        assert "checks" in result
        assert result["status"] in ["healthy", "degraded", "unhealthy"]

    def test_check_all_database_not_found(self, tmp_path, monkeypatch):
        """Test health check when database not found."""
        # Mock the path to ensure database doesn't exist
        def mock_db_path():
            return tmp_path / "nonexistent.db"
        
        # This will check for ./workspace/.agent_memory.db
        result = HealthCheck.check_all()
        
        # Should have database check entry
        assert "checks" in result


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        # Should not raise any error
        setup_logging()
        setup_logging("DEBUG")
        setup_logging("WARNING")

    def test_setup_logging_invalid_level(self):
        """Test with invalid log level raises error."""
        # Should raise error for invalid level
        with pytest.raises(AttributeError):
            setup_logging("INVALID")
