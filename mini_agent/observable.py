"""Observable mixin for monitoring and logging."""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ObservableMixin:
    """Mixin to add observability to classes."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._metrics: dict[str, Any] = {}

    def record_metric(self, name: str, value: Any) -> None:
        """Record a metric value."""
        self._metrics[name] = {
            "value": value,
            "timestamp": time.time(),
        }
        logger.debug(f"[METRIC] {self.__class__.__name__}.{name} = {value}")

    def get_metric(self, name: str) -> Any | None:
        """Get a recorded metric."""
        return self._metrics.get(name, {}).get("value")

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all recorded metrics."""
        return {k: v["value"] for k, v in self._metrics.items()}


def timed(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"[TIMED] {func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"[TIMED] {func.__name__} failed after {elapsed:.3f}s: {e}")
            raise

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"[TIMED] {func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"[TIMED] {func.__name__} failed after {elapsed:.3f}s: {e}")
            raise

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def logged(func: Callable) -> Callable:
    """Decorator to add logging to function calls."""
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any):
        logger.debug(f"[CALL] {func.__name__}(args={len(args)}, kwargs={list(kwargs.keys())})")
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"[OK] {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"[ERROR] {func.__name__}: {e}")
            raise

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any):
        logger.debug(f"[CALL] {func.__name__}(args={len(args)}, kwargs={list(kwargs.keys())})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"[OK] {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"[ERROR] {func.__name__}: {e}")
            raise

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


class HealthCheck:
    """Health check for the system."""

    @staticmethod
    def check_all() -> dict[str, Any]:
        """Run all health checks."""
        results = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {}
        }

        # Check database
        try:
            import sqlite3
            from .paths import get_default_memory_db_path
            
            db_path = get_default_memory_db_path()
            if db_path.exists():
                conn = sqlite3.connect(db_path)
                conn.execute("SELECT 1")
                conn.close()
                results["checks"]["database"] = "ok"
            else:
                results["checks"]["database"] = "not_found"
                results["status"] = "degraded"
        except Exception as e:
            results["checks"]["database"] = f"error: {e}"
            results["status"] = "unhealthy"

        # Check K-line data table (same DB as memory/session)
        try:
            from .paths import get_default_memory_db_path

            kline_path = get_default_memory_db_path()
            if kline_path.exists():
                conn = sqlite3.connect(kline_path)
                cursor = conn.execute("SELECT COUNT(*) FROM daily_kline")
                count = cursor.fetchone()[0]
                conn.close()
                results["checks"]["kline_db"] = f"ok ({count} records)"
            else:
                results["checks"]["kline_db"] = "not_found"
        except Exception as e:
            results["checks"]["kline_db"] = f"error: {e}"

        return results


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
