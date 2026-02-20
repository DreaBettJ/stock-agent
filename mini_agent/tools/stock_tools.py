"""A-share stock tools skeleton.

This module provides a conservative, extensible tool framework for:
- stock selection
- stock analysis
- trade action planning

Data acquisition methods are intentionally left as placeholders so callers can
plug in real market/fundamental/news providers later.

Backtest convention:
- `timestamp` is optional for all stock tools.
- If omitted, tools run in realtime mode (using current time).
- If provided, backend implementations should resolve data "as-of" that timestamp.
"""

from __future__ import annotations

import json
import asyncio
import contextlib
import io
import logging
import os
import random
import time
import re
from abc import ABC
from datetime import datetime, timedelta
from typing import Any

from .base import Tool, ToolResult

logger = logging.getLogger(__name__)


def normalize_timestamp(timestamp: str | None = None) -> str:
    """Normalize timestamp to ISO 8601 with UTC+08:00.

    Supported inputs:
    - YYYY-MM-DD
    - YYYY-MM-DDTHH:MM:SS
    - Any valid datetime.fromisoformat string (with/without timezone)
    """
    if timestamp is None:
        # Realtime mode default: use current local timestamp.
        return datetime.now().astimezone().isoformat()
    ts = timestamp.strip()
    if not ts:
        return datetime.now().astimezone().isoformat()

    if len(ts) == 10 and ts[4] == "-" and ts[7] == "-":
        return f"{ts}T00:00:00+08:00"

    try:
        dt = datetime.fromisoformat(ts)
    except ValueError as e:
        raise ValueError(
            "Invalid timestamp format. Use ISO 8601, e.g. 2025-01-15T10:30:00+08:00 or 2025-01-15"
        ) from e

    if dt.tzinfo is None:
        return dt.isoformat() + "+08:00"
    return dt.isoformat()


# 说明：
# Tool 层只负责参数校验和输出格式，不直接绑定具体数据源。
# 真实行情/财报/新闻抓取放到 AShareDataBackend，实现后可无缝替换。
class AShareDataBackend(ABC):
    """Pluggable backend for A-share data.

    Replace or subclass these methods to connect your own data source.
    """

    async def get_quote(self, ticker: str, timestamp: str | None = None) -> dict[str, Any]:
        raise NotImplementedError("TODO: implement quote fetch logic")

    async def get_fundamentals(self, ticker: str, timestamp: str | None = None, period: str = "ttm") -> dict[str, Any]:
        raise NotImplementedError("TODO: implement fundamentals fetch logic")

    async def get_news(self, timestamp: str | None = None, ticker: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        raise NotImplementedError("TODO: implement news fetch logic")

    async def screen_stocks(self, strategy: str, timestamp: str | None = None, max_results: int = 20) -> list[dict[str, Any]]:
        raise NotImplementedError("TODO: implement stock screening logic")


class PlaceholderAShareDataBackend(AShareDataBackend):
    """Default backend used before real data providers are wired in.

    默认占位后端：便于先把 Agent 工具链跑通，再逐步接入实际数据服务。
    """


class AkShareDataBackend(AShareDataBackend):
    """AkShare-based backend for A-share data."""

    # 进程内全局节流：多工具并发时也共用一个速率控制，降低被限流/封禁风险。
    _rate_limit_lock = asyncio.Lock()
    _last_call_monotonic = 0.0

    def __init__(self):
        try:
            import akshare as ak  # type: ignore
        except Exception as e:
            raise ImportError(
                "akshare is not installed. Run `uv sync` (or install akshare) before using stock tools."
            ) from e
        self.ak = ak
        # 可通过环境变量覆盖，便于在不同网络环境下调节频率。
        self.min_interval_seconds = float(os.getenv("AKSHARE_MIN_INTERVAL_SECONDS", "1.5"))
        self.jitter_seconds = float(os.getenv("AKSHARE_JITTER_SECONDS", "0.5"))
        self.max_retries = int(os.getenv("AKSHARE_MAX_RETRIES", "3"))
        self.retry_base_delay = float(os.getenv("AKSHARE_RETRY_BASE_DELAY_SECONDS", "2.0"))
        self.request_timeout_seconds = float(os.getenv("AKSHARE_REQUEST_TIMEOUT_SECONDS", "25"))
        self.slow_log_threshold_seconds = float(os.getenv("AKSHARE_SLOW_LOG_THRESHOLD_SECONDS", "4.0"))
        self.playwright_spot_timeout_seconds = float(os.getenv("PLAYWRIGHT_SPOT_TIMEOUT_SECONDS", "20"))
        self.playwright_spot_url = os.getenv(
            "PLAYWRIGHT_SPOT_URL",
            (
                "https://push2.eastmoney.com/api/qt/clist/get?"
                "pn=1&pz=5000&po=1&np=1&fltt=2&invt=2&fid=f3&"
                "fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&"
                "fields=f12,f14,f2,f3,f8,f7,f9,f23,f20,f6"
            ),
        )

    @staticmethod
    def _ticker_to_symbol(ticker: str) -> str:
        ts = ticker.strip().upper()
        if "." in ts:
            return ts.split(".")[0]
        if ts.startswith(("SH", "SZ")):
            return ts[2:]
        return ts

    async def _apply_rate_limit(self) -> None:
        """Apply global call spacing and random jitter for anti-ban behavior."""
        async with self._rate_limit_lock:
            now = asyncio.get_running_loop().time()
            elapsed = now - self._last_call_monotonic
            required_wait = max(0.0, self.min_interval_seconds - elapsed)
            if required_wait > 0:
                await asyncio.sleep(required_wait)
            if self.jitter_seconds > 0:
                await asyncio.sleep(random.uniform(0, self.jitter_seconds))
            self.__class__._last_call_monotonic = asyncio.get_running_loop().time()

    async def _run_akshare_call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        def _call():
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return fn(*args, **kwargs)

        call_name = getattr(fn, "__name__", str(fn))
        last_exception: Exception | None = None
        for attempt in range(self.max_retries + 1):
            await self._apply_rate_limit()
            started = time.perf_counter()
            try:
                result = await asyncio.wait_for(asyncio.to_thread(_call), timeout=self.request_timeout_seconds)
                elapsed = time.perf_counter() - started
                if elapsed >= self.slow_log_threshold_seconds:
                    logger.warning(
                        "AkShare slow call: fn=%s elapsed=%.2fs attempt=%s timeout=%.1fs",
                        call_name,
                        elapsed,
                        attempt + 1,
                        self.request_timeout_seconds,
                    )
                else:
                    logger.info("AkShare call ok: fn=%s elapsed=%.2fs attempt=%s", call_name, elapsed, attempt + 1)
                return result
            except Exception as e:
                last_exception = e
                elapsed = time.perf_counter() - started
                logger.warning(
                    "AkShare call failed: fn=%s elapsed=%.2fs attempt=%s/%s error=%s",
                    call_name,
                    elapsed,
                    attempt + 1,
                    self.max_retries + 1,
                    repr(e),
                )
                if attempt >= self.max_retries:
                    break
                # 指数退避 + 随机抖动，降低瞬时重试风暴。
                delay = self.retry_base_delay * (2**attempt)
                if self.jitter_seconds > 0:
                    delay += random.uniform(0, self.jitter_seconds)
                logger.info("AkShare retry scheduled: fn=%s next_delay=%.2fs", call_name, delay)
                await asyncio.sleep(delay)
        raise RuntimeError(f"AkShare call failed after {self.max_retries + 1} attempts: {last_exception}")

    @staticmethod
    def _parse_json_or_jsonp(raw_text: str) -> dict[str, Any]:
        text = (raw_text or "").strip()
        if not text:
            return {}
        if text.startswith("{"):
            return json.loads(text)
        # JSONP fallback: callback({...})
        match = re.search(r"\((\{.*\})\)\s*;?\s*$", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)

    async def _fetch_spot_with_playwright(self) -> Any:
        """Fetch A-share spot snapshot via Playwright request API."""
        try:
            from playwright.async_api import async_playwright  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Playwright not available. Install dependency and browser: "
                "`uv sync && uv run playwright install chromium`."
            ) from e

        await self._apply_rate_limit()
        started = time.perf_counter()
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                try:
                    response = await context.request.get(self.playwright_spot_url, timeout=int(self.playwright_spot_timeout_seconds * 1000))
                    body = await response.text()
                finally:
                    await context.close()
                    await browser.close()

            payload = self._parse_json_or_jsonp(body)
            diff = (((payload or {}).get("data") or {}).get("diff")) or []
            records = []
            for item in diff:
                records.append(
                    {
                        "代码": self._safe_value(item.get("f12")),
                        "名称": self._safe_value(item.get("f14")),
                        "最新价": self._safe_value(item.get("f2")),
                        "涨跌幅": self._safe_value(item.get("f3")),
                        "换手率": self._safe_value(item.get("f8")),
                        "振幅": self._safe_value(item.get("f7")),
                        "市盈率-动态": self._safe_value(item.get("f9")),
                        "市净率": self._safe_value(item.get("f23")),
                        "总市值": self._safe_value(item.get("f20")),
                        "成交额": self._safe_value(item.get("f6")),
                    }
                )

            import pandas as pd  # type: ignore

            df = pd.DataFrame(records)
            elapsed = time.perf_counter() - started
            if elapsed >= self.slow_log_threshold_seconds:
                logger.warning("Playwright slow call: fn=spot_snapshot elapsed=%.2fs rows=%s", elapsed, len(df))
            else:
                logger.info("Playwright call ok: fn=spot_snapshot elapsed=%.2fs rows=%s", elapsed, len(df))
            return df
        except Exception as e:
            elapsed = time.perf_counter() - started
            logger.warning("Playwright spot fetch failed: elapsed=%.2fs error=%s", elapsed, repr(e))
            raise

    @staticmethod
    def _pick_time_column(df: Any) -> str | None:
        for col in ("日期", "时间", "date", "Date", "datetime", "trade_date"):
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _safe_value(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        try:
            # pandas NaN -> None
            if value != value:  # noqa: PLR0124
                return None
        except Exception:
            pass
        return value

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in row.items():
            result[str(k)] = self._safe_value(v)
        return result

    @staticmethod
    def _to_numeric(df: Any, cols: list[str]) -> Any:
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(",", "", regex=False)
                df[col] = df[col].astype(float)
        return df

    @staticmethod
    def _score_desc(series: Any) -> Any:
        # Higher is better
        if series is None or len(series) == 0:
            return series
        return series.rank(pct=True, method="average")

    @staticmethod
    def _score_asc(series: Any) -> Any:
        # Lower is better
        if series is None or len(series) == 0:
            return series
        return (1 - series.rank(pct=True, method="average")).clip(lower=0, upper=1)

    def _asof_filter(self, df: Any, timestamp: str) -> Any:
        time_col = self._pick_time_column(df)
        if not time_col:
            return df
        work = df.copy()
        work[time_col] = work[time_col].astype(str)
        asof_dt = datetime.fromisoformat(timestamp).replace(tzinfo=None)
        parsed = None
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y%m%d"):
            try:
                parsed = work[time_col].map(lambda x: datetime.strptime(str(x)[:19], fmt))
                break
            except Exception:
                parsed = None
        if parsed is None:
            return work
        work = work.assign(__parsed_time=parsed)
        work = work[work["__parsed_time"] <= asof_dt]
        return work.drop(columns=["__parsed_time"]) if "__parsed_time" in work.columns else work

    async def get_quote(self, ticker: str, timestamp: str | None = None) -> dict[str, Any]:
        timestamp = normalize_timestamp(timestamp)
        symbol = self._ticker_to_symbol(ticker)
        asof_dt = datetime.fromisoformat(timestamp).replace(tzinfo=None)
        start_dt = asof_dt - timedelta(days=180)

        hist_df = await self._run_akshare_call(
            self.ak.stock_zh_a_hist,
            symbol=symbol,
            period="daily",
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=asof_dt.strftime("%Y%m%d"),
            adjust="",
        )
        if hist_df is None or hist_df.empty:
            return {"ticker": ticker, "timestamp": timestamp, "source": "akshare:stock_zh_a_hist", "quote": None}

        hist_df = self._asof_filter(hist_df, timestamp)
        if hist_df.empty:
            return {"ticker": ticker, "timestamp": timestamp, "source": "akshare:stock_zh_a_hist", "quote": None}

        last = hist_df.iloc[-1]
        quote = {
            "date": self._safe_value(last.get("日期")),
            "open": self._safe_value(last.get("开盘")),
            "high": self._safe_value(last.get("最高")),
            "low": self._safe_value(last.get("最低")),
            "close": self._safe_value(last.get("收盘")),
            "volume": self._safe_value(last.get("成交量")),
            "amount": self._safe_value(last.get("成交额")),
            "turnover_rate": self._safe_value(last.get("换手率")),
        }
        return {
            "ticker": ticker,
            "symbol": symbol,
            "timestamp": timestamp,
            "source": "akshare:stock_zh_a_hist",
            "quote": quote,
        }

    async def get_fundamentals(self, ticker: str, timestamp: str | None = None, period: str = "ttm") -> dict[str, Any]:
        timestamp = normalize_timestamp(timestamp)
        symbol = self._ticker_to_symbol(ticker)

        indicator_df = await self._run_akshare_call(self.ak.stock_financial_analysis_indicator, symbol=symbol)
        indicator_df = self._asof_filter(indicator_df, timestamp) if indicator_df is not None else indicator_df
        latest_indicator = (
            self._row_to_dict(indicator_df.iloc[-1]) if indicator_df is not None and not indicator_df.empty else None
        )

        valuation = None
        try:
            valuation_df = await self._run_akshare_call(self.ak.stock_a_lg_indicator, symbol=symbol)
            valuation_df = self._asof_filter(valuation_df, timestamp)
            if valuation_df is not None and not valuation_df.empty:
                valuation = self._row_to_dict(valuation_df.iloc[-1])
        except Exception:
            valuation = None

        return {
            "ticker": ticker,
            "symbol": symbol,
            "timestamp": timestamp,
            "period": period,
            "source": "akshare",
            "financial_indicator": latest_indicator,
            "valuation_indicator": valuation,
        }

    async def get_news(self, timestamp: str | None = None, ticker: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        timestamp = normalize_timestamp(timestamp)
        asof_dt = datetime.fromisoformat(timestamp).replace(tzinfo=None)
        news_df = None

        if ticker:
            symbol = self._ticker_to_symbol(ticker)
            news_df = await self._run_akshare_call(self.ak.stock_news_em, symbol=symbol)
        else:
            news_df = await self._run_akshare_call(self.ak.stock_info_global_em)

        if news_df is None or news_df.empty:
            return []

        time_col = self._pick_time_column(news_df)
        if time_col:
            work = news_df.copy()
            try:
                parsed = work[time_col].map(lambda x: datetime.fromisoformat(str(x).replace("/", "-")).replace(tzinfo=None))
                work = work.assign(__parsed_time=parsed)
                work = work[work["__parsed_time"] <= asof_dt]
                work = work.sort_values("__parsed_time", ascending=False)
                news_df = work.drop(columns=["__parsed_time"])
            except Exception:
                pass

        return [self._row_to_dict(row) for _, row in news_df.head(max(limit, 1)).iterrows()]

    async def screen_stocks(self, strategy: str, timestamp: str | None = None, max_results: int = 20) -> list[dict[str, Any]]:
        timestamp = normalize_timestamp(timestamp)
        # Use Playwright snapshot as primary source (with anti-ban rate limiting).
        # Fallback to AkShare endpoint only if Playwright path fails.
        try:
            spot_df = await self._fetch_spot_with_playwright()
        except Exception:
            logger.warning("Fallback to AkShare spot endpoint: stock_zh_a_spot_em")
            spot_df = await self._run_akshare_call(self.ak.stock_zh_a_spot_em)
        if spot_df is None or spot_df.empty:
            return []

        df = spot_df.copy()
        raw_count = len(df)
        if "代码" in df.columns:
            raw_codes = [str(x) for x in df["代码"].head(20).tolist()]
            logger.info("Screen raw universe: strategy=%s timestamp=%s count=%s sample_codes=%s", strategy, timestamp, raw_count, raw_codes)
        else:
            logger.info("Screen raw universe: strategy=%s timestamp=%s count=%s", strategy, timestamp, raw_count)
        df = self._to_numeric(df, ["最新价", "涨跌幅", "换手率", "振幅", "市盈率-动态", "市净率", "总市值", "成交额"])

        # Layer 1: tradability and risk hygiene filters
        before = len(df)
        if "名称" in df.columns:
            df = df[~df["名称"].astype(str).str.contains("ST", case=False, na=False)]
        if "代码" in df.columns:
            # Avoid BSE-heavy universe for this conservative A-share template.
            df = df[~df["代码"].astype(str).str.startswith(("4", "8"))]
        if "最新价" in df.columns:
            df = df[df["最新价"] > 0]
        if "成交额" in df.columns:
            df = df[df["成交额"] >= 2e8]
        if "总市值" in df.columns:
            df = df[df["总市值"] >= 8e9]
        if "换手率" in df.columns:
            df = df[(df["换手率"] >= 0.1) & (df["换手率"] <= 15)]
        logger.info("Screen layer1 tradability: before=%s after=%s removed=%s", before, len(df), before - len(df))

        if df.empty:
            return []

        # Layer 2: strategy-specific hard constraints
        before = len(df)
        if strategy == "quality":
            if {"市盈率-动态", "市净率"}.issubset(df.columns):
                df = df[(df["市盈率-动态"] > 0) & (df["市盈率-动态"] <= 40) & (df["市净率"] <= 6)]
        elif strategy == "dividend":
            if {"市盈率-动态", "市净率", "总市值"}.issubset(df.columns):
                df = df[(df["市盈率-动态"] > 0) & (df["市盈率-动态"] <= 25) & (df["市净率"] <= 3.5) & (df["总市值"] >= 2e10)]
        elif strategy == "low_vol":
            if {"振幅", "换手率"}.issubset(df.columns):
                df = df[(df["振幅"] <= 6) & (df["换手率"] <= 8)]
        elif strategy == "value":
            if {"市盈率-动态", "市净率"}.issubset(df.columns):
                df = df[(df["市盈率-动态"] > 0) & (df["市盈率-动态"] <= 22) & (df["市净率"] <= 2.8)]
        logger.info("Screen layer2 strategy_filter: strategy=%s before=%s after=%s removed=%s", strategy, before, len(df), before - len(df))

        if df.empty:
            return []

        # Layer 3: factor scoring (0~1), weighted by strategy style
        score_cols = []
        if "总市值" in df.columns:
            df["score_size"] = self._score_desc(df["总市值"])
            score_cols.append("score_size")
        if "成交额" in df.columns:
            df["score_liquidity"] = self._score_desc(df["成交额"])
            score_cols.append("score_liquidity")
        if "市盈率-动态" in df.columns:
            pe = df["市盈率-动态"].where(df["市盈率-动态"] > 0, other=df["市盈率-动态"].median())
            df["score_pe"] = self._score_asc(pe)
            score_cols.append("score_pe")
        if "市净率" in df.columns:
            pb = df["市净率"].where(df["市净率"] > 0, other=df["市净率"].median())
            df["score_pb"] = self._score_asc(pb)
            score_cols.append("score_pb")
        if "振幅" in df.columns:
            df["score_volatility"] = self._score_asc(df["振幅"])
            score_cols.append("score_volatility")
        if "换手率" in df.columns:
            # Conservative preference: avoid both too dead and too hot; center around 2.5
            centered = (df["换手率"] - 2.5).abs()
            df["score_turnover"] = self._score_asc(centered)
            score_cols.append("score_turnover")
        if "涨跌幅" in df.columns:
            # Penalize extreme short-term moves
            momentum_risk = df["涨跌幅"].abs()
            df["score_momentum_stability"] = self._score_asc(momentum_risk)
            score_cols.append("score_momentum_stability")

        weights_map = {
            "quality": {
                "score_size": 0.18,
                "score_liquidity": 0.20,
                "score_pe": 0.16,
                "score_pb": 0.16,
                "score_volatility": 0.14,
                "score_turnover": 0.10,
                "score_momentum_stability": 0.06,
            },
            "dividend": {
                "score_size": 0.22,
                "score_liquidity": 0.20,
                "score_pe": 0.22,
                "score_pb": 0.20,
                "score_volatility": 0.10,
                "score_turnover": 0.06,
            },
            "low_vol": {
                "score_size": 0.15,
                "score_liquidity": 0.18,
                "score_volatility": 0.30,
                "score_turnover": 0.17,
                "score_momentum_stability": 0.20,
            },
            "value": {
                "score_size": 0.14,
                "score_liquidity": 0.18,
                "score_pe": 0.30,
                "score_pb": 0.28,
                "score_volatility": 0.06,
                "score_momentum_stability": 0.04,
            },
        }
        weights = weights_map.get(strategy, weights_map["quality"])

        effective_weights = {k: v for k, v in weights.items() if k in score_cols}
        if not effective_weights:
            return []

        total_weight = sum(effective_weights.values())
        normalized_weights = {k: (v / total_weight) for k, v in effective_weights.items()}
        df["total_score"] = 0.0
        for col, w in normalized_weights.items():
            df["total_score"] = df["total_score"] + df[col] * w

        df = df.sort_values("total_score", ascending=False)
        top_df = df.head(max(max_results, 1))

        rows = []
        for _, row in top_df.iterrows():
            row_data = self._row_to_dict(row)
            factor_scores = {k: round(float(row_data.get(k, 0.0)), 4) for k in normalized_weights}
            item = {
                "代码": row_data.get("代码"),
                "名称": row_data.get("名称"),
                "最新价": row_data.get("最新价"),
                "涨跌幅": row_data.get("涨跌幅"),
                "换手率": row_data.get("换手率"),
                "振幅": row_data.get("振幅"),
                "市盈率-动态": row_data.get("市盈率-动态"),
                "市净率": row_data.get("市净率"),
                "总市值": row_data.get("总市值"),
                "成交额": row_data.get("成交额"),
                "strategy": strategy,
                "total_score": round(float(row_data.get("total_score", 0.0)), 4),
                "factor_scores": factor_scores,
                "timestamp": timestamp,
                "screen_note": "multi-layer screening: tradability filter + style constraints + weighted factor score (AkShare spot snapshot).",
            }
            rows.append(item)
        picked_codes = [str(x.get("代码")) for x in rows]
        picked_scores = [x.get("total_score") for x in rows]
        logger.info(
            "Screen final picks: strategy=%s requested=%s picked=%s codes=%s scores=%s",
            strategy,
            max_results,
            len(rows),
            picked_codes,
            picked_scores,
        )
        return rows


class AShareQuoteTool(Tool):
    """Get real-time or latest quote data for a ticker."""

    def __init__(self, backend: AShareDataBackend):
        self.backend = backend

    @property
    def name(self) -> str:
        return "get_a_share_quote"

    @property
    def description(self) -> str:
        return "Get quote data for an A-share ticker (e.g., 600519.SH, 000001.SZ)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "A-share ticker symbol, e.g. 600519.SH or 000001.SZ",
                },
                "timestamp": {
                    "type": "string",
                    "description": "Optional analysis time in ISO 8601. If omitted, realtime mode is used.",
                },
            },
            "required": ["ticker"],
        }

    async def execute(self, ticker: str, timestamp: str | None = None) -> ToolResult:
        try:
            normalized_ts = normalize_timestamp(timestamp)
            data = await self.backend.get_quote(ticker=ticker, timestamp=normalized_ts)
            return ToolResult(success=True, content=json.dumps(data, ensure_ascii=False, indent=2))
        except NotImplementedError as e:
            # 数据后端尚未实现时，向 LLM 返回明确错误，避免“静默失败”。
            return ToolResult(success=False, content="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to get quote: {e}")


class AShareFundamentalsTool(Tool):
    """Get fundamental metrics for a ticker."""

    def __init__(self, backend: AShareDataBackend):
        self.backend = backend

    @property
    def name(self) -> str:
        return "get_a_share_fundamentals"

    @property
    def description(self) -> str:
        return "Get fundamental data for an A-share ticker (valuation, profitability, leverage, cashflow)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "A-share ticker symbol, e.g. 600519.SH or 000001.SZ",
                },
                "timestamp": {
                    "type": "string",
                    "description": "Optional analysis time in ISO 8601. If omitted, realtime mode is used.",
                },
                "period": {
                    "type": "string",
                    "description": "Financial period preference (ttm, annual, quarterly)",
                    "default": "ttm",
                },
            },
            "required": ["ticker"],
        }

    async def execute(self, ticker: str, timestamp: str | None = None, period: str = "ttm") -> ToolResult:
        try:
            normalized_ts = normalize_timestamp(timestamp)
            data = await self.backend.get_fundamentals(ticker=ticker, timestamp=normalized_ts, period=period)
            return ToolResult(success=True, content=json.dumps(data, ensure_ascii=False, indent=2))
        except NotImplementedError as e:
            return ToolResult(success=False, content="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to get fundamentals: {e}")


class AShareNewsTool(Tool):
    """Get market or ticker-specific news/events."""

    def __init__(self, backend: AShareDataBackend):
        self.backend = backend

    @property
    def name(self) -> str:
        return "get_a_share_news"

    @property
    def description(self) -> str:
        return "Get recent A-share news. Can query market-wide or by ticker."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "Optional analysis time in ISO 8601. If omitted, realtime mode is used.",
                },
                "ticker": {
                    "type": "string",
                    "description": "Optional ticker symbol for stock-specific news.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of items to return.",
                    "default": 10,
                },
            },
        }

    async def execute(self, timestamp: str | None = None, ticker: str | None = None, limit: int = 10) -> ToolResult:
        try:
            normalized_ts = normalize_timestamp(timestamp)
            items = await self.backend.get_news(timestamp=normalized_ts, ticker=ticker, limit=limit)
            return ToolResult(success=True, content=json.dumps(items, ensure_ascii=False, indent=2))
        except NotImplementedError as e:
            return ToolResult(success=False, content="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to get news: {e}")


class AShareScreenTool(Tool):
    """Screen candidate stocks by conservative strategies."""

    def __init__(self, backend: AShareDataBackend):
        self.backend = backend

    @property
    def name(self) -> str:
        return "screen_a_share_stocks"

    @property
    def description(self) -> str:
        return "Screen A-share candidates using conservative strategies (quality, dividend, low_vol, value)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "Optional analysis time in ISO 8601. If omitted, realtime mode is used.",
                },
                "strategy": {
                    "type": "string",
                    "description": "Screening strategy.",
                    "enum": ["quality", "dividend", "low_vol", "value"],
                    "default": "quality",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max number of candidate stocks.",
                    "default": 20,
                },
            },
        }

    async def execute(self, timestamp: str | None = None, strategy: str = "quality", max_results: int = 20) -> ToolResult:
        try:
            normalized_ts = normalize_timestamp(timestamp)
            result = await self.backend.screen_stocks(strategy=strategy, timestamp=normalized_ts, max_results=max_results)
            return ToolResult(success=True, content=json.dumps(result, ensure_ascii=False, indent=2))
        except NotImplementedError as e:
            return ToolResult(success=False, content="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to screen stocks: {e}")


class ConservativeTradePlanTool(Tool):
    """Build a conservative trade plan based on supplied analysis inputs."""

    @property
    def name(self) -> str:
        return "build_conservative_trade_plan"

    @property
    def description(self) -> str:
        return "Build a conservative A-share trade plan with staged entries, stop, and risk controls."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol"},
                "timestamp": {
                    "type": "string",
                    "description": "Optional analysis time in ISO 8601. If omitted, realtime mode is used.",
                },
                "current_price": {"type": "number", "description": "Latest known price"},
                "support_price": {"type": "number", "description": "Technical support level"},
                "resistance_price": {"type": "number", "description": "Technical resistance level"},
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "default": "low",
                    "description": "User risk tolerance. Conservative default is low.",
                },
                "account_size": {"type": "number", "description": "Optional account size for sizing hints"},
            },
            "required": ["ticker", "current_price", "support_price"],
        }

    async def execute(
        self,
        ticker: str,
        current_price: float,
        support_price: float,
        timestamp: str | None = None,
        resistance_price: float | None = None,
        risk_level: str = "low",
        account_size: float | None = None,
    ) -> ToolResult:
        try:
            normalized_ts = normalize_timestamp(timestamp)
            # 保守止损：支撑位下方 2% 作为失效位（可在后续版本参数化）。
            stop_price = round(support_price * 0.98, 3)
            tranche_weights = [0.4, 0.35, 0.25]

            # 分批建仓价格带：从接近现价到靠近支撑位，越跌越小心加仓。
            entry_band = [
                round(current_price * 0.995, 3),
                round(current_price * 0.98, 3),
                round(max(support_price, current_price * 0.965), 3),
            ]

            # 单股风险 = 入场价与止损价的价差，用于估算可承受仓位。
            risk_per_share = max(current_price - stop_price, 0.0001)
            position_note = "Use small size (e.g., <= 10% of total capital for single stock)."
            if account_size and risk_level == "low":
                # Conservative: risk 0.5% of account per position idea.
                max_risk_budget = account_size * 0.005
                max_shares = int(max_risk_budget / risk_per_share)
                position_note = (
                    f"Max risk budget: {max_risk_budget:.2f}, suggested max shares: {max_shares} "
                    f"(computed by risk budget / (entry-stop))."
                )

            take_profit = None
            if resistance_price is not None:
                take_profit = round(resistance_price * 0.98, 3)
            else:
                # 若无明确压力位，给一个保守默认目标，避免过度乐观。
                take_profit = round(current_price * 1.08, 3)

            plan = {
                "ticker": ticker,
                "timestamp": normalized_ts,
                "style": "conservative",
                "risk_level": risk_level,
                "entry_plan": {
                    "tranches": [
                        {"weight": tranche_weights[0], "entry_price": entry_band[0]},
                        {"weight": tranche_weights[1], "entry_price": entry_band[1]},
                        {"weight": tranche_weights[2], "entry_price": entry_band[2]},
                    ],
                    "rule": "Only add next tranche if price stabilizes near support with no major negative catalyst.",
                },
                "risk_control": {
                    "stop_loss": stop_price,
                    "invalidation": "Break below support with expanding volume or adverse fundamental event.",
                },
                "profit_taking": {
                    "first_target": take_profit,
                    "de_risk_rule": "Trim 30-50% near first target or on momentum divergence.",
                },
                "position_sizing": position_note,
                "disclaimer": "For research/education only. Final decision and risk are user's responsibility.",
            }
            return ToolResult(success=True, content=json.dumps(plan, ensure_ascii=False, indent=2))
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to build trade plan: {e}")


def create_a_share_tools(backend: AShareDataBackend | None = None) -> list[Tool]:
    """Create all A-share related tools.

    Args:
        backend: Optional concrete backend implementation. If omitted,
            PlaceholderAShareDataBackend is used.
    """
    # 统一入口：后续只需替换 backend，不需要改 Agent 或 CLI 组装逻辑。
    if backend is not None:
        data_backend = backend
    else:
        try:
            data_backend = AkShareDataBackend()
        except ImportError:
            data_backend = PlaceholderAShareDataBackend()
    return [
        AShareScreenTool(data_backend),
        AShareQuoteTool(data_backend),
        AShareFundamentalsTool(data_backend),
        AShareNewsTool(data_backend),
        ConservativeTradePlanTool(),
    ]
