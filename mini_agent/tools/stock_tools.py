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
import sqlite3
from abc import ABC
from datetime import datetime, timedelta
from typing import Any

from mini_agent.paths import DEFAULT_MEMORY_DB_PATH

from .base import Tool, ToolResult
from .kline_db_tool import KLineDB

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

    async def get_technical_signals(
        self,
        ticker: str,
        timestamp: str | None = None,
        window: int = 120,
    ) -> dict[str, Any]:
        raise NotImplementedError("TODO: implement technical signals logic")


class PlaceholderAShareDataBackend(AShareDataBackend):
    """Default backend used before real data providers are wired in.

    默认占位后端：便于先把 Agent 工具链跑通，再逐步接入实际数据服务。
    """


class LocalAShareDataBackend(AShareDataBackend):
    """Local SQLite-backed backend using daily_kline snapshot."""

    def __init__(self, db_path: str = str(DEFAULT_MEMORY_DB_PATH)):
        self.kline_db = KLineDB(db_path=db_path)
        self.db_path = self.kline_db.db_path

    async def get_quote(self, ticker: str, timestamp: str | None = None) -> dict[str, Any]:
        normalized_ts = normalize_timestamp(timestamp)
        asof = datetime.fromisoformat(normalized_ts).replace(tzinfo=None).strftime("%Y-%m-%d")
        symbol = self.kline_db.normalize_ticker(ticker)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT ticker, date, open, high, low, close, volume, amount
                FROM daily_kline
                WHERE ticker = ? AND date <= ?
                ORDER BY date DESC
                LIMIT 1
                """,
                (symbol, asof),
            ).fetchone()
        if row is None:
            return {
                "ticker": ticker,
                "symbol": symbol,
                "timestamp": normalized_ts,
                "source": "local:daily_kline",
                "quote": None,
            }
        return {
            "ticker": ticker,
            "symbol": symbol,
            "timestamp": normalized_ts,
            "source": "local:daily_kline",
            "quote": {
                "date": row["date"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "amount": row["amount"],
                "turnover_rate": None,
            },
        }

    async def get_fundamentals(self, ticker: str, timestamp: str | None = None, period: str = "ttm") -> dict[str, Any]:
        raise NotImplementedError("Local fundamentals not available in daily_kline DB.")

    async def get_news(self, timestamp: str | None = None, ticker: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        raise NotImplementedError("Local news not available in daily_kline DB.")

    async def screen_stocks(self, strategy: str, timestamp: str | None = None, max_results: int = 20) -> list[dict[str, Any]]:
        normalized_ts = normalize_timestamp(timestamp)
        asof = datetime.fromisoformat(normalized_ts).replace(tzinfo=None).strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            day_row = conn.execute(
                "SELECT MAX(date) AS d FROM daily_kline WHERE date <= ?",
                (asof,),
            ).fetchone()
            effective_date = str(day_row["d"]) if day_row and day_row["d"] else None
            if not effective_date:
                return []
            universe_limit = max(max_results * 40, 400)
            rows = conn.execute(
                """
                SELECT ticker, date, open, high, low, close, volume, amount,
                       CASE WHEN open IS NOT NULL AND open != 0 THEN (close - open) / open * 100 ELSE NULL END AS change_pct
                FROM daily_kline
                WHERE date = ?
                ORDER BY amount DESC
                LIMIT ?
                """,
                (effective_date, universe_limit),
            ).fetchall()
        if not rows:
            return []

        candidates: list[dict[str, Any]] = []
        for row in rows:
            open_price = float(row["open"] or 0.0)
            high = float(row["high"] or 0.0)
            low = float(row["low"] or 0.0)
            close = float(row["close"] or 0.0)
            amount = float(row["amount"] or 0.0)
            change_pct = float(row["change_pct"] or 0.0)
            amplitude = ((high - low) / open_price * 100.0) if open_price > 0 else 0.0
            if close <= 0 or amount < 5e7:
                continue
            candidates.append(
                {
                    "代码": str(row["ticker"]),
                    "名称": str(row["ticker"]),
                    "最新价": close,
                    "涨跌幅": change_pct,
                    "换手率": None,
                    "振幅": amplitude,
                    "市盈率-动态": None,
                    "市净率": None,
                    "总市值": None,
                    "成交额": amount,
                }
            )
        if not candidates:
            return []

        filtered: list[dict[str, Any]] = []
        for item in candidates:
            change_pct = float(item["涨跌幅"] or 0.0)
            amplitude = float(item["振幅"] or 0.0)
            amount = float(item["成交额"] or 0.0)
            if strategy == "low_vol":
                if amplitude <= 5.0 and abs(change_pct) <= 3.0:
                    filtered.append(item)
            elif strategy == "value":
                if -3.5 <= change_pct <= 5.0 and amplitude <= 8.0:
                    filtered.append(item)
            elif strategy == "dividend":
                if abs(change_pct) <= 6.0 and amplitude <= 8.0 and amount >= 1e8:
                    filtered.append(item)
            elif strategy == "growth":
                # 高成长：放宽涨跌幅限制，关注流动性
                if abs(change_pct) <= 12.0 and amount >= 3e8:
                    filtered.append(item)
            else:  # quality
                if abs(change_pct) <= 9.5 and amplitude <= 12.0:
                    filtered.append(item)
        if not filtered:
            filtered = candidates[:]

        liquidity_vals = [float(x["成交额"] or 0.0) for x in filtered]
        stability_vals = [abs(float(x["涨跌幅"] or 0.0)) for x in filtered]
        volatility_vals = [float(x["振幅"] or 0.0) for x in filtered]
        mean_liquidity = (sum(liquidity_vals) / len(liquidity_vals)) if liquidity_vals else 1.0
        mean_stability = (sum(stability_vals) / len(stability_vals)) if stability_vals else 1.0
        mean_volatility = (sum(volatility_vals) / len(volatility_vals)) if volatility_vals else 1.0
        mean_liquidity = mean_liquidity if mean_liquidity > 1e-6 else 1.0
        mean_stability = mean_stability if mean_stability > 1e-6 else 1.0
        mean_volatility = mean_volatility if mean_volatility > 1e-6 else 1.0

        weights_map = {
            "quality": {"liquidity": 0.30, "stability": 0.30, "low_vol": 0.40},
            "dividend": {"liquidity": 0.45, "stability": 0.25, "low_vol": 0.30},
            "low_vol": {"liquidity": 0.20, "stability": 0.35, "low_vol": 0.45},
            "value": {"liquidity": 0.25, "stability": 0.20, "low_vol": 0.25, "value_pullback": 0.30},
        }
        weights = weights_map.get(strategy, weights_map["quality"])

        scored: list[dict[str, Any]] = []
        for item in filtered:
            change_pct = float(item["涨跌幅"] or 0.0)
            amount = float(item["成交额"] or 0.0)
            amplitude = float(item["振幅"] or 0.0)
            factor_scores = {
                "score_liquidity": min(1.0, amount / mean_liquidity),
                "score_stability": 1.0 / (1.0 + abs(change_pct) / mean_stability),
                "score_low_vol": 1.0 / (1.0 + amplitude / mean_volatility),
                "score_value_pullback": 1.0 / (1.0 + max(change_pct, 0.0)),
            }
            total_score = 0.0
            for k, w in weights.items():
                key = {
                    "liquidity": "score_liquidity",
                    "stability": "score_stability",
                    "low_vol": "score_low_vol",
                    "value_pullback": "score_value_pullback",
                }[k]
                total_score += factor_scores.get(key, 0.0) * float(w)

            scored.append(
                {
                    **item,
                    "strategy": strategy,
                    "total_score": round(total_score, 4),
                    "factor_scores": {k: round(float(v), 4) for k, v in factor_scores.items()},
                    "timestamp": normalized_ts,
                    "screen_note": (
                        "local kline snapshot screening with strategy-specific filters and proxy scoring "
                        "(liquidity/stability/volatility)."
                    ),
                }
            )

        scored.sort(key=lambda x: float(x.get("total_score") or 0.0), reverse=True)
        return scored[: max(max_results, 1)]

    async def get_technical_signals(
        self,
        ticker: str,
        timestamp: str | None = None,
        window: int = 120,
    ) -> dict[str, Any]:
        normalized_ts = normalize_timestamp(timestamp)
        asof_dt = datetime.fromisoformat(normalized_ts).replace(tzinfo=None)
        symbol = self.kline_db.normalize_ticker(ticker)
        start_dt = asof_dt - timedelta(days=max(int(window), 20) * 3)
        rows = self.kline_db.get_kline(symbol, start_dt.strftime("%Y-%m-%d"), asof_dt.strftime("%Y-%m-%d"))
        if not rows:
            return {
                "ticker": ticker,
                "symbol": symbol,
                "timestamp": normalized_ts,
                "source": "local:daily_kline",
                "signal_status": "no_data",
            }
        closes = [float(r.get("close") or 0.0) for r in rows if r.get("close") is not None]
        if len(closes) < 20:
            return {
                "ticker": ticker,
                "symbol": symbol,
                "timestamp": normalized_ts,
                "source": "local:daily_kline",
                "signal_status": "no_data",
            }
        ma5 = sum(closes[-5:]) / 5
        ma20 = sum(closes[-20:]) / 20
        trend = "up" if closes[-1] >= ma20 else "down"
        return {
            "ticker": ticker,
            "symbol": symbol,
            "timestamp": normalized_ts,
            "source": "local:daily_kline",
            "signal_status": "ok",
            "window": min(window, len(closes)),
            "trend": trend,
            "signals": {
                "ma_cross": "golden_cross" if ma5 > ma20 else "death_cross",
                "macd": "neutral",
                "rsi": "neutral",
                "breakout": "none",
            },
            "score": 60 if trend == "up" else 40,
        }

    async def get_market_index(self, timestamp: str | None = None) -> dict[str, Any]:
        """获取大盘指数，优先同日实时快照，失败后降级到日线数据。"""
        import warnings
        warnings.filterwarnings("ignore")
        
        normalized_ts = normalize_timestamp(timestamp)
        asof_dt = datetime.fromisoformat(normalized_ts)
        now_dt = datetime.now(asof_dt.tzinfo) if asof_dt.tzinfo else datetime.now()
        prefer_realtime = asof_dt.date() == now_dt.date()
        
        # 指数代码映射: 本地代码 -> (Yahoo代码, TuShare代码, AkShare代码, 名称)
        index_map = {
            "000001": ("^SSEC", "000001.SH", "sh000001", "上证指数"),
            "399001": ("399001.SZ", "399001.SZ", "sz399001", "深证成指"),
            "399006": ("399006.SZ", "399006.SZ", "sz399006", "创业板指"),
            "000300": ("^CSI300", "000300.SH", "sh000300", "沪深300"),
        }
        
        results = {}
        source_used = None
        
        # 1) 同日查询优先实时快照，避免在交易日返回“昨日收盘”造成误导。
        if prefer_realtime:
            logger.info("尝试获取大盘指数: AkShare 实时快照...")
            realtime_results = await self._get_akshare_index_spot(index_map, normalized_ts)
            if realtime_results:
                logger.info("✅ AkShare 实时快照获取成功")
                return {
                    "timestamp": normalized_ts,
                    "source": "akshare:spot",
                    "mode": "realtime",
                    "as_of_date": asof_dt.strftime("%Y-%m-%d"),
                    "indices": realtime_results,
                }

        # 2. 尝试 Yahoo Finance（日线）
        logger.info("尝试获取大盘指数: Yahoo Finance...")
        yahoo_results = await self._get_yahoo_index(index_map)
        if yahoo_results:
            results.update(yahoo_results)
            source_used = "yahoo_finance"
            logger.info("✅ Yahoo Finance 获取成功")
            await self._save_index_to_db(results, source_used)
            return {
                "timestamp": normalized_ts,
                "source": source_used,
                "mode": "daily",
                "as_of_date": max(str(v.get("date", "")) for v in results.values()) if results else None,
                "indices": results,
            }
        
        # 3. 尝试 TuShare (如果配置了token, 日线)
        logger.info("尝试获取大盘指数: TuShare...")
        tushare_results = await self._get_tushare_index(index_map)
        if tushare_results:
            results.update(tushare_results)
            source_used = "tushare"
            logger.info("✅ TuShare 获取成功")
            await self._save_index_to_db(results, source_used)
            return {
                "timestamp": normalized_ts,
                "source": source_used,
                "mode": "daily",
                "as_of_date": max(str(v.get("date", "")) for v in results.values()) if results else None,
                "indices": results,
            }
        
        # 4. 尝试 AkShare（日线）
        logger.info("尝试获取大盘指数: AkShare...")
        akshare_results = await self._get_akshare_index(index_map)
        if akshare_results:
            results.update(akshare_results)
            source_used = "akshare"
            logger.info("✅ AkShare 获取成功")
            await self._save_index_to_db(results, source_used)
            return {
                "timestamp": normalized_ts,
                "source": source_used,
                "mode": "daily",
                "as_of_date": max(str(v.get("date", "")) for v in results.values()) if results else None,
                "indices": results,
            }
        
        # 5. 降级到本地数据库（日线）
        logger.info("降级到本地数据库获取大盘指数...")
        local_results = await self._get_local_index(index_map, normalized_ts)
        
        return {
            "timestamp": normalized_ts,
            "source": "local:daily_kline",
            "mode": "daily",
            "as_of_date": max(str(v.get("date", "")) for v in local_results.values()) if local_results else None,
            "indices": local_results,
        }

    async def _get_akshare_index_spot(self, index_map: dict, timestamp: str) -> dict:
        """从 AkShare 实时快照接口获取指数数据。"""
        try:
            import akshare as ak
            import pandas as pd  # type: ignore
        except ImportError:
            return {}

        try:
            df = ak.stock_zh_index_spot_sina()
        except Exception as e:
            logger.debug(f"AkShare 实时指数获取失败: {e}")
            return {}

        if df is None or df.empty:
            return {}

        def _to_float(value: Any) -> float:
            try:
                if value is None:
                    return 0.0
                parsed = pd.to_numeric(value, errors="coerce")
                if pd.isna(parsed):
                    return 0.0
                return float(parsed)
            except Exception:
                return 0.0

        code_col = None
        for candidate in ("代码", "symbol", "指数代码", "code"):
            if candidate in df.columns:
                code_col = candidate
                break
        if code_col is None:
            return {}

        work = df.copy()
        work[code_col] = work[code_col].astype(str).str.lower().str.strip()

        results: dict[str, dict[str, Any]] = {}
        date_str = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")

        for local_code, (_, _, akshare_code, name) in index_map.items():
            candidates = {local_code.lower(), akshare_code.lower()}
            row_df = work[work[code_col].isin(candidates)]
            if row_df.empty:
                continue

            row = row_df.iloc[0]
            open_price = _to_float(row.get("今开", row.get("open", 0)))
            high_price = _to_float(row.get("最高", row.get("high", 0)))
            low_price = _to_float(row.get("最低", row.get("low", 0)))
            close_price = _to_float(row.get("最新价", row.get("close", 0)))
            volume = _to_float(row.get("成交量", row.get("volume", 0)))

            change_pct = _to_float(row.get("涨跌幅", row.get("change_percent", 0)))
            if abs(change_pct) > 1000:
                change_pct = change_pct / 100

            results[local_code] = {
                "name": name,
                "date": date_str,
                "close": close_price,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "volume": volume,
                "change_pct": round(change_pct, 2),
            }
        return results

    async def _get_tushare_index(self, index_map: dict) -> dict:
        """从 TuShare 获取指数数据"""
        try:
            import tushare as ts
            
            token = os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN")
            if not token:
                logger.debug("TuShare token 未配置")
                return {}
            
            ts.set_token(token)
            pro = ts.pro_api()
            
            results = {}
            ts_code_map = {"000001": "000001.SH", "399001": "399001.SZ", "399006": "399006.SZ", "000300": "000300.SH"}
            
            for code, (yahoo_code, ts_code, akshare_code, name) in index_map.items():
                try:
                    df = pro.index_daily(ts_code=ts_code_map.get(code))
                    if df is not None and not df.empty:
                        latest = df.iloc[0]
                        prev = df.iloc[1] if len(df) > 1 else latest
                        close = float(latest.get('close', 0))
                        prev_close = float(prev.get('close', close))
                        change_pct = ((close - prev_close) / prev_close * 100) if prev_close else 0
                        
                        results[code] = {
                            "name": name, "date": str(latest.get('trade_date', '')),
                            "close": close, "open": float(latest.get('open', 0)),
                            "high": float(latest.get('high', 0)), "low": float(latest.get('low', 0)),
                            "volume": float(latest.get('vol', 0)), "change_pct": round(change_pct, 2),
                        }
                except Exception as e:
                    logger.debug(f"TuShare index {code} failed: {e}")
                    continue
            
            return results
        except Exception as e:
            logger.debug(f"TuShare 获取失败: {e}")
            return {}

    async def _get_local_index(self, index_map: dict, timestamp: str) -> dict:
        """从本地数据库获取指数数据"""
        asof = datetime.fromisoformat(timestamp).replace(tzinfo=None).strftime("%Y-%m-%d")
        
        results = {}
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for code, (_, _, _, name) in index_map.items():
                row = conn.execute(
                    """SELECT ticker, date, open, high, low, close, volume, amount
                    FROM daily_kline WHERE ticker = ? AND date <= ? ORDER BY date DESC LIMIT 1""",
                    (code, asof),
                ).fetchone()

                if row:
                    prev_row = conn.execute(
                        """SELECT close FROM daily_kline WHERE ticker = ? AND date < ? ORDER BY date DESC LIMIT 1""",
                        (code, row["date"]),
                    ).fetchone()

                    prev_close = prev_row["close"] if prev_row else row["close"]
                    change_pct = ((row["close"] - prev_close) / prev_close * 100) if prev_close else 0

                    results[code] = {
                        "name": name, "date": row["date"], "close": row["close"],
                        "open": row["open"], "high": row["high"], "low": row["low"],
                        "volume": row["volume"], "change_pct": round(change_pct, 2),
                    }
        return results

    async def _get_yahoo_index(self, index_map: dict) -> dict:
        """从 Yahoo Finance 获取指数数据"""
        try:
            import yfinance as yf
            
            results = {}
            for code, (yahoo_code, _, _, name) in index_map.items():
                try:
                    ticker = yf.Ticker(yahoo_code)
                    hist = ticker.history(period="5d")
                    if hist is not None and not hist.empty:
                        latest = hist.iloc[-1]
                        prev = hist.iloc[-2] if len(hist) > 1 else latest
                        change_pct = ((latest['Close'] - prev['Close']) / prev['Close'] * 100) if prev['Close'] else 0
                        
                        results[code] = {
                            "name": name,
                            "date": str(latest.name.date()) if hasattr(latest.name, 'date') else str(datetime.now().date()),
                            "close": round(latest['Close'], 2),
                            "open": round(latest['Open'], 2),
                            "high": round(latest['High'], 2),
                            "low": round(latest['Low'], 2),
                            "volume": latest['Volume'],
                            "change_pct": round(change_pct, 2),
                        }
                except Exception as e:
                    logger.debug(f"Yahoo index {code} failed: {e}")
                    continue
            
            return results
        except ImportError:
            return {}

    async def _get_akshare_index(self, index_map: dict) -> dict:
        """从 AkShare 获取指数数据"""
        try:
            import akshare as ak
            
            results = {}
            for code, (_, _, akshare_code, name) in index_map.items():
                try:
                    df = ak.stock_zh_index_daily(symbol=akshare_code)
                    
                    if df is not None and not df.empty:
                        latest = df.iloc[-1]
                        prev = df.iloc[-2] if len(df) > 1 else latest
                        
                        close = float(latest.get('close', 0))
                        prev_close = float(prev.get('close', close))
                        change_pct = ((close - prev_close) / prev_close * 100) if prev_close else 0
                        
                        results[code] = {
                            "name": name,
                            "date": str(latest.get('date', '')),
                            "close": close,
                            "open": float(latest.get('open', 0)),
                            "high": float(latest.get('high', 0)),
                            "low": float(latest.get('low', 0)),
                            "volume": float(latest.get('volume', 0)),
                            "change_pct": round(change_pct, 2),
                        }
                except Exception as e:
                    logger.debug(f"AkShare index {code} failed: {e}")
                    continue
            
            return results
        except ImportError:
            return {}

    async def _save_index_to_db(self, indices: dict, source: str) -> None:
        """将指数数据存储到本地数据库"""
        from datetime import datetime
        
        records = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for code, data in indices.items():
            date = data.get("date", "")
            if not date:
                continue
                
            record = {
                "ticker": code,
                "date": date,
                "open": data.get("open", 0),
                "high": data.get("high", 0),
                "low": data.get("low", 0),
                "close": data.get("close", 0),
                "volume": data.get("volume", 0),
                "amount": data.get("close", 0) * data.get("volume", 0),
                "created_at": now,
            }
            records.append(record)
        
        if not records:
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                for record in records:
                    conn.execute("""
                        INSERT OR REPLACE INTO daily_kline 
                        (ticker, date, open, high, low, close, volume, amount, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record["ticker"], record["date"], record["open"], 
                        record["high"], record["low"], record["close"],
                        record["volume"], record["amount"], record["created_at"]
                    ))
                conn.commit()
                logger.info(f"已存储 {len(records)} 条指数数据到本地 (source: {source})")
        except Exception as e:
            logger.warning(f"存储指数数据失败: {e}")


class TuShareDataBackend(AShareDataBackend):
    """TuShare-based backend for A-share data.

    Notes:
    - Requires `tushare` package and a valid token.
    - Token can be passed directly or via env `TUSHARE_TOKEN` / `TS_TOKEN`.
    """

    _rate_limit_lock = asyncio.Lock()
    _last_call_monotonic = 0.0

    def __init__(self, token: str | None = None):
        try:
            import tushare as ts  # type: ignore
        except Exception as e:
            raise ImportError(
                "tushare is not installed. Run `uv add tushare` (or `uv sync` after adding dependency)."
            ) from e

        resolved_token = (token or os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN") or "").strip()
        if not resolved_token:
            raise ValueError(
                "TuShare token not configured. Set `tools.tushare_token` in config.yaml "
                "or env `TUSHARE_TOKEN` / `TS_TOKEN`."
            )

        self.ts = ts
        self.pro = ts.pro_api(resolved_token)
        self.min_interval_seconds = float(os.getenv("TUSHARE_MIN_INTERVAL_SECONDS", "0.35"))
        self.jitter_seconds = float(os.getenv("TUSHARE_JITTER_SECONDS", "0.15"))
        self.max_retries = int(os.getenv("TUSHARE_MAX_RETRIES", "2"))
        self.retry_base_delay = float(os.getenv("TUSHARE_RETRY_BASE_DELAY_SECONDS", "1.5"))

    @staticmethod
    def _ticker_to_symbol(ticker: str) -> str:
        ts = ticker.strip().upper()
        if "." in ts:
            return ts.split(".")[0]
        if ts.startswith(("SH", "SZ")):
            return ts[2:]
        return ts

    @staticmethod
    def _ticker_to_ts_code(ticker: str) -> str:
        ts = ticker.strip().upper()
        if "." in ts:
            symbol, suffix = ts.split(".", 1)
            suffix = suffix.upper()
            if suffix in {"SH", "SZ"}:
                return f"{symbol}.{suffix}"
            return f"{symbol}.SH" if symbol.startswith(("5", "6", "9")) else f"{symbol}.SZ"
        if ts.startswith(("SH", "SZ")):
            symbol = ts[2:]
            suffix = "SH" if ts.startswith("SH") else "SZ"
            return f"{symbol}.{suffix}"
        return f"{ts}.SH" if ts.startswith(("5", "6", "9")) else f"{ts}.SZ"

    async def _apply_rate_limit(self) -> None:
        async with self._rate_limit_lock:
            now = asyncio.get_running_loop().time()
            elapsed = now - self._last_call_monotonic
            required_wait = max(0.0, self.min_interval_seconds - elapsed)
            if required_wait > 0:
                await asyncio.sleep(required_wait)
            if self.jitter_seconds > 0:
                await asyncio.sleep(random.uniform(0, self.jitter_seconds))
            self.__class__._last_call_monotonic = asyncio.get_running_loop().time()

    async def _run_tushare_call(self, fn: Any, **kwargs: Any) -> Any:
        def _call():
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return fn(**kwargs)

        call_name = getattr(fn, "__name__", str(fn))
        last_exception: Exception | None = None
        for attempt in range(self.max_retries + 1):
            await self._apply_rate_limit()
            started = time.perf_counter()
            try:
                result = await asyncio.to_thread(_call)
                elapsed = time.perf_counter() - started
                logger.info("TuShare call ok: fn=%s elapsed=%.2fs attempt=%s", call_name, elapsed, attempt + 1)
                return result
            except Exception as e:
                last_exception = e
                elapsed = time.perf_counter() - started
                logger.warning(
                    "TuShare call failed: fn=%s elapsed=%.2fs attempt=%s/%s error=%s",
                    call_name,
                    elapsed,
                    attempt + 1,
                    self.max_retries + 1,
                    repr(e),
                )
                if attempt >= self.max_retries:
                    break
                delay = self.retry_base_delay * (2**attempt)
                if self.jitter_seconds > 0:
                    delay += random.uniform(0, self.jitter_seconds)
                logger.info("TuShare retry scheduled: fn=%s next_delay=%.2fs", call_name, delay)
                await asyncio.sleep(delay)
        raise RuntimeError(f"TuShare call failed after {self.max_retries + 1} attempts: {last_exception}")

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
            if value != value:  # noqa: PLR0124
                return None
        except Exception:
            pass
        return value

    @staticmethod
    def _to_numeric(df: Any, cols: list[str]) -> Any:
        for col in cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(",", "", regex=False)
                df[col] = df[col].astype(float)
        return df

    @staticmethod
    def _score_desc(series: Any) -> Any:
        if series is None or len(series) == 0:
            return series
        return series.rank(pct=True, method="average")

    @staticmethod
    def _score_asc(series: Any) -> Any:
        if series is None or len(series) == 0:
            return series
        return (1 - series.rank(pct=True, method="average")).clip(lower=0, upper=1)

    async def get_quote(self, ticker: str, timestamp: str | None = None) -> dict[str, Any]:
        timestamp = normalize_timestamp(timestamp)
        asof_dt = datetime.fromisoformat(timestamp).replace(tzinfo=None)
        trade_date = asof_dt.strftime("%Y%m%d")
        ts_code = self._ticker_to_ts_code(ticker)
        symbol = self._ticker_to_symbol(ticker)

        daily_df = await self._run_tushare_call(
            self.pro.daily,
            ts_code=ts_code,
            start_date=trade_date,
            end_date=trade_date,
        )
        if daily_df is None or daily_df.empty:
            return {"ticker": ticker, "symbol": symbol, "timestamp": timestamp, "source": "tushare:daily", "quote": None}

        row = daily_df.sort_values("trade_date").iloc[-1]
        quote = {
            "date": self._safe_value(row.get("trade_date")),
            "open": self._safe_value(row.get("open")),
            "high": self._safe_value(row.get("high")),
            "low": self._safe_value(row.get("low")),
            "close": self._safe_value(row.get("close")),
            "volume": self._safe_value(row.get("vol")),
            "amount": self._safe_value(row.get("amount")),
            "turnover_rate": None,
        }
        return {
            "ticker": ticker,
            "symbol": symbol,
            "timestamp": timestamp,
            "source": "tushare:daily",
            "quote": quote,
        }

    async def get_fundamentals(self, ticker: str, timestamp: str | None = None, period: str = "ttm") -> dict[str, Any]:
        # Keep fundamentals path on AkShare for now; caller will fallback if needed.
        raise NotImplementedError("TuShare fundamentals path is not wired yet.")

    async def get_news(self, timestamp: str | None = None, ticker: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        # Keep news path on AkShare for now; caller will fallback if needed.
        raise NotImplementedError("TuShare news path is not wired yet.")

    async def get_technical_signals(self, ticker: str, timestamp: str | None = None, window: int = 120) -> dict[str, Any]:
        # Keep technical path on AkShare for now; caller will fallback if needed.
        raise NotImplementedError("TuShare technical signals path is not wired yet.")

    async def screen_stocks(self, strategy: str, timestamp: str | None = None, max_results: int = 20) -> list[dict[str, Any]]:
        timestamp = normalize_timestamp(timestamp)
        asof_dt = datetime.fromisoformat(timestamp).replace(tzinfo=None)
        trade_date = asof_dt.strftime("%Y%m%d")

        stock_basic_df = await self._run_tushare_call(
            self.pro.stock_basic,
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name",
        )
        daily_df = await self._run_tushare_call(
            self.pro.daily,
            trade_date=trade_date,
            fields="ts_code,open,high,low,close,pct_chg,vol,amount",
        )
        daily_basic_df = await self._run_tushare_call(
            self.pro.daily_basic,
            trade_date=trade_date,
            fields="ts_code,turnover_rate,pe,pb,total_mv",
        )
        if daily_df is None or daily_df.empty or daily_basic_df is None or daily_basic_df.empty:
            return []

        import pandas as pd  # type: ignore

        df = daily_df.merge(daily_basic_df, on="ts_code", how="left")
        if stock_basic_df is not None and not stock_basic_df.empty:
            df = df.merge(stock_basic_df[["ts_code", "name", "symbol"]], on="ts_code", how="left")
        else:
            df["symbol"] = df["ts_code"].astype(str).str.split(".").str[0]
            df["name"] = df["symbol"]

        # Align TuShare fields to existing strategy pipeline field names.
        df["代码"] = df.get("symbol")
        df["名称"] = df.get("name")
        df["最新价"] = df.get("close")
        df["涨跌幅"] = df.get("pct_chg")
        df["换手率"] = df.get("turnover_rate")
        if {"high", "low", "close"}.issubset(df.columns):
            base = df["close"].replace(0, pd.NA).abs()
            df["振幅"] = ((df["high"] - df["low"]).abs() / base * 100).fillna(0)
        else:
            df["振幅"] = 0.0
        df["市盈率-动态"] = df.get("pe")
        df["市净率"] = df.get("pb")
        df["总市值"] = df.get("total_mv")
        # TuShare amount for daily is usually in 千元; normalize to 元 for filter consistency.
        df["成交额"] = pd.to_numeric(df.get("amount"), errors="coerce") * 1000

        df = self._to_numeric(df, ["最新价", "涨跌幅", "换手率", "振幅", "市盈率-动态", "市净率", "总市值", "成交额"])

        before = len(df)
        if "名称" in df.columns:
            df = df[~df["名称"].astype(str).str.contains("ST", case=False, na=False)]
        if "代码" in df.columns:
            df = df[~df["代码"].astype(str).str.startswith(("4", "8"))]
        if "最新价" in df.columns:
            df = df[df["最新价"] > 0]
        if "成交额" in df.columns:
            df = df[df["成交额"] >= 2e8]
        if "总市值" in df.columns:
            df = df[df["总市值"] >= 8e9]
        if "换手率" in df.columns:
            df = df[(df["换手率"] >= 0.1) & (df["换手率"] <= 15)]
        logger.info("TuShare screen layer1 tradability: before=%s after=%s removed=%s", before, len(df), before - len(df))

        if df.empty:
            return []

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
        elif strategy == "growth":
            # 高成长策略：放宽估值限制，关注流动性
            if {"市盈率-动态", "成交额"}.issubset(df.columns):
                df = df[(df["市盈率-动态"] > 0) & (df["市盈率-动态"] <= 80) & (df["成交额"] >= 3e8)]
        logger.info(
            "TuShare screen layer2 strategy_filter: strategy=%s before=%s after=%s removed=%s",
            strategy,
            before,
            len(df),
            before - len(df),
        )

        if df.empty:
            return []

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
            centered = (df["换手率"] - 2.5).abs()
            df["score_turnover"] = self._score_asc(centered)
            score_cols.append("score_turnover")
        if "涨跌幅" in df.columns:
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
            factor_scores = {k: round(float(row.get(k, 0.0)), 4) for k in normalized_weights}
            rows.append(
                {
                    "代码": self._safe_value(row.get("代码")),
                    "名称": self._safe_value(row.get("名称")),
                    "最新价": self._safe_value(row.get("最新价")),
                    "涨跌幅": self._safe_value(row.get("涨跌幅")),
                    "换手率": self._safe_value(row.get("换手率")),
                    "振幅": self._safe_value(row.get("振幅")),
                    "市盈率-动态": self._safe_value(row.get("市盈率-动态")),
                    "市净率": self._safe_value(row.get("市净率")),
                    "总市值": self._safe_value(row.get("总市值")),
                    "成交额": self._safe_value(row.get("成交额")),
                    "strategy": strategy,
                    "total_score": round(float(row.get("total_score", 0.0)), 4),
                    "factor_scores": factor_scores,
                    "timestamp": timestamp,
                    "screen_note": "multi-layer screening: tradability filter + style constraints + weighted factor score (TuShare snapshot).",
                }
            )
        return rows


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

        # 使用 stock_financial_benefit_new_ths 获取财务指标（已修复替代失效的 stock_financial_analysis_indicator）
        latest_indicator = None
        try:
            indicator_df = await self._run_akshare_call(self.ak.stock_financial_benefit_new_ths, symbol=symbol)
            if indicator_df is not None and not indicator_df.empty:
                # 转换为宽表格式，按指标名称展开
                latest_df = indicator_df[indicator_df["report_date"] == indicator_df["report_date"].max()]
                if not latest_df.empty:
                    # 将行转换为字典，保留指标名和值
                    indicator_dict = {}
                    for _, row in latest_df.iterrows():
                        metric = row.get("metric_name", "")
                        value = row.get("value")
                        yoy = row.get("yoy")
                        if metric and value is not None and value != "":
                            indicator_dict[metric] = {"value": value, "yoy": yoy}
                    latest_indicator = indicator_dict
        except Exception:
            latest_indicator = None

        # 使用 stock_individual_info_em 获取估值和基本信息（已修复替代失效的 stock_a_lg_indicator）
        valuation = None
        try:
            info_df = await self._run_akshare_call(self.ak.stock_individual_info_em, symbol=symbol)
            if info_df is not None and not info_df.empty:
                # 转换为字典格式
                valuation = {}
                for _, row in info_df.iterrows():
                    item = row.get("item", "")
                    value = row.get("value")
                    if item:
                        valuation[item] = value
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

    async def get_technical_signals(
        self,
        ticker: str,
        timestamp: str | None = None,
        window: int = 120,
    ) -> dict[str, Any]:
        timestamp = normalize_timestamp(timestamp)
        symbol = self._ticker_to_symbol(ticker)
        asof_dt = datetime.fromisoformat(timestamp).replace(tzinfo=None)
        lookback_days = max(int(window), 120) + 120
        start_dt = asof_dt - timedelta(days=lookback_days)

        hist_df = await self._run_akshare_call(
            self.ak.stock_zh_a_hist,
            symbol=symbol,
            period="daily",
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=asof_dt.strftime("%Y%m%d"),
            adjust="",
        )
        if hist_df is None or hist_df.empty:
            return {
                "ticker": ticker,
                "symbol": symbol,
                "timestamp": timestamp,
                "source": "akshare:stock_zh_a_hist",
                "signal_status": "no_data",
            }

        df = self._asof_filter(hist_df, timestamp)
        if df is None or df.empty:
            return {
                "ticker": ticker,
                "symbol": symbol,
                "timestamp": timestamp,
                "source": "akshare:stock_zh_a_hist",
                "signal_status": "no_data_as_of",
            }

        import pandas as pd  # type: ignore

        work = df.copy()
        for col in ("开盘", "最高", "最低", "收盘"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work.dropna(subset=["收盘"])
        if len(work) < 35:
            return {
                "ticker": ticker,
                "symbol": symbol,
                "timestamp": timestamp,
                "source": "akshare:stock_zh_a_hist",
                "signal_status": "insufficient_data",
                "available_rows": int(len(work)),
            }

        close = work["收盘"]
        high = work["最高"] if "最高" in work.columns else close
        low = work["最低"] if "最低" in work.columns else close

        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = (dif - dea) * 2

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi14 = 100 - (100 / (1 + rs))

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()

        support_20 = low.tail(20).min() if len(low) >= 20 else low.min()
        resistance_20 = high.tail(20).max() if len(high) >= 20 else high.max()
        support_60 = low.tail(60).min() if len(low) >= 60 else low.min()
        resistance_60 = high.tail(60).max() if len(high) >= 60 else high.max()

        latest_close = float(close.iloc[-1])
        latest_ma20 = float(ma20.iloc[-1]) if pd.notna(ma20.iloc[-1]) else latest_close
        latest_ma60 = float(ma60.iloc[-1]) if pd.notna(ma60.iloc[-1]) else latest_ma20
        latest_dif = float(dif.iloc[-1]) if pd.notna(dif.iloc[-1]) else 0.0
        latest_dea = float(dea.iloc[-1]) if pd.notna(dea.iloc[-1]) else 0.0
        latest_rsi14 = float(rsi14.iloc[-1]) if pd.notna(rsi14.iloc[-1]) else 50.0
        latest_atr14 = float(atr14.iloc[-1]) if pd.notna(atr14.iloc[-1]) else 0.0
        latest_macd_hist = float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else 0.0

        ma_cross = "neutral"
        if len(ma5) >= 2 and len(ma20) >= 2:
            prev_diff = ma5.iloc[-2] - ma20.iloc[-2]
            curr_diff = ma5.iloc[-1] - ma20.iloc[-1]
            if pd.notna(prev_diff) and pd.notna(curr_diff):
                if prev_diff <= 0 < curr_diff:
                    ma_cross = "golden_cross"
                elif prev_diff >= 0 > curr_diff:
                    ma_cross = "death_cross"

        macd_signal = "neutral"
        if latest_dif > latest_dea and latest_macd_hist > 0:
            macd_signal = "bullish"
        elif latest_dif < latest_dea and latest_macd_hist < 0:
            macd_signal = "bearish"

        rsi_signal = "neutral"
        if latest_rsi14 >= 70:
            rsi_signal = "overbought"
        elif latest_rsi14 <= 30:
            rsi_signal = "oversold"

        breakout = "none"
        if latest_close > float(resistance_20) * 1.001:
            breakout = "bullish_breakout"
        elif latest_close < float(support_20) * 0.999:
            breakout = "bearish_breakdown"

        if latest_close >= latest_ma20 >= latest_ma60:
            trend = "up"
        elif latest_close <= latest_ma20 <= latest_ma60:
            trend = "down"
        else:
            trend = "sideways"

        score = 50
        if trend == "up":
            score += 15
        elif trend == "down":
            score -= 15

        if ma_cross == "golden_cross":
            score += 10
        elif ma_cross == "death_cross":
            score -= 10

        if macd_signal == "bullish":
            score += 10
        elif macd_signal == "bearish":
            score -= 10

        if rsi_signal == "oversold":
            score += 5
        elif rsi_signal == "overbought":
            score -= 5

        if breakout == "bullish_breakout":
            score += 8
        elif breakout == "bearish_breakdown":
            score -= 8

        score = max(0, min(100, int(score)))

        return {
            "ticker": ticker,
            "symbol": symbol,
            "timestamp": timestamp,
            "source": "akshare:stock_zh_a_hist",
            "signal_status": "ok",
            "window": int(window),
            "trend": trend,
            "signals": {
                "ma_cross": ma_cross,
                "macd": macd_signal,
                "rsi": rsi_signal,
                "breakout": breakout,
            },
            "levels": {
                "support_20": round(float(support_20), 4),
                "resistance_20": round(float(resistance_20), 4),
                "support_60": round(float(support_60), 4),
                "resistance_60": round(float(resistance_60), 4),
            },
            "score": score,
            "raw": {
                "close": round(latest_close, 4),
                "ma5": round(float(ma5.iloc[-1]), 4) if pd.notna(ma5.iloc[-1]) else None,
                "ma10": round(float(ma10.iloc[-1]), 4) if pd.notna(ma10.iloc[-1]) else None,
                "ma20": round(latest_ma20, 4),
                "ma60": round(latest_ma60, 4),
                "dif": round(latest_dif, 6),
                "dea": round(latest_dea, 6),
                "macd_hist": round(latest_macd_hist, 6),
                "rsi14": round(latest_rsi14, 4),
                "atr14": round(latest_atr14, 4),
            },
        }

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
        elif strategy == "growth":
            # 高成长策略：放宽估值限制，关注流动性和适度涨幅
            if {"市盈率-动态", "市净率", "成交额"}.issubset(df.columns):
                df = df[(df["市盈率-动态"] > 0) & (df["市盈率-动态"] <= 80) & (df["成交额"] >= 3e8)]  # 成长股估值偏高，需放宽
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
            "growth": {
                # 高成长策略：看重流动性、适度市值、动量，宽松估值
                "score_size": 0.20,
                "score_liquidity": 0.30,  # 高流动性意味着市场关注度高
                "score_pe": 0.10,  # 成长股估值通常偏高，降低权重
                "score_pb": 0.08,
                "score_volatility": 0.12,  # 适度波动
                "score_turnover": 0.10,
                "score_momentum_stability": 0.10,  # 适度关注动量
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

    async def get_market_index(self, timestamp: str | None = None) -> dict[str, Any]:
        """Get major market indices - AkShare网络不稳定，返回提示信息"""
        return {
            "timestamp": normalize_timestamp(timestamp),
            "source": "akshare",
            "error": "AkShare network unavailable. Please use local data source.",
            "indices": {},
        }


class PreferredAShareDataBackend(AShareDataBackend):
    """Preferred backend chain: TuShare first, AkShare fallback."""

    def __init__(self, primary: AShareDataBackend, fallback: AShareDataBackend):
        self.primary = primary
        self.fallback = fallback

    @staticmethod
    def _is_empty_result(method_name: str, payload: Any) -> bool:
        if payload is None:
            return True
        if method_name == "get_quote" and isinstance(payload, dict):
            return payload.get("quote") is None
        if method_name == "screen_stocks" and isinstance(payload, list):
            return len(payload) == 0
        if method_name == "get_news" and isinstance(payload, list):
            return len(payload) == 0
        if method_name == "get_fundamentals" and isinstance(payload, dict):
            return payload.get("financial_indicator") is None and payload.get("valuation_indicator") is None
        if method_name == "get_technical_signals" and isinstance(payload, dict):
            status = str(payload.get("signal_status") or "").lower()
            return status.startswith("no_data")
        return False

    async def _call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        primary_method = getattr(self.primary, method_name)
        fallback_method = getattr(self.fallback, method_name)
        try:
            primary_result = await primary_method(*args, **kwargs)
            if self._is_empty_result(method_name, primary_result):
                logger.warning(
                    "Primary stock backend returned empty payload, fallback to secondary: method=%s",
                    method_name,
                )
                return await fallback_method(*args, **kwargs)
            return primary_result
        except Exception as exc:
            logger.warning(
                "Primary stock backend failed, fallback to secondary: method=%s error=%s",
                method_name,
                repr(exc),
            )
            return await fallback_method(*args, **kwargs)

    async def get_quote(self, ticker: str, timestamp: str | None = None) -> dict[str, Any]:
        return await self._call("get_quote", ticker=ticker, timestamp=timestamp)

    async def get_fundamentals(self, ticker: str, timestamp: str | None = None, period: str = "ttm") -> dict[str, Any]:
        return await self._call("get_fundamentals", ticker=ticker, timestamp=timestamp, period=period)

    async def get_news(self, timestamp: str | None = None, ticker: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        return await self._call("get_news", timestamp=timestamp, ticker=ticker, limit=limit)

    async def screen_stocks(self, strategy: str, timestamp: str | None = None, max_results: int = 20) -> list[dict[str, Any]]:
        return await self._call("screen_stocks", strategy=strategy, timestamp=timestamp, max_results=max_results)

    async def get_technical_signals(
        self,
        ticker: str,
        timestamp: str | None = None,
        window: int = 120,
    ) -> dict[str, Any]:
        return await self._call("get_technical_signals", ticker=ticker, timestamp=timestamp, window=window)

    async def get_market_index(self, timestamp: str | None = None) -> dict[str, Any]:
        return await self._call("get_market_index", timestamp=timestamp)


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
        return "使用保守策略筛选A股候选股票（quality、dividend、low_vol、value）。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "可选分析时点（ISO 8601）。不传则使用实时模式。",
                },
                "strategy": {
                    "type": "string",
                    "description": "筛选策略。",
                    "enum": ["quality", "dividend", "low_vol", "value"],
                    "default": "quality",
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回候选股票的最大数量。",
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


class AShareTechnicalSignalsTool(Tool):
    """Generate technical signals for one A-share ticker."""

    def __init__(self, backend: AShareDataBackend):
        self.backend = backend

    @property
    def name(self) -> str:
        return "get_a_share_technical_signals"

    @property
    def description(self) -> str:
        return "生成A股技术信号（趋势、均线、MACD、RSI、突破、支撑阻力）。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "A股代码，例如 600519.SH 或 000001.SZ",
                },
                "timestamp": {
                    "type": "string",
                    "description": "可选分析时点（ISO 8601）。不传则使用实时模式。",
                },
                "window": {
                    "type": "integer",
                    "description": "技术分析窗口（交易日），默认 120。",
                    "default": 120,
                },
            },
            "required": ["ticker"],
        }

    async def execute(self, ticker: str, timestamp: str | None = None, window: int = 120) -> ToolResult:
        try:
            normalized_ts = normalize_timestamp(timestamp)
            data = await self.backend.get_technical_signals(ticker=ticker, timestamp=normalized_ts, window=window)
            return ToolResult(success=True, content=json.dumps(data, ensure_ascii=False, indent=2))
        except NotImplementedError as e:
            return ToolResult(success=False, content="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to get technical signals: {e}")


class AShareMarketIndexTool(Tool):
    """Get major market indices: Shanghai, Shenzhen, ChiNext, CSI300"""

    def __init__(self, backend: AShareDataBackend):
        self.backend = backend

    @property
    def name(self) -> str:
        return "get_market_index"

    @property
    def description(self) -> str:
        return "Get major A-share market indices (Shanghai, Shenzhen, ChiNext, CSI300) with daily行情"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "Optional analysis time in ISO 8601. If omitted, realtime mode is used.",
                },
            },
        }

    async def execute(self, timestamp: str | None = None) -> ToolResult:
        try:
            normalized_ts = normalize_timestamp(timestamp)
            data = await self.backend.get_market_index(timestamp=normalized_ts)
            return ToolResult(success=True, content=json.dumps(data, ensure_ascii=False, indent=2))
        except NotImplementedError as e:
            return ToolResult(success=False, content="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, content=f"Failed to get market index: {e}")


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


def create_a_share_tools(
    backend: AShareDataBackend | None = None,
    tushare_token: str | None = None,
    kline_db_path: str | None = None,
) -> list[Tool]:
    """Create all A-share related tools.

    Args:
        backend: Optional concrete backend implementation. If omitted,
            PlaceholderAShareDataBackend is used.
        tushare_token: Optional TuShare token from config; env variables still supported as fallback.
    """
    # 统一入口：后续只需替换 backend，不需要改 Agent 或 CLI 组装逻辑。
    if backend is not None:
        data_backend = backend
    else:
        local_backend = LocalAShareDataBackend(db_path=kline_db_path or str(DEFAULT_MEMORY_DB_PATH))
        try:
            ak_backend = AkShareDataBackend()
        except ImportError:
            ak_backend = None

        tushare_backend = None
        try:
            # Prefer TuShare when token is configured.
            token = (tushare_token or "").strip() or None
            if token is None:
                tushare_backend = TuShareDataBackend()
            else:
                tushare_backend = TuShareDataBackend(token=token)
            logger.info("A-share backend selected: TuShare primary")
        except Exception as exc:
            logger.warning("TuShare backend unavailable, skip primary: %s", exc)

        if tushare_backend is not None and ak_backend is not None:
            remote_backend = PreferredAShareDataBackend(primary=tushare_backend, fallback=ak_backend)
            data_backend = PreferredAShareDataBackend(primary=local_backend, fallback=remote_backend)
            logger.info("A-share backend mode: preferred chain (Local -> TuShare -> AkShare)")
        elif tushare_backend is not None:
            data_backend = PreferredAShareDataBackend(primary=local_backend, fallback=tushare_backend)
            logger.info("A-share backend mode: preferred chain (Local -> TuShare)")
        elif ak_backend is not None:
            data_backend = PreferredAShareDataBackend(primary=local_backend, fallback=ak_backend)
            logger.info("A-share backend mode: preferred chain (Local -> AkShare)")
        else:
            data_backend = local_backend
            logger.warning("A-share backend mode: Local only (no TuShare/AkShare available)")
    return [
        AShareScreenTool(data_backend),
        AShareTechnicalSignalsTool(data_backend),
        AShareQuoteTool(data_backend),
        AShareFundamentalsTool(data_backend),
        AShareNewsTool(data_backend),
        AShareMarketIndexTool(data_backend),
        ConservativeTradePlanTool(),
    ]
