"""K-line database utility and query tool."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult


class KLineDB:
    """SQLite K-line database accessor."""

    def __init__(self, db_path: str = "./workspace/.agent_memory.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_kline (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kline_ticker ON daily_kline(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kline_date ON daily_kline(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kline_ticker_date ON daily_kline(ticker, date)")
            conn.commit()

    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        ts = ticker.strip().upper()
        if "." in ts:
            return ts.split(".")[0]
        if ts.startswith(("SH", "SZ")):
            return ts[2:]
        return ts

    def upsert_daily_kline(self, ticker: str, date: str, open_price: float, high: float, low: float, close: float, volume: float = 0.0, amount: float = 0.0) -> None:
        """Insert or update one K-line row."""
        symbol = self.normalize_ticker(ticker)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO daily_kline (ticker, date, open, high, low, close, volume, amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume,
                    amount=excluded.amount
                """,
                (symbol, date, open_price, high, low, close, volume, amount),
            )
            conn.commit()

    def get_kline(self, ticker: str, start: str, end: str) -> list[dict[str, Any]]:
        """Get historical K-line rows."""
        symbol = self.normalize_ticker(ticker)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ticker, date, open, high, low, close, volume, amount
                FROM daily_kline
                WHERE ticker = ? AND date BETWEEN ? AND ?
                ORDER BY date ASC
                """,
                (symbol, start, end),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_latest_price(self, ticker: str) -> float:
        """Get latest close price."""
        symbol = self.normalize_ticker(ticker)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT close FROM daily_kline WHERE ticker = ? ORDER BY date DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        if row is None or row["close"] is None:
            raise KeyError(f"kline not found for ticker: {symbol}")
        return float(row["close"])

    def get_next_open_price(self, ticker: str, as_of_date: str) -> float:
        """Get next trading day's open price, fallback to same-day close."""
        symbol = self.normalize_ticker(ticker)
        with self._connect() as conn:
            next_row = conn.execute(
                """
                SELECT open FROM daily_kline
                WHERE ticker = ? AND date > ?
                ORDER BY date ASC
                LIMIT 1
                """,
                (symbol, as_of_date),
            ).fetchone()
            if next_row and next_row["open"] is not None:
                return float(next_row["open"])

            same_day = conn.execute(
                "SELECT close FROM daily_kline WHERE ticker = ? AND date = ? LIMIT 1",
                (symbol, as_of_date),
            ).fetchone()
        if same_day and same_day["close"] is not None:
            return float(same_day["close"])
        raise KeyError(f"next open price not found for ticker={symbol} as_of_date={as_of_date}")

    def get_trading_days(self, start: str, end: str) -> list[str]:
        """Get trading day list within range."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT date
                FROM daily_kline
                WHERE date BETWEEN ? AND ?
                ORDER BY date ASC
                """,
                (start, end),
            ).fetchall()
        return [str(row["date"]) for row in rows]


class KLineQueryTool(Tool):
    """Tool to query historical daily K-line rows."""

    def __init__(self, db_path: str = "./workspace/.agent_memory.db"):
        self.kline_db = KLineDB(db_path=db_path)

    @property
    def name(self) -> str:
        return "query_kline"

    @property
    def description(self) -> str:
        return "Query historical daily K-line rows by ticker and date range."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock code, e.g. 600519.SH"},
                "start": {"type": "string", "description": "Start date, YYYY-MM-DD"},
                "end": {"type": "string", "description": "End date, YYYY-MM-DD"},
            },
            "required": ["ticker", "start", "end"],
        }

    async def execute(self, ticker: str, start: str, end: str) -> ToolResult:
        try:
            rows = self.kline_db.get_kline(ticker=ticker, start=start, end=end)
            if not rows:
                return ToolResult(success=True, content="[]")
            return ToolResult(success=True, content=json.dumps(rows, ensure_ascii=False, indent=2))
        except Exception as exc:
            return ToolResult(success=False, content="", error=f"Failed to query kline: {exc}")


def create_kline_tools(db_path: str = "./workspace/.agent_memory.db") -> list[Tool]:
    """Create K-line related tools."""
    return [KLineQueryTool(db_path=db_path)]
