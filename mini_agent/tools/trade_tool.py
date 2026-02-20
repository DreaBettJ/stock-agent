"""Trade Record Tool backed by SQLite.

This tool records buy/sell operations for stock trading with:
- session_id (for tracking which AI session made the decision)
- operation type (buy/sell)
- ticker (stock code)
- price (cost price)
- quantity (number of shares)
- reason (why this trade was made)
- timestamp (operation time)
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .base import Tool, ToolResult


class _TradeStore:
    """Lightweight SQLite trade store."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
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
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    amount REAL NOT NULL,
                    reason TEXT,
                    notes TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            conn.commit()

    def insert(
        self,
        session_id: str,
        operation: str,
        ticker: str,
        price: float,
        quantity: float,
        reason: str,
        notes: str | None = None,
    ) -> int:
        amount = price * quantity
        timestamp = datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO trades (session_id, timestamp, operation, ticker, price, quantity, amount, reason, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, timestamp, operation, ticker, price, quantity, amount, reason, notes),
            )
            conn.commit()
            return cursor.lastrowid

    def query(
        self,
        ticker: str | None = None,
        session_id: str | None = None,
        operation: str | None = None,
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        sql = "SELECT id, session_id, timestamp, operation, ticker, price, quantity, amount, reason, notes FROM trades"
        clauses = []
        params: list[Any] = []

        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if operation:
            clauses.append("operation = ?")
            params.append(operation)

        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, min(limit, 1000)))

        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            return list(cursor.fetchall())

    def get_positions(self, ticker: str | None = None) -> list[sqlite3.Row]:
        """Calculate current positions (holdings) from trade history."""
        sql = """
            SELECT 
                ticker,
                SUM(CASE WHEN operation = 'buy' THEN quantity ELSE -quantity END) as net_quantity,
                SUM(CASE WHEN operation = 'buy' THEN amount ELSE -amount END) as net_amount,
                COUNT(*) as trade_count
            FROM trades
        """
        params = []
        if ticker:
            sql += " WHERE ticker = ?"
            params.append(ticker)

        sql += " GROUP BY ticker HAVING net_quantity > 0"

        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            return list(cursor.fetchall())


class TradeRecordTool(Tool):
    """Tool for recording buy/sell trades into SQLite."""

    def __init__(self, memory_file: str = "./workspace/.agent_memory.db", session_id: str | None = None):
        self.memory_file = Path(memory_file)
        self.session_id = session_id or uuid4().hex
        self.store = _TradeStore(self.memory_file)

    @property
    def name(self) -> str:
        return "record_trade"

    @property
    def description(self) -> str:
        return (
            "Record a buy or sell trade operation. "
            "Stores: session_id, operation type, ticker, price, quantity, reason, timestamp. "
            "Use this to track all trading decisions made by the AI."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Operation type: 'buy' or 'sell'",
                    "enum": ["buy", "sell"],
                },
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., 600519.SH, 000001.SZ)",
                },
                "price": {
                    "type": "number",
                    "description": "Cost price per share",
                },
                "quantity": {
                    "type": "number",
                    "description": "Number of shares",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for this trade (e.g., '低估', '突破阻力位', '止盈', '止损')",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional additional notes",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional override session_id. If omitted, tool default session_id is used.",
                },
            },
            "required": ["operation", "ticker", "price", "quantity", "reason"],
        }

    async def execute(
        self,
        operation: str,
        ticker: str,
        price: float,
        quantity: float,
        reason: str,
        notes: str | None = None,
        session_id: str | None = None,
    ) -> ToolResult:
        try:
            sid = session_id or self.session_id
            
            # Validate operation
            operation = operation.lower()
            if operation not in ["buy", "sell"]:
                return ToolResult(success=False, content="", error=f"Invalid operation: {operation}. Must be 'buy' or 'sell'")
            
            # Validate inputs
            if price <= 0:
                return ToolResult(success=False, content="", error=f"Invalid price: {price}. Must be > 0")
            if quantity <= 0:
                return ToolResult(success=False, content="", error=f"Invalid quantity: {quantity}. Must be > 0")
            
            # Record trade
            trade_id = self.store.insert(
                session_id=sid,
                operation=operation,
                ticker=ticker.upper(),
                price=price,
                quantity=quantity,
                reason=reason,
                notes=notes,
            )
            
            amount = price * quantity
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            content = f"""✅ Trade Recorded:
- ID: {trade_id}
- Session: {sid[:8]}...
- Operation: {operation.upper()}
- Ticker: {ticker.upper()}
- Price: {price:.2f}
- Quantity: {quantity}
- Amount: {amount:,.2f}
- Reason: {reason}
- Time: {timestamp}"""
            
            if notes:
                content += f"\n- Notes: {notes}"
            
            return ToolResult(success=True, content=content)
            
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to record trade: {str(e)}")


class TradeHistoryTool(Tool):
    """Tool for querying trade history from SQLite."""

    def __init__(self, memory_file: str = "./workspace/.agent_memory.db", session_id: str | None = None):
        self.memory_file = Path(memory_file)
        self.session_id = session_id
        self.store = _TradeStore(self.memory_file)

    @property
    def name(self) -> str:
        return "get_trade_history"

    @property
    def description(self) -> str:
        return (
            "Query trade history from SQLite. "
            "Can filter by ticker, session_id, and operation type (buy/sell)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Optional: filter by stock ticker (e.g., 600519.SH)",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional: filter by session_id",
                },
                "operation": {
                    "type": "string",
                    "description": "Optional: filter by operation type ('buy' or 'sell')",
                    "enum": ["buy", "sell"],
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of trades to return (default 100)",
                    "default": 100,
                },
            },
        }

    async def execute(
        self,
        ticker: str | None = None,
        session_id: str | None = None,
        operation: str | None = None,
        limit: int = 100,
    ) -> ToolResult:
        try:
            trades = self.store.query(ticker=ticker, session_id=session_id, operation=operation, limit=limit)

            if not trades:
                filters = []
                if ticker:
                    filters.append(f"ticker={ticker}")
                if session_id:
                    filters.append(f"session={session_id[:8]}...")
                if operation:
                    filters.append(f"operation={operation}")
                filter_str = f" ({', '.join(filters)})" if filters else ""
                
                return ToolResult(success=True, content=f"No trades found{filter_str}.")

            formatted = []
            for trade in trades:
                op = trade["operation"].upper()
                t = trade["ticker"]
                p = trade["price"]
                q = trade["quantity"]
                a = trade["amount"]
                r = trade["reason"]
                ts = trade["timestamp"]
                sid = trade["session_id"]
                formatted.append(f"{op} {t} × {q} @ {p:.2f} = {a:,.2f} | {r} | {ts[:19]} | sid:{sid[:8]}")

            header = f"Trade History ({len(formatted)} records):"
            return ToolResult(success=True, content=header + "\n" + "\n".join(formatted))
            
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to query trades: {str(e)}")


class PositionTool(Tool):
    """Tool for getting current positions (holdings) from trade history."""

    def __init__(self, memory_file: str = "./workspace/.agent_memory.db", session_id: str | None = None):
        self.memory_file = Path(memory_file)
        self.session_id = session_id
        self.store = _TradeStore(self.memory_file)

    @property
    def name(self) -> str:
        return "get_positions"

    @property
    def description(self) -> str:
        return (
            "Get current positions (holdings) calculated from trade history. "
            "Shows net quantity and amount for each ticker."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Optional: filter by stock ticker",
                },
            },
        }

    async def execute(self, ticker: str | None = None) -> ToolResult:
        try:
            positions = self.store.get_positions(ticker=ticker)

            if not positions:
                return ToolResult(success=True, content="No open positions.")

            formatted = []
            total_value = 0
            for pos in positions:
                t = pos["ticker"]
                q = pos["net_quantity"]
                a = pos["net_amount"]
                avg_price = a / q if q > 0 else 0
                formatted.append(f"{t}: {q:.0f} shares @ avg {avg_price:.2f} = {a:,.2f}")
                total_value += a

            header = f"Current Positions ({len(formatted)} holdings, total: {total_value:,.2f}):"
            return ToolResult(success=True, content=header + "\n" + "\n".join(formatted))
            
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to get positions: {str(e)}")


def create_trade_tools(memory_file: str = "./workspace/.agent_memory.db", session_id: str | None = None) -> list[Tool]:
    """Create all trade-related tools.
    
    Args:
        memory_file: Path to SQLite database
        session_id: Current session identifier
    """
    return [
        TradeRecordTool(memory_file=memory_file, session_id=session_id),
        TradeHistoryTool(memory_file=memory_file, session_id=session_id),
        PositionTool(memory_file=memory_file, session_id=session_id),
    ]
