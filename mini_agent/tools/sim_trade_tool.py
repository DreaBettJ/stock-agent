"""Simulation trade tool based on session state + K-line data."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from mini_agent.paths import DEFAULT_MEMORY_DB_PATH
from mini_agent.session import SessionManager

from .base import Tool, ToolResult
from .kline_db_tool import KLineDB


class _SimTradeStore:
    """SQLite storage for simulation trades and positions."""

    def __init__(self, db_path: str = str(DEFAULT_MEMORY_DB_PATH)):
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
                CREATE TABLE IF NOT EXISTS sim_trades (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL,
                    price REAL,
                    quantity INT,
                    amount REAL,
                    fee REAL,
                    profit REAL,
                    trade_date TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sim_positions (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    quantity INT,
                    avg_cost REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT,
                    UNIQUE(session_id, ticker)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_trades_session ON sim_trades(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sim_positions_session ON sim_positions(session_id)")
            conn.commit()

    def get_position(self, session_id: str, ticker: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT session_id, ticker, quantity, avg_cost FROM sim_positions WHERE session_id = ? AND ticker = ?",
                (session_id, ticker),
            ).fetchone()

    def upsert_position(self, session_id: str, ticker: str, quantity: int, avg_cost: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sim_positions (session_id, ticker, quantity, avg_cost, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id, ticker) DO UPDATE SET
                    quantity=excluded.quantity,
                    avg_cost=excluded.avg_cost,
                    updated_at=excluded.updated_at
                """,
                (session_id, ticker, quantity, avg_cost, datetime.now().isoformat()),
            )
            conn.commit()

    def delete_position(self, session_id: str, ticker: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sim_positions WHERE session_id = ? AND ticker = ?", (session_id, ticker))
            conn.commit()

    def insert_trade(
        self,
        session_id: str,
        ticker: str,
        action: str,
        price: float,
        quantity: int,
        amount: float,
        fee: float,
        profit: float | None,
        trade_date: str,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sim_trades (session_id, ticker, action, price, quantity, amount, fee, profit, trade_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, ticker, action, price, quantity, amount, fee, profit, trade_date),
            )
            conn.commit()
            return int(cursor.lastrowid)


class SimulateTradeTool(Tool):
    """Execute buy/sell simulation against session cash and positions."""

    COMMISSION_RATE = 0.0003
    STAMP_DUTY_RATE = 0.001

    def __init__(
        self,
        db_path: str = str(DEFAULT_MEMORY_DB_PATH),
        kline_db_path: str = None,
    ):
        self.db_path = db_path
        self.session_manager = SessionManager(db_path=db_path)
        # K-line data is in separate database
        if kline_db_path:
            self.kline_db = KLineDB(db_path=kline_db_path)
        else:
            # Try to find kline DB in common locations
            import os
            workspace = os.path.dirname(os.path.abspath(db_path))
            default_kline = os.path.join(workspace, "stock_kline.db")
            if os.path.exists(default_kline):
                self.kline_db = KLineDB(db_path=default_kline)
            else:
                self.kline_db = KLineDB(db_path=db_path)
        self.store = _SimTradeStore(db_path=db_path)

    @property
    def name(self) -> str:
        return "simulate_trade"

    @property
    def description(self) -> str:
        return (
            "Execute simulation buy/sell with next-day open price, commission 0.03%, "
            "and stamp duty 0.1% on sell."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Experiment session id"},
                "action": {"type": "string", "enum": ["buy", "sell"], "description": "Trade action"},
                "ticker": {"type": "string", "description": "Stock code"},
                "quantity": {"type": "integer", "description": "Shares to trade"},
                "trade_date": {
                    "type": "string",
                    "description": "Reference date for execution price. Trade executes at next open.",
                },
            },
            "required": ["session_id", "action", "ticker", "quantity", "trade_date"],
        }

    async def execute(self, session_id: str, action: str, ticker: str, quantity: int, trade_date: str) -> ToolResult:
        try:
            if quantity <= 0:
                return ToolResult(success=False, content="", error="quantity must be > 0")

            session = self.session_manager.get_session(session_id)
            if session.mode != "simulation":
                return ToolResult(success=False, content="", error=f"session mode must be simulation, got {session.mode}")

            symbol = self.kline_db.normalize_ticker(ticker)
            action_normalized = action.strip().lower()
            
            # Buy uses next open price, Sell uses close price (today)
            if action_normalized == "buy":
                exec_price = self.kline_db.get_next_open_price(symbol, trade_date)
                gross_amount = exec_price * quantity
                fee = gross_amount * self.COMMISSION_RATE
                cash_delta = -(gross_amount + fee)
                if session.current_cash + cash_delta < -1e-9:
                    return ToolResult(success=False, content="", error="insufficient cash")

                old_position = self.store.get_position(session_id, symbol)
                old_qty = int(old_position["quantity"]) if old_position else 0
                old_cost = float(old_position["avg_cost"]) if old_position else 0.0
                new_qty = old_qty + quantity
                new_avg_cost = ((old_qty * old_cost) + gross_amount + fee) / new_qty
                self.store.upsert_position(session_id, symbol, new_qty, new_avg_cost)
                profit = None

            elif action_normalized == "sell":
                # For sell, use today's close price
                exec_price = self.kline_db.get_latest_price(symbol)
                gross_amount = exec_price * quantity
                old_position = self.store.get_position(session_id, symbol)
                if old_position is None or int(old_position["quantity"]) < quantity:
                    return ToolResult(success=False, content="", error="insufficient position")

                held_qty = int(old_position["quantity"])
                avg_cost = float(old_position["avg_cost"])
                fee = gross_amount * (self.COMMISSION_RATE + self.STAMP_DUTY_RATE)
                cash_delta = gross_amount - fee
                profit = (exec_price - avg_cost) * quantity - fee

                remaining_qty = held_qty - quantity
                if remaining_qty <= 0:
                    self.store.delete_position(session_id, symbol)
                else:
                    self.store.upsert_position(session_id, symbol, remaining_qty, avg_cost)

            else:
                return ToolResult(success=False, content="", error="action must be buy or sell")

            self.session_manager.update_current_cash(session_id, session.current_cash + cash_delta)
            trade_id = self.store.insert_trade(
                session_id=session_id,
                ticker=symbol,
                action=action_normalized,
                price=exec_price,
                quantity=quantity,
                amount=gross_amount,
                fee=fee,
                profit=profit,
                trade_date=trade_date,
            )

            content = (
                f"SIM_TRADE_OK id={trade_id} session={session_id} action={action_normalized} "
                f"ticker={symbol} qty={quantity} price={exec_price:.3f} amount={gross_amount:.2f} fee={fee:.2f}"
            )
            if profit is not None:
                content += f" profit={profit:.2f}"
            return ToolResult(success=True, content=content)
        except Exception as exc:
            return ToolResult(success=False, content="", error=f"simulate trade failed: {exc}")


def create_simulation_trade_tools(db_path: str = str(DEFAULT_MEMORY_DB_PATH)) -> list[Tool]:
    """Create simulation trade tools."""
    return [SimulateTradeTool(db_path=db_path)]
