"""Simulation trade tool based on session state + K-line data."""

from __future__ import annotations

import os
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
                    reason TEXT,
                    price_source TEXT,
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
            cols = conn.execute("PRAGMA table_info(sim_trades)").fetchall()
            col_names = {str(r["name"]) for r in cols}
            if "reason" not in col_names:
                conn.execute("ALTER TABLE sim_trades ADD COLUMN reason TEXT")
            if "price_source" not in col_names:
                conn.execute("ALTER TABLE sim_trades ADD COLUMN price_source TEXT")
            conn.commit()

    def get_position(self, session_id: int | str, ticker: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT session_id, ticker, quantity, avg_cost FROM sim_positions WHERE session_id = ? AND ticker = ?",
                (session_id, ticker),
            ).fetchone()

    def upsert_position(self, session_id: int | str, ticker: str, quantity: int, avg_cost: float) -> None:
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

    def delete_position(self, session_id: int | str, ticker: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sim_positions WHERE session_id = ? AND ticker = ?", (session_id, ticker))
            conn.commit()

    def insert_trade(
        self,
        session_id: int | str,
        ticker: str,
        action: str,
        price: float,
        quantity: int,
        amount: float,
        fee: float,
        profit: float | None,
        reason: str | None,
        price_source: str | None,
        trade_date: str,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sim_trades (session_id, ticker, action, price, quantity, amount, fee, profit, reason, price_source, trade_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, ticker, action, price, quantity, amount, fee, profit, reason, price_source, trade_date),
            )
            conn.commit()
            return int(cursor.lastrowid)


class SimulateTradeTool(Tool):
    """Execute buy/sell simulation against session cash and positions."""

    COMMISSION_RATE = 0.0003
    STAMP_DUTY_RATE = 0.001
    # Major A-share market index codes. They are not tradable stock tickers.
    BLOCKED_INDEX_TICKERS = {"000001", "399001", "399006", "000300"}

    def __init__(
        self,
        db_path: str = str(DEFAULT_MEMORY_DB_PATH),
        kline_db_path: str = None,
    ):
        self.db_path = db_path
        self.session_manager = SessionManager(db_path=db_path)
        # K-line data uses the same SQLite DB by default.
        if kline_db_path:
            self.kline_db = KLineDB(db_path=kline_db_path)
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

    def _resolve_execution_price(self, symbol: str, trade_date: str) -> tuple[float, str]:
        """Resolve execution price, preferring next-open then fallback to as-of close."""
        try:
            return self.kline_db.get_next_open_price(symbol, trade_date), "next_open"
        except Exception:
            return self.kline_db.get_price_on_or_before(symbol, trade_date), "asof_close"

    @staticmethod
    def _is_kechuang_ticker(symbol: str) -> bool:
        return str(symbol or "").startswith("688")

    @staticmethod
    def _kechuang_buy_enabled() -> bool:
        raw = str(os.getenv("MINI_AGENT_ENABLE_KECHUANG", "")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @classmethod
    def _is_blocked_index_ticker(cls, symbol: str) -> bool:
        return str(symbol or "").strip() in cls.BLOCKED_INDEX_TICKERS

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {
                    "oneOf": [{"type": "integer"}, {"type": "string"}],
                    "description": "Experiment session id",
                },
                "action": {"type": "string", "enum": ["buy", "sell"], "description": "Trade action"},
                "ticker": {"type": "string", "description": "Stock code"},
                "quantity": {"type": "integer", "description": "Shares to trade"},
                "trade_date": {
                    "type": "string",
                    "description": "Reference date for execution price. Trade executes at next open.",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional execution reason for audit trail.",
                },
            },
            "required": ["session_id", "action", "ticker", "quantity", "trade_date"],
        }

    async def execute(
        self,
        session_id: int | str,
        action: str,
        ticker: str,
        quantity: int,
        trade_date: str,
        reason: str | None = None,
    ) -> ToolResult:
        try:
            if quantity <= 0:
                return ToolResult(success=False, content="", error="quantity must be > 0")

            session = self.session_manager.get_session(session_id)
            if session.mode not in {"simulation", "backtest"}:
                return ToolResult(
                    success=False,
                    content="",
                    error=f"session mode must be simulation/backtest, got {session.mode}",
                )

            symbol = self.kline_db.normalize_ticker(ticker)
            if self._is_blocked_index_ticker(symbol):
                return ToolResult(
                    success=False,
                    content="",
                    error=f"market index ticker is not tradable: {symbol}",
                )
            action_normalized = action.strip().lower()
            
            # Buy/Sell both use next open price to avoid look-ahead bias in backtest.
            if action_normalized == "buy":
                if self._is_kechuang_ticker(symbol) and not self._kechuang_buy_enabled():
                    return ToolResult(
                        success=False,
                        content="",
                        error="market access denied: kechuang buy requires MINI_AGENT_ENABLE_KECHUANG=1",
                    )
                exec_price, price_source = self._resolve_execution_price(symbol, trade_date)
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
                profit = 0.0

            elif action_normalized == "sell":
                exec_price, price_source = self._resolve_execution_price(symbol, trade_date)
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
                reason=(str(reason).strip() if reason else f"auto:{action_normalized}:{symbol}:{trade_date}"),
                price_source=price_source,
                trade_date=trade_date,
            )

            content = (
                f"SIM_TRADE_OK id={trade_id} session={session_id} action={action_normalized} "
                f"ticker={symbol} qty={quantity} price={exec_price:.3f} amount={gross_amount:.2f} fee={fee:.2f} "
                f"price_source={price_source}"
            )
            if profit is not None:
                content += f" profit={profit:.2f}"
            if reason:
                content += f" reason={str(reason).strip()[:120]}"
            return ToolResult(success=True, content=content)
        except Exception as exc:
            return ToolResult(success=False, content="", error=f"simulate trade failed: {exc}")


def create_simulation_trade_tools(db_path: str = str(DEFAULT_MEMORY_DB_PATH)) -> list[Tool]:
    """Create simulation trade tools."""
    return [SimulateTradeTool(db_path=db_path)]
