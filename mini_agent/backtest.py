"""Backtest engine for historical simulation."""

from __future__ import annotations

import json
import inspect
import math
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .event_broadcaster import EventBroadcaster
from .session import SessionManager
from .tools.kline_db_tool import KLineDB


class PerformanceAnalyzer:
    """Calculate performance metrics for backtest."""

    @staticmethod
    def calculate(trades: list[dict], equity_curve: list[dict]) -> dict[str, Any]:
        """Calculate performance metrics."""
        if not equity_curve:
            return {"error": "No equity curve data"}

        initial_value = equity_curve[0]["value"]
        final_value = equity_curve[-1]["value"]
        
        # Total return
        total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0

        # Annual return
        start_date = datetime.fromisoformat(equity_curve[0]["date"])
        end_date = datetime.fromisoformat(equity_curve[-1]["date"])
        years = (end_date - start_date).days / 365.0
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Max drawdown
        peak = equity_curve[0]["value"]
        max_drawdown = 0.0
        for entry in equity_curve:
            if entry["value"] > peak:
                peak = entry["value"]
            dd = (peak - entry["value"]) / peak if peak > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        # Win rate
        sell_trades = [t for t in trades if t.get("action") == "sell" and t.get("profit") is not None]
        wins = sum(1 for t in sell_trades if t.get("profit", 0) > 0)
        win_rate = wins / len(sell_trades) if sell_trades else 0

        # Profit factor
        total_profit = sum(t["profit"] for t in sell_trades if t.get("profit", 0) > 0)
        total_loss = abs(sum(t["profit"] for t in sell_trades if t.get("profit", 0) < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        # Sharpe ratio (daily, risk-free rate assumed 0)
        daily_returns: list[float] = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]["value"]
            curr = equity_curve[i]["value"]
            if prev > 0:
                daily_returns.append((curr - prev) / prev)
        if daily_returns:
            mean_ret = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_ret) ** 2 for r in daily_returns) / len(daily_returns)
            std_ret = math.sqrt(variance)
            sharpe_ratio = (mean_ret / std_ret) * math.sqrt(252) if std_ret > 1e-12 else 0.0
        else:
            sharpe_ratio = 0.0

        return {
            "initial_capital": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": -max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(trades),
            "winning_trades": wins,
            "losing_trades": len(sell_trades) - wins,
        }


class HistoricalEventGenerator:
    """Generate historical events for backtest."""

    def __init__(self, kline_db: KLineDB):
        self.kline_db = kline_db

    def generate_daily_review_event(self, trading_date: str) -> dict[str, Any]:
        """Generate daily review event for a specific date."""
        # Get previous trading day
        prev_day = self._get_previous_trading_day(trading_date)
        if not prev_day:
            return {"type": "daily_review", "date": trading_date, "error": "No previous trading day"}

        # Get market data for that day
        try:
            # Get top gainers (simplified - just get some tickers with data)
            top_gainers = self._get_top_gainers(prev_day)
        except Exception:
            top_gainers = []

        return {
            "type": "daily_review",
            "date": trading_date,
            "previous_date": prev_day,
            "top_gainers": top_gainers,
        }

    def _get_previous_trading_day(self, current_date: str) -> str | None:
        """Get previous trading day."""
        days = self.kline_db.get_trading_days("1990-01-01", current_date)
        if len(days) < 2:
            return None
        return days[-2] if days[-1] == current_date else days[-1] if days else None

    def _get_top_gainers(self, trading_date: str, limit: int = 50) -> list[dict]:
        """Get top gainers for a trading day."""
        with sqlite3.connect(self.kline_db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT ticker, close, open,
                       (close - open) / open * 100 as change_pct
                FROM daily_kline
                WHERE date = ?
                ORDER BY change_pct DESC
                LIMIT ?
            """, (trading_date, limit)).fetchall()

        return [
            {"ticker": r["ticker"], "close": r["close"], "change_pct": r["change_pct"]}
            for r in rows if r["change_pct"] is not None
        ]


class BacktestEngine:
    """Backtest engine for running historical simulations."""

    def __init__(
        self,
        session_manager: SessionManager,
        kline_db: KLineDB,
        event_broadcaster: EventBroadcaster,
    ):
        self.session_manager = session_manager
        self.kline_db = kline_db
        self.event_broadcaster = event_broadcaster
        self.historical_generator = HistoricalEventGenerator(kline_db)

    async def run(
        self,
        session_id: int | str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Run backtest for a session."""
        # 1) 校验 session 与交易日历。
        session = self.session_manager.get_session(session_id)
        
        # Get trading days in range
        trading_days = self.kline_db.get_trading_days(start_date, end_date)
        if not trading_days:
            return {"error": f"No trading days in range {start_date} to {end_date}"}

        # 2) 初始化内存中的结果容器。
        equity_curve = []

        # 3) 保证重复回测可复现：清理该 session 旧的模拟仓位与成交。
        self._reset_session_state(session.session_id)

        # 4) 重置 session 运行态。
        initial_cash = session.initial_capital
        self.session_manager.update_current_cash(session.session_id, initial_cash)

        # 标记为运行中并开启监听，便于事件驱动回调。
        self.session_manager.start_session(session.session_id)

        # 5) 按交易日逐日回放。
        for trading_day in trading_days:
            self.session_manager.update_current_date(session.session_id, trading_day)
            
            # Generate event
            event = self.historical_generator.generate_daily_review_event(trading_day)
            
            # Trigger ONLY current session to avoid cross-session contamination.
            trigger_session = getattr(self.event_broadcaster, "trigger_session", None)
            if trigger_session is not None and inspect.iscoroutinefunction(trigger_session):
                await trigger_session(session, event)
            else:
                # 兼容兜底：自定义 broadcaster 没有 trigger_session 时走原广播接口。
                await self.event_broadcaster.broadcast(event)
            
            # 记录当日权益：按当日可见价格估值（避免前视偏差）。
            current_cash = self.session_manager.get_session(session.session_id).current_cash
            position_value = self._calculate_position_value(session.session_id, trading_day)
            total_value = current_cash + position_value
            
            equity_curve.append({
                "date": trading_day,
                "cash": current_cash,
                "position_value": position_value,
                "value": total_value,
            })

        # 6) 结束 session 并汇总结果。
        self.session_manager.finish_session(session.session_id)

        # Get all trades
        all_trades = self._get_session_trades(session.session_id)

        # Calculate performance
        performance = PerformanceAnalyzer.calculate(all_trades, equity_curve)

        return {
            "session_id": session.session_id,
            "start_date": start_date,
            "end_date": end_date,
            "trading_days": len(trading_days),
            "performance": performance,
            "equity_curve": equity_curve,
            "trades": all_trades,
        }

    def _calculate_position_value(self, session_id: int | str, pricing_date: str | None = None) -> float:
        """Calculate current position value."""
        with sqlite3.connect(self.session_manager.db_path) as conn:
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """
                    SELECT ticker, quantity, avg_cost
                    FROM sim_positions
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchall()
            except sqlite3.OperationalError:
                # 新库在首次交易前可能尚未创建模拟交易表。
                return 0.0

        total_value = 0.0
        for row in rows:
            try:
                if pricing_date:
                    # 回测路径：使用“截至当日”的最新收盘价估值持仓。
                    current_price = self.kline_db.get_price_on_or_before(row["ticker"], pricing_date)
                else:
                    # 运行态汇总路径：使用最新可用收盘价。
                    current_price = self.kline_db.get_latest_price(row["ticker"])
                total_value += current_price * row["quantity"]
            except Exception:
                # 单个标的估值失败时忽略，继续计算其他仓位。
                pass

        return total_value

    def _get_session_trades(self, session_id: int | str) -> list[dict]:
        """Get all trades for a session."""
        with sqlite3.connect(self.session_manager.db_path) as conn:
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM sim_trades
                    WHERE session_id = ?
                    ORDER BY trade_date
                    """,
                    (session_id,),
                ).fetchall()
            except sqlite3.OperationalError:
                # 新库在首次交易前可能尚未创建模拟交易表。
                return []

        return [dict(r) for r in rows]

    def _reset_session_state(self, session_id: int) -> None:
        """Clear one session's simulation positions/trades before a new backtest run."""
        with sqlite3.connect(self.session_manager.db_path) as conn:
            # 防御式建表，保证全新数据库首次回测也能直接运行。
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
            conn.execute("DELETE FROM sim_positions WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sim_trades WHERE session_id = ?", (session_id,))
            conn.commit()
