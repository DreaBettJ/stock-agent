"""Auto trading workflow for event-driven backtest/simulation."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from .agent import Agent
from .config import Config
from .event_broadcaster import EventBroadcaster
from .llm import LLMClient
from .schema import LLMProvider
from .session import SessionManager
from .tools.kline_db_tool import KLineDB


class AutoTradingWorkflow:
    """Auto trading workflow: event -> agent analysis -> execute trade."""

    def __init__(
        self,
        session_manager: SessionManager,
        kline_db: KLineDB,
        llm_provider: str = "anthropic",
        api_key: str = None,
        model: str = "MiniMax-M2.5",
    ):
        self.session_manager = session_manager
        self.kline_db = kline_db
        self.llm_provider = llm_provider
        
        # Load API key/model/provider from unified config if not provided.
        if api_key is None:
            try:
                cfg = Config.from_yaml(Config.get_default_config_path())
                api_key = cfg.llm.api_key
                self.llm_provider = cfg.llm.provider
                model = cfg.llm.model
            except Exception:
                # Keep passed-in defaults if config is unavailable.
                pass
        
        self.api_key = api_key
        self.model = model

    async def trigger_daily_review(
        self,
        session_id: int | str,
        trading_date: str | None = None,
        auto_execute: bool = False,
        trade_executor: Any | None = None,
    ) -> dict[str, Any]:
        """Trigger daily review for a session and optionally auto-execute trade."""
        
        # 1. Get session
        session = self.session_manager.get_session(session_id)
        
        # 2. Prepare market data
        trading_date = trading_date or datetime.now().strftime("%Y-%m-%d")
        market_data = self._prepare_market_data(trading_date)
        
        # 3. Prepare positions
        positions = self._get_positions(session_id, as_of_date=trading_date)
        
        # 4. Build prompt with strategy
        prompt = self._build_review_prompt(
            session=session,
            market_data=market_data,
            positions=positions,
            date=trading_date,
        )
        
        # 5. Call Agent for analysis
        llm = LLMClient(
            provider=LLMProvider(self.llm_provider),
            model=self.model,
            api_key=self.api_key,
        )
        
        agent = Agent(
            llm_client=llm,
            system_prompt=session.system_prompt,
            tools=[],  # Agent will analyze only, no tools
            max_steps=3,
        )
        
        # Add user message with context
        agent.add_user_message(prompt)
        
        # Run agent
        result = await agent.run()
        
        # 6. Parse agent response for trade signal
        trade_signal = self._parse_trade_signal(result, session)
        execution = None
        if auto_execute and trade_signal and trade_executor is not None:
            execution = await trade_executor.execute(
                session_id=session_id,
                action=trade_signal["action"],
                ticker=trade_signal["ticker"],
                quantity=trade_signal["quantity"],
                trade_date=trading_date,
            )
        
        return {
            "session_id": session_id,
            "date": trading_date,
            "agent_analysis": result,
            "trade_signal": trade_signal,
            "execution": execution.content if execution and execution.success else None,
            "execution_error": execution.error if execution and not execution.success else None,
        }

    def _prepare_market_data(self, date: str) -> dict[str, Any]:
        """Prepare market data for the day."""
        # Get top gainers
        try:
            top_gainers = self._get_top_gainers(date)
        except Exception:
            top_gainers = []
        
        # Get market summary
        market_summary = {
            "date": date,
            "top_gainers": top_gainers[:10],
            "total_stocks": len(top_gainers),
        }
        
        return market_summary

    def _get_top_gainers(self, date: str, limit: int = 50) -> list[dict]:
        """Get top gainers for a trading day."""
        import sqlite3
        
        with sqlite3.connect(self.kline_db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT ticker, close, open,
                       (close - open) / open * 100 as change_pct
                FROM daily_kline
                WHERE date = ?
                ORDER BY change_pct DESC
                LIMIT ?
            """, (date, limit)).fetchall()

        return [
            {"ticker": r["ticker"], "close": r["close"], "change_pct": r["change_pct"]}
            for r in rows if r["change_pct"] is not None
        ]

    def _get_positions(self, session_id: int | str, as_of_date: str | None = None) -> list[dict]:
        """Get current positions with kline data."""
        import sqlite3
        
        db_path = self.session_manager.db_path
        with sqlite3.connect(db_path) as conn:
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
                return []

        positions = []
        for r in rows:
            # Get current price
            try:
                if as_of_date:
                    current_price = self.kline_db.get_price_on_or_before(r["ticker"], as_of_date)
                else:
                    current_price = self.kline_db.get_latest_price(r["ticker"])
            except Exception:
                current_price = r["avg_cost"]
            
            profit = (current_price - r["avg_cost"]) * r["quantity"]
            profit_rate = (current_price - r["avg_cost"]) / r["avg_cost"] * 100
            
            # Get kline data (last 20 days)
            from datetime import datetime, timedelta
            if as_of_date:
                end_date = as_of_date
                dt = datetime.fromisoformat(as_of_date)
                start_date = (dt - timedelta(days=30)).strftime("%Y-%m-%d")
            else:
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            try:
                klines = self.kline_db.get_kline(r["ticker"], start_date, end_date)
            except Exception:
                klines = []
            
            positions.append({
                "ticker": r["ticker"],
                "quantity": r["quantity"],
                "avg_cost": r["avg_cost"],
                "current_price": current_price,
                "profit": profit,
                "profit_rate": profit_rate,
                "klines": klines[-20:] if klines else [],  # Last 20 days
            })
        
        return positions

    def _build_review_prompt(
        self,
        session,
        market_data: dict,
        positions: list[dict],
        date: str,
    ) -> str:
        """Build daily review prompt."""
        
        # Format positions with kline data
        positions_text = "无持仓"
        if positions:
            lines = []
            for p in positions:
                kline_info = ""
                if p.get("klines") and len(p["klines"]) >= 5:
                    # Get recent price trend
                    recent = p["klines"][-5:]
                    closes = [k["close"] for k in recent]
                    trend = "上涨" if closes[-1] > closes[0] else "下跌"
                    ma5 = sum(closes[-5:]) / 5
                    ma10 = sum(closes[-10:]) / 10 if len(closes) >= 10 else ma5
                    golden_cross = "金叉" if ma5 > ma10 else "死叉"
                    kline_info = f", 5日均线{ma5:.2f}, 10日均线{ma10:.2f}, {golden_cross}"
                
                lines.append(
                    f"- {p['ticker']}: 数量{p['quantity']}, "
                    f"成本{p['avg_cost']:.2f}, "
                    f"现价{p['current_price']:.2f}, "
                    f"盈亏{p['profit']:.2f}({p['profit_rate']:.1f}%){kline_info}"
                )
            positions_text = "\n".join(lines)
        
        # Format top gainers
        gainers_text = "无数据"
        if market_data.get("top_gainers"):
            gainers_text = "\n".join([
                f"- {s['ticker']}: {s['change_pct']:.1f}%"
                for s in market_data["top_gainers"][:10]
            ])
        
        prompt = f"""【时间】{date}
【事件】每日复盘

【当前持仓】
{positions_text}

【今日涨幅榜】
{gainers_text}

【你的策略】
{session.system_prompt}

请根据你的策略分析：
1. 持仓是否需要卖出？（死叉、涨幅>15%、跌幅>7%止损）
2. 是否有符合策略的买入机会？（金叉、回调企稳）
3. 风险提示

注意：如果不需要操作，请回复"不操作"。
如果需要操作，请格式化为：
买入:股票代码,数量
或
卖出:股票代码,数量
"""
        return prompt

    def _parse_trade_signal(self, agent_response: str, session) -> dict[str, Any] | None:
        """Parse trade signal from agent response."""
        response = agent_response.strip()
        
        # Check for "no operation" signal
        if "不操作" in response or "无操作" in response or "观望" in response:
            return None
        
        # Parse buy signal
        if "买入:" in response:
            parts = response.split("买入:")[1].strip().split(",")
            if len(parts) >= 2:
                return {
                    "action": "buy",
                    "ticker": parts[0].strip(),
                    "quantity": int(parts[1].strip()),
                }
        
        # Parse sell signal
        if "卖出:" in response:
            parts = response.split("卖出:")[1].strip().split(",")
            if len(parts) >= 2:
                return {
                    "action": "sell",
                    "ticker": parts[0].strip(),
                    "quantity": int(parts[1].strip()),
                }
        
        return None
