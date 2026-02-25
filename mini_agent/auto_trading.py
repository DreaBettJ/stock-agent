"""Auto trading workflow for event-driven backtest/simulation."""

from __future__ import annotations

import logging
import os
import json
import re
import sqlite3
from datetime import datetime
from typing import Any

from .app.prompt_guard import ensure_trade_policy_prompt, resolve_strategy_risk_profile
from .agent import Agent
from .config import Config
from .llm import LLMClient
from .schema import LLMProvider
from .session import SessionManager
from .tools.kline_db_tool import KLineDB

logger = logging.getLogger(__name__)


class AutoTradingWorkflow:
    """Auto trading workflow: event -> agent analysis -> execute trade."""
    # 该工作流负责把“每日事件”转为“可执行交易动作”：
    # 1) 读取会话与行情/持仓上下文
    # 2) 组织 prompt 交给 LLM 分析
    # 3) 解析结构化买卖信号
    # 4) 可选调用交易执行器落库

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
        event_id: str | None = None,
        auto_execute: bool = False,
        trade_executor: Any | None = None,
    ) -> dict[str, Any]:
        """Trigger daily review for a session and optionally auto-execute trade."""
        # 1) 读取 session（策略文本、模式、资金状态等）。
        session = self.session_manager.get_session(session_id)

        # 2) 准备当日市场上下文（如涨幅榜）。
        requested_date = trading_date or datetime.now().strftime("%Y-%m-%d")
        effective_date = requested_date
        market_data = self._prepare_market_data(effective_date)
        if not market_data.get("top_gainers"):
            fallback_date = self._find_latest_market_date_on_or_before(requested_date)
            if fallback_date and fallback_date != requested_date:
                logger.info(
                    "Daily review fallback date: session=%s requested=%s effective=%s",
                    session_id,
                    requested_date,
                    fallback_date,
                )
                effective_date = fallback_date
                market_data = self._prepare_market_data(effective_date)
        market_data["technical_signals"] = await self._build_technical_signals_for_market(market_data, effective_date)
        pipeline = self._build_daily_review_pipeline(session_id=session_id, trading_date=effective_date, market_data=market_data)

        # 3) 准备该会话当前持仓（含估值与近端 K 线片段）。
        positions = self._get_positions(session_id, as_of_date=effective_date)

        strategy_prompt, prompt_missing = ensure_trade_policy_prompt(
            prompt=session.system_prompt,
            role="A股自动交易代理",
            session=session,
            positions=positions,
        )
        if prompt_missing:
            logger.warning("Daily review strategy prompt missing sections: %s", ",".join(prompt_missing))

        # 4) 组装提示词：
        # - system prompt: 会话级策略与执行约束
        # - user prompt: 当次事件上下文（持仓/市场/分层）
        system_prompt = self._build_daily_review_system_prompt(
            strategy_prompt=strategy_prompt,
            allow_tool_execution=bool(auto_execute and trade_executor is not None),
        )
        prompt = self._build_review_prompt(
            session=session,
            market_data=market_data,
            positions=positions,
            date=effective_date,
            pipeline=pipeline,
        )
        logger.info(
            "Daily review prepared: session=%s, date=%s, positions=%d, top_gainers=%d",
            session_id,
            effective_date,
            len(positions),
            len(market_data.get("top_gainers", [])),
        )
        logger.info("System prompt to LLM (session=%s, date=%s):\n%s", session_id, effective_date, system_prompt)
        logger.info("Event prompt to LLM (session=%s, date=%s):\n%s", session_id, effective_date, prompt)
        
        # 5) 创建仅分析模式的 Agent（tools=[]），避免在分析阶段直接调用工具。
        llm = LLMClient(
            provider=LLMProvider(self.llm_provider),
            model=self.model,
            api_key=self.api_key,
        )

        agent_tools = []
        if auto_execute and trade_executor is not None:
            agent_tools = [trade_executor]

        agent = Agent(
            llm_client=llm,
            system_prompt=system_prompt,
            tools=agent_tools,
            session_id=session_id,
        )
        
        # 把构造好的上下文作为用户输入交给 Agent。
        agent.add_user_message(prompt)

        # 执行一次完整推理，得到自然语言分析文本。
        result = await agent.run()
        logger.info("LLM analysis completed (session=%s, date=%s): %s", session_id, effective_date, result)

        # 6) 解析结构化决策；若开启 auto_execute 则调用外部执行器实际下单。
        decision = self._parse_trade_decision(result, effective_date)
        trade_actions = list(decision.get("actions") or [])
        trade_signal = trade_actions[0] if trade_actions else None
        tool_executions: list[dict[str, Any]] = []
        tool_success_actions: list[dict[str, Any]] = []
        execution = None
        execution_results: list[dict[str, Any]] = []

        # Prefer Agent-initiated tool execution. Fallback to parser-triggered execution only if no tool trade happened.
        for msg in agent.messages:
            if msg.role != "tool":
                continue
            content = str(msg.content or "")
            if "SIM_TRADE_OK" in content:
                tool_executions.append({"success": True, "content": content})
                parsed = self._parse_sim_trade_tool_content(content)
                side = str(parsed.get("action") or "").strip().lower()
                ticker = str(parsed.get("ticker") or "").strip()
                qty = int(parsed.get("quantity") or 0)
                if side in {"buy", "sell"} and ticker and qty > 0:
                    tool_success_actions.append(
                        {
                            "action": side,
                            "ticker": ticker,
                            "quantity": qty,
                            "trade_date": effective_date,
                            "reason": "agent_tool_call",
                        }
                    )
                self.session_manager.record_trade_action_log(
                    session_id=session_id,
                    event_id=event_id,
                    event_type="daily_review",
                    action=parsed.get("action"),
                    ticker=parsed.get("ticker"),
                    quantity=parsed.get("quantity"),
                    trade_date=effective_date,
                    status="succeeded",
                    error_code=None,
                    reason="agent_tool_call",
                    request_payload=parsed,
                    response_payload={"content": content},
                )
            elif str(msg.name or "").strip() == "simulate_trade" or content.strip().lower().startswith("error:"):
                tool_executions.append({"success": False, "error": content})
                parsed = self._parse_sim_trade_tool_content(content)
                self.session_manager.record_trade_action_log(
                    session_id=session_id,
                    event_id=event_id,
                    event_type="daily_review",
                    action=parsed.get("action"),
                    ticker=parsed.get("ticker"),
                    quantity=parsed.get("quantity"),
                    trade_date=effective_date,
                    status="failed",
                    error_code="agent_tool_error",
                    reason="agent_tool_call",
                    request_payload=parsed,
                    response_payload={"error": content},
                )

        if tool_executions:
            last_exec = tool_executions[-1]
            if last_exec.get("success"):
                class _Execution:
                    success = True
                    content = str(last_exec.get("content") or "")
                    error = None

                execution = _Execution()
            else:
                class _ExecutionFail:
                    success = False
                    content = ""
                    error = str(last_exec.get("error") or "trade execution failed")

                execution = _ExecutionFail()
        # If tool execution already succeeded but parser has empty signal
        # (e.g. run ended by max_steps fallback text), recover actions from tool logs.
        if tool_success_actions:
            if not trade_actions:
                trade_actions = tool_success_actions
                trade_signal = trade_actions[0]
            if str(result).strip().startswith("Task couldn't be completed"):
                parts = [
                    f"{x['action']} {x['ticker']} x{x['quantity']}"
                    for x in tool_success_actions
                ]
                result = f"已执行交易：{'; '.join(parts)}"
            if decision.get("decision") != "trade":
                decision["decision"] = "trade"
                decision["actions"] = trade_actions
                decision["summary"] = str(decision.get("summary") or "tool execution succeeded")
        elif auto_execute and trade_actions and trade_executor is not None:
            for action in trade_actions:
                side = str(action.get("action") or "").strip().lower()
                ticker = str(action.get("ticker") or "").strip()
                qty = int(action.get("quantity") or 0)
                if side not in {"buy", "sell"} or not ticker or qty <= 0:
                    execution_results.append(
                        {
                            "success": False,
                            "error": f"invalid action payload: {action}",
                            "action": action,
                        }
                    )
                    continue
                action_trade_date = str(action.get("trade_date") or effective_date)
                action_reason = str(action.get("reason") or f"daily_review:{effective_date}").strip()
                exec_one = await trade_executor.execute(
                    session_id=session_id,
                    action=side,
                    ticker=ticker,
                    quantity=qty,
                    trade_date=action_trade_date,
                    reason=action_reason,
                )
                self.session_manager.record_trade_action_log(
                    session_id=session_id,
                    event_id=event_id,
                    event_type="daily_review",
                    action=side,
                    ticker=ticker,
                    quantity=qty,
                    trade_date=action_trade_date,
                    status="succeeded" if exec_one.success else "failed",
                    error_code=None if exec_one.success else "execute_failed",
                    reason=action_reason,
                    request_payload=action,
                    response_payload={
                        "success": bool(exec_one.success),
                        "content": str(exec_one.content or ""),
                        "error": str(exec_one.error or ""),
                    },
                )
                execution_results.append(
                    {
                        "success": bool(exec_one.success),
                        "content": str(exec_one.content or ""),
                        "error": str(exec_one.error or ""),
                        "action": action,
                    }
                )
            if execution_results:
                last_ok = next((x for x in reversed(execution_results) if x.get("success")), None)
                last = last_ok or execution_results[-1]
                if last.get("success"):
                    class _Execution:
                        success = True
                        content = str(last.get("content") or "")
                        error = None

                    execution = _Execution()
                else:
                    class _ExecutionFail:
                        success = False
                        content = ""
                        error = str(last.get("error") or "trade execution failed")

                    execution = _ExecutionFail()
        logger.info(
            "Daily review parsed signal: session=%s, date=%s, signal=%s, executed=%s",
            session_id,
            effective_date,
            trade_actions,
            bool(execution and execution.success),
        )
        
        return {
            "session_id": session_id,
            "event_id": event_id,
            "requested_date": requested_date,
            "date": effective_date,
            "agent_analysis": result,
            "daily_pipeline": pipeline,
            "decision": decision,
            "trade_actions": trade_actions,
            "trade_signal": trade_signal,
            "tool_executions": tool_executions,
            "execution_results": execution_results,
            "execution": execution.content if execution and execution.success else None,
            "execution_error": execution.error if execution and not execution.success else None,
        }

    def _find_latest_market_date_on_or_before(self, date: str) -> str | None:
        """Find latest date in local market DB on or before the given date."""
        import sqlite3

        with sqlite3.connect(self.kline_db.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(date) AS date FROM daily_kline WHERE date <= ?",
                (date,),
            ).fetchone()
        if not row:
            return None
        latest = row[0]
        return str(latest) if latest else None

    def _prepare_market_data(self, date: str) -> dict[str, Any]:
        """Prepare market data for the day."""
        # 容错处理：行情读取失败时返回空榜单，不中断整个决策链路。
        try:
            top_gainers = self._get_top_gainers(date)
        except Exception:
            top_gainers = []
        
        # 当前仅使用涨幅榜作为市场快照，可在此处继续扩展更多指标。
        market_summary = {
            "date": date,
            "top_gainers": top_gainers[:10],
            "total_stocks": len(top_gainers),
        }
        
        return market_summary

    async def _build_technical_signals_for_market(self, market_data: dict[str, Any], trading_date: str) -> list[dict[str, Any]]:
        """Build compact technical signals for top gainers from local K-line DB."""
        top_gainers = market_data.get("top_gainers") or []
        if not top_gainers:
            return []

        # Guard switch for fast local tests or offline mode.
        if os.getenv("MINI_AGENT_DISABLE_TECH_SIGNALS", "").strip() in {"1", "true", "TRUE"}:
            return []

        results: list[dict[str, Any]] = []
        for item in top_gainers[:5]:
            ticker = str(item.get("ticker") or "").strip()
            if not ticker:
                continue
            try:
                signal = self._build_technical_signal_from_kline(
                    ticker=ticker,
                    trading_date=trading_date,
                    window=120,
                )
            except Exception:
                continue

            if signal.get("signal_status") != "ok":
                continue

            results.append(
                {
                    "ticker": ticker,
                    "trend": signal.get("trend"),
                    "score": signal.get("score"),
                    "signals": signal.get("signals", {}),
                    "levels": signal.get("levels", {}),
                }
            )

        return results

    def _build_technical_signal_from_kline(self, ticker: str, trading_date: str, window: int = 120) -> dict[str, Any]:
        import pandas as pd  # type: ignore

        end_dt = datetime.fromisoformat(trading_date)
        start_dt = end_dt.replace(year=max(1991, end_dt.year - 2))
        rows = self.kline_db.get_kline(ticker, start_dt.strftime("%Y-%m-%d"), trading_date)
        if not rows:
            return {"ticker": ticker, "signal_status": "no_data"}

        df = pd.DataFrame(rows)
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        if len(df) < 35:
            return {"ticker": ticker, "signal_status": "insufficient_data"}

        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close

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

        trend = "sideways"
        if latest_close >= latest_ma20 >= latest_ma60:
            trend = "up"
        elif latest_close <= latest_ma20 <= latest_ma60:
            trend = "down"

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
            "signal_status": "ok",
            "window": int(window),
            "trend": trend,
            "score": score,
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
        }

    def _scan_market_movers(self, trading_date: str, limit: int = 300) -> list[dict[str, Any]]:
        """Scan daily movers and anomalies from K-line DB."""
        import sqlite3

        with sqlite3.connect(self.kline_db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT ticker, open, high, low, close, volume, amount,
                       (close - open) / open * 100 AS change_pct,
                       CASE WHEN open > 0 THEN (high - low) / open * 100 ELSE 0 END AS amplitude_pct
                FROM daily_kline
                WHERE date = ? AND open IS NOT NULL AND close IS NOT NULL
                ORDER BY ABS((close - open) / open) DESC, amount DESC
                LIMIT ?
                """,
                (trading_date, limit),
            ).fetchall()

        items: list[dict[str, Any]] = []
        for row in rows:
            change_pct = float(row["change_pct"] or 0.0)
            amplitude_pct = float(row["amplitude_pct"] or 0.0)
            amount = float(row["amount"] or 0.0)
            tags: list[str] = []
            if change_pct >= 5:
                tags.append("strong_up")
            if change_pct <= -5:
                tags.append("strong_down")
            if amplitude_pct >= 8:
                tags.append("high_volatility")
            if amount >= 1e9:
                tags.append("high_turnover")
            if not tags:
                tags.append("normal_move")

            items.append(
                {
                    "ticker": str(row["ticker"]),
                    "change_pct": round(change_pct, 2),
                    "amplitude_pct": round(amplitude_pct, 2),
                    "amount": amount,
                    "tags": tags,
                }
            )
        return items

    @staticmethod
    def _classify_ticker_group(ticker: str) -> str:
        code = str(ticker).strip()
        if code.startswith("688"):
            return "kechuang"
        if code.startswith(("300", "301")):
            return "chinext"
        if code.startswith(("600", "601", "603", "605")):
            return "sse_main"
        if code.startswith(("000", "001", "002", "003")):
            return "szse_main"
        return "other"

    def _classify_movers(self, movers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Cluster movers into coarse market groups (style/board buckets)."""
        buckets: dict[str, list[dict[str, Any]]] = {}
        for item in movers:
            group = self._classify_ticker_group(str(item.get("ticker") or ""))
            buckets.setdefault(group, []).append(item)

        clusters: list[dict[str, Any]] = []
        for group, items in buckets.items():
            if not items:
                continue
            avg_change = sum(float(x.get("change_pct", 0.0)) for x in items) / len(items)
            avg_amount = sum(float(x.get("amount", 0.0)) for x in items) / len(items)
            clusters.append(
                {
                    "group": group,
                    "count": len(items),
                    "avg_change_pct": round(avg_change, 2),
                    "avg_amount": round(avg_amount, 2),
                    "members": items[:20],
                }
            )
        clusters.sort(key=lambda x: (x["avg_change_pct"], x["count"], x["avg_amount"]), reverse=True)
        return clusters

    def _find_mainlines(self, clusters: list[dict[str, Any]], top_n: int = 3) -> list[dict[str, Any]]:
        """Pick main themes from clustered movers."""
        result: list[dict[str, Any]] = []
        for item in clusters[:top_n]:
            strength = float(item.get("avg_change_pct", 0.0)) * 2 + float(item.get("count", 0.0)) * 0.3
            result.append(
                {
                    "group": item.get("group"),
                    "strength": round(strength, 2),
                    "count": item.get("count"),
                    "avg_change_pct": item.get("avg_change_pct"),
                }
            )
        return result

    def _select_leaders(self, movers: list[dict[str, Any]], mainlines: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
        """Select leader candidates from mainline groups."""
        groups = {str(x.get("group")) for x in mainlines}
        candidates = []
        for item in movers:
            ticker = str(item.get("ticker") or "")
            group = self._classify_ticker_group(ticker)
            if group not in groups:
                continue
            score = float(item.get("change_pct", 0.0)) * 0.7 + float(item.get("amplitude_pct", 0.0)) * 0.3
            candidates.append(
                {
                    "ticker": ticker,
                    "group": group,
                    "change_pct": item.get("change_pct"),
                    "amplitude_pct": item.get("amplitude_pct"),
                    "leader_score": round(score, 2),
                }
            )
        candidates.sort(key=lambda x: x["leader_score"], reverse=True)
        return candidates[:top_n]

    def _analyze_positions(self, positions: list[dict[str, Any]], leaders: list[dict[str, Any]], mainlines: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze position health against mainline and leaders."""
        if not positions:
            return {
                "position_count": 0,
                "on_mainline": [],
                "off_mainline": [],
                "leader_overlap": [],
                "summary": "no_positions",
            }

        main_groups = {str(x.get("group")) for x in mainlines}
        leader_tickers = {str(x.get("ticker")) for x in leaders}
        on_mainline: list[str] = []
        off_mainline: list[str] = []
        overlap: list[str] = []

        for p in positions:
            ticker = str(p.get("ticker") or "")
            group = self._classify_ticker_group(ticker)
            if group in main_groups:
                on_mainline.append(ticker)
            else:
                off_mainline.append(ticker)
            if ticker in leader_tickers:
                overlap.append(ticker)

        return {
            "position_count": len(positions),
            "on_mainline": on_mainline,
            "off_mainline": off_mainline,
            "leader_overlap": overlap,
            "summary": "ok" if len(off_mainline) <= len(on_mainline) else "needs_rebalance",
        }

    def _build_rebalance_proposal(self, position_diagnostics: dict[str, Any], leaders: list[dict[str, Any]]) -> dict[str, Any]:
        """Build conservative rebalance proposal."""
        off_mainline = list(position_diagnostics.get("off_mainline") or [])
        leader_overlap = list(position_diagnostics.get("leader_overlap") or [])
        leaders_top = [str(x.get("ticker")) for x in leaders[:3]]

        actions: list[dict[str, Any]] = []
        if off_mainline:
            for ticker in off_mainline[:3]:
                actions.append(
                    {
                        "action": "trim_or_exit",
                        "ticker": ticker,
                        "reason": "off_mainline",
                        "size_hint": "reduce 30%-50% first",
                    }
                )
        if not leader_overlap and leaders_top:
            actions.append(
                {
                    "action": "consider_rotate",
                    "tickers": leaders_top,
                    "reason": "mainline_leaders",
                    "size_hint": "small trial position, staged entry",
                }
            )
        if not actions:
            actions.append({"action": "hold", "reason": "position_aligned_or_no_clear_edge"})

        return {"actions": actions}

    def _build_daily_review_pipeline(self, session_id: int | str, trading_date: str, market_data: dict[str, Any]) -> dict[str, Any]:
        """Build staged daily-review pipeline snapshot."""
        session = self.session_manager.get_session(session_id)
        movers = self._scan_market_movers(trading_date)
        clusters = self._classify_movers(movers)
        mainlines = self._find_mainlines(clusters)
        leaders = self._select_leaders(movers, mainlines)
        positions = self._get_positions(session_id, as_of_date=trading_date)
        position_diag = self._analyze_positions(positions, leaders, mainlines)
        position_risk = self._build_position_risk_snapshot(
            session_id=session_id,
            positions=positions,
            trading_date=trading_date,
        )
        rebalance = self._build_rebalance_proposal(position_diag, leaders)
        full_market_scan = self._build_full_market_scan_snapshot(trading_date)
        market_breadth = self._build_market_breadth_snapshot(trading_date)
        liquidity = self._build_liquidity_snapshot(trading_date)
        rotation = self._build_rotation_snapshot(trading_date)
        candidate_watchlists = self._build_candidate_watchlists(
            trading_date,
            mainlines,
            strategy_prompt=getattr(session, "system_prompt", ""),
        )
        rule_engine = self._build_rule_engine_snapshot(
            session_id=session_id,
            market_data=market_data,
            position_diagnostics=position_diag,
            trading_date=trading_date,
        )
        return {
            "scan_movers": movers[:50],
            "classify_clusters": clusters[:8],
            "mainlines": mainlines,
            "leaders": leaders,
            "position_diagnostics": position_diag,
            "position_risk": position_risk,
            "rebalance_proposal": rebalance,
            "full_market_scan": full_market_scan,
            "market_breadth": market_breadth,
            "liquidity": liquidity,
            "rotation": rotation,
            "candidate_watchlists": candidate_watchlists,
            "rule_engine": rule_engine,
            "technical_signals": market_data.get("technical_signals") or [],
        }

    def _build_market_breadth_snapshot(self, trading_date: str) -> dict[str, Any]:
        """Build market breadth snapshot from local daily_kline."""
        with sqlite3.connect(self.kline_db.db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(1) AS total,
                    SUM(CASE WHEN close > open THEN 1 ELSE 0 END) AS up_count,
                    SUM(CASE WHEN close < open THEN 1 ELSE 0 END) AS down_count,
                    SUM(CASE WHEN close = open THEN 1 ELSE 0 END) AS flat_count,
                    SUM(CASE WHEN open > 0 AND (close - open) / open * 100 >= 9.5 THEN 1 ELSE 0 END) AS limit_up_like,
                    SUM(CASE WHEN open > 0 AND (close - open) / open * 100 <= -9.5 THEN 1 ELSE 0 END) AS limit_down_like,
                    SUM(CASE WHEN open > 0 AND (high - low) / open * 100 >= 8 THEN 1 ELSE 0 END) AS high_vol_count
                FROM daily_kline
                WHERE date = ? AND open IS NOT NULL AND close IS NOT NULL
                """,
                (trading_date,),
            ).fetchone()
        if not row:
            return {"status": "no_data"}
        total = int(row[0] or 0)
        up_count = int(row[1] or 0)
        down_count = int(row[2] or 0)
        flat_count = int(row[3] or 0)
        return {
            "status": "ok",
            "total": total,
            "up_count": up_count,
            "down_count": down_count,
            "flat_count": flat_count,
            "up_down_ratio": round(up_count / max(down_count, 1), 3),
            "limit_up_like": int(row[4] or 0),
            "limit_down_like": int(row[5] or 0),
            "high_vol_count": int(row[6] or 0),
        }

    def _build_liquidity_snapshot(self, trading_date: str) -> dict[str, Any]:
        """Build turnover snapshot and compare to recent average."""
        with sqlite3.connect(self.kline_db.db_path) as conn:
            row_today = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM daily_kline WHERE date = ?",
                (trading_date,),
            ).fetchone()
            rows_hist = conn.execute(
                """
                SELECT date, COALESCE(SUM(amount), 0) AS total_amount
                FROM daily_kline
                WHERE date < ?
                GROUP BY date
                ORDER BY date DESC
                LIMIT 5
                """,
                (trading_date,),
            ).fetchall()
        total_today = float((row_today[0] if row_today else 0.0) or 0.0)
        hist_amounts = [float(r[1] or 0.0) for r in rows_hist]
        avg_5 = (sum(hist_amounts) / len(hist_amounts)) if hist_amounts else 0.0
        delta_pct = ((total_today - avg_5) / avg_5 * 100.0) if avg_5 > 0 else 0.0
        return {
            "status": "ok",
            "total_amount": round(total_today, 2),
            "avg_amount_5d": round(avg_5, 2),
            "delta_vs_5d_pct": round(delta_pct, 2),
            "history_days_used": len(hist_amounts),
        }

    def _build_rotation_snapshot(self, trading_date: str) -> dict[str, Any]:
        """Track simple group strength rotation over recent trading days."""
        with sqlite3.connect(self.kline_db.db_path) as conn:
            day_rows = conn.execute(
                """
                SELECT DISTINCT date
                FROM daily_kline
                WHERE date <= ?
                ORDER BY date DESC
                LIMIT 3
                """,
                (trading_date,),
            ).fetchall()
        days = [str(r[0]) for r in day_rows if r and r[0]]
        if not days:
            return {"status": "no_data", "groups": []}

        groups = ("sse_main", "szse_main", "chinext", "kechuang", "other")
        series: list[dict[str, Any]] = []
        for group in groups:
            vals: list[float] = []
            for day in reversed(days):
                movers = self._scan_market_movers(day, limit=300)
                items = [x for x in movers if self._classify_ticker_group(str(x.get("ticker") or "")) == group]
                if not items:
                    vals.append(0.0)
                    continue
                avg_change = sum(float(x.get("change_pct") or 0.0) for x in items) / len(items)
                vals.append(round(avg_change, 2))
            slope = vals[-1] - vals[0] if len(vals) >= 2 else 0.0
            series.append(
                {
                    "group": group,
                    "changes": vals,
                    "slope": round(slope, 2),
                }
            )
        series.sort(key=lambda x: x["slope"], reverse=True)
        return {
            "status": "ok",
            "days": list(reversed(days)),
            "groups": series,
        }

    def _build_candidate_watchlists(
        self,
        trading_date: str,
        mainlines: list[dict[str, Any]],
        strategy_prompt: str = "",
    ) -> dict[str, Any]:
        """Build pullback/breakout watchlists from liquid universe."""
        strategy_profile = resolve_strategy_risk_profile(strategy_prompt)
        chase_limit_pct = float(strategy_profile.get("chase_limit_pct") or 10.0)
        breakout_max_change = max(3.0, chase_limit_pct - 2.0)
        with sqlite3.connect(self.kline_db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT ticker, open, high, low, close, amount,
                       CASE WHEN open > 0 THEN (close - open) / open * 100 ELSE 0 END AS change_pct
                FROM daily_kline
                WHERE date = ? AND open IS NOT NULL AND close IS NOT NULL
                ORDER BY amount DESC
                LIMIT 60
                """,
                (trading_date,),
            ).fetchall()

        main_groups = {str(x.get("group") or "") for x in mainlines}
        breakout: list[dict[str, Any]] = []
        pullback: list[dict[str, Any]] = []
        avoid: list[dict[str, Any]] = []

        for row in rows:
            ticker = str(row["ticker"] or "").strip()
            if not ticker:
                continue
            try:
                sig = self._build_technical_signal_from_kline(ticker=ticker, trading_date=trading_date, window=120)
            except Exception:
                continue
            if sig.get("signal_status") != "ok":
                continue
            signals = sig.get("signals") or {}
            trend = str(sig.get("trend") or "")
            score = int(sig.get("score") or 0)
            change = float(row["change_pct"] or 0.0)
            group = self._classify_ticker_group(ticker)
            item = {
                "ticker": ticker,
                "group": group,
                "change_pct": round(change, 2),
                "score": score,
                "trend": trend,
                "rsi": str(signals.get("rsi") or ""),
                "macd": str(signals.get("macd") or ""),
                "ma": str(signals.get("ma_cross") or ""),
            }

            if change >= chase_limit_pct:
                avoid.append({**item, "reason": "chase_limit"})
                continue
            if trend == "down" or str(signals.get("ma_cross") or "") == "death_cross":
                avoid.append({**item, "reason": "trend_weak"})
                continue

            if (
                group in main_groups
                and trend == "up"
                and score >= 60
                and str(signals.get("macd") or "") == "bullish"
                and str(signals.get("rsi") or "") != "overbought"
                and 1.0 <= change <= breakout_max_change
                and len(breakout) < 8
            ):
                breakout.append(item)
                continue

            if (
                group in main_groups
                and trend in {"up", "sideways"}
                and score >= 55
                and str(signals.get("rsi") or "") in {"neutral", "oversold"}
                and -4.0 <= change <= 2.0
                and len(pullback) < 8
            ):
                pullback.append(item)

        return {
            "status": "ok",
            "policy": {
                "strategy_template": strategy_profile.get("strategy_template"),
                "chase_limit_pct": chase_limit_pct,
                "breakout_max_change": round(breakout_max_change, 2),
            },
            "pullback_setup": pullback,
            "breakout_setup": breakout,
            "avoid_chase": avoid[:10],
        }

    def _build_position_risk_snapshot(
        self,
        session_id: int | str,
        positions: list[dict[str, Any]],
        trading_date: str,
    ) -> dict[str, Any]:
        """Build deterministic position risk diagnostics tied to held positions."""
        if not positions:
            return {"status": "ok", "items": [], "actions": []}

        session = self.session_manager.get_session(session_id)
        stop_loss_pct = float(getattr(session, "stop_loss_pct", 7.0) or 7.0)
        take_profit_pct = float(getattr(session, "take_profit_pct", 15.0) or 15.0)

        items: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        for p in positions:
            ticker = str(p.get("ticker") or "").strip()
            qty = int(p.get("quantity") or 0)
            profit_rate = float(p.get("profit_rate") or 0.0)
            if not ticker or qty <= 0:
                continue

            trend = "unknown"
            score = 0
            ma_cross = "unknown"
            macd = "unknown"
            risk_tags: list[str] = []
            try:
                sig = self._build_technical_signal_from_kline(
                    ticker=ticker,
                    trading_date=trading_date,
                    window=120,
                )
                if sig.get("signal_status") == "ok":
                    trend = str(sig.get("trend") or "unknown")
                    score = int(sig.get("score") or 0)
                    signals = sig.get("signals") or {}
                    ma_cross = str(signals.get("ma_cross") or "unknown")
                    macd = str(signals.get("macd") or "unknown")
            except Exception:
                pass

            if profit_rate <= -stop_loss_pct:
                risk_tags.append("hard_stop_loss")
            if profit_rate >= take_profit_pct:
                risk_tags.append("take_profit_zone")
            if ma_cross == "death_cross":
                risk_tags.append("death_cross")
            if trend == "down":
                risk_tags.append("trend_down")
            if macd == "bearish":
                risk_tags.append("macd_bearish")

            if "hard_stop_loss" in risk_tags:
                actions.append(
                    {
                        "action": "sell",
                        "ticker": ticker,
                        "quantity": qty,
                        "trade_date": trading_date,
                        "reason": "position_risk:hard_stop_loss",
                    }
                )
            elif "take_profit_zone" in risk_tags:
                actions.append(
                    {
                        "action": "sell",
                        "ticker": ticker,
                        "quantity": max(100, (qty // 2 // 100) * 100),
                        "trade_date": trading_date,
                        "reason": "position_risk:take_profit_partial",
                    }
                )
            elif ("death_cross" in risk_tags or "trend_down" in risk_tags) and qty >= 100:
                actions.append(
                    {
                        "action": "sell",
                        "ticker": ticker,
                        "quantity": max(100, (qty // 3 // 100) * 100),
                        "trade_date": trading_date,
                        "reason": "position_risk:trend_weaken_trim",
                    }
                )

            items.append(
                {
                    "ticker": ticker,
                    "quantity": qty,
                    "profit_rate": round(profit_rate, 2),
                    "trend": trend,
                    "score": score,
                    "ma_cross": ma_cross,
                    "macd": macd,
                    "risk_tags": risk_tags,
                }
            )

        return {
            "status": "ok",
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "items": items,
            "actions": actions[:5],
        }

    def _build_full_market_scan_snapshot(self, trading_date: str) -> dict[str, Any]:
        """Build full-universe market scan summary for daily event context."""
        with sqlite3.connect(self.kline_db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            totals = conn.execute(
                """
                SELECT
                    COUNT(1) AS total,
                    AVG(CASE WHEN open > 0 THEN (close - open) / open * 100 END) AS mean_change,
                    MIN(CASE WHEN open > 0 THEN (close - open) / open * 100 END) AS min_change,
                    MAX(CASE WHEN open > 0 THEN (close - open) / open * 100 END) AS max_change,
                    AVG(CASE WHEN open > 0 THEN (high - low) / open * 100 END) AS mean_amplitude,
                    COALESCE(SUM(amount), 0) AS total_amount
                FROM daily_kline
                WHERE date = ? AND open IS NOT NULL AND close IS NOT NULL
                """,
                (trading_date,),
            ).fetchone()
            top_up_rows = conn.execute(
                """
                SELECT ticker,
                       CASE WHEN open > 0 THEN (close - open) / open * 100 ELSE 0 END AS change_pct,
                       amount
                FROM daily_kline
                WHERE date = ? AND open IS NOT NULL AND close IS NOT NULL
                ORDER BY change_pct DESC, amount DESC
                LIMIT 12
                """,
                (trading_date,),
            ).fetchall()
            top_down_rows = conn.execute(
                """
                SELECT ticker,
                       CASE WHEN open > 0 THEN (close - open) / open * 100 ELSE 0 END AS change_pct,
                       amount
                FROM daily_kline
                WHERE date = ? AND open IS NOT NULL AND close IS NOT NULL
                ORDER BY change_pct ASC, amount DESC
                LIMIT 12
                """,
                (trading_date,),
            ).fetchall()
            turnover_rows = conn.execute(
                """
                SELECT ticker,
                       CASE WHEN open > 0 THEN (close - open) / open * 100 ELSE 0 END AS change_pct,
                       amount
                FROM daily_kline
                WHERE date = ? AND open IS NOT NULL AND close IS NOT NULL
                ORDER BY amount DESC
                LIMIT 12
                """,
                (trading_date,),
            ).fetchall()

        if not totals or int(totals["total"] or 0) == 0:
            return {"status": "no_data"}

        def _pack(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
            return [
                {
                    "ticker": str(r["ticker"]),
                    "change_pct": round(float(r["change_pct"] or 0.0), 2),
                    "amount": round(float(r["amount"] or 0.0), 2),
                }
                for r in rows
            ]

        return {
            "status": "ok",
            "total_stocks": int(totals["total"] or 0),
            "mean_change_pct": round(float(totals["mean_change"] or 0.0), 3),
            "min_change_pct": round(float(totals["min_change"] or 0.0), 3),
            "max_change_pct": round(float(totals["max_change"] or 0.0), 3),
            "mean_amplitude_pct": round(float(totals["mean_amplitude"] or 0.0), 3),
            "total_amount": round(float(totals["total_amount"] or 0.0), 2),
            "top_up": _pack(top_up_rows),
            "top_down": _pack(top_down_rows),
            "top_turnover": _pack(turnover_rows),
        }

    def _build_rule_engine_snapshot(
        self,
        session_id: int | str,
        market_data: dict[str, Any],
        position_diagnostics: dict[str, Any],
        trading_date: str,
    ) -> dict[str, Any]:
        """Build deterministic rule-engine snapshot to constrain LLM freedom."""
        session = self.session_manager.get_session(session_id)
        top = list(market_data.get("top_gainers") or [])
        tech = {str(x.get("ticker") or ""): x for x in (market_data.get("technical_signals") or [])}
        if not top:
            return {"status": "no_data", "veto_reasons": ["no_market_snapshot"], "actions": []}

        cap_pct = float(getattr(session, "single_position_cap_pct", 15.0) or 15.0) / 100.0
        budget = max(float(getattr(session, "current_cash", 0.0) or 0.0) * cap_pct, 0.0)
        strategy_profile = resolve_strategy_risk_profile(getattr(session, "system_prompt", ""))
        chase_limit_pct = float(strategy_profile.get("chase_limit_pct") or 10.0)
        min_score = int(strategy_profile.get("min_score") or 55)
        allowed_trends = set(strategy_profile.get("allowed_trends") or ["up", "sideways"])
        veto_reasons: list[str] = []
        actions: list[dict[str, Any]] = []

        for row in top[:10]:
            ticker = str(row.get("ticker") or "").strip()
            if not ticker:
                continue
            change = float(row.get("change_pct") or 0.0)
            close = float(row.get("close") or 0.0)
            sig = tech.get(ticker) or {}
            trend = str(sig.get("trend") or "")
            score = int(sig.get("score") or 0)
            # Hard vetoes.
            if change >= chase_limit_pct:
                veto_reasons.append(f"{ticker}:chase_limit")
                continue
            if trend not in allowed_trends:
                veto_reasons.append(f"{ticker}:trend_weak")
                continue
            if score < min_score:
                veto_reasons.append(f"{ticker}:score_low")
                continue
            if close <= 0 or budget <= 0:
                continue
            qty = int(budget / close / 100) * 100
            if qty <= 0:
                continue
            actions.append(
                {
                    "action": "buy",
                    "ticker": ticker,
                    "quantity": qty,
                    "trade_date": trading_date,
                    "reason": "rule_engine_candidate",
                    "budget": round(budget, 2),
                    "close": close,
                }
            )
            if len(actions) >= 3:
                break

        return {
            "status": "ok",
            "policy": {
                "strategy_template": strategy_profile.get("strategy_template"),
                "chase_limit_pct": chase_limit_pct,
                "min_score": min_score,
                "allowed_trends": sorted(list(allowed_trends)),
            },
            "veto_reasons": veto_reasons[:20],
            "position_summary": position_diagnostics.get("summary"),
            "actions": actions,
        }

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
                # 首次运行、无成交记录时，sim_positions 表可能尚未创建。
                return []

        positions = []
        for r in rows:
            # 估值口径：若指定 as_of_date 则按“该日及之前”价格，避免前视偏差。
            try:
                if as_of_date:
                    current_price = self.kline_db.get_price_on_or_before(r["ticker"], as_of_date)
                else:
                    current_price = self.kline_db.get_latest_price(r["ticker"])
            except Exception:
                current_price = r["avg_cost"]
            
            profit = (current_price - r["avg_cost"]) * r["quantity"]
            profit_rate = (current_price - r["avg_cost"]) / r["avg_cost"] * 100
            
            # 取近窗 K 线供策略判断（趋势、均线等），默认截断到最近 20 根。
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

    def _build_daily_review_system_prompt(self, strategy_prompt: str, allow_tool_execution: bool = False) -> str:
        """Build daily review system prompt (session-level policy only)."""
        execution_rule = ""
        if allow_tool_execution:
            execution_rule = """
【执行约束】
- 你可以直接调用工具 `simulate_trade` 执行模拟交易。
- 如果你判断需要调仓，优先调用工具完成执行，不要只输出“买入/卖出文本信号”。
- 调用完成后，用简短中文总结执行结果与原因。
"""
        return f"{strategy_prompt}\n{execution_rule}".strip()

    def _build_review_prompt(
        self,
        session,
        market_data: dict,
        positions: list[dict],
        date: str,
        pipeline: dict[str, Any],
        current_time: str | None = None,
    ) -> str:
        """Build daily review user prompt (event context only)."""
        now_text = str(current_time or datetime.now().isoformat())
        
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

        # Format technical signals (for top gainers)
        technical_text = "无数据"
        technical_signals = market_data.get("technical_signals") or []
        if technical_signals:
            tech_lines = []
            for s in technical_signals:
                sig = s.get("signals") or {}
                tech_lines.append(
                    f"- {s.get('ticker')}: trend={s.get('trend')}, score={s.get('score')}, "
                    f"ma={sig.get('ma_cross')}, macd={sig.get('macd')}, rsi={sig.get('rsi')}, breakout={sig.get('breakout')}"
                )
            technical_text = "\n".join(tech_lines)

        # Format pipeline stages
        mainlines = pipeline.get("mainlines") or []
        leaders = pipeline.get("leaders") or []
        position_diag = pipeline.get("position_diagnostics") or {}
        rebalance = pipeline.get("rebalance_proposal") or {}
        rule_engine = pipeline.get("rule_engine") or {}
        position_risk = pipeline.get("position_risk") or {}
        full_market_scan = pipeline.get("full_market_scan") or {}
        market_breadth = pipeline.get("market_breadth") or {}
        liquidity = pipeline.get("liquidity") or {}
        rotation = pipeline.get("rotation") or {}
        watchlists = pipeline.get("candidate_watchlists") or {}

        mainline_text = "无明显主线"
        if mainlines:
            mainline_text = "\n".join(
                f"- {x.get('group')}: strength={x.get('strength')}, count={x.get('count')}, avg_change={x.get('avg_change_pct')}%"
                for x in mainlines
            )

        leader_text = "无明显龙头"
        if leaders:
            leader_text = "\n".join(
                f"- {x.get('ticker')} ({x.get('group')}): score={x.get('leader_score')}, change={x.get('change_pct')}%"
                for x in leaders
            )

        pos_text = (
            f"on_mainline={position_diag.get('on_mainline', [])}, "
            f"off_mainline={position_diag.get('off_mainline', [])}, "
            f"leader_overlap={position_diag.get('leader_overlap', [])}, "
            f"summary={position_diag.get('summary')}"
        )
        rebalance_text = json.dumps(rebalance, ensure_ascii=False)
        rule_engine_text = json.dumps(rule_engine, ensure_ascii=False)
        position_risk_text = json.dumps(position_risk, ensure_ascii=False)
        full_market_scan_text = json.dumps(full_market_scan, ensure_ascii=False)
        breadth_text = json.dumps(market_breadth, ensure_ascii=False)
        liquidity_text = json.dumps(liquidity, ensure_ascii=False)
        rotation_text = json.dumps(rotation, ensure_ascii=False)
        watchlist_text = json.dumps(watchlists, ensure_ascii=False)
        
        prompt = f"""【系统当前时间】{now_text}
【事件交易日】{date}
【事件】每日复盘

【当前持仓】
{positions_text}

【今日涨幅榜】
{gainers_text}

【技术信号摘要（候选）】
{technical_text}

【市场分层结果】
主线：
{mainline_text}
龙头：
{leader_text}
持仓诊断：
{pos_text}
调仓建议草案：
{rebalance_text}
规则引擎建议（硬约束预筛）：
{rule_engine_text}
持仓风控快照（止损/止盈/趋势）：
{position_risk_text}
全市场扫描快照（全A样本）：
{full_market_scan_text}
市场广度快照：
{breadth_text}
流动性快照：
{liquidity_text}
轮动快照（近3日）：
{rotation_text}
候选观察池（回调/突破/避开）：
{watchlist_text}

请根据你的策略分析：
1. 持仓是否需要卖出？（死叉、涨幅>15%、跌幅>7%止损）
2. 是否有符合策略的买入机会？（金叉、回调企稳）
3. 风险提示

请严格输出 JSON（不要输出 Markdown 表格）：
{{
  "decision": "trade" | "hold",
  "summary": "简短结论",
  "risk_warnings": ["风险1", "风险2"],
  "actions": [
    {{
      "action": "buy" | "sell",
      "ticker": "600519.SH",
      "quantity": 100,
      "trade_date": "{date}",
      "reason": "执行原因"
    }}
  ]
}}

要求：
- 无操作时：decision=hold, actions=[]
- 有操作时：decision=trade, actions 至少 1 条
- 所有 action 必须可直接落到 simulate_trade 参数
兼容：若无法输出 JSON，可退化输出“买入:代码,数量”或“不操作”。
"""
        return prompt

    def _parse_trade_decision(self, agent_response: str, default_trade_date: str) -> dict[str, Any]:
        """Parse structured decision JSON, fallback to legacy text protocol."""
        response = (agent_response or "").strip()
        if not response:
            return {"decision": "hold", "actions": [], "summary": "", "risk_warnings": []}

        json_payload = self._extract_json_payload(response)
        if json_payload is not None:
            actions_raw = json_payload.get("actions") if isinstance(json_payload, dict) else []
            actions = self._normalize_actions(actions_raw, default_trade_date)
            decision = str(json_payload.get("decision") or "").strip().lower() if isinstance(json_payload, dict) else ""
            if decision not in {"trade", "hold"}:
                decision = "trade" if actions else "hold"
            return {
                "decision": decision,
                "summary": str(json_payload.get("summary") or "") if isinstance(json_payload, dict) else "",
                "risk_warnings": list(json_payload.get("risk_warnings") or []) if isinstance(json_payload, dict) else [],
                "actions": actions,
            }

        # Legacy fallback.
        if "不操作" in response or "无操作" in response or "观望" in response:
            return {"decision": "hold", "summary": response, "risk_warnings": [], "actions": []}

        actions: list[dict[str, Any]] = []
        if "买入:" in response:
            parts = response.split("买入:")[1].strip().split(",")
            if len(parts) >= 2:
                try:
                    actions.append(
                        {
                            "action": "buy",
                            "ticker": parts[0].strip(),
                            "quantity": int(parts[1].strip()),
                            "trade_date": default_trade_date,
                            "reason": "legacy_text_signal",
                        }
                    )
                except Exception:
                    pass
        if "卖出:" in response:
            parts = response.split("卖出:")[1].strip().split(",")
            if len(parts) >= 2:
                try:
                    actions.append(
                        {
                            "action": "sell",
                            "ticker": parts[0].strip(),
                            "quantity": int(parts[1].strip()),
                            "trade_date": default_trade_date,
                            "reason": "legacy_text_signal",
                        }
                    )
                except Exception:
                    pass
        return {
            "decision": "trade" if actions else "hold",
            "summary": response[:400],
            "risk_warnings": [],
            "actions": actions,
        }

    def _extract_json_payload(self, text: str) -> dict[str, Any] | None:
        fenced = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        candidates: list[str] = []
        if fenced:
            candidates.append(fenced.group(1))
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidates.append(text[first : last + 1])
        for raw in candidates:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None

    def _normalize_actions(self, actions_raw: Any, default_trade_date: str) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        if not isinstance(actions_raw, list):
            return actions
        for item in actions_raw:
            if not isinstance(item, dict):
                continue
            side = str(item.get("action") or "").strip().lower()
            ticker = str(item.get("ticker") or "").strip()
            try:
                qty = int(item.get("quantity"))
            except Exception:
                qty = 0
            if side not in {"buy", "sell"} or not ticker or qty <= 0:
                continue
            actions.append(
                {
                    "action": side,
                    "ticker": ticker,
                    "quantity": qty,
                    "trade_date": str(item.get("trade_date") or default_trade_date),
                    "reason": str(item.get("reason") or "llm_decision").strip(),
                }
            )
        return actions

    def _parse_sim_trade_tool_content(self, text: str) -> dict[str, Any]:
        pairs = dict(re.findall(r"([a-zA-Z_]+)=([^\s]+)", str(text or "")))
        quantity_raw = str(pairs.get("qty") or "").strip()
        quantity = int(quantity_raw) if quantity_raw.isdigit() else None
        return {
            "action": str(pairs.get("action") or "").strip().lower() or None,
            "ticker": str(pairs.get("ticker") or "").strip() or None,
            "quantity": quantity,
            "price_source": pairs.get("price_source"),
            "raw": str(text or ""),
        }
