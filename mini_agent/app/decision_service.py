"""LLM decision orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_agent.auto_trading import AutoTradingWorkflow
from mini_agent.config import Config
from mini_agent.paths import resolve_kline_db_path
from mini_agent.session import SessionManager
from mini_agent.tools.kline_db_tool import KLineDB
from mini_agent.tools.sim_trade_tool import SimulateTradeTool


@dataclass(slots=True)
class DecisionRuntime:
    """Runtime dependencies for decision + execution."""

    session_manager: SessionManager
    kline_db: KLineDB
    workflow: AutoTradingWorkflow
    trade_tool: SimulateTradeTool


def build_decision_runtime(workspace_dir: Path, memory_db_path: Path) -> DecisionRuntime:
    """Build shared runtime components for LLM decision and simulated execution."""
    kline_db_path = resolve_kline_db_path(workspace_dir)
    if not kline_db_path.exists():
        raise FileNotFoundError(f"K-line database not found: {kline_db_path}")

    cfg = Config.from_yaml(Config.get_default_config_path())
    session_manager = SessionManager(db_path=str(memory_db_path))
    kline_db = KLineDB(db_path=str(kline_db_path))
    workflow = AutoTradingWorkflow(
        session_manager=session_manager,
        kline_db=kline_db,
        llm_provider=cfg.llm.provider,
        api_key=cfg.llm.api_key,
        model=cfg.llm.model,
    )
    trade_tool = SimulateTradeTool(db_path=str(memory_db_path), kline_db_path=str(kline_db_path))
    return DecisionRuntime(
        session_manager=session_manager,
        kline_db=kline_db,
        workflow=workflow,
        trade_tool=trade_tool,
    )


async def run_llm_decision(runtime: DecisionRuntime, session_id: int, trading_date: str) -> dict[str, Any]:
    """Run LLM decision and optional simulated execution."""
    return await runtime.workflow.trigger_daily_review(
        session_id=session_id,
        trading_date=trading_date,
        auto_execute=True,
        trade_executor=runtime.trade_tool,
    )

