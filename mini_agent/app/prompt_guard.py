"""Trading prompt guardrails for runtime system prompts."""

from __future__ import annotations

import os
import re
from typing import Any, Mapping, Sequence

from mini_agent.session import ExperimentSession

TRADE_POLICY_MARKER = "## 一、账户信息"


def extract_strategy_template_name(prompt: str) -> str:
    """Extract strategy template name from prompt text."""
    text = str(prompt or "")
    m = re.search(r"当前策略模板：([^\n\r]+)", text)
    if m:
        return str(m.group(1)).strip()
    return ""


def resolve_strategy_risk_profile(prompt: str) -> dict[str, Any]:
    """Resolve risk thresholds by strategy template."""
    strategy = extract_strategy_template_name(prompt)
    text = strategy.lower()
    profile = {
        "strategy_template": strategy or "default",
        "chase_limit_pct": 10.0,
        "min_score": 55,
        "allowed_trends": ["up", "sideways"],
    }
    if "激进" in strategy:
        profile.update(
            {
                "chase_limit_pct": 18.0,
                "min_score": 50,
                "allowed_trends": ["up", "sideways"],
            }
        )
    elif "自由策略" in strategy:
        profile.update(
            {
                "chase_limit_pct": 12.0,
                "min_score": 52,
                "allowed_trends": ["up", "sideways"],
            }
        )
    elif "高股息" in strategy:
        profile.update(
            {
                "chase_limit_pct": 8.0,
                "min_score": 50,
                "allowed_trends": ["up", "sideways"],
            }
        )
    elif "突破新高" in strategy:
        profile.update(
            {
                "chase_limit_pct": 12.0,
                "min_score": 58,
                "allowed_trends": ["up"],
            }
        )
    elif "动量" in strategy or "龙头" in strategy or "题材" in strategy:
        profile.update(
            {
                "chase_limit_pct": 12.0,
                "min_score": 55,
                "allowed_trends": ["up", "sideways"],
            }
        )
    elif "均线" in strategy or "趋势" in strategy:
        profile.update(
            {
                "chase_limit_pct": 10.0,
                "min_score": 58,
                "allowed_trends": ["up"],
            }
        )
    elif "quality" in text or "value" in text:
        profile.update(
            {
                "chase_limit_pct": 8.0,
                "min_score": 52,
                "allowed_trends": ["up", "sideways"],
            }
        )
    return profile


def _as_float(raw: str | None, default: float) -> float:
    try:
        if raw is None:
            return default
        return float(str(raw).strip())
    except Exception:
        return default


def _build_position_text(positions: Sequence[Mapping[str, Any]]) -> str:
    if not positions:
        return "空仓"
    lines: list[str] = []
    for row in positions[:12]:
        ticker = str(row.get("ticker") or row.get("代码") or "").strip() or "UNKNOWN"
        qty = row.get("quantity") or row.get("持仓") or 0
        avg_cost = row.get("avg_cost")
        if avg_cost is None:
            lines.append(f"{ticker} x {qty}股")
        else:
            lines.append(f"{ticker} x {qty}股 @ 均价{avg_cost}")
    return "; ".join(lines)


def _is_free_strategy_prompt(prompt: str) -> bool:
    text = (prompt or "").strip()
    if not text:
        return False
    return "当前策略模板：自由策略" in text or "自由策略" in text


def build_trade_policy_block(
    *,
    role: str,
    session: ExperimentSession,
    positions: Sequence[Mapping[str, Any]],
    constrained: bool = True,
) -> str:
    """Build trade policy block injected into system prompt."""
    risk_preference = str(getattr(session, "risk_preference", "") or os.getenv("MINI_AGENT_RISK_PREFERENCE", "medium")).strip().lower() or "medium"
    if risk_preference not in {"low", "medium", "high"}:
        risk_preference = "medium"

    single_position_cap_pct = _as_float(
        str(getattr(session, "single_position_cap_pct", "")) if getattr(session, "single_position_cap_pct", None) is not None else os.getenv("MINI_AGENT_SINGLE_POSITION_CAP_PCT"),
        25.0,
    )
    max_single_loss_pct = _as_float(
        str(getattr(session, "max_single_loss_pct", "")) if getattr(session, "max_single_loss_pct", None) is not None else os.getenv("MINI_AGENT_MAX_SINGLE_LOSS_PCT"),
        2.0,
    )
    stop_loss_pct = _as_float(
        str(getattr(session, "stop_loss_pct", "")) if getattr(session, "stop_loss_pct", None) is not None else os.getenv("MINI_AGENT_STOP_LOSS_PCT"),
        7.0,
    )
    take_profit_pct = _as_float(
        str(getattr(session, "take_profit_pct", "")) if getattr(session, "take_profit_pct", None) is not None else os.getenv("MINI_AGENT_TAKE_PROFIT_PCT"),
        15.0,
    )
    investment_horizon = str(
        getattr(session, "investment_horizon", "") or os.getenv("MINI_AGENT_INVESTMENT_HORIZON", "中线")
    ).strip() or "中线"
    position_text = _build_position_text(positions)
    strategy_profile = resolve_strategy_risk_profile(session.system_prompt)
    chase_limit_pct = float(strategy_profile.get("chase_limit_pct") or 10.0)

    base = (
        f"你是一个**{role}**，遵循以下规则：\n\n"
        "## 一、账户信息\n"
        f"- 账户总资金：{session.initial_capital:.2f}元\n"
        f"- 当前可用现金：{session.current_cash:.2f}元\n"
        f"- 当前持仓：{position_text}\n"
        f"- 风险偏好：{risk_preference}\n"
        f"- 最大单笔亏损：{max_single_loss_pct:.2f}%\n\n"
        "## 二、交易参数\n"
        f"- 单票仓位上限：{single_position_cap_pct:.2f}%\n"
        f"- 止损线：{stop_loss_pct:.2f}%\n"
        f"- 止盈线：{take_profit_pct:.2f}%\n"
        f"- 投资周期：{investment_horizon}\n\n"
    )
    if not constrained:
        return (
            base
            + "## 三、执行原则（自由策略）\n"
            + "1. 优先使用工具获取事实数据，再做决策\n"
            + "2. 在风险参数范围内自主决定是否交易、交易方向和仓位\n"
            + "3. 执行前补全交易要素（代码、方向、数量、交易日）\n"
            + "4. 执行后记录操作与理由\n"
        )
    return (
        base
        + "## 三、工作流程\n"
        + "1. 先用工具获取数据，工具失败再尝试备用数据源\n"
        + "2. 给出决策时必须包含：标的、仓位、止损、止盈\n"
        + "3. 执行前确认交易要素完整（代码、方向、数量、交易日）\n"
        + "4. 每笔交易后记录操作和原因\n\n"
        + "## 四、输出格式\n"
        + "每次操作原因必须包含：\n"
        + "- 标的代码 + 名称\n"
        + "- 买入/卖出价格\n"
        + "- 数量（股/金额）\n"
        + "- 止损价\n"
        + "- 止盈价\n"
        + "- 仓位占比\n"
        + "- 风险提示\n"
        + "调用tool进行交易\n\n"
        + "## 五、禁止事项\n"
        + f"- 不追高（涨幅>{chase_limit_pct:.1f}%不追）\n"
        + "- 不满仓（预留应急资金）\n"
        + "- 不听消息炒概念"
    )


def validate_trade_policy_block(prompt: str, *, constrained: bool = True) -> list[str]:
    """Return missing section titles for required policy block."""
    required_sections = [
        "## 一、账户信息",
        "## 二、交易参数",
    ]
    if constrained:
        required_sections.extend(
            [
                "## 三、工作流程",
                "## 四、输出格式",
                "## 五、禁止事项",
            ]
        )
    else:
        required_sections.append("## 三、执行原则（自由策略）")
    missing = [section for section in required_sections if section not in prompt]
    return missing


def ensure_trade_policy_prompt(
    *,
    prompt: str,
    role: str,
    session: ExperimentSession,
    positions: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str]]:
    """Ensure trading policy block is present and validated."""
    constrained = not _is_free_strategy_prompt(prompt)
    merged = prompt
    if TRADE_POLICY_MARKER not in merged:
        merged = (
            f"{prompt.strip()}\n\n"
            f"{build_trade_policy_block(role=role, session=session, positions=positions, constrained=constrained)}"
        )
    return merged, validate_trade_policy_block(merged, constrained=constrained)
