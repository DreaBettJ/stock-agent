"""Tests for prompt guardrail module."""

from __future__ import annotations

from mini_agent.app.prompt_guard import (
    ensure_trade_policy_prompt,
    resolve_strategy_risk_profile,
    validate_trade_policy_block,
)
from mini_agent.session import ExperimentSession


def _session() -> ExperimentSession:
    return ExperimentSession(
        session_id=1,
        name="s1",
        system_prompt="base prompt",
        mode="simulation",
        initial_capital=1000000.0,
        current_cash=800000.0,
        risk_preference="high",
        single_position_cap_pct=30.0,
        max_single_loss_pct=3.0,
        stop_loss_pct=6.0,
        take_profit_pct=18.0,
        investment_horizon="长线",
    )


def test_ensure_trade_policy_prompt_injects_sections():
    session = _session()
    merged, missing = ensure_trade_policy_prompt(
        prompt="你是交易助手",
        role="A股自动交易代理",
        session=session,
        positions=[{"ticker": "600519", "quantity": 100, "avg_cost": 1400.0}],
    )
    assert "## 一、账户信息" in merged
    assert "## 二、交易参数" in merged
    assert "## 三、工作流程" in merged
    assert "## 四、输出格式" in merged
    assert "## 五、禁止事项" in merged
    assert "风险偏好：high" in merged
    assert "单票仓位上限：30.00%" in merged
    assert "止损线：6.00%" in merged
    assert "止盈线：18.00%" in merged
    assert "投资周期：长线" in merged
    assert "不追高（涨幅>10.0%不追）" in merged
    assert missing == []


def test_validate_trade_policy_block_reports_missing():
    missing = validate_trade_policy_block("仅有普通提示词")
    assert "## 一、账户信息" in missing
    assert "## 五、禁止事项" in missing


def test_ensure_trade_policy_prompt_free_strategy_uses_light_constraints():
    session = _session()
    merged, missing = ensure_trade_policy_prompt(
        prompt="你是交易助手\n当前策略模板：自由策略",
        role="A股自动交易代理",
        session=session,
        positions=[],
    )
    assert "## 一、账户信息" in merged
    assert "## 二、交易参数" in merged
    assert "## 三、执行原则（自由策略）" in merged
    assert "## 五、禁止事项" not in merged
    assert missing == []


def test_resolve_strategy_risk_profile_aggressive():
    profile = resolve_strategy_risk_profile("当前策略模板：激进策略")
    assert profile["chase_limit_pct"] == 18.0
    assert profile["min_score"] == 50


def test_ensure_trade_policy_prompt_uses_dynamic_chase_limit():
    session = _session()
    session.system_prompt = "当前策略模板：激进策略"
    merged, _ = ensure_trade_policy_prompt(
        prompt="你是交易助手\n当前策略模板：激进策略",
        role="A股自动交易代理",
        session=session,
        positions=[],
    )
    assert "不追高（涨幅>18.0%不追）" in merged
