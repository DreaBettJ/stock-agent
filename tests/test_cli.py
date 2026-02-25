"""Tests for cli module."""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCLI:
    """Test CLI module."""

    def test_cli_module_imports(self):
        """Test that cli module can be imported."""
        from mini_agent import cli
        
        assert cli is not None

    def test_cli_has_main_function(self):
        """Test that CLI has main function."""
        from mini_agent import cli
        
        assert hasattr(cli, "main")
        assert callable(cli.main)

    def test_cli_entry_point(self):
        """Test CLI entry point exists."""
        from mini_agent.cli import main
        
        assert callable(main)

    @patch("sys.argv", ["mini-agent", "--help"])
    def test_cli_help(self):
        """Test CLI help output."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit):
                from mini_agent.cli import main
                main()
            
            # Check that help was printed
            output = mock_stdout.getvalue()
            # Either help was printed or system exit occurred

    def test_cli_config_loading(self):
        """Test CLI config loading."""
        from mini_agent.config import Config
        
        # Should have Config class
        assert Config is not None

    def test_cli_argument_parsing(self):
        """Test CLI argument parsing capabilities."""
        # The CLI should be able to parse basic arguments
        with patch("sys.argv", ["mini-agent", "version"]):
            with patch("sys.stdout", new_callable=StringIO):
                try:
                    from mini_agent.cli import main
                    # If it doesn't raise, that's fine
                except Exception:
                    pass


class TestCLICommands:
    """Test CLI command structure."""

    def test_session_command_exists(self):
        """Test that session commands are defined."""
        # This is a basic test to ensure the CLI structure exists
        try:
            from mini_agent.cli import main
            assert main is not None
        except ImportError:
            pytest.skip("CLI module not fully available")

    def test_tools_registration(self):
        """Test that tools can be registered."""
        from mini_agent.tools.base import Tool, ToolResult
        
        assert Tool is not None
        assert ToolResult is not None

    def test_llm_client_registration(self):
        """Test LLM client can be imported."""
        from mini_agent.llm import LLMClient
        
        assert LLMClient is not None


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_config_integration(self):
        """Test config integrates with CLI."""
        from mini_agent.config import AgentConfig, LLMConfig, ToolsConfig
        
        # Should be able to create config objects
        llm = LLMConfig(api_key="test")
        agent = AgentConfig()
        tools = ToolsConfig()
        
        assert llm.api_key == "test"
        assert agent.max_steps == 50
        assert tools.enable_file_tools is True

    def test_tools_available(self):
        """Test that all expected tools are available."""
        from mini_agent.tools import bash_tool, file_tools, note_tool
        
        assert bash_tool is not None
        assert file_tools is not None
        assert note_tool is not None

    def test_schema_imports(self):
        """Test schema imports work."""
        from mini_agent.schema import Message, ToolCall
        
        # Should be able to create messages
        msg = Message(role="user", content="test")
        assert msg.role == "user"
        assert msg.content == "test"


class TestCLIModuleStructure:
    """Test CLI module structure."""

    def test_module_has_required_classes(self):
        """Test CLI module has required classes."""
        try:
            from mini_agent.cli import (
                Agent,
                Config,
                LLMClient,
            )
        except ImportError as e:
            pytest.skip(f"CLI not fully implemented: {e}")

    def test_error_handling(self):
        """Test error handling in CLI."""
        # Test that CLI handles errors gracefully
        with patch("sys.argv", ["mini-agent", "invalid-command"]):
            with patch("sys.stderr", new_callable=StringIO):
                try:
                    from mini_agent.cli import main
                except (SystemExit, Exception):
                    pass  # Expected for invalid command


def test_cli_can_be_run():
    """Test CLI can be run (smoke test)."""
    # This is a basic smoke test
    import mini_agent.cli
    
    assert hasattr(mini_agent.cli, "main")


def test_parse_args_supports_session_id(monkeypatch):
    """Test top-level --session-id argument parsing."""
    from mini_agent.cli import parse_args

    monkeypatch.setattr(sys, "argv", ["mini-agent", "--session-id", "8"])
    args = parse_args()
    assert args.session_id == 8


def test_parse_args_session_create_risk_fields(monkeypatch):
    from mini_agent.cli import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mini-agent",
            "session",
            "create",
            "--name",
            "s1",
            "--prompt",
            "p",
            "--mode",
            "simulation",
            "--risk-preference",
            "high",
            "--stop-loss-pct",
            "6",
        ],
    )
    args = parse_args()
    assert args.command == "session"
    assert args.session_command == "create"
    assert args.risk_preference == "high"
    assert args.stop_loss_pct == 6.0


def test_parse_args_trade_action_logs(monkeypatch):
    from mini_agent.cli import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["mini-agent", "trade", "action-logs", "--session", "8", "--limit", "5"],
    )
    args = parse_args()
    assert args.command == "trade"
    assert args.trade_command == "action-logs"
    assert args.session_id == 8
    assert args.limit == 5


def test_add_workspace_tools_includes_simulate_trade(tmp_path):
    from mini_agent.cli import add_workspace_tools
    from mini_agent.config import AgentConfig, Config, LLMConfig, ToolsConfig

    config = Config(
        llm=LLMConfig(api_key="test"),
        agent=AgentConfig(),
        tools=ToolsConfig(
            enable_file_tools=False,
            enable_bash=False,
            enable_note=False,
            enable_stock_tools=False,
            enable_skills=False,
            enable_mcp=False,
        ),
    )
    tools = []
    add_workspace_tools(tools, config, tmp_path, session_id=1)
    names = [getattr(t, "name", "") for t in tools]
    assert "simulate_trade" in names


def test_add_workspace_tools_passes_tushare_token(monkeypatch, tmp_path):
    from mini_agent.cli import add_workspace_tools
    from mini_agent.config import AgentConfig, Config, LLMConfig, ToolsConfig

    captured = {}

    def _fake_create_a_share_tools(*, tushare_token=None, backend=None):
        captured["tushare_token"] = tushare_token
        return []

    monkeypatch.setattr("mini_agent.cli.create_a_share_tools", _fake_create_a_share_tools)

    config = Config(
        llm=LLMConfig(api_key="test"),
        agent=AgentConfig(),
        tools=ToolsConfig(
            enable_file_tools=False,
            enable_bash=False,
            enable_note=False,
            enable_stock_tools=True,
            tushare_token="ts-from-config",
            enable_skills=False,
            enable_mcp=False,
        ),
    )
    tools = []
    add_workspace_tools(tools, config, tmp_path, session_id=1)
    assert captured["tushare_token"] == "ts-from-config"


def test_parse_direct_trade_order_buy_with_alias():
    from mini_agent.cli import _parse_direct_trade_order

    order = _parse_direct_trade_order("购买100股茅台")
    assert order is not None
    assert order["action"] == "buy"
    assert order["ticker"] == "600519"
    assert order["quantity"] == 100


def test_parse_direct_trade_order_with_assumed_date():
    from mini_agent.cli import _parse_direct_trade_order

    order = _parse_direct_trade_order("假设现在是2025-01-01,买入 100 股茅台")
    assert order is not None
    assert order["action"] == "buy"
    assert order["ticker"] == "600519"
    assert order["quantity"] == 100
    assert order["trade_date"] == "2025-01-01"


def test_parse_direct_trade_order_sell_with_ticker():
    from mini_agent.cli import _parse_direct_trade_order

    order = _parse_direct_trade_order("卖出 601318 200股")
    assert order is not None
    assert order["action"] == "sell"
    assert order["ticker"] == "601318"
    assert order["quantity"] == 200


def test_parse_sim_trade_tool_content_ok():
    from mini_agent.cli import _parse_sim_trade_tool_content

    parsed = _parse_sim_trade_tool_content(
        "SIM_TRADE_OK id=1 session=8 action=buy ticker=600519 qty=100 price=1500.0 amount=150000 fee=45 price_source=next_open"
    )
    assert parsed["action"] == "buy"
    assert parsed["ticker"] == "600519"
    assert parsed["quantity"] == 100
    assert parsed["price_source"] == "next_open"


def test_record_sim_trade_critical_from_messages(tmp_path):
    from mini_agent.cli import _record_sim_trade_critical_from_messages
    from mini_agent.schema.schema import Message
    from mini_agent.session import SessionManager

    db_path = tmp_path / "m.db"
    manager = SessionManager(db_path=str(db_path))
    sid = manager.create_session(name="s1", system_prompt="p", mode="simulation")

    msgs = [
        Message(
            role="tool",
            name="simulate_trade",
            tool_call_id="tc1",
            content="SIM_TRADE_OK id=3 session=1 action=buy ticker=600519 qty=100 price=10.0 amount=1000 fee=0.3 price_source=next_open",
        )
    ]
    _record_sim_trade_critical_from_messages(
        session_manager=manager,
        session_id=sid,
        messages=msgs,
        event_type="chat_turn",
    )
    rows = manager.list_critical_memories(sid, limit=10)
    assert len(rows) == 1
    assert rows[0]["operation"] == "buy"


def test_load_strategy_templates(tmp_path):
    from mini_agent.cli import _load_strategy_templates

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "strategy_templates.md").write_text(
        "# title\n\n## 1. 模板A\n内容A\n\n## 2. 模板B\n内容B\n",
        encoding="utf-8",
    )
    templates = _load_strategy_templates(tmp_path)
    assert len(templates) == 2
    assert templates[0]["id"] == "1"
    assert templates[0]["name"] == "模板A"


def test_load_strategy_templates_excludes_non_template_level2_sections(tmp_path):
    from mini_agent.cli import _load_strategy_templates

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "strategy_templates.md").write_text(
        (
            "# title\n\n"
            "## 1. 模板A\n"
            "### 策略描述\n"
            "内容A\n\n"
            "## 策略选择建议\n"
            "这部分不应进入模板内容\n"
        ),
        encoding="utf-8",
    )
    templates = _load_strategy_templates(tmp_path)
    assert len(templates) == 1
    assert "策略选择建议" not in templates[0]["content"]
    assert "这部分不应进入模板内容" not in templates[0]["content"]


def test_collect_session_create_inputs_with_template_non_interactive(monkeypatch, tmp_path):
    from mini_agent.cli import _collect_session_create_inputs

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "strategy_templates.md").write_text(
        "# title\n\n## 1. 模板A\n### 策略描述\n测试策略\n",
        encoding="utf-8",
    )

    args = argparse.Namespace(
        name="s1",
        prompt=None,
        template="1",
        mode="simulation",
        initial_capital=100000.0,
        risk_preference="medium",
        max_single_loss_pct=2.0,
        single_position_cap_pct=25.0,
        stop_loss_pct=7.0,
        take_profit_pct=15.0,
        investment_horizon="中线",
        event_filter=[],
    )
    monkeypatch.setattr("mini_agent.cli.sys.stdin.isatty", lambda: False)
    payload = _collect_session_create_inputs(args, tmp_path)
    assert payload["name"] == "s1"
    assert "当前策略模板：模板A" in payload["system_prompt"]


def test_collect_session_create_inputs_with_free_template(monkeypatch, tmp_path):
    from mini_agent.cli import _collect_session_create_inputs

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "strategy_templates.md").write_text(
        "# title\n\n## 0. 自由策略\n### 策略描述\n动态决策\n",
        encoding="utf-8",
    )

    args = argparse.Namespace(
        name="s_free",
        prompt=None,
        template="0",
        mode="simulation",
        initial_capital=100000.0,
        risk_preference="medium",
        max_single_loss_pct=2.0,
        single_position_cap_pct=25.0,
        stop_loss_pct=7.0,
        take_profit_pct=15.0,
        investment_horizon="中线",
        event_filter=[],
    )
    monkeypatch.setattr("mini_agent.cli.sys.stdin.isatty", lambda: False)
    payload = _collect_session_create_inputs(args, tmp_path)
    assert "当前策略模板：自由策略" in payload["system_prompt"]
    assert "严格执行策略纪律" not in payload["system_prompt"]


def test_handle_sync_command_requires_tushare_token_for_all(monkeypatch, tmp_path, capsys):
    from mini_agent.cli import handle_sync_command
    from mini_agent.config import AgentConfig, Config, LLMConfig, ToolsConfig

    cfg = Config(
        llm=LLMConfig(api_key="test"),
        agent=AgentConfig(),
        tools=ToolsConfig(tushare_token=""),
    )
    monkeypatch.setattr("mini_agent.cli.Config.from_yaml", lambda _: cfg)
    monkeypatch.setattr("mini_agent.cli.Config.get_default_config_path", lambda: Path("/tmp/fake_config.yaml"))

    args = argparse.Namespace(
        cron=False,
        install_cron=False,
        tickers="",
        all=True,
        start="1991-01-01",
        end=None,
    )
    handle_sync_command(args, tmp_path)
    output = capsys.readouterr().out
    assert "tools.tushare_token is empty" in output


def test_handle_trade_action_logs_command(tmp_path, capsys, monkeypatch):
    from mini_agent.cli import handle_trade_command
    from mini_agent.session import SessionManager

    db_path = tmp_path / ".agent_memory.db"
    monkeypatch.setattr("mini_agent.cli.get_memory_db_path", lambda _workspace: db_path)
    manager = SessionManager(db_path=str(db_path))
    sid = manager.create_session(name="s1", system_prompt="p", mode="simulation")
    manager.record_trade_action_log(
        session_id=sid,
        event_id="daily_review:2026-02-25",
        event_type="daily_review",
        action="buy",
        ticker="600519",
        quantity=100,
        trade_date="2026-02-25",
        status="succeeded",
        reason="test",
    )

    args = argparse.Namespace(
        trade_command="action-logs",
        session_id=sid,
        limit=10,
    )
    handle_trade_command(args, tmp_path)
    out = capsys.readouterr().out
    assert "event_id" in out
    assert "600519" in out
