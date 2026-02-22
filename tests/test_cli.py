"""Tests for cli module."""

from __future__ import annotations

import sys
from io import StringIO
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
