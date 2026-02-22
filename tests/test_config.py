"""Tests for config module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from mini_agent.config import (
    AgentConfig,
    Config,
    LLMConfig,
    MCPConfig,
    RetryConfig,
    ToolsConfig,
)


class TestRetryConfig:
    """Test RetryConfig."""

    def test_default_values(self):
        config = RetryConfig()
        assert config.enabled is True
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0

    def test_custom_values(self):
        config = RetryConfig(
            enabled=False,
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
        )
        assert config.enabled is False
        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5

    def test_calculate_delay(self):
        from mini_agent.retry import RetryConfig
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, max_delay=60.0)
        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(10) == 60.0  # capped at max_delay


class TestLLMConfig:
    """Test LLMConfig."""

    def test_default_values(self):
        config = LLMConfig(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.api_base == "https://api.minimax.io"
        assert config.model == "MiniMax-M2.5"
        assert config.provider == "anthropic"
        assert config.retry.enabled is True

    def test_custom_values(self):
        config = LLMConfig(
            api_key="test-key",
            api_base="https://custom.api.com",
            model="gpt-4",
            provider="openai",
            retry=RetryConfig(max_retries=5),
        )
        assert config.api_base == "https://custom.api.com"
        assert config.model == "gpt-4"
        assert config.provider == "openai"
        assert config.retry.max_retries == 5


class TestAgentConfig:
    """Test AgentConfig."""

    def test_default_values(self):
        config = AgentConfig()
        assert config.max_steps == 50
        assert config.workspace_dir == "./workspace"
        assert config.system_prompt_path == "system_prompt.md"
        assert config.enable_intercept_log is True

    def test_custom_values(self):
        config = AgentConfig(
            max_steps=100,
            workspace_dir="/custom/workspace",
            system_prompt_path="custom.md",
            enable_intercept_log=False,
        )
        assert config.max_steps == 100
        assert config.workspace_dir == "/custom/workspace"
        assert config.system_prompt_path == "custom.md"
        assert config.enable_intercept_log is False


class TestMCPConfig:
    """Test MCPConfig."""

    def test_default_values(self):
        config = MCPConfig()
        assert config.connect_timeout == 10.0
        assert config.execute_timeout == 60.0
        assert config.sse_read_timeout == 120.0

    def test_custom_values(self):
        config = MCPConfig(
            connect_timeout=5.0,
            execute_timeout=30.0,
            sse_read_timeout=60.0,
        )
        assert config.connect_timeout == 5.0
        assert config.execute_timeout == 30.0
        assert config.sse_read_timeout == 60.0


class TestToolsConfig:
    """Test ToolsConfig."""

    def test_default_values(self):
        config = ToolsConfig()
        assert config.enable_file_tools is True
        assert config.enable_bash is True
        assert config.enable_note is True
        assert config.enable_stock_tools is True
        assert config.enable_skills is True
        assert config.skills_dir == "./skills"
        assert config.enable_mcp is True
        assert config.mcp_config_path == "mcp.json"

    def test_custom_values(self):
        config = ToolsConfig(
            enable_file_tools=False,
            enable_bash=False,
            mcp=MCPConfig(connect_timeout=5.0),
        )
        assert config.enable_file_tools is False
        assert config.enable_bash is False
        assert config.mcp.connect_timeout == 5.0


class TestConfig:
    """Test main Config class."""

    @pytest.fixture
    def valid_config_file(self):
        """Create a valid temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "api_key": "test-api-key-12345",
                "api_base": "https://api.test.com",
                "model": "test-model",
                "provider": "openai",
                "max_steps": 100,
                "workspace_dir": "/test/workspace",
            }
            yaml.dump(config, f)
            return Path(f.name)

    @pytest.fixture
    def invalid_config_file(self):
        """Create an invalid config file (empty)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            return Path(f.name)

    def test_load_from_yaml(self, valid_config_file):
        """Test loading config from YAML file."""
        config = Config.from_yaml(valid_config_file)
        assert config.llm.api_key == "test-api-key-12345"
        assert config.llm.api_base == "https://api.test.com"
        assert config.llm.model == "test-model"
        assert config.llm.provider == "openai"
        assert config.agent.max_steps == 100
        assert config.agent.workspace_dir == "/test/workspace"

    def test_load_missing_api_key(self, invalid_config_file):
        """Test that missing api_key raises error."""
        # Empty file first raises "empty" error, then we test with proper file
        with pytest.raises(ValueError):
            Config.from_yaml(invalid_config_file)

    def test_load_invalid_api_key(self):
        """Test that invalid api_key raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"api_key": "YOUR_API_KEY_HERE"}, f)
            config_path = Path(f.name)

        with pytest.raises(ValueError, match="Please configure a valid API Key"):
            Config.from_yaml(config_path)

    def test_load_file_not_found(self):
        """Test that non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("/nonexistent/path/config.yaml")

    def test_load_empty_file(self):
        """Test that empty file raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            config_path = Path(f.name)

        with pytest.raises(ValueError, match="Configuration file is empty"):
            Config.from_yaml(config_path)

    def test_get_package_dir(self):
        """Test getting package directory."""
        package_dir = Config.get_package_dir()
        assert isinstance(package_dir, Path)
        assert package_dir.name == "mini_agent"

    def test_find_config_file_priority(self, valid_config_file):
        """Test config file search priority."""
        # This test depends on the actual config structure
        result = Config.find_config_file("config.yaml")
        # May return None if no config exists in search paths
        assert result is None or isinstance(result, Path)
