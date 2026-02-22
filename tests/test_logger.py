"""Tests for logger module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mini_agent.logger import AgentLogger
from mini_agent.schema import Message, ToolCall, ToolCall


class TestAgentLogger:
    """Test AgentLogger class."""

    @pytest.fixture
    def logger_with_temp_dir(self):
        """Create logger with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                logger = AgentLogger(enabled=True)
                yield logger

    @pytest.fixture
    def disabled_logger(self):
        """Create disabled logger."""
        return AgentLogger(enabled=False)

    def test_init_enabled(self):
        """Test logger initialization when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                logger = AgentLogger(enabled=True)
                assert logger.enabled is True
                assert logger.log_dir.exists()

    def test_init_disabled(self):
        """Test logger initialization when disabled."""
        logger = AgentLogger(enabled=False)
        assert logger.enabled is False
        assert logger.log_file is None

    def test_start_new_run(self, logger_with_temp_dir):
        """Test starting a new run."""
        logger = logger_with_temp_dir
        logger.start_new_run()
        
        assert logger.log_file is not None
        assert logger.intercept_log_file is not None
        assert logger.log_index == 0
        assert logger.log_file.exists()

    def test_start_new_run_disabled(self, disabled_logger):
        """Test starting new run when disabled."""
        disabled_logger.start_new_run()
        assert disabled_logger.log_file is None

    def test_log_request(self, logger_with_temp_dir):
        """Test logging LLM request."""
        logger = logger_with_temp_dir
        logger.start_new_run()
        
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        
        logger.log_request(messages)
        assert logger.log_index == 1

    def test_log_response(self, logger_with_temp_dir):
        """Test logging LLM response."""
        logger = logger_with_temp_dir
        logger.start_new_run()
        
        logger.log_response(
            content="Test response",
            thinking="Thinking process",
            finish_reason="stop",
        )
        assert logger.log_index == 1

    def test_log_response_with_tool_calls(self, logger_with_temp_dir):
        """Test logging response with tool calls."""
        logger = logger_with_temp_dir
        logger.start_new_run()
        
        tool_calls = [
            ToolCall(
                id="call_123",
                type="function",
                function={"name": "bash", "arguments": {"command": "ls"}},
            )
        ]
        
        logger.log_response(
            content="Using tool",
            tool_calls=tool_calls,
            finish_reason="tool_use",
        )
        assert logger.log_index == 1

    def test_log_tool_result_success(self, logger_with_temp_dir):
        """Test logging successful tool result."""
        logger = logger_with_temp_dir
        logger.start_new_run()
        
        logger.log_tool_result(
            tool_name="bash",
            arguments={"command": "ls"},
            result_success=True,
            result_content="file1.txt\nfile2.txt",
        )
        assert logger.log_index == 1

    def test_log_tool_result_failure(self, logger_with_temp_dir):
        """Test logging failed tool result."""
        logger = logger_with_temp_dir
        logger.start_new_run()
        
        logger.log_tool_result(
            tool_name="bash",
            arguments={"command": "invalid"},
            result_success=False,
            result_error="Command failed",
        )
        assert logger.log_index == 1

    def test_log_intercept_event(self, logger_with_temp_dir):
        """Test logging intercept event."""
        logger = logger_with_temp_dir
        logger.start_new_run()
        
        logger.log_intercept_event("request", {"model": "test-model"})
        
        assert logger.intercept_log_file.exists()
        
        # Verify JSONL format
        with open(logger.intercept_log_file) as f:
            line = f.readline()
            event_data = json.loads(line)
            assert event_data["event"] == "request"
            assert event_data["model"] == "test-model"
            assert "timestamp" in event_data

    def test_log_intercept_event_disabled(self, disabled_logger):
        """Test logging intercept event when disabled."""
        disabled_logger.start_new_run()
        disabled_logger.log_intercept_event("test", {"key": "value"})
        # Should not raise any error

    def test_get_log_file_path(self, logger_with_temp_dir):
        """Test getting log file path."""
        logger = logger_with_temp_dir
        logger.start_new_run()
        
        path = logger.get_log_file_path()
        assert isinstance(path, Path)
        assert path.name.startswith("agent_run_")

    def test_get_intercept_log_file_path(self, logger_with_temp_dir):
        """Test getting intercept log file path."""
        logger = logger_with_temp_dir
        logger.start_new_run()
        
        path = logger.get_intercept_log_file_path()
        assert isinstance(path, Path)
        assert path.name.startswith("agent_intercept_")

    def test_multiple_runs(self, logger_with_temp_dir):
        """Test multiple runs create separate log files."""
        import time
        logger = logger_with_temp_dir
        
        logger.start_new_run()
        first_log = logger.log_file
        
        time.sleep(1.1)  # Wait to ensure different timestamp
        logger.start_new_run()
        second_log = logger.log_file
        
        assert first_log != second_log

    def test_write_log_type(self, logger_with_temp_dir):
        """Test writing different log types."""
        logger = logger_with_temp_dir
        logger.start_new_run()
        
        # Test REQUEST type
        logger._write_log("REQUEST", "Test request content")
        
        # Test RESPONSE type
        logger._write_log("RESPONSE", "Test response content")
        
        # Test TOOL_RESULT type
        logger._write_log("TOOL_RESULT", "Test tool result content")
        
        # Verify all were written to log file
        with open(logger.log_file) as f:
            content = f.read()
            assert "REQUEST" in content
            assert "RESPONSE" in content
            assert "TOOL_RESULT" in content

    def test_disabled_logger_no_op(self, disabled_logger):
        """Test that disabled logger does nothing."""
        # These should all be no-ops
        disabled_logger.start_new_run()
        disabled_logger.log_request([])
        disabled_logger.log_response("content")
        disabled_logger.log_tool_result("tool", {}, True)
        disabled_logger._write_log("TEST", "content")
        
        # Should not raise any errors
        assert disabled_logger.log_file is None
