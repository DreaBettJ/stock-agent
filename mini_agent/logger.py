"""Agent run logger"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .schema import Message, ToolCall


class AgentLogger:
    """Agent run logger

    Responsible for recording the complete interaction process of each agent run, including:
    - LLM requests and responses
    - Tool calls and results
    """

    def __init__(self, enabled: bool = True, session_id: int | str | None = None):
        """Initialize logger

        Logs are stored in <project_root>/log/ directory
        """
        self.enabled = enabled
        # Use project root /log directory for logs
        self.log_dir = Path(__file__).resolve().parent.parent / "log"
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = None
        self.intercept_log_file = None
        self.log_index = 0
        self.session_id: str | None = str(session_id) if session_id is not None else None

    def set_session_id(self, session_id: int | str | None) -> None:
        """Bind logger output to one session id."""
        self.session_id = str(session_id) if session_id is not None else None

    def start_new_run(self):
        """Start new run, create new log file"""
        if not self.enabled:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sid = self.session_id or "unknown"
        # Keep file names simple and filesystem-safe.
        sid = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in sid)
        log_filename = f"agent_run_s{sid}_{timestamp}.log"
        intercept_filename = f"agent_intercept_s{sid}_{timestamp}.jsonl"
        self.log_file = self.log_dir / log_filename
        self.intercept_log_file = self.log_dir / intercept_filename
        self.log_index = 0

        # Write log header
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Agent Run Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session ID: {self.session_id or 'unknown'}\n")
            f.write("=" * 80 + "\n\n")

    def log_intercept_event(self, event: str, data: dict[str, Any]):
        """Log one interception event as a single JSONL line."""
        if not self.enabled or self.intercept_log_file is None:
            return

        event_data: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
        }
        event_data.update(data)

        with open(self.intercept_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_data, ensure_ascii=False) + "\n")

    def log_request(self, messages: list[Message], tools: list[Any] | None = None):
        """Log LLM request

        Args:
            messages: Message list
            tools: Tool list (optional)
        """
        self.log_index += 1

        # Build complete request data structure
        request_data = {
            "messages": [],
            "tools": [],
        }

        # Convert messages to JSON serializable format
        for msg in messages:
            msg_dict = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.thinking:
                msg_dict["thinking"] = msg.thinking
            if msg.tool_calls:
                msg_dict["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if msg.name:
                msg_dict["name"] = msg.name

            request_data["messages"].append(msg_dict)

        # Only record tool names
        if tools:
            request_data["tools"] = [tool.name for tool in tools]

        # Format as JSON
        content = "LLM Request:\n\n"
        content += json.dumps(request_data, indent=2, ensure_ascii=False)

        self._write_log("REQUEST", content)

    def log_response(
        self,
        content: str,
        thinking: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        finish_reason: str | None = None,
    ):
        """Log LLM response

        Args:
            content: Response content
            thinking: Thinking content (optional)
            tool_calls: Tool call list (optional)
            finish_reason: Finish reason (optional)
        """
        self.log_index += 1

        # Build complete response data structure
        response_data = {
            "content": content,
        }

        if thinking:
            response_data["thinking"] = thinking

        if tool_calls:
            response_data["tool_calls"] = [tc.model_dump() for tc in tool_calls]

        if finish_reason:
            response_data["finish_reason"] = finish_reason

        # Format as JSON
        log_content = "LLM Response:\n\n"
        log_content += json.dumps(response_data, indent=2, ensure_ascii=False)

        self._write_log("RESPONSE", log_content)

    def log_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result_success: bool,
        result_content: str | None = None,
        result_error: str | None = None,
    ):
        """Log tool execution result

        Args:
            tool_name: Tool name
            arguments: Tool arguments
            result_success: Whether successful
            result_content: Result content (on success)
            result_error: Error message (on failure)
        """
        self.log_index += 1

        # Build complete tool execution result data structure
        tool_result_data = {
            "tool_name": tool_name,
            "arguments": arguments,
            "success": result_success,
        }

        if result_success:
            tool_result_data["result"] = result_content
        else:
            tool_result_data["error"] = result_error

        # Format as JSON
        content = "Tool Execution:\n\n"
        content += json.dumps(tool_result_data, indent=2, ensure_ascii=False)

        self._write_log("TOOL_RESULT", content)

    def _write_log(self, log_type: str, content: str):
        """Write log entry

        Args:
            log_type: Log type (REQUEST, RESPONSE, TOOL_RESULT)
            content: Log content
        """
        if not self.enabled or self.log_file is None:
            return

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"[{self.log_index}] {log_type}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write("-" * 80 + "\n")
            f.write(content + "\n")

    def get_log_file_path(self) -> Path:
        """Get current log file path"""
        return self.log_file

    def get_intercept_log_file_path(self) -> Path:
        """Get current interception log file path"""
        return self.intercept_log_file
