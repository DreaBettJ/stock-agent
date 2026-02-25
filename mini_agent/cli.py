"""
Mini Agent - Interactive Runtime Example

Usage:
    mini-agent [--workspace DIR] [--task TASK] [--session-id ID]

Examples:
    mini-agent                              # Use current directory as workspace (interactive mode)
    mini-agent --workspace /path/to/dir     # Use specific workspace directory (interactive mode)
    mini-agent --task "create a file"       # Execute a task non-interactively
    mini-agent --session-id 8               # Attach to existing session 8
"""

import argparse
import asyncio
import json
import logging
import os
import platform
import re
import sqlite3
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from mini_agent import LLMClient
from mini_agent.agent import Agent
from mini_agent.app.decision_service import build_decision_runtime, run_llm_decision
from mini_agent.app.evolution_service import EvolutionUseCaseService
from mini_agent.app.memory_service import (
    build_session_memory_snapshot,
    load_critical_session_memories,
    load_recent_session_memories,
)
from mini_agent.app.prompt_guard import ensure_trade_policy_prompt
from mini_agent.app.runtime_factory import build_runtime_session_context
from mini_agent.app.sync_service import build_cron_lines, install_cron_lines, resolve_ticker_universe, sync_kline_data
from mini_agent.config import Config
from mini_agent.event_broadcaster import EventBroadcaster
from mini_agent.paths import get_default_memory_db_path, resolve_kline_db_path
from mini_agent.schema import LLMProvider
from mini_agent.session import SessionManager
from mini_agent.tools.base import Tool
from mini_agent.tools.bash_tool import BashKillTool, BashOutputTool, BashTool
from mini_agent.tools.file_tools import EditTool, ReadTool, WriteTool
from mini_agent.tools.mcp_loader import cleanup_mcp_connections, load_mcp_tools_async, set_mcp_timeout_config
from mini_agent.tools.note_tool import RecallNoteTool, SessionNoteTool
from mini_agent.tools.sim_trade_tool import SimulateTradeTool
from mini_agent.tools.stock_tools import create_a_share_tools
from mini_agent.tools.skill_tool import create_skill_tools
from mini_agent.tools.kline_db_tool import KLineDB
from mini_agent.utils import calculate_display_width


# ANSI color codes
class Colors:
    """Terminal color definitions"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


def get_log_directory() -> Path:
    """Get the log directory path."""
    return Path(__file__).resolve().parent.parent / "log"


def setup_logging() -> None:
    """Initialize process logging for tool/backend diagnostics."""
    level_name = os.getenv("MINI_AGENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def show_log_directory(open_file_manager: bool = True) -> None:
    """Show log directory contents and optionally open file manager.

    Args:
        open_file_manager: Whether to open the system file manager
    """
    log_dir = get_log_directory()

    print(f"\n{Colors.BRIGHT_CYAN}📁 Log Directory: {log_dir}{Colors.RESET}")

    if not log_dir.exists() or not log_dir.is_dir():
        print(f"{Colors.RED}Log directory does not exist: {log_dir}{Colors.RESET}\n")
        return

    log_files = list(log_dir.glob("*.log"))

    if not log_files:
        print(f"{Colors.YELLOW}No log files found in directory.{Colors.RESET}\n")
        return

    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BRIGHT_YELLOW}Available Log Files (newest first):{Colors.RESET}")

    for i, log_file in enumerate(log_files[:10], 1):
        mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
        size = log_file.stat().st_size
        size_str = f"{size:,}" if size < 1024 else f"{size / 1024:.1f}K"
        print(f"  {Colors.GREEN}{i:2d}.{Colors.RESET} {Colors.BRIGHT_WHITE}{log_file.name}{Colors.RESET}")
        print(f"      {Colors.DIM}Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}, Size: {size_str}{Colors.RESET}")

    if len(log_files) > 10:
        print(f"  {Colors.DIM}... and {len(log_files) - 10} more files{Colors.RESET}")

    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")

    # Open file manager
    if open_file_manager:
        _open_directory_in_file_manager(log_dir)

    print()


def _open_directory_in_file_manager(directory: Path) -> None:
    """Open directory in system file manager (cross-platform)."""
    system = platform.system()

    try:
        if system == "Darwin":
            subprocess.run(["open", str(directory)], check=False)
        elif system == "Windows":
            subprocess.run(["explorer", str(directory)], check=False)
        elif system == "Linux":
            subprocess.run(["xdg-open", str(directory)], check=False)
    except FileNotFoundError:
        print(f"{Colors.YELLOW}Could not open file manager. Please navigate manually.{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.YELLOW}Error opening file manager: {e}{Colors.RESET}")


def read_log_file(filename: str) -> None:
    """Read and display a specific log file.

    Args:
        filename: The log filename to read
    """
    log_dir = get_log_directory()
    log_file = log_dir / filename

    if not log_file.exists() or not log_file.is_file():
        print(f"\n{Colors.RED}❌ Log file not found: {log_file}{Colors.RESET}\n")
        return

    print(f"\n{Colors.BRIGHT_CYAN}📄 Reading: {log_file}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}")

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        print(content)
        print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}")
        print(f"\n{Colors.GREEN}✅ End of file{Colors.RESET}\n")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error reading file: {e}{Colors.RESET}\n")


def print_banner():
    """Print welcome banner with proper alignment"""
    BOX_WIDTH = 58
    banner_text = f"{Colors.BOLD}🤖 Mini Agent - Multi-turn Interactive Session{Colors.RESET}"
    banner_width = calculate_display_width(banner_text)

    # Center the text with proper padding
    total_padding = BOX_WIDTH - banner_width
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding

    print()
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}╔{'═' * BOX_WIDTH}╗{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.BRIGHT_CYAN}║{Colors.RESET}{' ' * left_padding}{banner_text}{' ' * right_padding}{Colors.BOLD}{Colors.BRIGHT_CYAN}║{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}╚{'═' * BOX_WIDTH}╝{Colors.RESET}")
    print()


def print_help():
    """Print help information"""
    help_text = f"""
{Colors.BOLD}{Colors.BRIGHT_YELLOW}Available Commands:{Colors.RESET}
  {Colors.BRIGHT_GREEN}/help{Colors.RESET}      - Show this help message
  {Colors.BRIGHT_GREEN}/clear{Colors.RESET}     - Clear session history (keep system prompt)
  {Colors.BRIGHT_GREEN}/history{Colors.RESET}   - Show current session message count
  {Colors.BRIGHT_GREEN}/stats{Colors.RESET}     - Show session statistics
  {Colors.BRIGHT_GREEN}/log{Colors.RESET}       - Show log directory and recent files
  {Colors.BRIGHT_GREEN}/log <file>{Colors.RESET} - Read a specific log file
  {Colors.BRIGHT_GREEN}/exit{Colors.RESET}      - Exit program (also: exit, quit, q)

{Colors.BOLD}{Colors.BRIGHT_YELLOW}Keyboard Shortcuts:{Colors.RESET}
  {Colors.BRIGHT_CYAN}Esc{Colors.RESET}        - Cancel current agent execution
  {Colors.BRIGHT_CYAN}Ctrl+C{Colors.RESET}     - Exit program
  {Colors.BRIGHT_CYAN}Ctrl+U{Colors.RESET}     - Clear current input line
  {Colors.BRIGHT_CYAN}Ctrl+L{Colors.RESET}     - Clear screen
  {Colors.BRIGHT_CYAN}Ctrl+J{Colors.RESET}     - Insert newline (also Ctrl+Enter)
  {Colors.BRIGHT_CYAN}Tab{Colors.RESET}        - Auto-complete commands
  {Colors.BRIGHT_CYAN}↑/↓{Colors.RESET}        - Browse command history
  {Colors.BRIGHT_CYAN}→{Colors.RESET}          - Accept auto-suggestion

{Colors.BOLD}{Colors.BRIGHT_YELLOW}Usage:{Colors.RESET}
  - Enter your task directly, Agent will help you complete it
  - Agent remembers all conversation content in this session
  - Use {Colors.BRIGHT_GREEN}/clear{Colors.RESET} to start a new session
  - Press {Colors.BRIGHT_CYAN}Enter{Colors.RESET} to submit your message
  - Use {Colors.BRIGHT_CYAN}Ctrl+J{Colors.RESET} to insert line breaks within your message
"""
    print(help_text)


def print_session_info(agent: Agent, workspace_dir: Path, model: str):
    """Print session information with proper alignment"""
    BOX_WIDTH = 58

    def print_info_line(text: str):
        """Print a single info line with proper padding"""
        # Account for leading space
        text_width = calculate_display_width(text)
        padding = max(0, BOX_WIDTH - 1 - text_width)
        print(f"{Colors.DIM}│{Colors.RESET} {text}{' ' * padding}{Colors.DIM}│{Colors.RESET}")

    # Top border
    print(f"{Colors.DIM}┌{'─' * BOX_WIDTH}┐{Colors.RESET}")

    # Header (centered)
    header_text = f"{Colors.BRIGHT_CYAN}Session Info{Colors.RESET}"
    header_width = calculate_display_width(header_text)
    header_padding_total = BOX_WIDTH - 1 - header_width  # -1 for leading space
    header_padding_left = header_padding_total // 2
    header_padding_right = header_padding_total - header_padding_left
    print(f"{Colors.DIM}│{Colors.RESET} {' ' * header_padding_left}{header_text}{' ' * header_padding_right}{Colors.DIM}│{Colors.RESET}")

    # Divider
    print(f"{Colors.DIM}├{'─' * BOX_WIDTH}┤{Colors.RESET}")

    # Info lines
    print_info_line(f"Model: {model}")
    print_info_line(f"Workspace: {workspace_dir}")
    print_info_line(f"Message History: {len(agent.messages)} messages")
    print_info_line(f"Available Tools: {len(agent.tools)} tools")

    # Bottom border
    print(f"{Colors.DIM}└{'─' * BOX_WIDTH}┘{Colors.RESET}")
    print()
    print(f"{Colors.DIM}Type {Colors.BRIGHT_GREEN}/help{Colors.DIM} for help, {Colors.BRIGHT_GREEN}/exit{Colors.DIM} to quit{Colors.RESET}")
    print()


def print_stats(agent: Agent, session_start: datetime):
    """Print session statistics"""
    duration = datetime.now() - session_start
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    # Count different types of messages
    user_msgs = sum(1 for m in agent.messages if m.role == "user")
    assistant_msgs = sum(1 for m in agent.messages if m.role == "assistant")
    tool_msgs = sum(1 for m in agent.messages if m.role == "tool")

    print(f"\n{Colors.BOLD}{Colors.BRIGHT_CYAN}Session Statistics:{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")
    print(f"  Session Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"  Total Messages: {len(agent.messages)}")
    print(f"    - User Messages: {Colors.BRIGHT_GREEN}{user_msgs}{Colors.RESET}")
    print(f"    - Assistant Replies: {Colors.BRIGHT_BLUE}{assistant_msgs}{Colors.RESET}")
    print(f"    - Tool Calls: {Colors.BRIGHT_YELLOW}{tool_msgs}{Colors.RESET}")
    print(f"  Available Tools: {len(agent.tools)}")
    if agent.api_total_tokens > 0:
        print(f"  API Tokens Used: {Colors.BRIGHT_MAGENTA}{agent.api_total_tokens:,}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}\n")


def get_memory_db_path(workspace_dir: Path) -> Path:
    """Get unified SQLite db path."""
    _ = workspace_dir
    return get_default_memory_db_path()


async def _persist_turn_memory(
    tools: List[Tool],
    *,
    session_id: int,
    category: str,
    content: str,
    importance: int = 1,
) -> None:
    """Persist one conversation turn into memories via record_memory tool."""
    if not content.strip():
        return

    recorder = next((t for t in tools if getattr(t, "name", "") == "record_memory"), None)
    if recorder is None:
        return

    try:
        result = await recorder.execute(
            content=content,
            category=category,
            importance=importance,
            session_id=str(session_id),
        )
        if not result.success:
            logging.getLogger(__name__).warning("Memory persist failed (session=%s): %s", session_id, result.error)
    except Exception as exc:  # pragma: no cover - defensive
        logging.getLogger(__name__).exception("Memory persist exception (session=%s): %s", session_id, exc)


def _load_prompt_memories(memory_db_path: Path, session_id: int, recent_limit: int = 12, critical_limit: int = 8) -> list[dict[str, str]]:
    """Load prompt memories with critical memories first, then recent conversation memories."""
    critical = load_critical_session_memories(memory_db_path, session_id=session_id, limit=critical_limit)
    recent = load_recent_session_memories(memory_db_path, session_id=session_id, limit=recent_limit)
    return [*critical, *recent]


def _strategy_templates_path(workspace_dir: Path) -> Path:
    candidate = workspace_dir / "docs" / "strategy_templates.md"
    if candidate.exists():
        return candidate
    return Path(__file__).resolve().parent.parent / "docs" / "strategy_templates.md"


def _load_strategy_templates(workspace_dir: Path) -> list[dict[str, str]]:
    path = _strategy_templates_path(workspace_dir)
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    templates: list[dict[str, str]] = []
    current_id = ""
    current_name = ""
    buffer: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"^##\s+(\d+)\.\s+(.+?)\s*$", stripped)
        if m:
            if current_id and current_name:
                templates.append(
                    {
                        "id": current_id,
                        "name": current_name,
                        "content": "\n".join(buffer).strip(),
                    }
                )
            current_id = m.group(1)
            current_name = m.group(2)
            buffer = []
            continue
        # End current template when entering a non-template level-2 section.
        if current_id and stripped.startswith("## "):
            templates.append(
                {
                    "id": current_id,
                    "name": current_name,
                    "content": "\n".join(buffer).strip(),
                }
            )
            current_id = ""
            current_name = ""
            buffer = []
            continue
        if current_id:
            buffer.append(line)

    if current_id and current_name:
        templates.append(
            {
                "id": current_id,
                "name": current_name,
                "content": "\n".join(buffer).strip(),
            }
        )
    return templates


def _build_prompt_from_template(template: dict[str, str]) -> str:
    name = str(template.get("name") or "策略模板").strip()
    body = str(template.get("content") or "").strip()
    if name == "自由策略":
        return (
            f"你是一个A股自动交易代理。\n\n"
            f"当前策略模板：{name}\n\n"
            f"{body}\n\n"
            f"执行原则：\n"
            f"- 你可以基于当下市场数据自主制定与调整策略。\n"
            f"- 不预设固定买卖规则，由你自行判断是否交易与仓位分配。\n"
            f"- 优先调用工具获取事实数据，再做可执行决策。"
        )
    return (
        f"你是一个严格执行策略纪律的A股交易代理。\n\n"
        f"当前策略模板：{name}\n\n"
        f"{body}\n\n"
        f"执行要求：\n"
        f"- 交易信号必须可落地到工具调用。\n"
        f"- 决策必须包含仓位、止损、止盈、风险提示。\n"
        f"- 参数缺失时先澄清，不允许凭空猜测。"
    )


def _pick_strategy_template(templates: list[dict[str, str]], selector: str) -> dict[str, str] | None:
    key = (selector or "").strip()
    if not key:
        return None
    for item in templates:
        if key == str(item.get("id")):
            return item
    for item in templates:
        if key == str(item.get("name")):
            return item
    return None


def _pick_default_conservative_template(templates: list[dict[str, str]]) -> dict[str, str] | None:
    """Pick conservative default template, prefer 高股息策略 then 龙头股趋势策略."""
    preferred_names = ("高股息策略", "龙头股趋势策略")
    for name in preferred_names:
        matched = _pick_strategy_template(templates, name)
        if matched is not None:
            return matched
    # Fallback by id when current template library keeps high-dividend as #6.
    matched = _pick_strategy_template(templates, "6")
    if matched is not None:
        return matched
    return templates[0] if templates else None


def _prompt_with_default(label: str, default: str | None = None, input_func=input) -> str:
    prompt = f"{label}"
    if default is not None and str(default).strip():
        prompt += f" [{default}]"
    prompt += ": "
    raw = input_func(prompt).strip()
    return raw if raw else str(default or "")


def _collect_session_create_inputs(args: argparse.Namespace, workspace_dir: Path, input_func=input) -> dict[str, Any]:
    """Collect session create parameters from args + interactive prompts."""
    interactive = sys.stdin.isatty()
    templates = _load_strategy_templates(workspace_dir)

    name = args.name
    mode = args.mode or "simulation"
    initial_capital = args.initial_capital
    risk_preference = args.risk_preference or "low"
    max_single_loss_pct = args.max_single_loss_pct
    single_position_cap_pct = args.single_position_cap_pct
    stop_loss_pct = args.stop_loss_pct
    take_profit_pct = args.take_profit_pct
    investment_horizon = args.investment_horizon or "中线"
    event_filter = args.event_filter or []
    prompt = args.prompt
    template_selector = args.template

    if interactive:
        if not name:
            name = _prompt_with_default("Session 名称", datetime.now().strftime("strategy-%Y%m%d-%H%M%S"), input_func)
        if not mode:
            mode = _prompt_with_default("模式 (simulation/backtest)", "simulation", input_func)
        if initial_capital is None:
            initial_capital = float(_prompt_with_default("初始资金", "1000000", input_func))
        if not template_selector and not prompt and templates:
            print("可用策略模板:")
            for item in templates:
                print(f"  {item['id']}. {item['name']}")
            default_template = _pick_default_conservative_template(templates)
            default_selector = str(default_template["id"]) if default_template else ""
            template_selector = _prompt_with_default("选择模板编号(回车默认保守策略)", default_selector, input_func)
        if not prompt and template_selector:
            picked = _pick_strategy_template(templates, str(template_selector))
            if picked:
                prompt = _build_prompt_from_template(picked)
                print(f"{Colors.GREEN}✅ Selected strategy template: {picked['id']}. {picked['name']}{Colors.RESET}")
        if not prompt and templates:
            picked = _pick_default_conservative_template(templates)
            if picked:
                prompt = _build_prompt_from_template(picked)
                print(f"{Colors.GREEN}✅ Selected default conservative template: {picked['id']}. {picked['name']}{Colors.RESET}")
        if not prompt:
            prompt = _prompt_with_default("系统提示词", "", input_func)
        if not risk_preference:
            risk_preference = _prompt_with_default("风险偏好(low/medium/high)", "low", input_func)
        if max_single_loss_pct is None:
            max_single_loss_pct = float(_prompt_with_default("最大单笔亏损(%)", "1.5", input_func))
        if single_position_cap_pct is None:
            single_position_cap_pct = float(_prompt_with_default("单票仓位上限(%)", "15", input_func))
        if stop_loss_pct is None:
            stop_loss_pct = float(_prompt_with_default("止损线(%)", "6", input_func))
        if take_profit_pct is None:
            take_profit_pct = float(_prompt_with_default("止盈线(%)", "12", input_func))
        if not investment_horizon:
            investment_horizon = _prompt_with_default("投资周期", "中线", input_func)
    else:
        if not name:
            name = datetime.now().strftime("strategy-%Y%m%d-%H%M%S")
        if initial_capital is None:
            initial_capital = 1000000.0
        if max_single_loss_pct is None:
            max_single_loss_pct = 1.5
        if single_position_cap_pct is None:
            single_position_cap_pct = 15.0
        if stop_loss_pct is None:
            stop_loss_pct = 6.0
        if take_profit_pct is None:
            take_profit_pct = 12.0
        # non-interactive mode: prompt/template still required (or use default conservative template if available)
        missing = []
        if not prompt and not template_selector:
            picked_default = _pick_default_conservative_template(templates)
            if picked_default is not None:
                prompt = _build_prompt_from_template(picked_default)
            else:
                missing.append("--prompt/--template")
        if missing:
            raise ValueError(f"Non-interactive create missing required args: {', '.join(missing)}")
        if not prompt and template_selector:
            picked = _pick_strategy_template(templates, str(template_selector))
            if not picked:
                raise ValueError(f"Template not found: {template_selector}")
            prompt = _build_prompt_from_template(picked)

    if not prompt:
        raise ValueError("system prompt is required")

    return {
        "name": str(name).strip(),
        "system_prompt": str(prompt),
        "mode": str(mode).strip().lower(),
        "initial_capital": float(initial_capital),
        "risk_preference": str(risk_preference).strip().lower(),
        "max_single_loss_pct": float(max_single_loss_pct),
        "single_position_cap_pct": float(single_position_cap_pct),
        "stop_loss_pct": float(stop_loss_pct),
        "take_profit_pct": float(take_profit_pct),
        "investment_horizon": str(investment_horizon).strip() or "中线",
        "event_filter": event_filter,
    }


def handle_session_command(args: argparse.Namespace, workspace_dir: Path) -> None:
    """Handle session subcommands."""
    manager = SessionManager(db_path=str(get_memory_db_path(workspace_dir)))

    if args.session_command == "create":
        payload = _collect_session_create_inputs(args, workspace_dir)
        session_id = manager.create_session(
            name=payload["name"],
            system_prompt=payload["system_prompt"],
            mode=payload["mode"],
            initial_capital=payload["initial_capital"],
            risk_preference=payload["risk_preference"],
            max_single_loss_pct=payload["max_single_loss_pct"],
            single_position_cap_pct=payload["single_position_cap_pct"],
            stop_loss_pct=payload["stop_loss_pct"],
            take_profit_pct=payload["take_profit_pct"],
            investment_horizon=payload["investment_horizon"],
            event_filter=payload["event_filter"],
        )
        print(f"{Colors.GREEN}✅ Session created{Colors.RESET}")
        print(f"session_id: {session_id}")
        print(f"name: {payload['name']}")
        print(f"mode: {payload['mode']}")
        print(f"initial_capital: {payload['initial_capital']:.2f}")
        print(f"risk_preference: {payload['risk_preference']}")
        print(f"max_single_loss_pct: {payload['max_single_loss_pct']:.2f}")
        print(f"single_position_cap_pct: {payload['single_position_cap_pct']:.2f}")
        print(f"stop_loss_pct: {payload['stop_loss_pct']:.2f}")
        print(f"take_profit_pct: {payload['take_profit_pct']:.2f}")
        print(f"investment_horizon: {payload['investment_horizon']}")
        return

    if args.session_command == "list":
        sessions = manager.list_sessions()
        if not sessions:
            print(f"{Colors.YELLOW}No sessions found.{Colors.RESET}")
            return
        online_session_ids = set(manager.list_online_session_ids())
        print("session_id\tname\tmode\tstatus\tlistening")
        for session in sessions:
            is_online = session.session_id in online_session_ids
            status = "running" if is_online else "stopped"
            listening = "yes" if is_online else "no"
            print(f"{session.session_id}\t{session.name}\t{session.mode}\t{status}\t{listening}")
        return

    try:
        if args.session_command == "start":
            manager.start_session(args.session_id)
            print(f"{Colors.GREEN}✅ Session started: {args.session_id}{Colors.RESET}")
        elif args.session_command == "stop":
            manager.stop_session(args.session_id)
            print(f"{Colors.GREEN}✅ Session stopped: {args.session_id}{Colors.RESET}")
        elif args.session_command == "update":
            updated = False
            if args.prompt is not None:
                manager.update_system_prompt(args.session_id, args.prompt)
                updated = True
            manager.update_risk_profile(
                args.session_id,
                risk_preference=args.risk_preference,
                max_single_loss_pct=args.max_single_loss_pct,
                single_position_cap_pct=args.single_position_cap_pct,
                stop_loss_pct=args.stop_loss_pct,
                take_profit_pct=args.take_profit_pct,
                investment_horizon=args.investment_horizon,
            )
            if any(
                value is not None
                for value in (
                    args.risk_preference,
                    args.max_single_loss_pct,
                    args.single_position_cap_pct,
                    args.stop_loss_pct,
                    args.take_profit_pct,
                    args.investment_horizon,
                )
            ):
                updated = True
            if not updated:
                print(f"{Colors.YELLOW}⚠️ No fields provided. Nothing updated.{Colors.RESET}")
                return
            print(f"{Colors.GREEN}✅ Session updated: {args.session_id}{Colors.RESET}")
        elif args.session_command == "delete":
            manager.delete_session(args.session_id)
            print(f"{Colors.GREEN}✅ Session deleted: {args.session_id}{Colors.RESET}")
    except KeyError as exc:
        print(f"{Colors.RED}❌ {exc}{Colors.RESET}")
    except ValueError as exc:
        print(f"{Colors.RED}❌ {exc}{Colors.RESET}")


def _read_sim_positions(db_path: Path, session_id: str) -> list[sqlite3.Row]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT ticker, quantity, avg_cost
                FROM sim_positions
                WHERE session_id = ?
                ORDER BY ticker
                """,
                (session_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return list(rows)


def _read_realized_profit(db_path: Path, session_id: str) -> float:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(profit), 0) AS realized FROM sim_trades WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return 0.0
    return float(row["realized"] if row else 0.0)


def _resolve_trade_date_for_execution(memory_db_path: Path) -> str:
    """Prefer latest available kline date for deterministic simulation execution."""
    with sqlite3.connect(memory_db_path) as conn:
        row = conn.execute("SELECT MAX(date) AS d FROM daily_kline").fetchone()
    if row and row[0]:
        return str(row[0])
    return datetime.now().date().isoformat()


def _parse_direct_trade_order(user_input: str) -> dict | None:
    """Parse explicit natural-language buy/sell instruction from user input."""
    text = (user_input or "").strip()
    if not text:
        return None

    buy_keywords = ("买入", "购买", "买")
    sell_keywords = ("卖出", "卖", "减仓")
    action = None
    if any(k in text for k in buy_keywords):
        action = "buy"
    elif any(k in text for k in sell_keywords):
        action = "sell"
    if action is None:
        return None

    qty_match = re.search(r"(\d+)\s*股", text)
    if not qty_match:
        return None
    quantity = int(qty_match.group(1))
    if quantity <= 0:
        return None

    ticker_match = re.search(r"\b(\d{6})(?:\.(?:SH|SZ))?\b", text, flags=re.IGNORECASE)
    ticker = ticker_match.group(1) if ticker_match else ""
    if not ticker:
        alias_map = {
            "贵州茅台": "600519",
            "茅台": "600519",
            "平安银行": "000001",
            "中国平安": "601318",
            "宁德时代": "300750",
        }
        for alias, code in alias_map.items():
            if alias in text:
                ticker = code
                break

    if not ticker:
        return None
    date_match = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
    trade_date = date_match.group(1) if date_match else None

    return {"action": action, "ticker": ticker, "quantity": quantity, "trade_date": trade_date}


def _parse_sim_trade_tool_content(content: str) -> dict[str, Any]:
    """Parse simulate_trade tool output text into structured fields."""
    text = str(content or "").strip()
    pairs = dict(re.findall(r"([a-zA-Z_]+)=([^\s]+)", text))
    action = str(pairs.get("action") or "").strip().lower() or None
    ticker = str(pairs.get("ticker") or "").strip() or None
    quantity_raw = str(pairs.get("qty") or "").strip()
    quantity = int(quantity_raw) if quantity_raw.isdigit() else None
    return {
        "raw": text,
        "action": action,
        "ticker": ticker,
        "quantity": quantity,
        "price_source": pairs.get("price_source"),
    }


def _record_sim_trade_critical_from_messages(
    *,
    session_manager: SessionManager,
    session_id: int,
    messages: list[Any],
    event_type: str,
    event_id: str | None = None,
) -> None:
    """Persist critical memories for simulate_trade tool calls in a message slice."""
    for msg in messages:
        if msg.role != "tool" or str(msg.name or "") != "simulate_trade":
            continue
        parsed = _parse_sim_trade_tool_content(str(msg.content or ""))
        raw = str(parsed.get("raw") or "")
        if "SIM_TRADE_OK" in raw:
            op = str(parsed.get("action") or "trade")
        else:
            action = str(parsed.get("action") or "").strip()
            op = f"{action}_failed" if action else "trade_failed"
        session_manager.record_critical_memory(
            session_id=session_id,
            event_id=(event_id or None),
            event_type=event_type,
            operation=op,
            reason="simulate_trade_tool",
            content=json.dumps(parsed, ensure_ascii=False),
        )


def handle_trade_command(args: argparse.Namespace, workspace_dir: Path) -> None:
    """Handle trade subcommands."""
    db_path = get_memory_db_path(workspace_dir)
    kline_db_path = resolve_kline_db_path(workspace_dir)
    trade_tool = SimulateTradeTool(
        db_path=str(db_path),
        kline_db_path=str(kline_db_path) if kline_db_path.exists() else None,
    )

    if args.trade_command in {"buy", "sell"}:
        result = asyncio.run(
            trade_tool.execute(
                session_id=args.session_id,
                action=args.trade_command,
                ticker=args.ticker,
                quantity=args.quantity,
                trade_date=args.trade_date,
            )
        )
        if result.success:
            print(f"{Colors.GREEN}{result.content}{Colors.RESET}")
        else:
            print(f"{Colors.RED}❌ {result.error}{Colors.RESET}")
        return

    try:
        session = trade_tool.session_manager.get_session(args.session_id)
    except KeyError as exc:
        print(f"{Colors.RED}❌ {exc}{Colors.RESET}")
        return

    positions = _read_sim_positions(db_path=db_path, session_id=args.session_id)

    if args.trade_command == "positions":
        if not positions:
            print("No open positions.")
            return
        print(f"session: {args.session_id}")
        print(f"cash: {session.current_cash:.2f}")
        print("ticker\tquantity\tavg_cost")
        for row in positions:
            print(f"{row['ticker']}\t{int(row['quantity'])}\t{float(row['avg_cost']):.3f}")
        return

    if args.trade_command == "profit":
        kline_db = KLineDB(db_path=str(kline_db_path))
        realized = _read_realized_profit(db_path=db_path, session_id=args.session_id)
        unrealized = 0.0
        market_value = 0.0
        missing_prices: list[str] = []

        for row in positions:
            ticker = str(row["ticker"])
            quantity = int(row["quantity"])
            avg_cost = float(row["avg_cost"])
            try:
                latest = kline_db.get_latest_price(ticker)
            except KeyError:
                missing_prices.append(ticker)
                continue
            market_value += latest * quantity
            unrealized += (latest - avg_cost) * quantity

        total_assets = session.current_cash + market_value
        total_profit = total_assets - session.initial_capital
        total_return_pct = (total_profit / session.initial_capital * 100) if session.initial_capital else 0.0

        print(f"session: {args.session_id}")
        print(f"initial_capital: {session.initial_capital:.2f}")
        print(f"cash: {session.current_cash:.2f}")
        print(f"market_value: {market_value:.2f}")
        print(f"realized_profit: {realized:.2f}")
        print(f"unrealized_profit: {unrealized:.2f}")
        print(f"total_profit: {total_profit:.2f}")
        print(f"total_return: {total_return_pct:.2f}%")
        if missing_prices:
            print(
                f"{Colors.YELLOW}⚠️ Missing latest price for: {', '.join(missing_prices)}; "
                f"their unrealized pnl is excluded.{Colors.RESET}"
            )
        return

    if args.trade_command == "action-logs":
        rows = trade_tool.session_manager.list_trade_action_logs(args.session_id, limit=int(args.limit))
        if not rows:
            print("No trade action logs.")
            return
        print("id\tevent_id\taction\tticker\tqty\tdate\tstatus\terror_code\treason")
        for row in rows:
            print(
                f"{row.get('id')}\t{row.get('event_id') or ''}\t{row.get('action') or ''}\t"
                f"{row.get('ticker') or ''}\t{row.get('quantity') or ''}\t{row.get('trade_date') or ''}\t"
                f"{row.get('status') or ''}\t{row.get('error_code') or ''}\t{row.get('reason') or ''}"
            )
        return


async def _default_event_trigger(session, event: dict) -> str:
    """Default event callback used by CLI trigger command."""
    return f"TRIGGERED session={session.session_id} event={event.get('type')}"


def _build_event_from_args(args: argparse.Namespace) -> dict:
    """Build normalized event payload from CLI args."""
    event = {"type": args.event_type, "triggered_at": datetime.now().isoformat()}
    if args.payload:
        try:
            payload = json.loads(args.payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON payload: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError("--payload must be a JSON object")

        event.update(payload)
    if "date" not in event:
        event["date"] = datetime.now().date().isoformat()
    if "event_id" not in event or not str(event.get("event_id")).strip():
        event["event_id"] = f"{event.get('type')}:{event.get('date')}"
    return event


def _match_single_session_for_event(session, event_type: str | None) -> tuple[bool, str | None]:
    """Check whether one session should receive an event."""
    if not session.is_listening:
        return False, "session is not listening; run `mini-agent session start <id>` first"

    if event_type and session.event_filter and event_type not in session.event_filter:
        return False, f"session event_filter does not include '{event_type}'"

    return True, None


async def _run_auto_event_for_sessions(runtime, session_ids: list[int], trading_date: str) -> list[dict]:
    """Run auto decision for multiple sessions concurrently."""
    tasks = [
        asyncio.create_task(
            run_llm_decision(
                runtime=runtime,
                session_id=session_id,
                trading_date=trading_date,
                event_id=f"daily_review:{trading_date}",
            )
        )
        for session_id in session_ids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    records: list[dict] = []
    for session_id, result in zip(session_ids, results):
        if isinstance(result, Exception):
            records.append({"session_id": session_id, "success": False, "error": str(result)})
            continue
        records.append({"session_id": session_id, "success": True, "result": result})
    return records


def handle_backtest_command(args: argparse.Namespace, workspace_dir: Path) -> None:
    """Handle backtest subcommands."""
    from .backtest import BacktestEngine, PerformanceAnalyzer

    try:
        runtime = build_decision_runtime(workspace_dir, get_memory_db_path(workspace_dir))
    except FileNotFoundError as exc:
        print(f"{Colors.RED}❌ {exc}{Colors.RESET}")
        return

    async def _llm_backtest_trigger(session, event: dict) -> str:
        result = await run_llm_decision(
            runtime=runtime,
            session_id=session.session_id,
            trading_date=str(event.get("date") or ""),
            event_id=str(event.get("event_id") or "") or None,
        )
        if result.get("execution_error"):
            return f"LLM_DECISION_FAIL {result['execution_error']}"
        if result.get("execution"):
            return f"LLM_DECISION_EXECUTED {result['execution']}"
        return f"LLM_DECISION_NO_TRADE {result.get('agent_analysis', '')[:120]}"

    async def _mock_backtest_trigger(session, event: dict) -> str:
        _ = session
        return f"MOCK_DECISION_NO_TRADE date={event.get('date')}"

    trigger = _mock_backtest_trigger if getattr(args, "no_llm", False) else _llm_backtest_trigger
    broadcaster = EventBroadcaster(session_manager=runtime.session_manager, trigger=trigger)
    engine = BacktestEngine(runtime.session_manager, runtime.kline_db, broadcaster)

    if args.backtest_command == "run":
        import random

        random.seed(args.seed)
        print(f"{Colors.CYAN}🔄 Running backtest...{Colors.RESET}")
        print(f"session: {args.session}")
        print(f"period: {args.start} ~ {args.end}")
        print(f"seed: {args.seed}")
        if getattr(args, "no_llm", False):
            print(f"{Colors.YELLOW}mode: mock decision (--no-llm){Colors.RESET}")
        else:
            print("mode: llm decision")

        result = asyncio.run(engine.run(args.session, args.start, args.end))

        if "error" in result:
            print(f"{Colors.RED}❌ {result['error']}{Colors.RESET}")
            return

        perf = result["performance"]
        print(f"\n{Colors.GREEN}✅ Backtest completed!{Colors.RESET}")
        print(f"\n{'='*40}")
        print(f"{Colors.BOLD}Performance Summary{Colors.RESET}")
        print(f"{'='*40}")
        print(f"Period:        {result['start_date']} ~ {result['end_date']}")
        print(f"Trading Days:  {result['trading_days']}")
        print(f"Total Trades:  {perf.get('total_trades', 0)}")
        print(f"{'-'*40}")
        print(f"Initial:       ¥{perf.get('initial_capital', 0):,.2f}")
        print(f"Final:         ¥{perf.get('final_value', 0):,.2f}")
        print(f"Total Return:  {perf.get('total_return', 0)*100:.2f}%")
        print(f"Annual Return: {perf.get('annual_return', 0)*100:.2f}%")
        print(f"Sharpe:       {perf.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {perf.get('max_drawdown', 0)*100:.2f}%")
        print(f"Win Rate:     {perf.get('win_rate', 0)*100:.1f}%")
        print(f"Profit Factor:{perf.get('profit_factor', 0):.2f}")
        print(f"{'='*40}")

        if getattr(args, "gate", False):
            gate_failures: list[str] = []
            total_trades = int(perf.get("total_trades", 0) or 0)
            win_rate = float(perf.get("win_rate", 0.0) or 0.0)
            max_drawdown = float(perf.get("max_drawdown", 0.0) or 0.0)
            profit_factor = float(perf.get("profit_factor", 0.0) or 0.0)

            if total_trades < int(args.gate_min_trades):
                gate_failures.append(f"total_trades {total_trades} < {int(args.gate_min_trades)}")
            if win_rate < float(args.gate_min_win_rate):
                gate_failures.append(f"win_rate {win_rate:.3f} < {float(args.gate_min_win_rate):.3f}")
            if max_drawdown > float(args.gate_max_drawdown):
                gate_failures.append(f"max_drawdown {max_drawdown:.3f} > {float(args.gate_max_drawdown):.3f}")
            if profit_factor < float(args.gate_min_profit_factor):
                gate_failures.append(
                    f"profit_factor {profit_factor:.3f} < {float(args.gate_min_profit_factor):.3f}"
                )

            if gate_failures:
                print(f"{Colors.RED}❌ Backtest gate failed{Colors.RESET}")
                for line in gate_failures:
                    print(f"- {line}")
                raise SystemExit(2)
            print(f"{Colors.GREEN}✅ Backtest gate passed{Colors.RESET}")

    elif args.backtest_command == "result":
        # Show stored backtest result
        print(f"{Colors.CYAN}Backtest result for session: {args.session}{Colors.RESET}")
        # For now, just show session info
        try:
            session = session_manager.get_session(args.session)
            print(f"Mode: {session.mode}")
            print(f"Status: {session.status}")
            print(f"Initial capital: ¥{session.initial_capital:,.2f}")
            print(f"Current cash: ¥{session.current_cash:,.2f}")
        except KeyError:
            print(f"{Colors.RED}❌ Session not found: {args.session}{Colors.RESET}")





def handle_event_command(args: argparse.Namespace, workspace_dir: Path) -> None:
    """Handle event subcommands."""
    if args.event_command != "trigger":
        return

    try:
        event = _build_event_from_args(args)
    except ValueError as exc:
        print(f"{Colors.RED}❌ {exc}{Colors.RESET}")
        return
    if getattr(args, "debug", False):
        base_event_id = str(event.get("event_id") or f"{event.get('type')}:{event.get('date')}")
        nonce = uuid.uuid4().hex[:8]
        event["event_id"] = f"{base_event_id}:debug:{datetime.now().strftime('%Y%m%d%H%M%S')}{nonce}"
        print(f"{Colors.YELLOW}⚠️ debug mode enabled: idempotency bypass via unique event_id{Colors.RESET}")

    manager = SessionManager(db_path=str(get_memory_db_path(workspace_dir)))
    if args.all_sessions:
        target_session_ids = manager.list_online_session_ids()
    else:
        try:
            manager.get_session(args.session_id)
        except KeyError as exc:
            print(f"{Colors.RED}❌ {exc}{Colors.RESET}")
            return
        if not manager.is_session_online(args.session_id):
            print(f"event_type: {event.get('type')}")
            print(f"session: {args.session_id}")
            print(f"{Colors.YELLOW}⚠️ skipped: session runtime is offline{Colors.RESET}")
            return
        target_session_ids = [int(args.session_id)]

    print(f"event_type: {event.get('type')}")
    print(f"target_sessions: {len(target_session_ids)}")
    if not target_session_ids:
        print(f"{Colors.YELLOW}⚠️ No online session runtimes found.{Colors.RESET}")
        return

    queued = 0
    duplicated = 0
    for sid in target_session_ids:
        inserted, event_id = manager.enqueue_event(sid, event)
        if inserted:
            queued += 1
            print(f"- {sid}: queued event_id={event_id}")
        else:
            duplicated += 1
            print(f"- {sid}: duplicate event_id={event_id} (ignored)")

    print(f"queued={queued}, duplicate={duplicated}")


def handle_evolve_command(args: argparse.Namespace, workspace_dir: Path) -> None:
    """Handle low-coupling evolution use-case commands."""
    service = EvolutionUseCaseService(db_path=str(get_memory_db_path(workspace_dir)))
    cmd = args.evolve_command
    if cmd == "scan":
        records: list[dict[str, Any]] = []
        if args.use_llm:
            try:
                cfg = Config.from_yaml(Config.get_default_config_path())
                llm = LLMClient(
                    api_key=cfg.llm.api_key,
                    provider=LLMProvider(cfg.llm.provider),
                    api_base=cfg.llm.api_base,
                    model=cfg.llm.model,
                    retry_config=cfg.llm.retry,
                )
                records = asyncio.run(service.generate_use_cases_with_llm(llm, limit=args.limit))
            except Exception as exc:
                print(f"{Colors.YELLOW}⚠️ LLM use-case generation failed, fallback to rule templates: {exc}{Colors.RESET}")
                records = []
        if not records:
            records = service.generate_use_cases_with_rules(limit=args.limit)
        if not records:
            print(f"{Colors.YELLOW}No new use cases generated.{Colors.RESET}")
            return
        print(f"{Colors.GREEN}Generated use cases: {len(records)}{Colors.RESET}")
        for item in records:
            print(
                f"- {item['use_case_id']}\tissue={item['issue_type']}\tcount={item['count']}\t{item['title']}"
            )
        return

    if cmd == "trace-sync":
        result = service.ingest_all_intercept_logs(session_id=args.session_id)
        print(
            f"{Colors.GREEN}trace synced: files={result['files']} inserted={result['inserted']} skipped={result['skipped']} errors={result['errors']}{Colors.RESET}"
        )
        return

    if cmd == "trace-summary":
        summary = service.trace_summary(session_id=args.session_id, limit=args.limit)
        print(f"total_rows: {summary['total_rows']}")
        print("events:")
        for row in summary["events"]:
            print(f"- {row['event']}: {row['cnt']}")
        print("tool_failures:")
        for row in summary["tool_failures"]:
            print(f"- {row.get('tool_name') or 'unknown'}: {row['fail_count']}")
        return

    if cmd == "list":
        enabled_filter = None
        if args.enabled == "on":
            enabled_filter = 1
        elif args.enabled == "off":
            enabled_filter = 0
        rows = service.list_use_cases(enabled=enabled_filter, limit=args.limit)
        if not rows:
            print(f"{Colors.YELLOW}No use cases found.{Colors.RESET}")
            return
        print("use_case_id\tenabled\tissue_type\ttitle")
        for row in rows:
            print(
                f"{row['use_case_id']}\t{('on' if int(row['enabled']) == 1 else 'off')}\t{row['issue_type']}\t{row['title']}"
            )
        return

    try:
        if cmd == "enable":
            service.set_use_case_enabled(args.use_case_id, True)
            print(f"{Colors.GREEN}✅ Enabled use_case: {args.use_case_id}{Colors.RESET}")
            return
        if cmd == "disable":
            service.set_use_case_enabled(args.use_case_id, False)
            print(f"{Colors.GREEN}✅ Disabled use_case: {args.use_case_id}{Colors.RESET}")
            return
        if cmd == "prompt":
            block = service.render_prompt_block(limit=args.limit)
            if not block:
                print(f"{Colors.YELLOW}No enabled use cases for prompt injection.{Colors.RESET}")
                return
            print("[Execution Use Cases]")
            print(block)
            return
    except (KeyError, ValueError) as exc:
        print(f"{Colors.RED}❌ {exc}{Colors.RESET}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Mini Agent - AI assistant with file tools and MCP support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mini-agent                              # Use current directory as workspace
  mini-agent --workspace /path/to/dir     # Use specific workspace directory
  mini-agent log                          # Show log directory and recent files
  mini-agent log agent_run_xxx.log        # Read a specific log file
  mini-agent session create --name test --prompt "..." --mode simulation
  mini-agent trade buy 600519 100 --session <id>
  mini-agent event trigger daily_review --all
        """,
    )
    parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        default=None,
        help="Workspace directory (default: current directory)",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default=None,
        help="Execute a task non-interactively and exit",
    )
    parser.add_argument(
        "--session-id",
        type=int,
        default=None,
        help="Attach to an existing session ID instead of creating a new runtime session",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="mini-agent 0.1.0",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # log subcommand
    log_parser = subparsers.add_parser("log", help="Show log directory or read log files")
    log_parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Log filename to read (optional, shows directory if omitted)",
    )

    # session subcommands
    session_parser = subparsers.add_parser("session", help="Manage experiment sessions")
    session_subparsers = session_parser.add_subparsers(dest="session_command", help="Session actions")

    session_create = session_subparsers.add_parser("create", help="Create a session")
    session_create.add_argument("--name", required=False, default=None, help="Session name (interactive if omitted)")
    session_create.add_argument("--prompt", required=False, default=None, help="System prompt (interactive if omitted)")
    session_create.add_argument(
        "--template",
        required=False,
        default=None,
        help="Strategy template id/name from docs/strategy_templates.md",
    )
    session_create.add_argument(
        "--mode",
        default=None,
        choices=["simulation", "backtest"],
        help="Session mode",
    )
    session_create.add_argument(
        "--initial-capital",
        "--capital",
        dest="initial_capital",
        type=float,
        default=None,
        help="Initial capital (interactive if omitted)",
    )
    session_create.add_argument(
        "--event-filter",
        nargs="*",
        default=[],
        help="Optional event types this session listens to",
    )
    session_create.add_argument(
        "--risk-preference",
        choices=["low", "medium", "high"],
        default=None,
        help="Session risk preference (interactive if omitted)",
    )
    session_create.add_argument(
        "--max-single-loss-pct",
        type=float,
        default=None,
        help="Max single trade loss percent (interactive if omitted)",
    )
    session_create.add_argument(
        "--single-position-cap-pct",
        type=float,
        default=None,
        help="Single position cap percent (interactive if omitted)",
    )
    session_create.add_argument(
        "--stop-loss-pct",
        type=float,
        default=None,
        help="Stop loss percent (interactive if omitted)",
    )
    session_create.add_argument(
        "--take-profit-pct",
        type=float,
        default=None,
        help="Take profit percent (interactive if omitted)",
    )
    session_create.add_argument(
        "--investment-horizon",
        default=None,
        help="Investment horizon label, e.g. 短线/中线/长线 (interactive if omitted)",
    )

    session_subparsers.add_parser("list", help="List sessions")

    session_start = session_subparsers.add_parser("start", help="Start one session")
    session_start.add_argument("session_id", type=int, help="Session ID")

    session_stop = session_subparsers.add_parser("stop", help="Stop one session")
    session_stop.add_argument("session_id", type=int, help="Session ID")

    session_update = session_subparsers.add_parser("update", help="Update session prompt/risk profile")
    session_update.add_argument("session_id", type=int, help="Session ID")
    session_update.add_argument("--prompt", default=None, help="New system prompt (optional)")
    session_update.add_argument(
        "--risk-preference",
        choices=["low", "medium", "high"],
        default=None,
        help="Session risk preference",
    )
    session_update.add_argument("--max-single-loss-pct", type=float, default=None, help="Max single trade loss percent")
    session_update.add_argument("--single-position-cap-pct", type=float, default=None, help="Single position cap percent")
    session_update.add_argument("--stop-loss-pct", type=float, default=None, help="Stop loss percent")
    session_update.add_argument("--take-profit-pct", type=float, default=None, help="Take profit percent")
    session_update.add_argument("--investment-horizon", default=None, help="Investment horizon label")

    session_delete = session_subparsers.add_parser("delete", help="Delete one session")
    session_delete.add_argument("session_id", type=int, help="Session ID")

    # trade subcommands
    trade_parser = subparsers.add_parser("trade", help="Simulation trade commands")
    trade_subparsers = trade_parser.add_subparsers(dest="trade_command", help="Trade actions")

    trade_buy = trade_subparsers.add_parser("buy", help="Buy one ticker")
    trade_buy.add_argument("ticker", help="Ticker")
    trade_buy.add_argument("quantity", type=int, help="Quantity")
    trade_buy.add_argument("--session", required=True, dest="session_id", type=int, help="Session ID")
    trade_buy.add_argument(
        "--date",
        dest="trade_date",
        default=datetime.now().date().isoformat(),
        help="Trade date for execution price lookup (YYYY-MM-DD)",
    )

    trade_sell = trade_subparsers.add_parser("sell", help="Sell one ticker")
    trade_sell.add_argument("ticker", help="Ticker")
    trade_sell.add_argument("quantity", type=int, help="Quantity")
    trade_sell.add_argument("--session", required=True, dest="session_id", type=int, help="Session ID")
    trade_sell.add_argument(
        "--date",
        dest="trade_date",
        default=datetime.now().date().isoformat(),
        help="Trade date for execution price lookup (YYYY-MM-DD)",
    )

    trade_positions = trade_subparsers.add_parser("positions", help="Show current positions")
    trade_positions.add_argument("--session", required=True, dest="session_id", type=int, help="Session ID")

    trade_profit = trade_subparsers.add_parser("profit", help="Show profit summary")
    trade_profit.add_argument("--session", required=True, dest="session_id", type=int, help="Session ID")

    trade_action_logs = trade_subparsers.add_parser("action-logs", help="Show trade action execution logs")
    trade_action_logs.add_argument("--session", required=True, dest="session_id", type=int, help="Session ID")
    trade_action_logs.add_argument("--limit", type=int, default=20, help="Max rows to show")

    # backtest subcommands
    backtest_parser = subparsers.add_parser("backtest", help="Backtest commands")
    backtest_subparsers = backtest_parser.add_subparsers(dest="backtest_command", help="Backtest actions")

    backtest_run = backtest_subparsers.add_parser("run", help="Run backtest")
    backtest_run.add_argument("--session", required=True, type=int, help="Session ID for backtest")
    backtest_run.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    backtest_run.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    backtest_run.add_argument("--seed", type=int, default=42, help="Random seed for reproducible local runs")
    backtest_run.add_argument(
        "--no-llm",
        action="store_true",
        help="Run backtest with mock decision (no external LLM call, for fast local verification)",
    )
    backtest_run.add_argument("--gate", action="store_true", help="Enable quality gate check after backtest")
    backtest_run.add_argument("--gate-min-trades", type=int, default=1, help="Gate: minimum total trades")
    backtest_run.add_argument("--gate-min-win-rate", type=float, default=0.30, help="Gate: minimum win rate (0~1)")
    backtest_run.add_argument("--gate-max-drawdown", type=float, default=0.35, help="Gate: maximum drawdown (0~1)")
    backtest_run.add_argument("--gate-min-profit-factor", type=float, default=0.8, help="Gate: minimum profit factor")

    backtest_result = backtest_subparsers.add_parser("result", help="Show backtest result")
    backtest_result.add_argument("--session", required=True, type=int, help="Session ID")

    # event subcommands
    event_parser = subparsers.add_parser("event", help="Event broadcasting commands")
    event_subparsers = event_parser.add_subparsers(dest="event_command", help="Event actions")

    event_trigger = event_subparsers.add_parser("trigger", help="Trigger one event")
    event_trigger.add_argument("event_type", help="Event type, e.g. daily_review")
    target_group = event_trigger.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--all", action="store_true", dest="all_sessions", help="Enqueue to all online CLI sessions")
    target_group.add_argument("--session", dest="session_id", type=int, help="Enqueue to one online session")
    event_trigger.add_argument("--payload", default=None, help="Optional JSON payload object")
    event_trigger.add_argument("--debug", action="store_true", help="Debug mode: bypass duplicate check by forcing unique event_id")

    # health check command
    health_parser = subparsers.add_parser("health", help="Health check commands")
    health_subparsers = health_parser.add_subparsers(dest="health_command", help="Health actions")
    health_check = health_subparsers.add_parser("check", help="Run health check")

    # sync command
    sync_parser = subparsers.add_parser("sync", help="Sync A-share daily K-line data")
    sync_parser.add_argument("tickers", nargs="?", default="", help="Comma-separated tickers, e.g. 600519,000001")
    sync_parser.add_argument("--all", action="store_true", help="Sync all A-share tickers")
    sync_parser.add_argument("--start", default="1991-01-01", help="Start date YYYY-MM-DD")
    sync_parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    sync_parser.add_argument("--cron", action="store_true", help="Print recommended crontab entries")
    sync_parser.add_argument("--install-cron", action="store_true", help="Install/update recommended crontab entries")

    # evolve command
    evolve_parser = subparsers.add_parser("evolve", help="Execution use-case prompt management")
    evolve_subparsers = evolve_parser.add_subparsers(dest="evolve_command", help="Evolution actions")

    evolve_scan = evolve_subparsers.add_parser("scan", help="Scan runtime data and generate execution use cases")
    evolve_scan.add_argument("--limit", type=int, default=200, help="Scan recent failed event logs")
    evolve_scan.add_argument("--llm", dest="use_llm", action="store_true", help="Use LLM to generate intelligent proposals")
    evolve_scan.add_argument(
        "--no-llm",
        dest="use_llm",
        action="store_false",
        help="Disable LLM generation and use rule templates only",
    )
    evolve_scan.set_defaults(use_llm=True)

    evolve_list = evolve_subparsers.add_parser("list", help="List use cases")
    evolve_list.add_argument(
        "--enabled",
        default=None,
        choices=["on", "off"],
        help="Optional enabled filter",
    )
    evolve_list.add_argument("--limit", type=int, default=50, help="Maximum use cases to list")

    evolve_enable = evolve_subparsers.add_parser("enable", help="Enable one use case")
    evolve_enable.add_argument("use_case_id", help="Use case ID")

    evolve_disable = evolve_subparsers.add_parser("disable", help="Disable one use case")
    evolve_disable.add_argument("use_case_id", help="Use case ID")

    evolve_prompt = evolve_subparsers.add_parser("prompt", help="Render enabled use cases as one prompt block")
    evolve_prompt.add_argument("--limit", type=int, default=12, help="Maximum enabled use cases in prompt block")

    evolve_trace_sync = evolve_subparsers.add_parser("trace-sync", help="Ingest intercept JSONL into structured reasoning traces")
    evolve_trace_sync.add_argument("--session-id", type=int, default=None, help="Optional one session id")

    evolve_trace_summary = evolve_subparsers.add_parser("trace-summary", help="Show structured reasoning trace summary")
    evolve_trace_summary.add_argument("--session-id", type=int, default=None, help="Optional one session id")
    evolve_trace_summary.add_argument("--limit", type=int, default=20, help="Top items limit")

    return parser.parse_args()


async def initialize_base_tools(config: Config):
    """Initialize base tools (independent of workspace)

    These tools are loaded from package configuration and don't depend on workspace.
    Note: File tools are now workspace-dependent and initialized in add_workspace_tools()

    Args:
        config: Configuration object

    Returns:
        Tuple of (list of tools, skill loader if skills enabled)
    """

    # 学习提示：这里先加载“与工作目录无关”的工具。
    # 例如 MCP/Skills 的连接与发现，不依赖当前 workspace。
    tools = []
    skill_loader = None

    # 1. Bash auxiliary tools (output monitoring and kill)
    # 学习提示：真正执行命令的 BashTool 会在 add_workspace_tools() 里创建，
    # 因为它需要绑定 cwd=workspace_dir。
    # Note: BashTool itself is created in add_workspace_tools() with workspace_dir as cwd
    if config.tools.enable_bash:
        bash_output_tool = BashOutputTool()
        tools.append(bash_output_tool)
        print(f"{Colors.GREEN}✅ Loaded Bash Output tool{Colors.RESET}")

        bash_kill_tool = BashKillTool()
        tools.append(bash_kill_tool)
        print(f"{Colors.GREEN}✅ Loaded Bash Kill tool{Colors.RESET}")

    # 3. Claude Skills (loaded from package directory)
    # 学习提示：这里只注入技能“索引”和 get_skill 工具，完整技能内容按需加载。
    if config.tools.enable_skills:
        print(f"{Colors.BRIGHT_CYAN}Loading Claude Skills...{Colors.RESET}")
        try:
            # Resolve skills directory with priority search
            # Expand ~ to user home directory for portability
            skills_path = Path(config.tools.skills_dir).expanduser()
            if skills_path.is_absolute():
                skills_dir = str(skills_path)
            else:
                # Search in priority order:
                # 1. Current directory (dev mode: ./skills or ./mini_agent/skills)
                # 2. Package directory (installed: site-packages/mini_agent/skills)
                search_paths = [
                    skills_path,  # ./skills for backward compatibility
                    Path("mini_agent") / skills_path,  # ./mini_agent/skills
                    Config.get_package_dir() / skills_path,  # site-packages/mini_agent/skills
                ]

                # Find first existing path
                skills_dir = str(skills_path)  # default
                for path in search_paths:
                    if path.exists():
                        skills_dir = str(path.resolve())
                        break

            skill_tools, skill_loader = create_skill_tools(skills_dir)
            if skill_tools:
                tools.extend(skill_tools)
                print(f"{Colors.GREEN}✅ Loaded Skill tool (get_skill){Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}⚠️  No available Skills found{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️  Failed to load Skills: {e}{Colors.RESET}")

    # 4. MCP tools (loaded with priority search)
    # 学习提示：MCP 是外部工具协议层，这里负责连接并把远程工具适配成统一 Tool 接口。
    if config.tools.enable_mcp:
        print(f"{Colors.BRIGHT_CYAN}Loading MCP tools...{Colors.RESET}")
        try:
            # Apply MCP timeout configuration from config.yaml
            mcp_config = config.tools.mcp
            set_mcp_timeout_config(
                connect_timeout=mcp_config.connect_timeout,
                execute_timeout=mcp_config.execute_timeout,
                sse_read_timeout=mcp_config.sse_read_timeout,
            )
            print(
                f"{Colors.DIM}  MCP timeouts: connect={mcp_config.connect_timeout}s, "
                f"execute={mcp_config.execute_timeout}s, sse_read={mcp_config.sse_read_timeout}s{Colors.RESET}"
            )

            # Use priority search for mcp.json
            mcp_config_path = Config.find_config_file(config.tools.mcp_config_path)
            if mcp_config_path:
                mcp_tools = await load_mcp_tools_async(str(mcp_config_path))
                if mcp_tools:
                    tools.extend(mcp_tools)
                    print(f"{Colors.GREEN}✅ Loaded {len(mcp_tools)} MCP tools (from: {mcp_config_path}){Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}⚠️  No available MCP tools found{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}⚠️  MCP config file not found: {config.tools.mcp_config_path}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️  Failed to load MCP tools: {e}{Colors.RESET}")

    print()  # Empty line separator
    return tools, skill_loader


def add_workspace_tools(tools: List[Tool], config: Config, workspace_dir: Path, session_id: int):
    """Add workspace-dependent tools

    These tools need to know the workspace directory.

    Args:
        tools: Existing tools list to add to
        config: Configuration object
        workspace_dir: Workspace directory path
        session_id: Current runtime session identifier
    """
    # 学习提示：从这里开始，工具会感知“当前项目目录”。
    # 这样 read/write/edit/bash 的相对路径都会落在 workspace 下。
    # Ensure workspace directory exists
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Bash tool - needs workspace as cwd for command execution
    if config.tools.enable_bash:
        bash_tool = BashTool(workspace_dir=str(workspace_dir))
        tools.append(bash_tool)
        print(f"{Colors.GREEN}✅ Loaded Bash tool (cwd: {workspace_dir}){Colors.RESET}")

    # File tools - need workspace to resolve relative paths
    if config.tools.enable_file_tools:
        tools.extend(
            [
                ReadTool(workspace_dir=str(workspace_dir)),
                WriteTool(workspace_dir=str(workspace_dir)),
                EditTool(workspace_dir=str(workspace_dir)),
            ]
        )
        print(f"{Colors.GREEN}✅ Loaded file operation tools (workspace: {workspace_dir}){Colors.RESET}")

    # Session note tool - needs workspace to store memory file
    if config.tools.enable_note:
        memory_db = str(get_memory_db_path(workspace_dir))
        tools.append(SessionNoteTool(memory_file=memory_db, session_id=session_id))
        tools.append(RecallNoteTool(memory_file=memory_db, session_id=session_id))
        print(f"{Colors.GREEN}✅ Loaded session note tools (SQLite, session_id: {session_id}){Colors.RESET}")

    # A-share stock tools - selection/analysis/action planning skeleton
    if config.tools.enable_stock_tools:
        stock_tools = create_a_share_tools(tushare_token=config.tools.tushare_token)
        tools.extend(stock_tools)
        print(f"{Colors.GREEN}✅ Loaded {len(stock_tools)} A-share stock tools (skeleton){Colors.RESET}")

    # NOTE:
    # We intentionally keep a single simulation trading pipeline
    # (sim_trades/sim_positions via SimulateTradeTool) to avoid dual trade ledgers.
    memory_db = str(get_memory_db_path(workspace_dir))
    kline_db_path = str(resolve_kline_db_path(workspace_dir))
    tools.append(SimulateTradeTool(db_path=memory_db, kline_db_path=kline_db_path))
    print(f"{Colors.GREEN}✅ Loaded simulation trade tool (simulate_trade, session_id: {session_id}){Colors.RESET}")


async def _quiet_cleanup():
    """Clean up MCP connections, suppressing noisy asyncgen teardown tracebacks."""
    # Silence the asyncgen finalization noise that anyio/mcp emits when
    # stdio_client's task group is torn down across tasks.  The handler is
    # intentionally NOT restored: asyncgen finalization happens during
    # asyncio.run() shutdown (after run_agent returns), so restoring the
    # handler here would still let the noise through.  Since this runs
    # right before process exit, swallowing late exceptions is safe.
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(lambda _loop, _ctx: None)
    try:
        await cleanup_mcp_connections()
    except Exception:
        pass


async def run_agent(workspace_dir: Path, task: str = None, attach_session_id: int | None = None):
    """Run Agent in interactive or non-interactive mode.

    Args:
        workspace_dir: Workspace directory path
        task: If provided, execute this task and exit (non-interactive mode)
    """
    # 学习提示：run_agent 是 CLI 的主编排函数，核心链路如下：
    # 1) 读配置 -> 2) 初始化 LLM -> 3) 装配工具 -> 4) 构建 Agent
    # 5) 进入非交互/交互模式 -> 6) 清理 MCP 连接
    session_start = datetime.now()

    # 1. Load configuration from package directory
    config_path = Config.get_default_config_path()

    if not config_path.exists():
        print(f"{Colors.RED}❌ Configuration file not found{Colors.RESET}")
        print()
        print(f"{Colors.BRIGHT_CYAN}📦 Configuration Search Path:{Colors.RESET}")
        print(f"  {Colors.DIM}1) mini_agent/config/config.yaml{Colors.RESET} (development)")
        print(f"  {Colors.DIM}2) ~/.mini-agent/config/config.yaml{Colors.RESET} (user)")
        print(f"  {Colors.DIM}3) <package>/config/config.yaml{Colors.RESET} (installed)")
        print()
        print(f"{Colors.BRIGHT_YELLOW}🚀 Quick Setup (Recommended):{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_GREEN}curl -fsSL https://raw.githubusercontent.com/MiniMax-AI/Mini-Agent/main/scripts/setup-config.sh | bash{Colors.RESET}"
        )
        print()
        print(f"{Colors.DIM}  This will automatically:{Colors.RESET}")
        print(f"{Colors.DIM}    • Create ~/.mini-agent/config/{Colors.RESET}")
        print(f"{Colors.DIM}    • Download configuration files{Colors.RESET}")
        print(f"{Colors.DIM}    • Guide you to add your API Key{Colors.RESET}")
        print()
        print(f"{Colors.BRIGHT_YELLOW}📝 Manual Setup:{Colors.RESET}")
        user_config_dir = Path.home() / ".mini-agent" / "config"
        example_config = Config.get_package_dir() / "config" / "config-example.yaml"
        print(f"  {Colors.DIM}mkdir -p {user_config_dir}{Colors.RESET}")
        print(f"  {Colors.DIM}cp {example_config} {user_config_dir}/config.yaml{Colors.RESET}")
        print(f"  {Colors.DIM}# Then edit {user_config_dir}/config.yaml to add your API Key{Colors.RESET}")
        print()
        return

    try:
        config = Config.from_yaml(config_path)
    except FileNotFoundError:
        print(f"{Colors.RED}❌ Error: Configuration file not found: {config_path}{Colors.RESET}")
        return
    except ValueError as e:
        print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")
        print(f"{Colors.YELLOW}Please check the configuration file format{Colors.RESET}")
        return
    except Exception as e:
        print(f"{Colors.RED}❌ Error: Failed to load configuration file: {e}{Colors.RESET}")
        return

    # 2. Initialize LLM client
    # 学习提示：Config 的 retry 字段会被转换成底层 retry 配置对象，
    # 并通过回调把重试过程打印到终端。
    from mini_agent.retry import RetryConfig as RetryConfigBase

    # Convert configuration format
    retry_config = RetryConfigBase(
        enabled=config.llm.retry.enabled,
        max_retries=config.llm.retry.max_retries,
        initial_delay=config.llm.retry.initial_delay,
        max_delay=config.llm.retry.max_delay,
        exponential_base=config.llm.retry.exponential_base,
        retryable_exceptions=(Exception,),
    )

    # Create retry callback function to display retry information in terminal
    def on_retry(exception: Exception, attempt: int):
        """Retry callback function to display retry information"""
        print(f"\n{Colors.BRIGHT_YELLOW}⚠️  LLM call failed (attempt {attempt}): {str(exception)}{Colors.RESET}")
        next_delay = retry_config.calculate_delay(attempt - 1)
        print(f"{Colors.DIM}   Retrying in {next_delay:.1f}s (attempt {attempt + 1})...{Colors.RESET}")

    # Convert provider string to LLMProvider enum
    provider = LLMProvider.ANTHROPIC if config.llm.provider.lower() == "anthropic" else LLMProvider.OPENAI

    llm_client = LLMClient(
        api_key=config.llm.api_key,
        provider=provider,
        api_base=config.llm.api_base,
        model=config.llm.model,
        retry_config=retry_config if config.llm.retry.enabled else None,
    )

    # Set retry callback
    if config.llm.retry.enabled:
        llm_client.retry_callback = on_retry
        print(f"{Colors.GREEN}✅ LLM retry mechanism enabled (max {config.llm.retry.max_retries} retries){Colors.RESET}")

    # 3. Initialize base tools (independent of workspace)
    tools, skill_loader = await initialize_base_tools(config)

    # 4. Load System Prompt (with priority search)
    # 学习提示：System Prompt 是 Agent 的“长期行为约束”。
    system_prompt_path = Config.find_config_file(config.agent.system_prompt_path)
    if system_prompt_path and system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text(encoding="utf-8")
        print(f"{Colors.GREEN}✅ Loaded system prompt (from: {system_prompt_path}){Colors.RESET}")
    else:
        system_prompt = "You are Mini-Agent, an intelligent assistant powered by MiniMax M2.5 that can help users complete various tasks."
        print(f"{Colors.YELLOW}⚠️  System prompt not found, using default{Colors.RESET}")

    # 5. Inject Skills Metadata into System Prompt (Progressive Disclosure - Level 1)
    # 学习提示：这里只注入技能名称+描述，避免一次性把所有技能正文塞进上下文。
    if skill_loader:
        skills_metadata = skill_loader.get_skills_metadata_prompt()
        if skills_metadata:
            # Replace placeholder with actual metadata
            system_prompt = system_prompt.replace("{SKILLS_METADATA}", skills_metadata)
            print(f"{Colors.GREEN}✅ Injected {len(skill_loader.loaded_skills)} skills metadata into system prompt{Colors.RESET}")
        else:
            # Remove placeholder if no skills
            system_prompt = system_prompt.replace("{SKILLS_METADATA}", "")
    else:
        # Remove placeholder if skills not enabled
        system_prompt = system_prompt.replace("{SKILLS_METADATA}", "")

    # 5.5 Inject enabled execution use cases (low-coupling evolution module)
    evo_service = EvolutionUseCaseService(db_path=str(get_memory_db_path(workspace_dir)))
    use_case_block = evo_service.render_prompt_block(limit=12)
    if use_case_block:
        system_prompt = f"{system_prompt.strip()}\n\n【Execution Use Cases】\n{use_case_block}"
        print(f"{Colors.GREEN}✅ Injected execution use cases into system prompt{Colors.RESET}")

    create_profile: dict[str, Any] | None = None
    if attach_session_id is None and task is None and sys.stdin.isatty():
        try:
            create_profile = _collect_session_create_inputs(
                argparse.Namespace(
                    name=None,
                    prompt=None,
                    template=None,
                    mode=None,
                    initial_capital=None,
                    risk_preference=None,
                    max_single_loss_pct=None,
                    single_position_cap_pct=None,
                    stop_loss_pct=None,
                    take_profit_pct=None,
                    investment_horizon=None,
                    event_filter=[],
                ),
                workspace_dir,
            )
        except ValueError as exc:
            print(f"{Colors.RED}❌ Interactive session setup failed: {exc}{Colors.RESET}")
            return

    # 6. Create/start runtime session (or attach existing one) with process binding.
    try:
        runtime_ctx = build_runtime_session_context(
            memory_db_path=get_memory_db_path(workspace_dir),
            system_prompt=(create_profile or {}).get("system_prompt", system_prompt),
            attach_session_id=attach_session_id,
            runtime_type="cli",
            session_name_prefix="cli",
            session_name=(create_profile or {}).get("name"),
            mode=(create_profile or {}).get("mode", "simulation"),
            initial_capital=float((create_profile or {}).get("initial_capital", 100000.0)),
            create_session_kwargs=(
                {
                    "risk_preference": (create_profile or {}).get("risk_preference"),
                    "max_single_loss_pct": (create_profile or {}).get("max_single_loss_pct"),
                    "single_position_cap_pct": (create_profile or {}).get("single_position_cap_pct"),
                    "stop_loss_pct": (create_profile or {}).get("stop_loss_pct"),
                    "take_profit_pct": (create_profile or {}).get("take_profit_pct"),
                    "investment_horizon": (create_profile or {}).get("investment_horizon"),
                    "event_filter": (create_profile or {}).get("event_filter", []),
                }
                if create_profile is not None
                else None
            ),
        )
    except KeyError:
        print(f"{Colors.RED}❌ Session not found: {attach_session_id}{Colors.RESET}")
        return
    except ValueError as exc:
        print(f"{Colors.RED}❌ {exc}{Colors.RESET}")
        return

    session_manager = runtime_ctx.session_manager
    session_id = runtime_ctx.session_id
    session_name = runtime_ctx.session_name
    system_prompt = runtime_ctx.effective_system_prompt
    session_state = session_manager.get_session(session_id)
    position_rows = _read_sim_positions(get_memory_db_path(workspace_dir), str(session_id))
    position_payload = [
        {
            "ticker": str(row["ticker"]),
            "quantity": int(row["quantity"]),
            "avg_cost": float(row["avg_cost"]) if row["avg_cost"] is not None else None,
        }
        for row in position_rows
    ]
    system_prompt, prompt_missing = ensure_trade_policy_prompt(
        prompt=system_prompt,
        role="A股自动交易代理",
        session=session_state,
        positions=position_payload,
    )
    if prompt_missing:
        print(
            f"{Colors.YELLOW}⚠️ Trading system prompt missing sections: {', '.join(prompt_missing)}; auto-injected guardrail block.{Colors.RESET}"
        )
    if runtime_ctx.attached:
        print(f"{Colors.GREEN}✅ Attached to existing session: {session_id} ({session_name}){Colors.RESET}")
    else:
        print(f"{Colors.GREEN}✅ Runtime session started: {session_id} ({session_name}){Colors.RESET}")

    auto_trade_rule = (
        "【交易执行规则】\n"
        "- 当前为 simulation（模拟交易）环境。\n"
        "- 当用户给出明确交易指令（如“买入/卖出 + 标的 + 数量”）时，必须直接调用工具 `simulate_trade` 执行。\n"
        "- 不要反问“是否确认执行”，不要只给建议不执行。\n"
        "- 若参数缺失（标的/数量/日期）再提问补全；若参数完整则立即执行并回报成交结果。"
    )
    if auto_trade_rule not in system_prompt:
        system_prompt = f"{system_prompt.strip()}\n\n{auto_trade_rule}"

    # 7. Add workspace-dependent tools
    add_workspace_tools(tools, config, workspace_dir, session_id=session_id)

    # 8. Create Agent
    agent = Agent(
        llm_client=llm_client,
        system_prompt=system_prompt,
        tools=tools,
        max_steps=config.agent.max_steps,
        workspace_dir=str(workspace_dir),
        enable_intercept_log=config.agent.enable_intercept_log,
        session_id=session_id,
    )

    # 9. Display welcome information
    if not task:
        print_banner()
        print_session_info(agent, workspace_dir, config.llm.model)

    # 9.5 Non-interactive mode: execute task and exit
    # 学习提示：--task 会直接跑一次 agent.run() 然后退出，不进入 REPL 循环。
    if task:
        direct_order = _parse_direct_trade_order(task)
        if direct_order:
            trade_tool = next((t for t in tools if getattr(t, "name", "") == "simulate_trade"), None)
            if trade_tool is None:
                print(f"\n{Colors.RED}❌ simulate_trade tool not available{Colors.RESET}")
            else:
                trade_date = str(direct_order.get("trade_date") or _resolve_trade_date_for_execution(get_memory_db_path(workspace_dir)))
                result = await trade_tool.execute(
                    session_id=session_id,
                    action=direct_order["action"],
                    ticker=direct_order["ticker"],
                    quantity=direct_order["quantity"],
                    trade_date=trade_date,
                    reason="direct_user_instruction",
                )
                content = result.content if result.success else f"simulate trade failed: {result.error}"
                await _persist_turn_memory(tools, session_id=session_id, category="conversation_user", content=task)
                await _persist_turn_memory(tools, session_id=session_id, category="conversation_assistant", content=content)
                session_manager.record_critical_memory(
                    session_id=session_id,
                    event_type="manual_trade",
                    operation=direct_order["action"],
                    reason="direct_user_instruction",
                    content=json.dumps(
                        {
                            "ticker": direct_order["ticker"],
                            "quantity": direct_order["quantity"],
                            "trade_date": trade_date,
                            "result": content,
                        },
                        ensure_ascii=False,
                    ),
                )
                if result.success:
                    print(f"\n{Colors.GREEN}{content}{Colors.RESET}\n")
                else:
                    print(f"\n{Colors.RED}{content}{Colors.RESET}\n")
            print_stats(agent, session_start)
            runtime_ctx.close(stop_session=True)
            await _quiet_cleanup()
            return

        print(f"\n{Colors.BRIGHT_BLUE}Agent{Colors.RESET} {Colors.DIM}›{Colors.RESET} {Colors.DIM}Executing task...{Colors.RESET}\n")
        await _persist_turn_memory(tools, session_id=session_id, category="conversation_user", content=task)
        mem_snapshot = build_session_memory_snapshot(
            _load_prompt_memories(get_memory_db_path(workspace_dir), session_id=session_id, recent_limit=12, critical_limit=8)
        )
        if mem_snapshot:
            agent.add_user_message(f"[Session Memory]\n{mem_snapshot}\n\n[User Request]\n{task}")
        else:
            agent.add_user_message(task)
        try:
            before_msg_count = len(agent.messages)
            final_reply = await agent.run()
            await _persist_turn_memory(
                tools,
                session_id=session_id,
                category="conversation_assistant",
                content=final_reply or "",
            )
            _record_sim_trade_critical_from_messages(
                session_manager=session_manager,
                session_id=session_id,
                messages=agent.messages[before_msg_count:],
                event_type="chat_turn",
            )
        except Exception as e:
            print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}")
        finally:
            print_stats(agent, session_start)

        runtime_ctx.close(stop_session=True)
        # Cleanup MCP connections
        await _quiet_cleanup()
        return

    # 9. Setup prompt_toolkit session
    # Command completer
    command_completer = WordCompleter(
        ["/help", "/clear", "/history", "/stats", "/log", "/exit", "/quit", "/q"],
        ignore_case=True,
        sentence=True,
    )

    # Custom style for prompt
    prompt_style = Style.from_dict(
        {
            "prompt": "#00ff00 bold",  # Green and bold
            "separator": "#666666",  # Gray
        }
    )

    # Custom key bindings
    kb = KeyBindings()

    @kb.add("c-u")  # Ctrl+U: Clear current line
    def _(event):
        """Clear the current input line"""
        event.current_buffer.reset()

    @kb.add("c-l")  # Ctrl+L: Clear screen (optional bonus)
    def _(event):
        """Clear the screen"""
        event.app.renderer.clear()

    @kb.add("c-j")  # Ctrl+J (对应 Ctrl+Enter)
    def _(event):
        """Insert a newline"""
        event.current_buffer.insert_text("\n")

    # Create prompt session with history and auto-suggest
    # Use FileHistory for persistent history across sessions (stored in user's home directory)
    history_file = Path.home() / ".mini-agent" / ".history"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=command_completer,
        style=prompt_style,
        key_bindings=kb,
    )

    agent_run_lock = asyncio.Lock()
    event_poller_stop = asyncio.Event()

    async def _process_inbox_event(item: dict) -> None:
        event = item.get("event_payload")
        if not isinstance(event, dict):
            session_manager.complete_inbox_event(int(item["id"]), success=False, error="invalid event payload")
            return

        begin_token = session_manager.begin_event_processing(session_id, event)
        if not begin_token.get("process"):
            reason = f"idempotent_skip status={begin_token.get('status')}"
            session_manager.mark_event_skipped(session_id, event, reason)
            session_manager.complete_inbox_event(int(item["id"]), success=True)
            print(f"\n{Colors.YELLOW}⚠️ Event skipped: {reason}{Colors.RESET}\n")
            return

        event_prompt = (
            "[System Event]\n"
            f"type: {event.get('type')}\n"
            f"event_id: {event.get('event_id')}\n"
            f"payload: {json.dumps(event, ensure_ascii=False)}\n\n"
            "This event is delivered by background dispatcher. "
            "Please analyze and decide whether simulated trade actions are needed."
        )
        event_type = str(event.get("type") or "")
        trading_date = str(event.get("date") or datetime.now().date().isoformat())

        async with agent_run_lock:
            print(
                f"\n{Colors.BRIGHT_BLUE}Event{Colors.RESET} {Colors.DIM}›{Colors.RESET} "
                f"{Colors.DIM}Processing {event.get('type')} ({event.get('event_id')})...{Colors.RESET}\n"
            )
            try:
                if event_type == "daily_review":
                    runtime = build_decision_runtime(workspace_dir, get_memory_db_path(workspace_dir))
                    decision = await run_llm_decision(
                        runtime=runtime,
                        session_id=session_id,
                        trading_date=trading_date,
                        event_id=str(event.get("event_id") or "") or None,
                    )
                    final_reply = str(decision.get("agent_analysis") or "")
                    trade_actions = list(decision.get("trade_actions") or [])
                    signal = decision.get("trade_signal")
                    execution = decision.get("execution")
                    execution_error = decision.get("execution_error")
                    primary_action = trade_actions[0] if trade_actions else (signal if isinstance(signal, dict) else None)
                    signal_action = str(primary_action.get("action")) if isinstance(primary_action, dict) else None
                    signal_ticker = str(primary_action.get("ticker")) if isinstance(primary_action, dict) else None
                    signal_qty = primary_action.get("quantity") if isinstance(primary_action, dict) else None
                    summary = (
                        f"[daily_review] actions={trade_actions or signal} execution={execution or execution_error or 'no_trade'} "
                        f"date={trading_date}"
                    )
                    await _persist_turn_memory(
                        tools,
                        session_id=session_id,
                        category="conversation_user",
                        content=f"[event]{event_prompt}",
                    )
                    await _persist_turn_memory(
                        tools,
                        session_id=session_id,
                        category="conversation_assistant",
                        content=f"{final_reply}\n\n{summary}".strip(),
                    )
                    op: str | None = None
                    if execution_error:
                        op = f"{signal_action}_failed" if signal_action else "trade_failed"
                    elif execution:
                        if signal_action:
                            op = signal_action
                        else:
                            parsed_exec = _parse_sim_trade_tool_content(str(execution))
                            parsed_action = str(parsed_exec.get("action") or "").strip()
                            op = parsed_action or "trade_executed"
                    elif signal_action:
                        op = f"{signal_action}_planned"
                    reason = final_reply.strip().replace("\n", " ")[:500]
                    critical_content = json.dumps(
                        {
                            "date": trading_date,
                            "signal": signal,
                            "trade_actions": trade_actions,
                            "execution": execution,
                            "execution_error": execution_error,
                            "ticker": signal_ticker,
                            "quantity": signal_qty,
                        },
                        ensure_ascii=False,
                    )
                    if op:
                        session_manager.record_critical_memory(
                            session_id=session_id,
                            event_id=str(event.get("event_id") or ""),
                            event_type=event_type,
                            operation=op,
                            reason=reason or None,
                            content=critical_content,
                        )
                    success = execution_error is None
                    err = execution_error
                    result_text = summary
                else:
                    await _persist_turn_memory(
                        tools,
                        session_id=session_id,
                        category="conversation_user",
                        content=f"[event]{event_prompt}",
                    )
                    mem_snapshot = build_session_memory_snapshot(
                        _load_prompt_memories(
                            get_memory_db_path(workspace_dir),
                            session_id=session_id,
                            recent_limit=12,
                            critical_limit=8,
                        )
                    )
                    if mem_snapshot:
                        agent.add_user_message(f"[Session Memory]\n{mem_snapshot}\n\n[Event]\n{event_prompt}")
                    else:
                        agent.add_user_message(event_prompt)
                    before_msg_count = len(agent.messages)
                    final_reply = await agent.run()
                    await _persist_turn_memory(
                        tools,
                        session_id=session_id,
                        category="conversation_assistant",
                        content=final_reply or "",
                    )
                    _record_sim_trade_critical_from_messages(
                        session_manager=session_manager,
                        session_id=session_id,
                        messages=agent.messages[before_msg_count:],
                        event_type=event_type or "event",
                        event_id=str(event.get("event_id") or "") or None,
                    )
                    success = True
                    err = None
                    result_text = (final_reply or "")[:2000]

                await _persist_turn_memory(
                    tools,
                    session_id=session_id,
                    category="event_result",
                    content=f"type={event_type} date={trading_date} success={success}",
                )
                session_manager.finalize_event_processing(
                    session_id,
                    event,
                    status="succeeded" if success else "failed",
                    success=success,
                    result=result_text,
                    error=err,
                    started_at=begin_token.get("started_at"),
                )
                session_manager.complete_inbox_event(int(item["id"]), success=success, error=err)
            except Exception as exc:
                error = str(exc)
                session_manager.finalize_event_processing(
                    session_id,
                    event,
                    status="failed",
                    success=False,
                    error=error,
                    started_at=begin_token.get("started_at"),
                )
                session_manager.complete_inbox_event(int(item["id"]), success=False, error=error)
                print(f"\n{Colors.RED}❌ Event processing failed: {error}{Colors.RESET}\n")

    async def _event_poller_loop() -> None:
        while not event_poller_stop.is_set():
            try:
                runtime_ctx.heartbeat()
                item = session_manager.claim_next_pending_event(session_id)
                if item is None:
                    await asyncio.sleep(1.0)
                    continue
                await _process_inbox_event(item)
            except Exception as exc:
                print(f"\n{Colors.YELLOW}⚠️ Event poller warning: {exc}{Colors.RESET}\n")
                await asyncio.sleep(1.0)

    event_poller_task = asyncio.create_task(_event_poller_loop())

    # 10. Interactive loop
    try:
        while True:
            try:
                user_input = await session.prompt_async(
                    [
                        ("class:prompt", "You"),
                        ("", " › "),
                    ],
                    multiline=False,
                    enable_history_search=True,
                )
                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    command = user_input.lower()
                    if command in ["/exit", "/quit", "/q"]:
                        print(f"\n{Colors.BRIGHT_YELLOW}👋 Goodbye! Thanks for using Mini Agent{Colors.RESET}\n")
                        print_stats(agent, session_start)
                        break
                    if command == "/help":
                        print_help()
                        continue
                    if command == "/clear":
                        old_count = len(agent.messages)
                        agent.messages = [agent.messages[0]]
                        print(f"{Colors.GREEN}✅ Cleared {old_count - 1} messages, starting new session{Colors.RESET}\n")
                        continue
                    if command == "/history":
                        print(f"\n{Colors.BRIGHT_CYAN}Current session message count: {len(agent.messages)}{Colors.RESET}\n")
                        continue
                    if command == "/stats":
                        print_stats(agent, session_start)
                        continue
                    if command == "/log" or command.startswith("/log "):
                        parts = user_input.split(maxsplit=1)
                        if len(parts) == 1:
                            show_log_directory(open_file_manager=True)
                        else:
                            filename = parts[1].strip("\"'")
                            read_log_file(filename)
                        continue
                    print(f"{Colors.RED}❌ Unknown command: {user_input}{Colors.RESET}")
                    print(f"{Colors.DIM}Type /help to see available commands{Colors.RESET}\n")
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    print(f"\n{Colors.BRIGHT_YELLOW}👋 Goodbye! Thanks for using Mini Agent{Colors.RESET}\n")
                    print_stats(agent, session_start)
                    break

                direct_order = _parse_direct_trade_order(user_input)
                if direct_order:
                    async with agent_run_lock:
                        trade_tool = next((t for t in tools if getattr(t, "name", "") == "simulate_trade"), None)
                        if trade_tool is None:
                            print(f"\n{Colors.RED}❌ simulate_trade tool not available{Colors.RESET}\n")
                            continue
                        trade_date = str(
                            direct_order.get("trade_date")
                            or _resolve_trade_date_for_execution(get_memory_db_path(workspace_dir))
                        )
                        result = await trade_tool.execute(
                            session_id=session_id,
                            action=direct_order["action"],
                            ticker=direct_order["ticker"],
                            quantity=direct_order["quantity"],
                            trade_date=trade_date,
                            reason="direct_user_instruction",
                        )
                        content = result.content if result.success else f"simulate trade failed: {result.error}"
                        await _persist_turn_memory(
                            tools,
                            session_id=session_id,
                            category="conversation_user",
                            content=user_input,
                        )
                        await _persist_turn_memory(
                            tools,
                            session_id=session_id,
                            category="conversation_assistant",
                            content=content,
                        )
                        session_manager.record_critical_memory(
                            session_id=session_id,
                            event_type="manual_trade",
                            operation=direct_order["action"],
                            reason="direct_user_instruction",
                            content=json.dumps(
                                {
                                    "ticker": direct_order["ticker"],
                                    "quantity": direct_order["quantity"],
                                    "trade_date": trade_date,
                                    "result": content,
                                },
                                ensure_ascii=False,
                            ),
                        )
                    if result.success:
                        print(f"\n{Colors.GREEN}{content}{Colors.RESET}\n")
                    else:
                        print(f"\n{Colors.RED}{content}{Colors.RESET}\n")
                    print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}\n")
                    continue

                async with agent_run_lock:
                    print(
                        f"\n{Colors.BRIGHT_BLUE}Agent{Colors.RESET} {Colors.DIM}›{Colors.RESET} {Colors.DIM}Thinking...{Colors.RESET}\n"
                    )
                    await _persist_turn_memory(
                        tools,
                        session_id=session_id,
                        category="conversation_user",
                        content=user_input,
                    )
                    mem_snapshot = build_session_memory_snapshot(
                        _load_prompt_memories(
                            get_memory_db_path(workspace_dir),
                            session_id=session_id,
                            recent_limit=12,
                            critical_limit=8,
                        )
                    )
                    if mem_snapshot:
                        agent.add_user_message(f"[Session Memory]\n{mem_snapshot}\n\n[User Request]\n{user_input}")
                    else:
                        agent.add_user_message(user_input)

                    before_msg_count = len(agent.messages)
                    final_reply = await agent.run()
                    await _persist_turn_memory(
                        tools,
                        session_id=session_id,
                        category="conversation_assistant",
                        content=final_reply or "",
                    )
                    _record_sim_trade_critical_from_messages(
                        session_manager=session_manager,
                        session_id=session_id,
                        messages=agent.messages[before_msg_count:],
                        event_type="chat_turn",
                    )
                print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}\n")

            except KeyboardInterrupt:
                print(f"\n\n{Colors.BRIGHT_YELLOW}👋 Interrupt signal detected, exiting...{Colors.RESET}\n")
                print_stats(agent, session_start)
                break
            except Exception as e:
                print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}")
                print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}\n")
    finally:
        event_poller_stop.set()
        event_poller_task.cancel()
        try:
            await event_poller_task
        except asyncio.CancelledError:
            pass
        runtime_ctx.close(stop_session=True)
        # 11. Cleanup MCP connections
        await _quiet_cleanup()


def handle_health_command(args: argparse.Namespace, workspace_dir: Path) -> None:
    """Handle health check commands."""
    from .observable import HealthCheck

    if args.health_command == "check":
        print(f"{Colors.CYAN}🔍 Running health check...{Colors.RESET}\n")
        result = HealthCheck.check_all()

        status = result["status"]
        status_color = Colors.GREEN if status == "healthy" else Colors.RED if status == "unhealthy" else Colors.YELLOW

        print(f"{status_color}Status: {status.upper()}{Colors.RESET}\n")

        print(f"{Colors.BOLD}Checks:{Colors.RESET}")
        for check_name, check_result in result["checks"].items():
            print(f"  {check_name}: {check_result}")


def handle_sync_command(args: argparse.Namespace, workspace_dir: Path) -> None:
    """Sync A-share daily K-line data into unified DB."""
    db_path = get_default_memory_db_path()
    kline_db = KLineDB(db_path=str(db_path))
    tushare_token = ""
    try:
        cfg = Config.from_yaml(Config.get_default_config_path())
        tushare_token = (cfg.tools.tushare_token or "").strip()
    except Exception:
        tushare_token = ""
    config_path = Config.get_default_config_path()

    if args.cron:
        print("Recommended crontab entries:")
        for line in build_cron_lines(Path.cwd(), args.start):
            print(line)
        return
    if args.install_cron:
        ok, detail = install_cron_lines(Path.cwd(), args.start)
        if ok:
            print(f"{Colors.GREEN}✅ {detail}{Colors.RESET}")
        else:
            print(f"{Colors.RED}❌ Failed to install cron: {detail}{Colors.RESET}")
        return

    if args.all and not tushare_token:
        print(f"{Colors.RED}❌ tools.tushare_token is empty in config: {config_path}{Colors.RESET}")
        print(f"{Colors.YELLOW}Please set tools.tushare_token before running `sync --all`.{Colors.RESET}")
        return

    start_date = args.start
    end_date = args.end
    if args.all and args.start == "1991-01-01" and args.end is None:
        # Default full-range is too heavy for day-to-day usage; switch to daily incremental.
        today = datetime.now().date().isoformat()
        start_date = today
        end_date = today
        print(f"{Colors.YELLOW}ℹ️ --all default changed to daily incremental: {today}{Colors.RESET}")

    try:
        tickers = resolve_ticker_universe(
            args.tickers or "",
            args.all,
            kline_db=kline_db,
            tushare_token=tushare_token,
        )
    except Exception as exc:
        print(f"{Colors.RED}❌ Failed to fetch ticker universe: {exc}{Colors.RESET}")
        return
    if not tickers:
        print(f"{Colors.RED}❌ Provide tickers (e.g. 600519,000001) or use --all{Colors.RESET}")
        return

    print(f"{Colors.CYAN}🔄 Syncing K-line data -> {db_path}{Colors.RESET}")
    result = sync_kline_data(
        kline_db,
        tickers=tickers,
        start=start_date,
        end=end_date,
        tushare_token=tushare_token,
    )
    print(f"tickers={len(tickers)}, range={start_date}~{result.end_date}")

    print(f"{Colors.GREEN}✅ Sync done{Colors.RESET}")
    print(f"success={result.success_count}, failed={len(result.failed)}, rows={result.total_rows}")
    if result.failed:
        print(f"{Colors.YELLOW}failed tickers: {', '.join(result.failed[:20])}{Colors.RESET}")


def main():
    """Main entry point for CLI"""
    setup_logging()

    # 学习提示：main 很薄，只做参数解析、workspace 解析，然后把控制权交给 run_agent。
    # Parse command line arguments
    args = parse_args()

    # Handle log subcommand
    if args.command == "log":
        if args.filename:
            read_log_file(args.filename)
        else:
            show_log_directory(open_file_manager=True)
        return

    # Determine workspace directory
    # Expand ~ to user home directory for portability
    if args.workspace:
        workspace_dir = Path(args.workspace).expanduser().absolute()
    else:
        # Use current working directory
        workspace_dir = Path.cwd()

    # Ensure workspace directory exists
    workspace_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "session":
        if not args.session_command:
            print(f"{Colors.RED}❌ Missing session action. Use: create/list/start/stop/delete{Colors.RESET}")
            return
        handle_session_command(args, workspace_dir)
        return

    if args.command == "trade":
        if not args.trade_command:
            print(f"{Colors.RED}❌ Missing trade action. Use: buy/sell/positions/profit{Colors.RESET}")
            return
        handle_trade_command(args, workspace_dir)
        return

    if args.command == "backtest":
        if not args.backtest_command:
            print(f"{Colors.RED}❌ Missing backtest action. Use: run/result{Colors.RESET}")
            return
        handle_backtest_command(args, workspace_dir)
        return

    if args.command == "event":
        if not args.event_command:
            print(f"{Colors.RED}❌ Missing event action. Use: trigger{Colors.RESET}")
            return
        handle_event_command(args, workspace_dir)
        return

    if args.command == "health":
        if not args.health_command:
            print(f"{Colors.RED}❌ Missing health action. Use: check{Colors.RESET}")
            return
        handle_health_command(args, workspace_dir)
        return

    if args.command == "sync":
        handle_sync_command(args, workspace_dir)
        return

    if args.command == "evolve":
        if not args.evolve_command:
            print(f"{Colors.RED}❌ Missing evolve action. Use: scan/list/enable/disable/prompt/trace-sync/trace-summary{Colors.RESET}")
            return
        handle_evolve_command(args, workspace_dir)
        return

    # Run the agent (config always loaded from package directory)
    asyncio.run(run_agent(workspace_dir, task=args.task, attach_session_id=args.session_id))


if __name__ == "__main__":
    main()
