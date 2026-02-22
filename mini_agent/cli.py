"""
Mini Agent - Interactive Runtime Example

Usage:
    mini-agent [--workspace DIR] [--task TASK]

Examples:
    mini-agent                              # Use current directory as workspace (interactive mode)
    mini-agent --workspace /path/to/dir     # Use specific workspace directory (interactive mode)
    mini-agent --task "create a file"       # Execute a task non-interactively
"""

import argparse
import asyncio
import json
import logging
import os
import platform
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from mini_agent import LLMClient
from mini_agent.agent import Agent
from mini_agent.app.decision_service import build_decision_runtime, run_llm_decision
from mini_agent.app.memory_service import build_session_memory_snapshot, load_recent_session_memories
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


def handle_session_command(args: argparse.Namespace, workspace_dir: Path) -> None:
    """Handle session subcommands."""
    manager = SessionManager(db_path=str(get_memory_db_path(workspace_dir)))

    if args.session_command == "create":
        session_id = manager.create_session(
            name=args.name,
            system_prompt=args.prompt,
            mode=args.mode,
            initial_capital=args.initial_capital,
            event_filter=args.event_filter or [],
        )
        print(f"{Colors.GREEN}✅ Session created{Colors.RESET}")
        print(f"session_id: {session_id}")
        print(f"name: {args.name}")
        print(f"mode: {args.mode}")
        print(f"initial_capital: {args.initial_capital:.2f}")
        return

    if args.session_command == "list":
        sessions = manager.list_sessions()
        if not sessions:
            print(f"{Colors.YELLOW}No sessions found.{Colors.RESET}")
            return
        print("session_id\tname\tmode\tstatus\tlistening")
        for session in sessions:
            listening = "yes" if session.is_listening else "no"
            print(f"{session.session_id}\t{session.name}\t{session.mode}\t{session.status}\t{listening}")
        return

    try:
        if args.session_command == "start":
            manager.start_session(args.session_id)
            print(f"{Colors.GREEN}✅ Session started: {args.session_id}{Colors.RESET}")
        elif args.session_command == "stop":
            manager.stop_session(args.session_id)
            print(f"{Colors.GREEN}✅ Session stopped: {args.session_id}{Colors.RESET}")
        elif args.session_command == "delete":
            manager.delete_session(args.session_id)
            print(f"{Colors.GREEN}✅ Session deleted: {args.session_id}{Colors.RESET}")
    except KeyError as exc:
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


async def _default_event_trigger(session, event: dict) -> str:
    """Default event callback used by CLI trigger command."""
    return f"TRIGGERED session={session.session_id} event={event.get('type')}"


def _build_event_from_args(args: argparse.Namespace) -> dict:
    """Build normalized event payload from CLI args."""
    event = {"type": args.event_type, "triggered_at": datetime.now().isoformat()}
    if not args.payload:
        return event

    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("--payload must be a JSON object")

    event.update(payload)
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
        asyncio.create_task(run_llm_decision(runtime=runtime, session_id=session_id, trading_date=trading_date))
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

    event_type = str(event.get("type") or "").strip() or None
    manager = SessionManager(db_path=str(get_memory_db_path(workspace_dir)))
    if args.all_sessions:
        target_sessions = manager.broadcast_event(event)
    else:
        try:
            session = manager.get_session(args.session_id)
        except KeyError as exc:
            print(f"{Colors.RED}❌ {exc}{Colors.RESET}")
            return

        matched, reason = _match_single_session_for_event(session, event_type)
        if not matched:
            print(f"event_type: {event.get('type')}")
            print(f"session: {args.session_id}")
            print(f"{Colors.YELLOW}⚠️ skipped: {reason}{Colors.RESET}")
            return
        target_sessions = [session]

    print(f"event_type: {event.get('type')}")
    print(f"matched_sessions: {len(target_sessions)}")
    if not target_sessions:
        print(f"{Colors.YELLOW}⚠️ No listening sessions matched this event.{Colors.RESET}")
        return

    print(f"{Colors.CYAN}🤖 Auto trading mode (default){Colors.RESET}")
    try:
        runtime = build_decision_runtime(workspace_dir, get_memory_db_path(workspace_dir))
    except FileNotFoundError as exc:
        print(f"{Colors.RED}❌ {exc}{Colors.RESET}")
        return

    trading_date = str(event.get("date") or datetime.now().date().isoformat())
    session_ids = [session.session_id for session in target_sessions]
    auto_records = asyncio.run(_run_auto_event_for_sessions(runtime, session_ids, trading_date))
    for item in auto_records:
        sid = item["session_id"]
        if not item.get("success"):
            error_message = item.get("error", "unknown error")
            manager.record_event_result(
                session_id=sid,
                event=event,
                success=False,
                error=error_message,
            )
            print(f"- {sid}: failed {error_message}")
            continue

        result = item["result"]
        signal = result.get("trade_signal")
        if signal:
            summary = f"{signal['action']} {signal['ticker']} x{signal['quantity']}"
        else:
            summary = "no_trade"

        execution = result.get("execution")
        execution_error = result.get("execution_error")
        detail = execution if execution else execution_error if execution_error else summary
        manager.record_event_result(
            session_id=sid,
            event=event,
            success=execution_error is None,
            result=detail,
            error=execution_error,
        )
        print(f"- {sid}: ok {summary} | {detail}")


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
    session_create.add_argument("--name", required=True, help="Session name")
    session_create.add_argument("--prompt", required=True, help="System prompt")
    session_create.add_argument(
        "--mode",
        default="simulation",
        choices=["simulation", "backtest"],
        help="Session mode",
    )
    session_create.add_argument(
        "--initial-capital",
        "--capital",
        dest="initial_capital",
        type=float,
        default=100000.0,
        help="Initial capital",
    )
    session_create.add_argument(
        "--event-filter",
        nargs="*",
        default=[],
        help="Optional event types this session listens to",
    )

    session_subparsers.add_parser("list", help="List sessions")

    session_start = session_subparsers.add_parser("start", help="Start one session")
    session_start.add_argument("session_id", type=int, help="Session ID")

    session_stop = session_subparsers.add_parser("stop", help="Stop one session")
    session_stop.add_argument("session_id", type=int, help="Session ID")

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

    backtest_result = backtest_subparsers.add_parser("result", help="Show backtest result")
    backtest_result.add_argument("--session", required=True, type=int, help="Session ID")

    # event subcommands
    event_parser = subparsers.add_parser("event", help="Event broadcasting commands")
    event_subparsers = event_parser.add_subparsers(dest="event_command", help="Event actions")

    event_trigger = event_subparsers.add_parser("trigger", help="Trigger one event")
    event_trigger.add_argument("event_type", help="Event type, e.g. daily_review")
    target_group = event_trigger.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--all", action="store_true", dest="all_sessions", help="Trigger all listening sessions")
    target_group.add_argument("--session", dest="session_id", type=int, help="Trigger a specific session")
    event_trigger.add_argument("--payload", default=None, help="Optional JSON payload object")

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
        stock_tools = create_a_share_tools()
        tools.extend(stock_tools)
        print(f"{Colors.GREEN}✅ Loaded {len(stock_tools)} A-share stock tools (skeleton){Colors.RESET}")

    # NOTE:
    # We intentionally keep a single simulation trading pipeline
    # (sim_trades/sim_positions via SimulateTradeTool) to avoid dual trade ledgers.


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


async def run_agent(workspace_dir: Path, task: str = None):
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

    # 6. Create and start a runtime session (interactive CLI itself is one session).
    session_manager = SessionManager(db_path=str(get_memory_db_path(workspace_dir)))
    session_name = f"cli-{session_start.strftime('%Y%m%d-%H%M%S')}"
    session_id = session_manager.create_session(
        name=session_name,
        system_prompt=system_prompt,
        mode="simulation",
        initial_capital=100000.0,
    )
    session_manager.start_session(session_id)
    print(f"{Colors.GREEN}✅ Runtime session started: {session_id} ({session_name}){Colors.RESET}")

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
        print(f"\n{Colors.BRIGHT_BLUE}Agent{Colors.RESET} {Colors.DIM}›{Colors.RESET} {Colors.DIM}Executing task...{Colors.RESET}\n")
        await _persist_turn_memory(tools, session_id=session_id, category="conversation_user", content=task)
        mem_snapshot = build_session_memory_snapshot(
            load_recent_session_memories(get_memory_db_path(workspace_dir), session_id=session_id, limit=12)
        )
        if mem_snapshot:
            agent.add_user_message(f"[Session Memory]\n{mem_snapshot}\n\n[User Request]\n{task}")
        else:
            agent.add_user_message(task)
        try:
            final_reply = await agent.run()
            await _persist_turn_memory(
                tools,
                session_id=session_id,
                category="conversation_assistant",
                content=final_reply or "",
            )
        except Exception as e:
            print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}")
        finally:
            print_stats(agent, session_start)

        session_manager.stop_session(session_id)
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

    # 10. Interactive loop
    # 学习提示：这里是一个 REPL：
    # - 先处理 /help /clear 等命令
    # - 再把普通输入交给 Agent.run()
    # - 每轮输入都共享同一个 agent.messages（多轮记忆）
    while True:
        try:
            # Get user input using prompt_toolkit
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

            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()

                if command in ["/exit", "/quit", "/q"]:
                    print(f"\n{Colors.BRIGHT_YELLOW}👋 Goodbye! Thanks for using Mini Agent{Colors.RESET}\n")
                    print_stats(agent, session_start)
                    break

                elif command == "/help":
                    print_help()
                    continue

                elif command == "/clear":
                    # Clear message history but keep system prompt
                    old_count = len(agent.messages)
                    agent.messages = [agent.messages[0]]  # Keep only system message
                    print(f"{Colors.GREEN}✅ Cleared {old_count - 1} messages, starting new session{Colors.RESET}\n")
                    continue

                elif command == "/history":
                    print(f"\n{Colors.BRIGHT_CYAN}Current session message count: {len(agent.messages)}{Colors.RESET}\n")
                    continue

                elif command == "/stats":
                    print_stats(agent, session_start)
                    continue

                elif command == "/log" or command.startswith("/log "):
                    # Parse /log command
                    parts = user_input.split(maxsplit=1)
                    if len(parts) == 1:
                        # /log - show log directory
                        show_log_directory(open_file_manager=True)
                    else:
                        # /log <filename> - read specific log file
                        filename = parts[1].strip("\"'")
                        read_log_file(filename)
                    continue

                else:
                    print(f"{Colors.RED}❌ Unknown command: {user_input}{Colors.RESET}")
                    print(f"{Colors.DIM}Type /help to see available commands{Colors.RESET}\n")
                    continue

            # Normal conversation - exit check
            if user_input.lower() in ["exit", "quit", "q"]:
                print(f"\n{Colors.BRIGHT_YELLOW}👋 Goodbye! Thanks for using Mini Agent{Colors.RESET}\n")
                print_stats(agent, session_start)
                break

            # Run Agent with Esc cancellation support
            # 学习提示：Esc 的取消不是强杀线程，而是设置 cancel_event。
            # Agent 在安全检查点读取该事件，保证消息状态一致。
            print(
                f"\n{Colors.BRIGHT_BLUE}Agent{Colors.RESET} {Colors.DIM}›{Colors.RESET} {Colors.DIM}Thinking... (Esc to cancel){Colors.RESET}\n"
            )
            await _persist_turn_memory(
                tools,
                session_id=session_id,
                category="conversation_user",
                content=user_input,
            )
            mem_snapshot = build_session_memory_snapshot(
                load_recent_session_memories(get_memory_db_path(workspace_dir), session_id=session_id, limit=12)
            )
            if mem_snapshot:
                agent.add_user_message(f"[Session Memory]\n{mem_snapshot}\n\n[User Request]\n{user_input}")
            else:
                agent.add_user_message(user_input)

            # Create cancellation event
            cancel_event = asyncio.Event()
            agent.cancel_event = cancel_event

            # Esc key listener thread
            esc_listener_stop = threading.Event()
            esc_cancelled = [False]  # Mutable container for thread access

            def esc_key_listener():
                """Listen for Esc key in a separate thread."""
                # 学习提示：按键监听放在线程里，避免阻塞 asyncio 主事件循环。
                if platform.system() == "Windows":
                    try:
                        import msvcrt

                        while not esc_listener_stop.is_set():
                            if msvcrt.kbhit():
                                char = msvcrt.getch()
                                if char == b"\x1b":  # Esc
                                    print(f"\n{Colors.BRIGHT_YELLOW}⏹️  Esc pressed, cancelling...{Colors.RESET}")
                                    esc_cancelled[0] = True
                                    cancel_event.set()
                                    break
                            esc_listener_stop.wait(0.05)
                    except Exception:
                        pass
                    return

                # Unix/macOS
                try:
                    import select
                    import termios
                    import tty

                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)

                    try:
                        tty.setcbreak(fd)
                        while not esc_listener_stop.is_set():
                            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                            if rlist:
                                char = sys.stdin.read(1)
                                if char == "\x1b":  # Esc
                                    print(f"\n{Colors.BRIGHT_YELLOW}⏹️  Esc pressed, cancelling...{Colors.RESET}")
                                    esc_cancelled[0] = True
                                    cancel_event.set()
                                    break
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass

            # Start Esc listener thread
            esc_thread = threading.Thread(target=esc_key_listener, daemon=True)
            esc_thread.start()

            # Run agent with periodic cancellation check
            # 学习提示：外层轮询负责把线程侧的“按下 Esc”同步到 asyncio.Event。
            try:
                agent_task = asyncio.create_task(agent.run())

                # Poll for cancellation while agent runs
                while not agent_task.done():
                    if esc_cancelled[0]:
                        cancel_event.set()
                    await asyncio.sleep(0.1)

                # Get result
                final_reply = agent_task.result()
                await _persist_turn_memory(
                    tools,
                    session_id=session_id,
                    category="conversation_assistant",
                    content=final_reply or "",
                )

            except asyncio.CancelledError:
                print(f"\n{Colors.BRIGHT_YELLOW}⚠️  Agent execution cancelled{Colors.RESET}")
            finally:
                agent.cancel_event = None
                esc_listener_stop.set()
                esc_thread.join(timeout=0.2)

            # Visual separation
            print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}\n")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.BRIGHT_YELLOW}👋 Interrupt signal detected, exiting...{Colors.RESET}\n")
            print_stats(agent, session_start)
            break

        except Exception as e:
            print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}")
            print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}\n")

    session_manager.stop_session(session_id)
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

    try:
        tickers = resolve_ticker_universe(args.tickers or "", args.all)
    except Exception as exc:
        print(f"{Colors.RED}❌ Failed to fetch ticker universe: {exc}{Colors.RESET}")
        return
    if not tickers:
        print(f"{Colors.RED}❌ Provide tickers (e.g. 600519,000001) or use --all{Colors.RESET}")
        return

    print(f"{Colors.CYAN}🔄 Syncing K-line data -> {db_path}{Colors.RESET}")
    result = sync_kline_data(kline_db, tickers=tickers, start=args.start, end=args.end)
    print(f"tickers={len(tickers)}, range={args.start}~{result.end_date}")

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

    # Run the agent (config always loaded from package directory)
    asyncio.run(run_agent(workspace_dir, task=args.task))


if __name__ == "__main__":
    main()
