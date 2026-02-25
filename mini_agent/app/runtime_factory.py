"""Runtime session factory for CLI and non-CLI entrypoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mini_agent.session import SessionManager


@dataclass(slots=True)
class RuntimeSessionContext:
    """Bound runtime session context for one process."""

    session_manager: SessionManager
    session_id: int
    session_name: str
    runtime_type: str
    pid: int
    attached: bool
    effective_system_prompt: str

    def heartbeat(self) -> None:
        """Refresh runtime heartbeat for this process binding."""
        self.session_manager.heartbeat_session_runtime(
            self.session_id,
            runtime_type=self.runtime_type,
            pid=self.pid,
        )

    def close(self, *, stop_session: bool = True) -> None:
        """Unregister runtime and optionally stop session."""
        self.session_manager.unregister_session_runtime(
            self.session_id,
            runtime_type=self.runtime_type,
            pid=self.pid,
        )
        if stop_session:
            self.session_manager.stop_session(self.session_id)


def build_runtime_session_context(
    *,
    memory_db_path: Path,
    system_prompt: str,
    attach_session_id: int | None = None,
    runtime_type: str = "cli",
    pid: int | None = None,
    session_name_prefix: str = "runtime",
    session_name: str | None = None,
    mode: str = "simulation",
    initial_capital: float = 100000.0,
    create_session_kwargs: dict[str, Any] | None = None,
) -> RuntimeSessionContext:
    """Create or attach session and bind runtime to current process."""
    session_manager = SessionManager(db_path=str(memory_db_path))
    runtime_pid = int(pid or os.getpid())

    if attach_session_id is not None:
        attached = session_manager.get_session(attach_session_id)
        session_id = attached.session_id
        session_name = attached.name
        effective_prompt = attached.system_prompt or system_prompt
        attached_mode = (attached.mode or "").strip().lower()
        if mode and attached_mode and attached_mode != mode:
            raise ValueError(
                f"session mode mismatch: expected {mode}, got {attached.mode} (session_id={session_id})"
            )
        is_attached = True
    else:
        session_name = session_name or f"{session_name_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        extra_kwargs = dict(create_session_kwargs or {})
        session_id = session_manager.create_session(
            name=session_name,
            system_prompt=system_prompt,
            mode=mode,
            initial_capital=initial_capital,
            **extra_kwargs,
        )
        effective_prompt = system_prompt
        is_attached = False

    session_manager.start_session(session_id)
    session_manager.register_session_runtime(session_id, runtime_type=runtime_type, pid=runtime_pid)

    return RuntimeSessionContext(
        session_manager=session_manager,
        session_id=session_id,
        session_name=session_name,
        runtime_type=runtime_type,
        pid=runtime_pid,
        attached=is_attached,
        effective_system_prompt=effective_prompt,
    )
