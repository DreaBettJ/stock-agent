"""Event broadcasting for experiment sessions."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from .session import ExperimentSession, SessionManager

TriggerCallable = Callable[[ExperimentSession, dict[str, Any]], Awaitable[str]]


class EventBroadcaster:
    """Broadcasts one event to listening sessions asynchronously."""

    def __init__(
        self,
        session_manager: SessionManager,
        trigger: TriggerCallable,
    ):
        self.session_manager = session_manager
        self.trigger = trigger
        self.event_results: list[dict[str, Any]] = []

    async def broadcast(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Broadcast event to matching listening sessions."""
        target_sessions = self.session_manager.broadcast_event(event)
        tasks = [asyncio.create_task(self.trigger_session(session, event)) for session in target_sessions]

        if not tasks:
            return []

        return await asyncio.gather(*tasks)

    async def trigger_session(self, session: ExperimentSession, event: dict[str, Any]) -> dict[str, Any]:
        """Trigger one session and record result."""
        try:
            output = await self.trigger(session, event)
            record = {
                "session_id": session.session_id,
                "event_type": event.get("type"),
                "success": True,
                "result": output,
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            record = {
                "session_id": session.session_id,
                "event_type": event.get("type"),
                "success": False,
                "error": str(exc),
            }

        self.event_results.append(record)
        return record
