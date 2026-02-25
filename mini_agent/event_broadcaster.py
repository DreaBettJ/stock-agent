"""Event broadcasting for experiment sessions."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from .session import ExperimentSession, SessionManager

TriggerCallable = Callable[[ExperimentSession, dict[str, Any]], Awaitable[str]]


class EventBroadcaster:
    """Broadcasts one event to listening sessions asynchronously."""
    # 事件广播器职责：
    # 1) 从 SessionManager 获取应接收事件的会话
    # 2) 并发触发每个会话的回调
    # 3) 统一收集并返回结构化结果

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
        # 先由 session 层完成过滤（listening + event_filter）。
        target_sessions = self.session_manager.broadcast_event(event)
        # 每个会话一个异步任务，并发触发，互不阻塞。
        tasks = [asyncio.create_task(self.trigger_session(session, event)) for session in target_sessions]

        if not tasks:
            return []

        # 返回每个会话的执行结果（包含 success / error / result）。
        return await asyncio.gather(*tasks)

    async def trigger_session(self, session: ExperimentSession, event: dict[str, Any]) -> dict[str, Any]:
        """Trigger one session and record result."""
        begin_token = self.session_manager.begin_event_processing(session.session_id, event)
        if not begin_token.get("process"):
            reason = f"idempotent_skip status={begin_token.get('status')}"
            record = {
                "session_id": session.session_id,
                "event_type": event.get("type"),
                "success": False,
                "error": reason,
            }
            self.session_manager.mark_event_skipped(session.session_id, event, reason)
            self.event_results.append(record)
            return record

        try:
            # 调用外部注入的 trigger 回调，保持广播器本身与具体业务解耦。
            output = await self.trigger(session, event)
            record = {
                "session_id": session.session_id,
                "event_type": event.get("type"),
                "success": True,
                "result": output,
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            # 单个会话失败不应影响其他会话，错误转为结构化结果返回。
            record = {
                "session_id": session.session_id,
                "event_type": event.get("type"),
                "success": False,
                "error": str(exc),
            }

        self.session_manager.finalize_event_processing(
            session_id=session.session_id,
            event=event,
            status="succeeded" if record.get("success") else "failed",
            success=bool(record.get("success")),
            result=record.get("result"),
            error=record.get("error"),
            started_at=begin_token.get("started_at"),
        )

        # 保存历史结果，便于上层调试/观测。
        self.event_results.append(record)
        return record
