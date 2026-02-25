"""Session-scoped memory read helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def load_recent_session_memories(db_path: Path, session_id: int, limit: int = 12) -> list[dict[str, str]]:
    """Load recent non-deleted memories for one session."""
    if limit <= 0:
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT category, content, created_at
            FROM memories
            WHERE is_deleted = 0
              AND session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (str(session_id), limit),
        ).fetchall()

    # Reverse for chronological display.
    return [
        {
            "category": str(r["category"] or "general"),
            "content": str(r["content"] or ""),
            "created_at": str(r["created_at"] or ""),
        }
        for r in reversed(rows)
        if str(r["content"] or "").strip()
    ]


def load_critical_session_memories(db_path: Path, session_id: int, limit: int = 8) -> list[dict[str, str]]:
    """Load latest critical memories for one session."""
    if limit <= 0:
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT event_type, operation, reason, content, created_at
            FROM critical_memories
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(session_id), limit),
        ).fetchall()
    items: list[dict[str, str]] = []
    for r in reversed(rows):
        event_type = str(r["event_type"] or "event")
        operation = str(r["operation"] or "decision")
        reason = str(r["reason"] or "")
        content = str(r["content"] or "")
        snippet = f"[{event_type}/{operation}] {content}"
        if reason:
            snippet += f" | reason={reason}"
        items.append(
            {
                "category": "critical_memory",
                "content": snippet,
                "created_at": str(r["created_at"] or ""),
            }
        )
    return items


def build_session_memory_snapshot(memories: list[dict[str, str]]) -> str:
    """Format memories into one compact snapshot block for prompt injection."""
    if not memories:
        return ""
    lines: list[str] = []
    for i, m in enumerate(memories, 1):
        lines.append(f"{i}. [{m['category']}] {m['content']}")
    return "\n".join(lines)
