"""Session Note Tool backed by SQLite.

This tool stores notes in a local SQLite database for durable persistence.
Each note is tagged with a session_id to support session-scoped recall.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from mini_agent.paths import DEFAULT_MEMORY_DB_PATH

from .base import Tool, ToolResult


class _NoteStore:
    """Lightweight SQLite note store."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            # New memory table aligned with PRD.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'note',
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance INTEGER DEFAULT 0,
                    is_deleted INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id, is_deleted)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")

            # Legacy compatibility: migrate old notes table into memories.
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='notes' LIMIT 1").fetchone():
                conn.execute(
                    """
                    INSERT INTO memories (session_id, role, category, content, importance, is_deleted, created_at)
                    SELECT n.session_id, 'note', n.category, n.content, 0, 0, n.timestamp
                    FROM notes n
                    WHERE NOT EXISTS (
                        SELECT 1 FROM memories m
                        WHERE m.session_id = n.session_id
                          AND m.category = n.category
                          AND m.content = n.content
                          AND m.created_at = n.timestamp
                    )
                    """
                )
            conn.commit()

    def insert(
        self,
        session_id: str,
        category: str,
        content: str,
        timestamp: str,
        importance: int = 0,
        role: str = "note",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (session_id, role, category, content, importance, is_deleted, created_at)
                VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (session_id, role, category, content, int(importance), timestamp),
            )
            conn.commit()

    def query(
        self,
        category: str | None = None,
        session_id: str | None = None,
        limit: int = 200,
    ) -> list[sqlite3.Row]:
        sql = """
            SELECT id, session_id, created_at, category, content, importance, role
            FROM memories
            WHERE is_deleted = 0
        """
        clauses = []
        params: list[Any] = []

        if category:
            clauses.append("category = ?")
            params.append(category)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)

        if clauses:
            sql += " AND " + " AND ".join(clauses)

        sql += " ORDER BY id ASC LIMIT ?"
        params.append(max(1, min(limit, 2000)))

        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            return list(cursor.fetchall())


class SessionNoteTool(Tool):
    """Tool for recording session notes into SQLite."""

    def __init__(self, memory_file: str = str(DEFAULT_MEMORY_DB_PATH), session_id: str | None = None):
        self.memory_file = Path(memory_file)
        self.session_id = session_id or uuid4().hex
        self.store = _NoteStore(self.memory_file)

    @property
    def name(self) -> str:
        return "record_memory"

    @property
    def description(self) -> str:
        return (
            "Record important information as session notes for future reference. "
            "Each note is persisted with a session_id and timestamp in SQLite."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to record as a note. Be concise but specific.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category/tag for this note (e.g., user_preference, project_info, decision)",
                },
                "importance": {
                    "type": "integer",
                    "description": "Importance score (0-10). High-importance memory should be retained.",
                    "default": 0,
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional override session_id. If omitted, tool default session_id is used.",
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str,
        category: str = "general",
        importance: int = 0,
        session_id: str | None = None,
    ) -> ToolResult:
        try:
            sid = session_id or self.session_id
            ts = datetime.now().isoformat()
            importance = max(0, min(int(importance), 10))
            self.store.insert(
                session_id=sid,
                category=category,
                content=content,
                timestamp=ts,
                importance=importance,
                role="note",
            )

            return ToolResult(
                success=True,
                content=f"Recorded memory: {content} (category: {category}, importance: {importance}, session_id: {sid})",
            )
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to record memory: {str(e)}")


class RecallNoteTool(Tool):
    """Tool for recalling notes from SQLite."""

    def __init__(self, memory_file: str = str(DEFAULT_MEMORY_DB_PATH), session_id: str | None = None):
        self.memory_file = Path(memory_file)
        self.session_id = session_id
        self.store = _NoteStore(self.memory_file)

    @property
    def name(self) -> str:
        return "recall_memories"

    @property
    def description(self) -> str:
        return (
            "Recall recorded session notes from SQLite. "
            "Can filter by category and/or session_id."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional: filter notes by category",
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional: filter notes by session_id. If omitted, returns all sessions.",
                },
                "current_session_only": {
                    "type": "boolean",
                    "description": "If true, only return notes from current tool session_id.",
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of notes to return (default 200).",
                    "default": 200,
                },
            },
        }

    async def execute(
        self,
        category: str | None = None,
        session_id: str | None = None,
        current_session_only: bool = False,
        limit: int = 200,
    ) -> ToolResult:
        try:
            effective_session_id = session_id
            if current_session_only:
                if not self.session_id:
                    return ToolResult(success=False, content="", error="current_session_only=true but no session_id bound to tool")
                effective_session_id = self.session_id

            notes = self.store.query(category=category, session_id=effective_session_id, limit=limit)

            if not notes:
                if effective_session_id:
                    return ToolResult(success=True, content=f"No notes found for session_id: {effective_session_id}")
                if category:
                    return ToolResult(success=True, content=f"No notes found in category: {category}")
                return ToolResult(success=True, content="No notes recorded yet.")

            formatted = []
            for idx, note in enumerate(notes, 1):
                timestamp = note["created_at"]
                cat = note["category"]
                sid = note["session_id"]
                content = note["content"]
                importance = int(note["importance"]) if note["importance"] is not None else 0
                formatted.append(
                    f"{idx}. [{cat}] {content}\n"
                    f"   (session_id: {sid}, importance: {importance}, recorded at {timestamp})"
                )

            return ToolResult(success=True, content="Recorded Notes:\n" + "\n".join(formatted))
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to recall notes: {str(e)}")
