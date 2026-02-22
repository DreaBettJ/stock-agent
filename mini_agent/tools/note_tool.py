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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_session ON notes(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_category ON notes(category)")
            conn.commit()

    def insert(self, session_id: str, category: str, content: str, timestamp: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO notes (session_id, timestamp, category, content) VALUES (?, ?, ?, ?)",
                (session_id, timestamp, category, content),
            )
            conn.commit()

    def query(
        self,
        category: str | None = None,
        session_id: str | None = None,
        limit: int = 200,
    ) -> list[sqlite3.Row]:
        sql = "SELECT id, session_id, timestamp, category, content FROM notes"
        clauses = []
        params: list[Any] = []

        if category:
            clauses.append("category = ?")
            params.append(category)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)

        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

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
        return "record_note"

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
                "session_id": {
                    "type": "string",
                    "description": "Optional override session_id. If omitted, tool default session_id is used.",
                },
            },
            "required": ["content"],
        }

    async def execute(self, content: str, category: str = "general", session_id: str | None = None) -> ToolResult:
        try:
            sid = session_id or self.session_id
            ts = datetime.now().isoformat()
            self.store.insert(session_id=sid, category=category, content=content, timestamp=ts)

            return ToolResult(
                success=True,
                content=f"Recorded note: {content} (category: {category}, session_id: {sid})",
            )
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to record note: {str(e)}")


class RecallNoteTool(Tool):
    """Tool for recalling notes from SQLite."""

    def __init__(self, memory_file: str = str(DEFAULT_MEMORY_DB_PATH), session_id: str | None = None):
        self.memory_file = Path(memory_file)
        self.session_id = session_id
        self.store = _NoteStore(self.memory_file)

    @property
    def name(self) -> str:
        return "recall_notes"

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
                timestamp = note["timestamp"]
                cat = note["category"]
                sid = note["session_id"]
                content = note["content"]
                formatted.append(f"{idx}. [{cat}] {content}\n   (session_id: {sid}, recorded at {timestamp})")

            return ToolResult(success=True, content="Recorded Notes:\n" + "\n".join(formatted))
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to recall notes: {str(e)}")
