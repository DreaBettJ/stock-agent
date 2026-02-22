"""Session management for stock experiment workflows."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from .paths import DEFAULT_MEMORY_DB_PATH

SessionMode = Literal["simulation", "backtest"]
SessionStatus = Literal["running", "stopped", "finished"]


@dataclass(slots=True)
class ExperimentSession:
    """Runtime model for one experiment session."""

    session_id: int
    name: str
    system_prompt: str
    mode: SessionMode
    initial_capital: float = 100000.0
    current_cash: float = 100000.0
    positions: dict[str, dict[str, float]] = field(default_factory=dict)
    status: SessionStatus = "stopped"
    is_listening: bool = False
    backtest_start: date | None = None
    backtest_end: date | None = None
    current_date: date | None = None
    event_filter: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SessionManager:
    """SQLite-backed session manager."""

    def __init__(self, db_path: str = str(DEFAULT_MEMORY_DB_PATH)):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            if not self._table_exists(conn, "sessions"):
                self._create_sessions_table(conn)
            elif self._needs_session_id_migration(conn):
                self._migrate_sessions_table(conn)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_mode ON sessions(mode)")
            conn.commit()

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            (table_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _create_sessions_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                mode TEXT NOT NULL,
                initial_capital REAL DEFAULT 100000,
                current_cash REAL,
                status TEXT DEFAULT 'stopped',
                is_listening INTEGER DEFAULT 0,
                backtest_start TEXT,
                backtest_end TEXT,
                current_date TEXT,
                event_filter TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT,
                legacy_session_id TEXT
            )
            """
        )

    @staticmethod
    def _needs_session_id_migration(conn: sqlite3.Connection) -> bool:
        rows = conn.execute("PRAGMA table_info(sessions)").fetchall()
        for row in rows:
            if row["name"] == "session_id":
                col_type = str(row["type"] or "").upper()
                return "INT" not in col_type
        return True

    def _migrate_sessions_table(self, conn: sqlite3.Connection) -> None:
        conn.execute("DROP TABLE IF EXISTS sessions_new")
        self._create_sessions_table_with_name(conn, "sessions_new")
        conn.execute(
            """
            INSERT INTO sessions_new (
                name, system_prompt, mode, initial_capital, current_cash, status, is_listening,
                backtest_start, backtest_end, current_date, event_filter, created_at, updated_at, legacy_session_id
            )
            SELECT
                name,
                system_prompt,
                mode,
                COALESCE(initial_capital, 100000),
                COALESCE(current_cash, COALESCE(initial_capital, 100000)),
                COALESCE(status, 'stopped'),
                COALESCE(is_listening, 0),
                backtest_start,
                backtest_end,
                current_date,
                event_filter,
                created_at,
                updated_at,
                CAST(session_id AS TEXT)
            FROM sessions
            ORDER BY COALESCE(created_at, updated_at, '')
            """
        )
        conn.execute("DROP TABLE sessions")
        conn.execute("ALTER TABLE sessions_new RENAME TO sessions")

    @staticmethod
    def _create_sessions_table_with_name(conn: sqlite3.Connection, table_name: str) -> None:
        conn.execute(
            f"""
            CREATE TABLE {table_name} (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                mode TEXT NOT NULL,
                initial_capital REAL DEFAULT 100000,
                current_cash REAL,
                status TEXT DEFAULT 'stopped',
                is_listening INTEGER DEFAULT 0,
                backtest_start TEXT,
                backtest_end TEXT,
                current_date TEXT,
                event_filter TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT,
                legacy_session_id TEXT
            )
            """
        )

    @staticmethod
    def _normalize_session_id(session_id: int | str) -> int:
        if isinstance(session_id, int):
            return session_id
        normalized = str(session_id).strip()
        if normalized.isdigit():
            return int(normalized)
        raise ValueError(f"session_id must be integer, got: {session_id}")

    @staticmethod
    def _normalize_mode(mode: str) -> SessionMode:
        normalized = mode.strip().lower()
        if normalized not in {"simulation", "backtest"}:
            raise ValueError("mode must be either 'simulation' or 'backtest'")
        return normalized  # type: ignore[return-value]

    @staticmethod
    def _normalize_status(status: str) -> SessionStatus:
        normalized = status.strip().lower()
        if normalized not in {"running", "stopped", "finished"}:
            raise ValueError("status must be one of: running, stopped, finished")
        return normalized  # type: ignore[return-value]

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        if not value:
            return None
        return date.fromisoformat(value)

    @staticmethod
    def _serialize_date(value: date | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, date):
            return value.isoformat()
        return date.fromisoformat(value).isoformat()

    def _row_to_session(self, row: sqlite3.Row) -> ExperimentSession:
        event_filter_raw = row["event_filter"]
        event_filter: list[str] = []
        if event_filter_raw:
            parsed = json.loads(event_filter_raw)
            if isinstance(parsed, list):
                event_filter = [str(item) for item in parsed if str(item).strip()]

        return ExperimentSession(
            session_id=int(row["session_id"]),
            name=row["name"],
            system_prompt=row["system_prompt"],
            mode=self._normalize_mode(row["mode"]),
            initial_capital=float(row["initial_capital"]),
            current_cash=float(row["current_cash"]),
            status=self._normalize_status(row["status"]),
            is_listening=bool(row["is_listening"]),
            backtest_start=self._parse_date(row["backtest_start"]),
            backtest_end=self._parse_date(row["backtest_end"]),
            current_date=self._parse_date(row["current_date"]),
            event_filter=event_filter,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )

    def create_session(self, name: str, system_prompt: str, mode: str, **kwargs: Any) -> int:
        """Create a new session and return integer session_id."""
        normalized_mode = self._normalize_mode(mode)
        session_id_raw = kwargs.get("session_id")
        initial_capital = float(kwargs.get("initial_capital", 100000.0))
        current_cash = float(kwargs.get("current_cash", initial_capital))
        backtest_start = self._serialize_date(kwargs.get("backtest_start"))
        backtest_end = self._serialize_date(kwargs.get("backtest_end"))
        current_date = self._serialize_date(kwargs.get("current_date"))
        event_filter = kwargs.get("event_filter") or []
        now = datetime.now().isoformat()

        if not isinstance(event_filter, list):
            raise ValueError("event_filter must be a list[str]")

        with self._connect() as conn:
            if session_id_raw is None:
                cursor = conn.execute(
                    """
                    INSERT INTO sessions (
                        name, system_prompt, mode,
                        initial_capital, current_cash, status, is_listening,
                        backtest_start, backtest_end, current_date,
                        event_filter, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'stopped', 0, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        system_prompt,
                        normalized_mode,
                        initial_capital,
                        current_cash,
                        backtest_start,
                        backtest_end,
                        current_date,
                        json.dumps(event_filter, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                session_id = int(cursor.lastrowid)
            else:
                session_id = self._normalize_session_id(session_id_raw)
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, name, system_prompt, mode,
                        initial_capital, current_cash, status, is_listening,
                        backtest_start, backtest_end, current_date,
                        event_filter, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'stopped', 0, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        name,
                        system_prompt,
                        normalized_mode,
                        initial_capital,
                        current_cash,
                        backtest_start,
                        backtest_end,
                        current_date,
                        json.dumps(event_filter, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            conn.commit()
        return session_id

    def start_session(self, session_id: int | str) -> None:
        """Start session and mark listening enabled."""
        self._update_session(self._normalize_session_id(session_id), status="running", is_listening=True)

    def stop_session(self, session_id: int | str) -> None:
        """Stop session and disable event listening."""
        self._update_session(self._normalize_session_id(session_id), status="stopped", is_listening=False)

    def finish_session(self, session_id: int | str) -> None:
        """Mark session as finished and disable listening."""
        self._update_session(self._normalize_session_id(session_id), status="finished", is_listening=False)

    def list_sessions(self) -> list[ExperimentSession]:
        """List all sessions sorted by create time descending."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
        return [self._row_to_session(row) for row in rows]

    def get_session(self, session_id: int | str) -> ExperimentSession:
        """Get one session by id."""
        sid = self._normalize_session_id(session_id)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (sid,)).fetchone()
        if row is None:
            raise KeyError(f"session not found: {sid}")
        return self._row_to_session(row)

    def get_listening_sessions(self, event_type: str | None = None) -> list[ExperimentSession]:
        """Get listening sessions, optionally filtered by event type."""
        sessions = [session for session in self.list_sessions() if session.is_listening]
        if not event_type:
            return sessions
        return [
            session
            for session in sessions
            if not session.event_filter or event_type in session.event_filter
        ]

    def update_event_filter(self, session_id: int | str, event_filter: list[str]) -> None:
        """Update session event filter."""
        if not isinstance(event_filter, list):
            raise ValueError("event_filter must be a list[str]")
        self._update_session(self._normalize_session_id(session_id), event_filter=json.dumps(event_filter, ensure_ascii=False))

    def update_current_cash(self, session_id: int | str, current_cash: float) -> None:
        """Update session cash."""
        self._update_session(self._normalize_session_id(session_id), current_cash=float(current_cash))

    def update_current_date(self, session_id: int | str, current_date: date | str | None) -> None:
        """Update current backtest date."""
        self._update_session(self._normalize_session_id(session_id), current_date=self._serialize_date(current_date))

    def delete_session(self, session_id: int | str) -> None:
        """Delete session."""
        sid = self._normalize_session_id(session_id)
        with self._connect() as conn:
            self._delete_if_table_exists(conn, "sim_positions", sid)
            self._delete_if_table_exists(conn, "sim_trades", sid)
            self._delete_if_table_exists(conn, "trades", sid)
            self._delete_if_table_exists(conn, "notes", sid)
            cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
            conn.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"session not found: {sid}")

    def broadcast_event(self, event: dict[str, Any]) -> list[ExperimentSession]:
        """Return listening sessions that should receive event."""
        event_type = str(event.get("type") or "").strip() or None
        return self.get_listening_sessions(event_type)

    @staticmethod
    def _delete_if_table_exists(conn: sqlite3.Connection, table_name: str, session_id: int) -> None:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
            (table_name,),
        ).fetchone()
        if exists:
            conn.execute(f"DELETE FROM {table_name} WHERE session_id = ?", (session_id,))

    def _update_session(self, session_id: int, **fields: Any) -> None:
        if not fields:
            return

        field_names = [name for name in fields.keys() if name != "session_id"]
        assignments = [f"{name} = ?" for name in field_names]
        values = [fields[name] for name in field_names]
        values.append(datetime.now().isoformat())
        values.append(session_id)

        sql = f"UPDATE sessions SET {', '.join(assignments)}, updated_at = ? WHERE session_id = ?"

        with self._connect() as conn:
            cursor = conn.execute(sql, values)
            conn.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"session not found: {session_id}")
