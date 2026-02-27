"""Session management for stock experiment workflows."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
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
    risk_preference: str = "medium"
    max_single_loss_pct: float = 2.0
    single_position_cap_pct: float = 25.0
    stop_loss_pct: float = 7.0
    take_profit_pct: float = 15.0
    investment_horizon: str = "中线"
    trade_notice_enabled: bool = False
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

            self._create_session_runtime_table(conn)
            self._create_event_inbox_table(conn)
            self._create_event_logs_table(conn)
            self._create_critical_memories_table(conn)
            self._create_trade_action_logs_table(conn)
            self._ensure_sessions_schema(conn)
            self._ensure_event_inbox_schema(conn)
            self._ensure_event_logs_schema(conn)
            self._ensure_trade_action_logs_schema(conn)
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
                risk_preference TEXT DEFAULT 'medium',
                max_single_loss_pct REAL DEFAULT 2.0,
                single_position_cap_pct REAL DEFAULT 25.0,
                stop_loss_pct REAL DEFAULT 7.0,
                take_profit_pct REAL DEFAULT 15.0,
                investment_horizon TEXT DEFAULT '中线',
                trade_notice_enabled INTEGER DEFAULT 0,
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
    def _create_event_logs_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                event_id TEXT NOT NULL,
                event_type TEXT,
                event_payload TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                retry_count INTEGER NOT NULL DEFAULT 0,
                success INTEGER,
                result TEXT,
                error TEXT,
                triggered_at TEXT,
                queued_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                latency_ms INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_logs_session ON event_logs(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_logs_event_id ON event_logs(event_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_logs_type ON event_logs(event_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_logs_created_at ON event_logs(created_at)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_event_logs_session_event_id ON event_logs(session_id, event_id)")

    @staticmethod
    def _create_session_runtime_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_runtime (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                runtime_type TEXT NOT NULL DEFAULT 'cli',
                pid INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'running',
                started_at TEXT,
                last_heartbeat TEXT,
                updated_at TEXT,
                UNIQUE(session_id, runtime_type, pid)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session_runtime_session ON session_runtime(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session_runtime_status ON session_runtime(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session_runtime_heartbeat ON session_runtime(last_heartbeat)")

    @staticmethod
    def _create_event_inbox_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_inbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                event_id TEXT NOT NULL,
                event_type TEXT,
                event_payload TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                retry_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT,
                updated_at TEXT,
                locked_at TEXT,
                processed_at TEXT,
                UNIQUE(session_id, event_id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_inbox_session ON event_inbox(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_inbox_status ON event_inbox(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_inbox_created ON event_inbox(created_at)")

    @staticmethod
    def _create_critical_memories_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS critical_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                event_id TEXT,
                event_type TEXT,
                operation TEXT,
                reason TEXT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_critical_memories_session ON critical_memories(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_critical_memories_event ON critical_memories(event_id)")

    @staticmethod
    def _create_trade_action_logs_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_action_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                event_id TEXT,
                event_type TEXT,
                action TEXT,
                ticker TEXT,
                quantity INTEGER,
                trade_date TEXT,
                status TEXT NOT NULL,
                error_code TEXT,
                reason TEXT,
                request_payload TEXT,
                response_payload TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_action_logs_session ON trade_action_logs(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_action_logs_event ON trade_action_logs(event_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_action_logs_status ON trade_action_logs(status)")

    @staticmethod
    def _ensure_event_inbox_schema(conn: sqlite3.Connection) -> None:
        if not SessionManager._table_exists(conn, "event_inbox"):
            return
        rows = conn.execute("PRAGMA table_info(event_inbox)").fetchall()
        existing = {str(row["name"]) for row in rows}
        desired: list[tuple[str, str]] = [
            ("retry_count", "INTEGER NOT NULL DEFAULT 0"),
            ("locked_at", "TEXT"),
            ("processed_at", "TEXT"),
        ]
        for name, ddl in desired:
            if name not in existing:
                conn.execute(f"ALTER TABLE event_inbox ADD COLUMN {name} {ddl}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_inbox_status ON event_inbox(status)")

    @staticmethod
    def _ensure_event_logs_schema(conn: sqlite3.Connection) -> None:
        if not SessionManager._table_exists(conn, "event_logs"):
            return

        rows = conn.execute("PRAGMA table_info(event_logs)").fetchall()
        existing = {str(row["name"]) for row in rows}
        desired: list[tuple[str, str]] = [
            ("event_id", "TEXT"),
            ("status", "TEXT NOT NULL DEFAULT 'pending'"),
            ("retry_count", "INTEGER NOT NULL DEFAULT 0"),
            ("queued_at", "TEXT"),
            ("started_at", "TEXT"),
            ("finished_at", "TEXT"),
            ("latency_ms", "INTEGER"),
        ]
        for name, ddl in desired:
            if name not in existing:
                conn.execute(f"ALTER TABLE event_logs ADD COLUMN {name} {ddl}")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_logs_event_id ON event_logs(event_id)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_event_logs_session_event_id ON event_logs(session_id, event_id)")

    @staticmethod
    def _ensure_trade_action_logs_schema(conn: sqlite3.Connection) -> None:
        if not SessionManager._table_exists(conn, "trade_action_logs"):
            return
        rows = conn.execute("PRAGMA table_info(trade_action_logs)").fetchall()
        existing = {str(row["name"]) for row in rows}
        desired: list[tuple[str, str]] = [
            ("event_id", "TEXT"),
            ("event_type", "TEXT"),
            ("action", "TEXT"),
            ("ticker", "TEXT"),
            ("quantity", "INTEGER"),
            ("trade_date", "TEXT"),
            ("status", "TEXT NOT NULL DEFAULT 'unknown'"),
            ("error_code", "TEXT"),
            ("reason", "TEXT"),
            ("request_payload", "TEXT"),
            ("response_payload", "TEXT"),
            ("created_at", "TEXT"),
        ]
        for name, ddl in desired:
            if name not in existing:
                conn.execute(f"ALTER TABLE trade_action_logs ADD COLUMN {name} {ddl}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_action_logs_session ON trade_action_logs(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_action_logs_event ON trade_action_logs(event_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_action_logs_status ON trade_action_logs(status)")

    @staticmethod
    def _ensure_sessions_schema(conn: sqlite3.Connection) -> None:
        if not SessionManager._table_exists(conn, "sessions"):
            return
        rows = conn.execute("PRAGMA table_info(sessions)").fetchall()
        existing = {str(row["name"]) for row in rows}
        desired: list[tuple[str, str]] = [
            ("risk_preference", "TEXT DEFAULT 'medium'"),
            ("max_single_loss_pct", "REAL DEFAULT 2.0"),
            ("single_position_cap_pct", "REAL DEFAULT 25.0"),
            ("stop_loss_pct", "REAL DEFAULT 7.0"),
            ("take_profit_pct", "REAL DEFAULT 15.0"),
            ("investment_horizon", "TEXT DEFAULT '中线'"),
            ("trade_notice_enabled", "INTEGER DEFAULT 0"),
        ]
        for name, ddl in desired:
            if name not in existing:
                conn.execute(f"ALTER TABLE sessions ADD COLUMN {name} {ddl}")

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
                name, system_prompt, mode, initial_capital, current_cash,
                risk_preference, max_single_loss_pct, single_position_cap_pct, stop_loss_pct, take_profit_pct, investment_horizon,
                trade_notice_enabled,
                status, is_listening,
                backtest_start, backtest_end, current_date, event_filter, created_at, updated_at, legacy_session_id
            )
            SELECT
                name,
                system_prompt,
                mode,
                COALESCE(initial_capital, 100000),
                COALESCE(current_cash, COALESCE(initial_capital, 100000)),
                'medium',
                2.0,
                25.0,
                7.0,
                15.0,
                '中线',
                0,
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
                risk_preference TEXT DEFAULT 'medium',
                max_single_loss_pct REAL DEFAULT 2.0,
                single_position_cap_pct REAL DEFAULT 25.0,
                stop_loss_pct REAL DEFAULT 7.0,
                take_profit_pct REAL DEFAULT 15.0,
                investment_horizon TEXT DEFAULT '中线',
                trade_notice_enabled INTEGER DEFAULT 0,
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

    @staticmethod
    def _parse_db_datetime(value: str | None) -> datetime | None:
        """Parse DB datetime with UTC->UTC+8 normalization for SQLite CURRENT_TIMESTAMP rows.

        Rules:
        - SQLite CURRENT_TIMESTAMP style (`YYYY-MM-DD HH:MM:SS[.fff]`) is treated as UTC and converted to UTC+8.
        - ISO 8601 with timezone is converted to UTC+8.
        - ISO 8601 without timezone is kept as-is (application-local historical behavior).
        """
        if not value:
            return None
        raw = str(value).strip()
        if not raw:
            return None

        # SQLite CURRENT_TIMESTAMP format (UTC).
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                utc_dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
                return utc_dt.astimezone(timezone(timedelta(hours=8))).replace(tzinfo=None)
            except ValueError:
                pass

        # ISO 8601 string.
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt
            return dt.astimezone(timezone(timedelta(hours=8))).replace(tzinfo=None)
        except ValueError:
            return None

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
            risk_preference=str(row["risk_preference"] or "medium"),
            max_single_loss_pct=float(row["max_single_loss_pct"] if row["max_single_loss_pct"] is not None else 2.0),
            single_position_cap_pct=float(
                row["single_position_cap_pct"] if row["single_position_cap_pct"] is not None else 25.0
            ),
            stop_loss_pct=float(row["stop_loss_pct"] if row["stop_loss_pct"] is not None else 7.0),
            take_profit_pct=float(row["take_profit_pct"] if row["take_profit_pct"] is not None else 15.0),
            investment_horizon=str(row["investment_horizon"] or "中线"),
            trade_notice_enabled=bool(row["trade_notice_enabled"] if row["trade_notice_enabled"] is not None else 0),
            status=self._normalize_status(row["status"]),
            is_listening=bool(row["is_listening"]),
            backtest_start=self._parse_date(row["backtest_start"]),
            backtest_end=self._parse_date(row["backtest_end"]),
            current_date=self._parse_date(row["current_date"]),
            event_filter=event_filter,
            created_at=self._parse_db_datetime(row["created_at"]),
            updated_at=self._parse_db_datetime(row["updated_at"]),
        )

    def create_session(self, name: str, system_prompt: str, mode: str, **kwargs: Any) -> int:
        """Create a new session and return integer session_id."""
        normalized_mode = self._normalize_mode(mode)
        session_id_raw = kwargs.get("session_id")
        initial_capital = float(kwargs.get("initial_capital", 100000.0))
        current_cash = float(kwargs.get("current_cash", initial_capital))
        risk_preference = str(kwargs.get("risk_preference", "medium")).strip().lower() or "medium"
        max_single_loss_pct = float(kwargs.get("max_single_loss_pct", 2.0))
        single_position_cap_pct = float(kwargs.get("single_position_cap_pct", 25.0))
        stop_loss_pct = float(kwargs.get("stop_loss_pct", 7.0))
        take_profit_pct = float(kwargs.get("take_profit_pct", 15.0))
        investment_horizon = str(kwargs.get("investment_horizon", "中线")).strip() or "中线"
        trade_notice_enabled = bool(kwargs.get("trade_notice_enabled", False))
        backtest_start = self._serialize_date(kwargs.get("backtest_start"))
        backtest_end = self._serialize_date(kwargs.get("backtest_end"))
        current_date = self._serialize_date(kwargs.get("current_date"))
        event_filter = kwargs.get("event_filter") or []
        now = datetime.now().isoformat()

        if not isinstance(event_filter, list):
            raise ValueError("event_filter must be a list[str]")
        if risk_preference not in {"low", "medium", "high"}:
            raise ValueError("risk_preference must be one of: low, medium, high")

        with self._connect() as conn:
            if session_id_raw is None:
                cursor = conn.execute(
                    """
                    INSERT INTO sessions (
                        name, system_prompt, mode,
                        initial_capital, current_cash, risk_preference,
                        max_single_loss_pct, single_position_cap_pct, stop_loss_pct, take_profit_pct, investment_horizon,
                        trade_notice_enabled,
                        status, is_listening,
                        backtest_start, backtest_end, current_date,
                        event_filter, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'stopped', 0, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        system_prompt,
                        normalized_mode,
                        initial_capital,
                        current_cash,
                        risk_preference,
                        max_single_loss_pct,
                        single_position_cap_pct,
                        stop_loss_pct,
                        take_profit_pct,
                        investment_horizon,
                        1 if trade_notice_enabled else 0,
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
                        initial_capital, current_cash, risk_preference,
                        max_single_loss_pct, single_position_cap_pct, stop_loss_pct, take_profit_pct, investment_horizon,
                        trade_notice_enabled,
                        status, is_listening,
                        backtest_start, backtest_end, current_date,
                        event_filter, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'stopped', 0, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        name,
                        system_prompt,
                        normalized_mode,
                        initial_capital,
                        current_cash,
                        risk_preference,
                        max_single_loss_pct,
                        single_position_cap_pct,
                        stop_loss_pct,
                        take_profit_pct,
                        investment_horizon,
                        1 if trade_notice_enabled else 0,
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

    def update_system_prompt(self, session_id: int | str, system_prompt: str) -> None:
        """Update session system prompt."""
        self._update_session(self._normalize_session_id(session_id), system_prompt=system_prompt)

    def update_risk_profile(
        self,
        session_id: int | str,
        *,
        risk_preference: str | None = None,
        max_single_loss_pct: float | None = None,
        single_position_cap_pct: float | None = None,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        investment_horizon: str | None = None,
        trade_notice_enabled: bool | None = None,
    ) -> None:
        """Update session-level trading risk configuration."""
        fields: dict[str, Any] = {}
        if risk_preference is not None:
            normalized = str(risk_preference).strip().lower()
            if normalized not in {"low", "medium", "high"}:
                raise ValueError("risk_preference must be one of: low, medium, high")
            fields["risk_preference"] = normalized
        if max_single_loss_pct is not None:
            fields["max_single_loss_pct"] = float(max_single_loss_pct)
        if single_position_cap_pct is not None:
            fields["single_position_cap_pct"] = float(single_position_cap_pct)
        if stop_loss_pct is not None:
            fields["stop_loss_pct"] = float(stop_loss_pct)
        if take_profit_pct is not None:
            fields["take_profit_pct"] = float(take_profit_pct)
        if investment_horizon is not None:
            fields["investment_horizon"] = str(investment_horizon).strip() or "中线"
        if trade_notice_enabled is not None:
            fields["trade_notice_enabled"] = 1 if bool(trade_notice_enabled) else 0
        if not fields:
            return
        self._update_session(self._normalize_session_id(session_id), **fields)

    def delete_session(self, session_id: int | str) -> None:
        """Delete session."""
        sid = self._normalize_session_id(session_id)
        with self._connect() as conn:
            self._delete_if_table_exists(conn, "sim_positions", sid)
            self._delete_if_table_exists(conn, "sim_trades", sid)
            self._delete_if_table_exists(conn, "trades", sid)
            self._delete_if_table_exists(conn, "notes", sid)
            self._delete_if_table_exists(conn, "event_logs", sid)
            self._delete_if_table_exists(conn, "event_inbox", sid)
            self._delete_if_table_exists(conn, "session_runtime", sid)
            self._delete_if_table_exists(conn, "critical_memories", sid)
            self._delete_if_table_exists(conn, "trade_action_logs", sid)
            cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
            conn.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"session not found: {sid}")

    def broadcast_event(self, event: dict[str, Any]) -> list[ExperimentSession]:
        """Return listening sessions that should receive event."""
        event_type = str(event.get("type") or "").strip() or None
        return self.get_listening_sessions(event_type)

    def register_session_runtime(self, session_id: int | str, *, runtime_type: str = "cli", pid: int = 0) -> None:
        """Register one running session runtime (foreground or background CLI)."""
        sid = self._normalize_session_id(session_id)
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_runtime (
                    session_id, runtime_type, pid, status, started_at, last_heartbeat, updated_at
                ) VALUES (?, ?, ?, 'running', ?, ?, ?)
                ON CONFLICT(session_id, runtime_type, pid) DO UPDATE SET
                    status='running',
                    last_heartbeat=excluded.last_heartbeat,
                    updated_at=excluded.updated_at
                """,
                (sid, runtime_type, int(pid), now, now, now),
            )
            conn.commit()

    def heartbeat_session_runtime(self, session_id: int | str, *, runtime_type: str = "cli", pid: int = 0) -> None:
        """Refresh runtime heartbeat."""
        sid = self._normalize_session_id(session_id)
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE session_runtime
                SET status='running', last_heartbeat=?, updated_at=?
                WHERE session_id=? AND runtime_type=? AND pid=?
                """,
                (now, now, sid, runtime_type, int(pid)),
            )
            if cursor.rowcount == 0:
                conn.execute(
                    """
                    INSERT INTO session_runtime (
                        session_id, runtime_type, pid, status, started_at, last_heartbeat, updated_at
                    ) VALUES (?, ?, ?, 'running', ?, ?, ?)
                    """,
                    (sid, runtime_type, int(pid), now, now, now),
                )
            conn.commit()

    def unregister_session_runtime(self, session_id: int | str, *, runtime_type: str = "cli", pid: int = 0) -> None:
        """Mark runtime offline."""
        sid = self._normalize_session_id(session_id)
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE session_runtime
                SET status='stopped', updated_at=?
                WHERE session_id=? AND runtime_type=? AND pid=?
                """,
                (now, sid, runtime_type, int(pid)),
            )
            conn.commit()

    def list_online_session_ids(self, heartbeat_ttl_seconds: int = 60) -> list[int]:
        """Return session ids with active runtime heartbeat."""
        cutoff = (datetime.now().timestamp() - heartbeat_ttl_seconds)
        cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT session_id
                FROM session_runtime
                WHERE status='running' AND last_heartbeat >= ?
                ORDER BY session_id ASC
                """,
                (cutoff_iso,),
            ).fetchall()
        return [int(row["session_id"]) for row in rows]

    def is_session_online(self, session_id: int | str, heartbeat_ttl_seconds: int = 60) -> bool:
        """Check whether one session currently has active runtime."""
        sid = self._normalize_session_id(session_id)
        return sid in set(self.list_online_session_ids(heartbeat_ttl_seconds=heartbeat_ttl_seconds))

    def enqueue_event(self, session_id: int | str, event: dict[str, Any]) -> tuple[bool, str]:
        """Enqueue one event for session; idempotent by (session_id, event_id)."""
        sid = self._normalize_session_id(session_id)
        now = datetime.now().isoformat()
        event_id = self._normalize_event_id(event)
        event_type = str(event.get("type") or "")
        payload = json.dumps(event, ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO event_inbox (
                    session_id, event_id, event_type, event_payload,
                    status, retry_count, error, created_at, updated_at, locked_at, processed_at
                ) VALUES (?, ?, ?, ?, 'pending', 0, NULL, ?, ?, NULL, NULL)
                """,
                (sid, event_id, event_type, payload, now, now),
            )
            conn.commit()
        return (cursor.rowcount > 0, event_id)

    def claim_next_pending_event(self, session_id: int | str) -> dict[str, Any] | None:
        """Claim next pending inbox event for processing."""
        sid = self._normalize_session_id(session_id)
        now = datetime.now().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, session_id, event_id, event_type, event_payload, status, retry_count,
                       error, created_at, updated_at, locked_at, processed_at
                FROM event_inbox
                WHERE session_id=? AND status='pending'
                ORDER BY id ASC
                LIMIT 1
                """,
                (sid,),
            ).fetchone()
            if row is None:
                return None
            cursor = conn.execute(
                """
                UPDATE event_inbox
                SET status='running', locked_at=?, updated_at=?
                WHERE id=? AND status='pending'
                """,
                (now, now, int(row["id"])),
            )
            conn.commit()
        if cursor.rowcount == 0:
            return None
        item = dict(row)
        payload_raw = item.get("event_payload")
        if isinstance(payload_raw, str) and payload_raw:
            try:
                item["event_payload"] = json.loads(payload_raw)
            except json.JSONDecodeError:
                pass
        return item

    def complete_inbox_event(self, inbox_id: int, *, success: bool, error: str | None = None) -> None:
        """Mark claimed inbox event as completed."""
        now = datetime.now().isoformat()
        status = "succeeded" if success else "failed"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE event_inbox
                SET status=?, error=?, processed_at=?, updated_at=?
                WHERE id=?
                """,
                (status, error, now, now, int(inbox_id)),
            )
            conn.commit()

    def requeue_inbox_event(self, inbox_id: int, *, error: str) -> None:
        """Requeue failed running inbox event for retry."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE event_inbox
                SET status='pending', retry_count=retry_count+1, error=?, updated_at=?
                WHERE id=?
                """,
                (error, now, int(inbox_id)),
            )
            conn.commit()

    def list_event_inbox(
        self,
        session_id: int | str | None = None,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List inbox events for observability/testing."""
        if limit <= 0:
            return []
        sql = [
            "SELECT id, session_id, event_id, event_type, event_payload, status, retry_count, error,",
            "created_at, updated_at, locked_at, processed_at",
            "FROM event_inbox",
        ]
        params: list[Any] = []
        where: list[str] = []
        if session_id is not None:
            where.append("session_id = ?")
            params.append(self._normalize_session_id(session_id))
        if status:
            where.append("status = ?")
            params.append(status)
        if where:
            sql.append("WHERE " + " AND ".join(where))
        sql.append("ORDER BY id DESC LIMIT ?")
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(" ".join(sql), tuple(params)).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            payload_raw = item.get("event_payload")
            if isinstance(payload_raw, str) and payload_raw:
                try:
                    item["event_payload"] = json.loads(payload_raw)
                except json.JSONDecodeError:
                    pass
            results.append(item)
        return results

    def record_critical_memory(
        self,
        session_id: int | str,
        *,
        content: str,
        operation: str | None = None,
        reason: str | None = None,
        event_id: str | None = None,
        event_type: str | None = None,
    ) -> int:
        """Persist non-prunable critical memory for trading actions/decisions.

        Idempotency rule:
        - when event_id is provided, avoid duplicating the same (session_id, event_id, operation)
        """
        sid = self._normalize_session_id(session_id)
        now = datetime.now().isoformat()
        normalized_event_id = (event_id or "").strip() or None
        normalized_event_type = (event_type or "").strip() or None
        normalized_operation = (operation or "").strip() or None
        with self._connect() as conn:
            if normalized_event_id:
                existing = conn.execute(
                    """
                    SELECT id
                    FROM critical_memories
                    WHERE session_id = ?
                      AND event_id = ?
                      AND COALESCE(operation, '') = COALESCE(?, '')
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (sid, normalized_event_id, normalized_operation),
                ).fetchone()
                if existing is not None:
                    return int(existing["id"])

            cursor = conn.execute(
                """
                INSERT INTO critical_memories (
                    session_id, event_id, event_type, operation, reason, content, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    normalized_event_id,
                    normalized_event_type,
                    normalized_operation,
                    reason,
                    content,
                    now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def list_critical_memories(self, session_id: int | str, limit: int = 20) -> list[dict[str, Any]]:
        """List latest critical memories for one session."""
        sid = self._normalize_session_id(session_id)
        if limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, event_id, event_type, operation, reason, content, created_at
                FROM critical_memories
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (sid, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _normalize_event_id(event: dict[str, Any]) -> str:
        event_id = str(event.get("event_id") or "").strip()
        if event_id:
            return event_id

        event_type = str(event.get("type") or "event").strip() or "event"
        event_date = str(event.get("date") or "").strip()
        if event_date:
            return f"{event_type}:{event_date}"
        triggered_at = str(event.get("triggered_at") or datetime.now().isoformat()).strip()
        return f"{event_type}:{triggered_at}"

    def _fetch_event_log_row(self, session_id: int, event_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT id, session_id, event_id, event_type, event_payload, status, retry_count,
                       success, result, error, triggered_at, queued_at, started_at, finished_at, latency_ms, created_at
                FROM event_logs
                WHERE session_id = ? AND event_id = ?
                LIMIT 1
                """,
                (session_id, event_id),
            ).fetchone()

    def begin_event_processing(self, session_id: int | str, event: dict[str, Any]) -> dict[str, Any]:
        """Reserve event processing slot with idempotency on (session_id, event_id)."""
        sid = self._normalize_session_id(session_id)
        event_id = self._normalize_event_id(event)
        now = datetime.now().isoformat()
        event_type = str(event.get("type") or "")
        triggered_at = str(event.get("triggered_at") or now)
        payload = json.dumps(event, ensure_ascii=False)

        existing = self._fetch_event_log_row(sid, event_id)
        if existing is None:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO event_logs (
                        session_id, event_id, event_type, event_payload,
                        status, retry_count, success, result, error,
                        triggered_at, queued_at, started_at, finished_at, latency_ms, created_at
                    ) VALUES (?, ?, ?, ?, 'running', 0, NULL, NULL, NULL, ?, ?, ?, NULL, NULL, ?)
                    """,
                    (sid, event_id, event_type, payload, triggered_at, now, now, now),
                )
                conn.commit()
            return {
                "process": True,
                "event_id": event_id,
                "status": "running",
                "retry_count": 0,
                "queued_at": now,
                "started_at": now,
            }

        existing_status = str(existing["status"] or "")
        if existing_status in {"succeeded", "running"}:
            return {
                "process": False,
                "event_id": event_id,
                "status": existing_status,
                "retry_count": int(existing["retry_count"] or 0),
                "queued_at": existing["queued_at"],
                "started_at": existing["started_at"],
            }

        retry_count = int(existing["retry_count"] or 0) + 1
        queued_at = existing["queued_at"] or now
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE event_logs
                SET event_type = ?, event_payload = ?, status = 'running',
                    retry_count = ?, success = NULL, result = NULL, error = NULL,
                    triggered_at = ?, queued_at = ?, started_at = ?, finished_at = NULL, latency_ms = NULL
                WHERE session_id = ? AND event_id = ?
                """,
                (event_type, payload, retry_count, triggered_at, queued_at, now, sid, event_id),
            )
            conn.commit()
        return {
            "process": True,
            "event_id": event_id,
            "status": "running",
            "retry_count": retry_count,
            "queued_at": queued_at,
            "started_at": now,
        }

    def finalize_event_processing(
        self,
        session_id: int | str,
        event: dict[str, Any],
        *,
        status: str,
        success: bool | None,
        result: str | None = None,
        error: str | None = None,
        started_at: str | None = None,
    ) -> None:
        """Finalize one event processing attempt."""
        sid = self._normalize_session_id(session_id)
        event_id = self._normalize_event_id(event)
        now = datetime.now()
        finished_at = now.isoformat()
        started_dt = None
        if started_at:
            try:
                started_dt = datetime.fromisoformat(started_at)
            except ValueError:
                started_dt = None
        latency_ms = int((now - started_dt).total_seconds() * 1000) if started_dt else None

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE event_logs
                SET status = ?, success = ?, result = ?, error = ?, finished_at = ?, latency_ms = ?
                WHERE session_id = ? AND event_id = ?
                """,
                (status, None if success is None else (1 if success else 0), result, error, finished_at, latency_ms, sid, event_id),
            )
            conn.commit()

    def mark_event_skipped(self, session_id: int | str, event: dict[str, Any], reason: str) -> None:
        """Persist skipped event caused by idempotency or filtering."""
        sid = self._normalize_session_id(session_id)
        token = self.begin_event_processing(sid, event)
        if token["process"]:
            self.finalize_event_processing(
                sid,
                event,
                status="skipped",
                success=False,
                error=reason,
                started_at=token.get("started_at"),
            )
            return
        # Already processed/running: update existing as skipped note only when not succeeded.
        if token.get("status") != "succeeded":
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE event_logs
                    SET status = 'skipped', success = 0, error = ?
                    WHERE session_id = ? AND event_id = ?
                    """,
                    (reason, sid, token["event_id"]),
                )
                conn.commit()

    def record_event_result(
        self,
        session_id: int | str,
        event: dict[str, Any],
        *,
        success: bool,
        result: str | None = None,
        error: str | None = None,
    ) -> int:
        """Backward-compatible helper: begin + finalize in one call."""
        token = self.begin_event_processing(session_id, event)
        if not token.get("process"):
            return -1
        self.finalize_event_processing(
            session_id,
            event,
            status="succeeded" if success else "failed",
            success=success,
            result=result,
            error=error,
            started_at=token.get("started_at"),
        )
        row = self._fetch_event_log_row(self._normalize_session_id(session_id), token["event_id"])
        return int(row["id"]) if row else -1

    def list_event_logs(self, session_id: int | str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List recent event logs, optionally filtered by session_id."""
        if limit <= 0:
            return []

        with self._connect() as conn:
            if session_id is None:
                rows = conn.execute(
                    """
                    SELECT id, session_id, event_id, event_type, event_payload, status, retry_count,
                           success, result, error, triggered_at, queued_at, started_at, finished_at, latency_ms, created_at
                    FROM event_logs
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                sid = self._normalize_session_id(session_id)
                rows = conn.execute(
                    """
                    SELECT id, session_id, event_id, event_type, event_payload, status, retry_count,
                           success, result, error, triggered_at, queued_at, started_at, finished_at, latency_ms, created_at
                    FROM event_logs
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (sid, limit),
                ).fetchall()

        logs: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            payload_raw = item.get("event_payload")
            if isinstance(payload_raw, str) and payload_raw:
                try:
                    item["event_payload"] = json.loads(payload_raw)
                except json.JSONDecodeError:
                    pass
            item["success"] = bool(item.get("success"))
            logs.append(item)
        return logs

    def record_trade_action_log(
        self,
        session_id: int | str,
        *,
        status: str,
        event_id: str | None = None,
        event_type: str | None = None,
        action: str | None = None,
        ticker: str | None = None,
        quantity: int | None = None,
        trade_date: str | None = None,
        error_code: str | None = None,
        reason: str | None = None,
        request_payload: dict[str, Any] | None = None,
        response_payload: dict[str, Any] | None = None,
    ) -> int:
        """Persist one action-level execution log for trade observability."""
        sid = self._normalize_session_id(session_id)
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trade_action_logs (
                    session_id, event_id, event_type, action, ticker, quantity, trade_date,
                    status, error_code, reason, request_payload, response_payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    (event_id or None),
                    (event_type or None),
                    (action or None),
                    (ticker or None),
                    int(quantity) if quantity is not None else None,
                    (trade_date or None),
                    str(status).strip() or "unknown",
                    (error_code or None),
                    (reason or None),
                    json.dumps(request_payload, ensure_ascii=False) if request_payload is not None else None,
                    json.dumps(response_payload, ensure_ascii=False) if response_payload is not None else None,
                    now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def list_trade_action_logs(self, session_id: int | str, limit: int = 100) -> list[dict[str, Any]]:
        """List action-level trade logs for one session."""
        sid = self._normalize_session_id(session_id)
        if limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, event_id, event_type, action, ticker, quantity, trade_date,
                       status, error_code, reason, request_payload, response_payload, created_at
                FROM trade_action_logs
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (sid, limit),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("request_payload", "response_payload"):
                raw = item.get(key)
                if isinstance(raw, str) and raw:
                    try:
                        item[key] = json.loads(raw)
                    except json.JSONDecodeError:
                        pass
            items.append(item)
        return items

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
