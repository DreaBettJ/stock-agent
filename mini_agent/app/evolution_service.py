"""Execution use-case prompt service (low-coupling evolution module)."""

from __future__ import annotations

import json
import hashlib
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from mini_agent.schema import Message


class EvolutionUseCaseService:
    """Manage execution use-case prompt snippets for trading runtime."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
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
                CREATE TABLE IF NOT EXISTS evolution_use_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    use_case_id TEXT NOT NULL UNIQUE,
                    issue_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    trigger_pattern TEXT NOT NULL,
                    prompt_snippet TEXT NOT NULL,
                    source_kind TEXT NOT NULL DEFAULT 'event_logs',
                    source_refs TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_evolution_use_cases_enabled ON evolution_use_cases(enabled)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_evolution_use_cases_issue ON evolution_use_cases(issue_type)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reasoning_trace_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signature TEXT NOT NULL UNIQUE,
                    session_id INTEGER NOT NULL,
                    run_index INTEGER NOT NULL DEFAULT 1,
                    event TEXT NOT NULL,
                    event_step INTEGER,
                    tool_name TEXT,
                    tool_call_id TEXT,
                    finish_reason TEXT,
                    success INTEGER,
                    content_chars INTEGER,
                    thinking_chars INTEGER,
                    result_chars INTEGER,
                    error_chars INTEGER,
                    source_file TEXT NOT NULL,
                    source_ts TEXT,
                    raw_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reasoning_trace_session ON reasoning_trace_events(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reasoning_trace_event ON reasoning_trace_events(event)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reasoning_trace_tool ON reasoning_trace_events(tool_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reasoning_trace_run ON reasoning_trace_events(session_id, run_index)")
            conn.commit()

    @staticmethod
    def _project_log_dir() -> Path:
        return Path(__file__).resolve().parent.parent.parent / "log"

    @staticmethod
    def _compute_trace_signature(session_id: int, item: dict[str, Any], source_file: str, run_index: int) -> str:
        payload = {
            "session_id": session_id,
            "source_file": source_file,
            "run_index": run_index,
            "timestamp": item.get("timestamp"),
            "event": item.get("event"),
            "step": item.get("step"),
            "tool_call_id": item.get("tool_call_id"),
            "tool_name": item.get("tool_name"),
            "finish_reason": item.get("finish_reason"),
            "success": item.get("success"),
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def ingest_intercept_log(self, session_id: int, log_path: Path) -> dict[str, int]:
        """Parse one intercept jsonl log into reasoning_trace_events."""
        if not log_path.exists():
            return {"inserted": 0, "skipped": 0, "errors": 0}

        inserted = 0
        skipped = 0
        errors = 0
        run_index = 0
        with self._connect() as conn:
            for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    errors += 1
                    continue
                if not isinstance(item, dict):
                    errors += 1
                    continue

                event = str(item.get("event") or "").strip()
                step = item.get("step")
                if event == "before_send" and int(step or 0) == 1:
                    run_index += 1
                if run_index <= 0:
                    run_index = 1

                signature = self._compute_trace_signature(session_id=session_id, item=item, source_file=log_path.name, run_index=run_index)
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO reasoning_trace_events (
                        signature, session_id, run_index, event, event_step, tool_name, tool_call_id,
                        finish_reason, success, content_chars, thinking_chars, result_chars, error_chars,
                        source_file, source_ts, raw_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signature,
                        int(session_id),
                        int(run_index),
                        event,
                        int(step) if step is not None else None,
                        str(item.get("tool_name") or "") or None,
                        str(item.get("tool_call_id") or "") or None,
                        str(item.get("finish_reason") or "") or None,
                        None if item.get("success") is None else (1 if bool(item.get("success")) else 0),
                        int(item.get("content_chars")) if item.get("content_chars") is not None else None,
                        int(item.get("thinking_chars")) if item.get("thinking_chars") is not None else None,
                        int(item.get("result_chars")) if item.get("result_chars") is not None else None,
                        int(item.get("error_chars")) if item.get("error_chars") is not None else None,
                        log_path.name,
                        str(item.get("timestamp") or "") or None,
                        json.dumps(item, ensure_ascii=False),
                        datetime.now().isoformat(),
                    ),
                )
                if cursor.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1
            conn.commit()
        return {"inserted": inserted, "skipped": skipped, "errors": errors}

    def ingest_all_intercept_logs(self, session_id: int | None = None) -> dict[str, int]:
        """Ingest intercept logs into structured reasoning table."""
        log_dir = self._project_log_dir()
        if not log_dir.exists():
            return {"files": 0, "inserted": 0, "skipped": 0, "errors": 0}
        pattern = r"agent_intercept_s(\d+)\.jsonl$"
        totals = {"files": 0, "inserted": 0, "skipped": 0, "errors": 0}
        for path in sorted(log_dir.glob("agent_intercept_s*.jsonl")):
            m = re.match(pattern, path.name)
            if not m:
                continue
            sid = int(m.group(1))
            if session_id is not None and sid != int(session_id):
                continue
            result = self.ingest_intercept_log(session_id=sid, log_path=path)
            totals["files"] += 1
            totals["inserted"] += result["inserted"]
            totals["skipped"] += result["skipped"]
            totals["errors"] += result["errors"]
        return totals

    def trace_summary(self, session_id: int | None = None, limit: int = 20) -> dict[str, Any]:
        """Return compact summary from structured reasoning traces."""
        where = []
        params: list[Any] = []
        if session_id is not None:
            where.append("session_id = ?")
            params.append(int(session_id))
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        with self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(1) AS c FROM reasoning_trace_events {where_sql}", tuple(params)).fetchone()
            fail_where = where + ["event='after_tool'", "success=0"]
            fail_sql = f"WHERE {' AND '.join(fail_where)}" if fail_where else ""
            fail_rows = conn.execute(
                f"""
                SELECT tool_name, COUNT(1) AS fail_count
                FROM reasoning_trace_events
                {fail_sql}
                GROUP BY tool_name
                ORDER BY fail_count DESC, tool_name
                LIMIT ?
                """,
                tuple(params + [limit]),
            ).fetchall()
            event_rows = conn.execute(
                f"""
                SELECT event, COUNT(1) AS cnt
                FROM reasoning_trace_events
                {where_sql}
                GROUP BY event
                ORDER BY cnt DESC, event
                LIMIT ?
                """,
                tuple(params + [limit]),
            ).fetchall()
        return {
            "total_rows": int(total["c"] if total else 0),
            "events": [dict(r) for r in event_rows],
            "tool_failures": [dict(r) for r in fail_rows],
        }

    @staticmethod
    def _classify_error(error: str) -> tuple[str, str, str]:
        text = (error or "").lower()
        if "session_id must be integer" in text or "session not found" in text:
            return (
                "session_binding_error",
                "交易必须绑定当前整数 session_id",
                "当触发交易工具时，session_id 只能使用当前运行会话ID（整数），禁止自造或猜测 session_id。",
            )
        if "next open price not found" in text:
            return (
                "trade_date_invalid",
                "交易日期无效时自动回退可交易日",
                "当 trade_date 对应不到可执行价格时，自动回退到最近可用交易日再执行一次。",
            )
        if "insufficient cash" in text:
            return (
                "insufficient_cash_guard",
                "资金不足时先给可执行降档方案",
                "当资金不足无法按原数量买入时，先返回可买最大股数方案，不要重复提交同样失败订单。",
            )
        if "source: not found" in text:
            return (
                "shell_compatibility",
                "命令兼容性约束",
                "避免使用 source；在 /bin/sh 环境使用 POSIX 兼容写法（例如用 '.' 代替 source）。",
            )
        return (
            "execution_failure_general",
            "通用执行失败降级策略",
            "关键工具失败时，先输出失败原因与下一步降级动作，不要无穷重试同参数。",
        )

    def list_failed_event_errors(self, limit: int = 200) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, event_id, event_type, error, created_at
                FROM event_logs
                WHERE status='failed' AND error IS NOT NULL AND TRIM(error) <> ''
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def create_use_case(
        self,
        *,
        use_case_id: str,
        issue_type: str,
        title: str,
        trigger_pattern: str,
        prompt_snippet: str,
        source_kind: str = "event_logs",
        source_refs: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            exists = conn.execute(
                """
                SELECT use_case_id
                FROM evolution_use_cases
                WHERE issue_type = ? AND trigger_pattern = ? AND enabled = 1
                ORDER BY id DESC
                LIMIT 1
                """,
                (issue_type, trigger_pattern),
            ).fetchone()
            if exists is not None:
                return (False, str(exists["use_case_id"]))

            dup = conn.execute("SELECT use_case_id FROM evolution_use_cases WHERE use_case_id = ? LIMIT 1", (use_case_id,)).fetchone()
            if dup is not None:
                return (False, str(dup["use_case_id"]))

            conn.execute(
                """
                INSERT INTO evolution_use_cases (
                    use_case_id, issue_type, title, trigger_pattern, prompt_snippet,
                    source_kind, source_refs, enabled, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    use_case_id,
                    issue_type,
                    title,
                    trigger_pattern,
                    prompt_snippet,
                    source_kind,
                    json.dumps(source_refs or [], ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            conn.commit()
        return (True, use_case_id)

    def list_use_cases(self, enabled: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        sql = [
            "SELECT id, use_case_id, issue_type, title, trigger_pattern, prompt_snippet, source_kind, source_refs, enabled,",
            "metadata, created_at, updated_at FROM evolution_use_cases",
        ]
        params: list[Any] = []
        if enabled is not None:
            sql.append("WHERE enabled = ?")
            params.append(int(enabled))
        sql.append("ORDER BY id DESC LIMIT ?")
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(" ".join(sql), tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def set_use_case_enabled(self, use_case_id: str, enabled: bool) -> None:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE evolution_use_cases
                SET enabled = ?, updated_at = ?
                WHERE use_case_id = ?
                """,
                (1 if enabled else 0, now, use_case_id),
            )
            conn.commit()
        if cur.rowcount == 0:
            raise KeyError(f"use_case not found: {use_case_id}")

    def render_prompt_block(self, limit: int = 12) -> str:
        rows = self.list_use_cases(enabled=1, limit=limit)
        if not rows:
            return ""
        lines: list[str] = []
        for i, row in enumerate(reversed(rows), 1):
            lines.append(
                f"{i}. [{row['issue_type']}] trigger={row['trigger_pattern']} -> {row['prompt_snippet']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _extract_json_array(text: str) -> list[dict[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        except Exception:
            pass
        match = re.search(r"```(?:json)?\s*(\[[\s\S]*\])\s*```", raw, flags=re.IGNORECASE)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, list):
                    return [x for x in data if isinstance(x, dict)]
            except Exception:
                return []
        return []

    async def generate_use_cases_with_llm(self, llm_client: Any, limit: int = 200) -> list[dict[str, Any]]:
        # Keep reasoning traces fresh for this scan run.
        self.ingest_all_intercept_logs()
        rows = self.list_failed_event_errors(limit=limit)
        if not rows:
            return []
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            err = str(row.get("error") or "").strip()
            if not err:
                continue
            b = grouped.setdefault(err, {"count": 0, "ids": [], "event_type": str(row.get("event_type") or "")})
            b["count"] += 1
            b["ids"].append(int(row.get("id") or 0))
        payload = [
            {"error": err, "count": g["count"], "event_type": g["event_type"], "sample_ids": g["ids"][:10]}
            for err, g in sorted(grouped.items(), key=lambda kv: kv[1]["count"], reverse=True)[:20]
        ]
        if not payload:
            return []

        system_prompt = (
            "你是交易执行系统的运行复盘助手。请把失败模式总结成可执行 use case 提示词。"
            "只输出 JSON 数组，不要其他文本。"
        )
        user_prompt = (
            "请输出最多8条 use case，每条字段："
            "issue_type,title,trigger_pattern,prompt_snippet,source_issue_ids(array[int])。"
            "prompt_snippet 要短小、可直接拼进系统提示词。"
            f"失败聚合数据: {json.dumps(payload, ensure_ascii=False)}"
        )
        response = await llm_client.generate(
            [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]
        )
        result = self._extract_json_array(response.content or "")
        if not result:
            return []

        day_key = datetime.now().strftime("%Y%m%d")
        created: list[dict[str, Any]] = []
        for idx, item in enumerate(result, 1):
            issue_type = str(item.get("issue_type") or f"llm_use_case_{idx}").strip().lower()
            issue_slug = re.sub(r"[^a-z0-9_]+", "_", issue_type).strip("_") or f"llm_use_case_{idx}"
            use_case_id = f"uc-{issue_slug}-{day_key}"
            source_ids_raw = item.get("source_issue_ids")
            source_ids: list[int] = []
            if isinstance(source_ids_raw, list):
                for v in source_ids_raw:
                    try:
                        source_ids.append(int(v))
                    except Exception:
                        continue
            ok, uid = self.create_use_case(
                use_case_id=use_case_id,
                issue_type=issue_slug,
                title=str(item.get("title") or issue_slug)[:200],
                trigger_pattern=str(item.get("trigger_pattern") or "execution_failure")[:200],
                prompt_snippet=str(item.get("prompt_snippet") or "")[:500],
                source_kind="event_logs",
                source_refs=source_ids,
                metadata={"generator": "llm", "raw_item": item},
            )
            if ok:
                created.append(
                    {
                        "use_case_id": uid,
                        "issue_type": issue_slug,
                        "title": str(item.get("title") or issue_slug),
                        "count": len(source_ids),
                    }
                )
        return created

    def generate_use_cases_with_rules(self, limit: int = 200) -> list[dict[str, Any]]:
        # Keep reasoning traces fresh for this scan run.
        self.ingest_all_intercept_logs()
        rows = self.list_failed_event_errors(limit=limit)
        if not rows:
            return []
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            err = str(row.get("error") or "").strip()
            if not err:
                continue
            issue_type, title, snippet = self._classify_error(err)
            b = grouped.setdefault(
                issue_type,
                {
                    "count": 0,
                    "ids": [],
                    "title": title,
                    "snippet": snippet,
                    "trigger_pattern": issue_type,
                },
            )
            b["count"] += 1
            b["ids"].append(int(row.get("id") or 0))
        day_key = datetime.now().strftime("%Y%m%d")
        created: list[dict[str, Any]] = []
        for issue_type, meta in grouped.items():
            use_case_id = f"uc-{issue_type}-{day_key}"
            ok, uid = self.create_use_case(
                use_case_id=use_case_id,
                issue_type=issue_type,
                title=str(meta["title"]),
                trigger_pattern=str(meta["trigger_pattern"]),
                prompt_snippet=str(meta["snippet"]),
                source_kind="event_logs",
                source_refs=list(meta["ids"]),
                metadata={"generator": "rule", "count": meta["count"]},
            )
            if ok:
                created.append(
                    {
                        "use_case_id": uid,
                        "issue_type": issue_type,
                        "title": str(meta["title"]),
                        "count": int(meta["count"]),
                    }
                )
        return created
