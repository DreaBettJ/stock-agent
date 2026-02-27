from mini_agent.session import SessionManager


def test_session_created_at_sqlite_utc_is_converted_to_utc8_on_read(tmp_path):
    db_path = tmp_path / "session_tz.db"
    manager = SessionManager(db_path=str(db_path))
    sid = manager.create_session(name="tz", system_prompt="p", mode="simulation")

    with manager._connect() as conn:  # noqa: SLF001 - testing internal conversion behavior
        conn.execute("UPDATE sessions SET created_at = ? WHERE session_id = ?", ("2026-02-26 10:14:40", sid))
        conn.commit()

    session = manager.get_session(sid)
    assert session.created_at is not None
    assert session.created_at.strftime("%Y-%m-%d %H:%M:%S") == "2026-02-26 18:14:40"
