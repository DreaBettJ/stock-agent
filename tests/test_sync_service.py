"""Tests for sync service fallback chain."""

from __future__ import annotations

from mini_agent.app.sync_service import resolve_ticker_universe, sync_kline_data
from mini_agent.tools.kline_db_tool import KLineDB


def test_resolve_ticker_universe_fallback_to_local_db(monkeypatch, tmp_path):
    db = KLineDB(db_path=str(tmp_path / "kline.db"))
    db.upsert_daily_kline("600519", "2026-02-13", 1, 1, 1, 1, 1, 1)
    db.upsert_daily_kline("000001", "2026-02-13", 1, 1, 1, 1, 1, 1)

    monkeypatch.setattr(
        "mini_agent.app.sync_service._resolve_all_tickers_with_tushare",
        lambda token: (_ for _ in ()).throw(RuntimeError("ts down")),
    )
    monkeypatch.setattr(
        "mini_agent.app.sync_service._resolve_all_tickers_with_akshare",
        lambda: (_ for _ in ()).throw(RuntimeError("ak down")),
    )

    tickers = resolve_ticker_universe("", True, kline_db=db, tushare_token="x")
    assert "600519" in tickers
    assert "000001" in tickers


def test_sync_kline_data_prefers_tushare_token(monkeypatch, tmp_path):
    db = KLineDB(db_path=str(tmp_path / "kline2.db"))

    called = {"ts": 0, "ak": 0}

    def _ts(kline_db, ticker, start, end, token):
        called["ts"] += 1
        return 3

    def _ak(kline_db, ticker, start, end):
        called["ak"] += 1
        return 1

    monkeypatch.setattr("mini_agent.app.sync_service.sync_one_ticker_with_tushare", _ts)
    monkeypatch.setattr("mini_agent.app.sync_service.sync_one_ticker_with_akshare", _ak)

    result = sync_kline_data(db, tickers=["600519"], start="2026-01-01", end="2026-02-13", tushare_token="abc")
    assert result.success_count == 1
    assert result.total_rows == 3
    assert called["ts"] == 1
    assert called["ak"] == 0


def test_sync_kline_data_uses_batch_tushare_for_daily_incremental(monkeypatch, tmp_path):
    db = KLineDB(db_path=str(tmp_path / "kline3.db"))
    tickers = [f"{i:06d}" for i in range(2000)]
    called = {"batch": 0, "single": 0}

    def _batch(kline_db, trade_date, token):
        called["batch"] += 1
        return 1500, 1500

    def _single(kline_db, ticker, start, end, token):
        called["single"] += 1
        return 1

    monkeypatch.setattr("mini_agent.app.sync_service.sync_all_for_trade_date_with_tushare", _batch)
    monkeypatch.setattr("mini_agent.app.sync_service.sync_one_ticker_with_tushare", _single)

    result = sync_kline_data(
        db,
        tickers=tickers,
        start="2026-02-24",
        end="2026-02-24",
        tushare_token="abc",
    )
    assert result.success_count == 1500
    assert result.total_rows == 1500
    assert called["batch"] == 1
    assert called["single"] == 0
