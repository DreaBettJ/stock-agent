"""K-line synchronization service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import sqlite3

from mini_agent.tools.kline_db_tool import KLineDB


def normalize_tickers(raw: str) -> list[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def sync_one_ticker_with_akshare(kline_db: KLineDB, ticker: str, start: str, end: str) -> int:
    import akshare as ak

    symbol = ticker.split(".")[0]
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        adjust="qfq",
    )
    if df is None or len(df) == 0:
        return 0

    inserted = 0
    for _, row in df.iterrows():
        kline_db.upsert_daily_kline(
            ticker=symbol,
            date=str(row["日期"]),
            open_price=float(row["开盘"]),
            high=float(row["最高"]),
            low=float(row["最低"]),
            close=float(row["收盘"]),
            volume=float(row["成交量"]) if row["成交量"] is not None else 0.0,
            amount=float(row["成交额"]) if row["成交额"] is not None else 0.0,
        )
        inserted += 1
    return inserted


def _ticker_to_ts_code(ticker: str) -> str:
    raw = (ticker or "").strip().upper()
    if "." in raw:
        return raw
    if raw.startswith(("SH", "SZ")) and len(raw) >= 8:
        code = raw[2:]
        market = "SH" if raw.startswith("SH") else "SZ"
        return f"{code}.{market}"
    if raw.startswith(("5", "6", "9")):
        return f"{raw}.SH"
    return f"{raw}.SZ"


def sync_one_ticker_with_tushare(
    kline_db: KLineDB,
    ticker: str,
    start: str,
    end: str,
    tushare_token: str,
) -> int:
    import tushare as ts  # type: ignore

    token = (tushare_token or "").strip()
    if not token:
        raise ValueError("tushare_token is empty")
    pro = ts.pro_api(token)

    ts_code = _ticker_to_ts_code(ticker)
    df = pro.daily(
        ts_code=ts_code,
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
    )
    if df is None or len(df) == 0:
        return 0

    inserted = 0
    for _, row in df.iterrows():
        trade_date = str(row["trade_date"])
        date_fmt = f"{trade_date[0:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
        amount = float(row["amount"]) * 1000 if row.get("amount") is not None else 0.0
        kline_db.upsert_daily_kline(
            ticker=ticker.split(".")[0],
            date=date_fmt,
            open_price=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["vol"]) if row.get("vol") is not None else 0.0,
            amount=amount,
        )
        inserted += 1
    return inserted


def sync_all_for_trade_date_with_tushare(kline_db: KLineDB, trade_date: str, tushare_token: str) -> tuple[int, int]:
    """Batch sync one trading day using TuShare daily(trade_date=...)."""
    import tushare as ts  # type: ignore

    token = (tushare_token or "").strip()
    if not token:
        raise ValueError("tushare_token is empty")

    pro = ts.pro_api(token)
    df = pro.daily(trade_date=trade_date.replace("-", ""))
    if df is None or len(df) == 0:
        return 0, 0

    success_tickers: set[str] = set()
    rows = 0
    for _, row in df.iterrows():
        ts_code = str(row.get("ts_code") or "")
        ticker = ts_code.split(".")[0] if "." in ts_code else ts_code
        if not ticker:
            continue
        trade = str(row["trade_date"])
        date_fmt = f"{trade[0:4]}-{trade[4:6]}-{trade[6:8]}"
        amount = float(row["amount"]) * 1000 if row.get("amount") is not None else 0.0
        kline_db.upsert_daily_kline(
            ticker=ticker,
            date=date_fmt,
            open_price=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["vol"]) if row.get("vol") is not None else 0.0,
            amount=amount,
        )
        success_tickers.add(ticker)
        rows += 1
    return len(success_tickers), rows


def _resolve_all_tickers_with_tushare(tushare_token: str) -> list[str]:
    import tushare as ts  # type: ignore

    token = (tushare_token or "").strip()
    if not token:
        raise ValueError("tushare_token is empty")
    pro = ts.pro_api(token)
    df = pro.stock_basic(exchange="", list_status="L", fields="symbol")
    if df is None or len(df) == 0:
        return []
    return [str(v).strip() for v in df["symbol"].tolist() if str(v).strip()]


def _resolve_all_tickers_with_akshare() -> list[str]:
    import akshare as ak

    spot = ak.stock_zh_a_spot_em()
    col = "代码" if "代码" in spot.columns else spot.columns[1]
    return [str(v).strip() for v in spot[col].tolist() if str(v).strip()]


def _resolve_all_tickers_from_local_db(kline_db: KLineDB | None) -> list[str]:
    if kline_db is None:
        return []
    with sqlite3.connect(kline_db.db_path) as conn:
        rows = conn.execute("SELECT DISTINCT ticker FROM daily_kline ORDER BY ticker").fetchall()
    return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]


def resolve_ticker_universe(
    tickers_arg: str,
    sync_all: bool,
    *,
    kline_db: KLineDB | None = None,
    tushare_token: str | None = None,
) -> list[str]:
    if sync_all:
        errors: list[str] = []
        token = (tushare_token or "").strip()
        if token:
            try:
                tickers = _resolve_all_tickers_with_tushare(token)
                if tickers:
                    return tickers
            except Exception as exc:
                errors.append(f"tushare: {exc}")
        try:
            tickers = _resolve_all_tickers_with_akshare()
            if tickers:
                return tickers
        except Exception as exc:
            errors.append(f"akshare: {exc}")
        local = _resolve_all_tickers_from_local_db(kline_db)
        if local:
            return local
        detail = "; ".join(errors) if errors else "no source available"
        raise RuntimeError(f"failed to resolve ticker universe: {detail}")
    return normalize_tickers(tickers_arg or "")


@dataclass(slots=True)
class SyncResult:
    success_count: int
    failed: list[str]
    total_rows: int
    end_date: str


def sync_kline_data(
    kline_db: KLineDB,
    tickers: list[str],
    start: str,
    end: str | None = None,
    *,
    tushare_token: str | None = None,
) -> SyncResult:
    end_date = end or datetime.now().date().isoformat()
    success = 0
    failed: list[str] = []
    total_rows = 0
    token = (tushare_token or "").strip()

    # Fast path: daily incremental sync for all tickers via one TuShare API call.
    if token and start == end_date and len(tickers) >= 1000:
        try:
            success_count, rows = sync_all_for_trade_date_with_tushare(kline_db, end_date, token)
            return SyncResult(success_count=success_count, failed=[], total_rows=rows, end_date=end_date)
        except Exception:
            pass

    for ticker in tickers:
        try:
            if token:
                try:
                    rows = sync_one_ticker_with_tushare(kline_db, ticker, start, end_date, token)
                except Exception:
                    rows = sync_one_ticker_with_akshare(kline_db, ticker, start, end_date)
            else:
                rows = sync_one_ticker_with_akshare(kline_db, ticker, start, end_date)
            total_rows += rows
            success += 1
        except Exception:
            failed.append(ticker)
    return SyncResult(success_count=success, failed=failed, total_rows=total_rows, end_date=end_date)


def build_cron_lines(cwd: Path, start: str | None = None) -> list[str]:
    _ = start
    return [
        f"0 16 * * 1-5 cd {cwd} && big-a-helper sync --all",
        f"5 16 * * 1-5 cd {cwd} && big-a-helper event trigger daily_review --all",
    ]


def install_cron_lines(cwd: Path, start: str | None = None) -> tuple[bool, str]:
    """Install/replace big-a-helper cron block in current user's crontab."""
    begin = "# >>> big-a-helper auto-sync >>>"
    end = "# <<< big-a-helper auto-sync <<<"
    block_lines = [begin, *build_cron_lines(cwd, start), end]
    new_block = "\n".join(block_lines)

    # Read current crontab. "no crontab for user" is treated as empty.
    current = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if current.returncode != 0:
        stderr = (current.stderr or "").lower()
        if "no crontab" in stderr:
            old_text = ""
        else:
            return False, (current.stderr or current.stdout or "failed to read current crontab").strip()
    else:
        old_text = current.stdout or ""

    lines = old_text.splitlines()
    kept: list[str] = []
    inside_block = False
    for line in lines:
        stripped = line.strip()
        if stripped == begin:
            inside_block = True
            continue
        if stripped == end:
            inside_block = False
            continue
        if not inside_block:
            kept.append(line)

    merged = "\n".join([*kept, "", new_block]).strip() + "\n"
    write = subprocess.run(["crontab", "-"], input=merged, capture_output=True, text=True)
    if write.returncode != 0:
        return False, (write.stderr or write.stdout or "failed to write crontab").strip()
    return True, "cron installed"
