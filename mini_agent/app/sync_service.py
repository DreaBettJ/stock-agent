"""K-line synchronization service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess

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


def resolve_ticker_universe(tickers_arg: str, sync_all: bool) -> list[str]:
    if sync_all:
        import akshare as ak

        spot = ak.stock_zh_a_spot_em()
        col = "代码" if "代码" in spot.columns else spot.columns[1]
        return [str(v).strip() for v in spot[col].tolist() if str(v).strip()]
    return normalize_tickers(tickers_arg or "")


@dataclass(slots=True)
class SyncResult:
    success_count: int
    failed: list[str]
    total_rows: int
    end_date: str


def sync_kline_data(kline_db: KLineDB, tickers: list[str], start: str, end: str | None = None) -> SyncResult:
    end_date = end or datetime.now().date().isoformat()
    success = 0
    failed: list[str] = []
    total_rows = 0
    for ticker in tickers:
        try:
            rows = sync_one_ticker_with_akshare(kline_db, ticker, start, end_date)
            total_rows += rows
            success += 1
        except Exception:
            failed.append(ticker)
    return SyncResult(success_count=success, failed=failed, total_rows=total_rows, end_date=end_date)


def build_cron_lines(cwd: Path, start: str) -> list[str]:
    return [
        f"0 16 * * 1-5 cd {cwd} && mini-agent sync --all --start {start}",
        f"5 16 * * 1-5 cd {cwd} && mini-agent event trigger daily_review --all",
    ]


def install_cron_lines(cwd: Path, start: str) -> tuple[bool, str]:
    """Install/replace mini-agent cron block in current user's crontab."""
    begin = "# >>> mini-agent auto-sync >>>"
    end = "# <<< mini-agent auto-sync <<<"
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
