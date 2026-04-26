"""Binance archive downloader — v2.0 30m bulk historical.

Spec: Project Spec 30min.md §6.1, §6.3 + Appendix C row 1.1
+ Decision Log v2.36 (clean v2.0 glue; NOT v1.0 fetcher.py renamed).

Fetches monthly .zip archives from data.binance.vision (geo-open public host;
Binance main api/fapi return HTTP 451 from VPS region — see §6.1).

Per Decision v2.28: 30m is the only fetched timeframe; 4H/1D are aggregated
in-pipeline at feature build (§6.4 Step A).
Per Decision v2.35: v2.0 universe is BTC/SOL/LINK; TAO/HYPE are Phase 4 candidates.

Output:
  - Postgres `ohlcv_30m` (primary; reuses v1.0 instance per Decision v2.27)
  - Parquet at data/storage/binance/30m/<SYMBOLSLUG>/<SYMBOLSLUG>-30m-YYYY-MM.parquet
    (snapshots for builder.py reproducibility)

Usage:
    python -m data.collectors.binance_archive                    # all configured symbols, 3yr
    python -m data.collectors.binance_archive --symbol BTC/USDT
    python -m data.collectors.binance_archive --years 3
    python -m data.collectors.binance_archive --no-resume        # full backfill
"""
from __future__ import annotations

import argparse
import io
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from data import db
from data.collectors.storage import OHLCV_COLS, symbol_slug
from utils.config import load_config, resolve_path
from utils.logging_setup import get_logger

log = get_logger("binance_archive")

ARCHIVE_HOST_DEFAULT = "https://data.binance.vision/data/spot/monthly/klines"
COLLECTOR = "binance_archive_30m"
TIMEFRAME = "30m"
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def month_url(host: str, slug: str, interval: str, year: int, month: int) -> str:
    return f"{host}/{slug}/{interval}/{slug}-{interval}-{year:04d}-{month:02d}.zip"


def download_month(url: str, retries: int = 3, backoff: float = 2.0) -> bytes | None:
    """Return bytes of the .zip, or None on 404 (archive not yet published)."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
        except requests.RequestException as exc:
            if attempt == retries - 1:
                log.error(f"download failed after {retries}: {url} — {exc}")
                raise
            time.sleep(backoff * (attempt + 1))
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content
    return None


def parse_month_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            # Binance added a CSV header row to some 2025+ archives. Auto-detect:
            # if the first byte after open is non-numeric, skip header.
            head = f.read(1)
            f.seek(0)
            header = 0 if head and not head.decode("utf-8", "ignore").isdigit() else None
            df = pd.read_csv(f, header=header, names=KLINE_COLS if header is None else None)
            if header == 0:
                df = df.rename(columns=dict(zip(df.columns, KLINE_COLS)))

    open_time = df["open_time"].astype("int64")
    # Binance switched archive timestamps from ms (≤2024) to µs (≥2025).
    # Normalize to ms for internal consistency. Threshold 10^14 ≈ year 5138 in ms,
    # well above any plausible ms value.
    if open_time.max() > 10**14:
        open_time = open_time // 1000

    out = pd.DataFrame({
        "timestamp": open_time,  # always ms post-normalization
        "open":   df["open"].astype(float),
        "high":   df["high"].astype(float),
        "low":    df["low"].astype(float),
        "close":  df["close"].astype(float),
        "volume": df["volume"].astype(float),
    })[OHLCV_COLS]
    return out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")


def months_in_range(start: datetime, end: datetime) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    cur = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cur <= end:
        out.append((cur.year, cur.month))
        cur = cur.replace(year=cur.year + 1, month=1) if cur.month == 12 else cur.replace(month=cur.month + 1)
    return out


def fetch_symbol(
    symbol_ccxt: str,
    start_dt: datetime,
    end_dt: datetime,
    storage_dir: Path,
    archive_host: str = ARCHIVE_HOST_DEFAULT,
    interval: str = TIMEFRAME,
    resume: bool = True,
) -> int:
    slug = symbol_slug(symbol_ccxt)

    if resume:
        last_ts = db.latest_ohlcv_ts(slug)
        if last_ts is not None:
            new_start = last_ts + timedelta(minutes=30)
            if new_start >= end_dt:
                log.info(f"{slug}: up-to-date (last bar @ {last_ts.isoformat()})")
                db.mark_collector_state(COLLECTOR, slug, last_ts, "up_to_date")
                return 0
            log.info(f"{slug}: resuming from {new_start.isoformat()}")
            start_dt = new_start

    months = months_in_range(start_dt, end_dt)
    log.info(f"{slug}: {len(months)} monthly archives queued ({months[0]} → {months[-1]})")

    parquet_dir = storage_dir / interval / slug
    parquet_dir.mkdir(parents=True, exist_ok=True)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    total = 0

    for (y, m) in tqdm(months, desc=slug, unit="month"):
        url = month_url(archive_host, slug, interval, y, m)
        content = download_month(url)
        if content is None:
            log.warning(f"{slug}: archive 404 for {y}-{m:02d} (likely current month)")
            continue
        df = parse_month_zip(content)
        df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]
        if df.empty:
            continue

        total += db.upsert_ohlcv(df, symbol=slug)
        df.to_parquet(parquet_dir / f"{slug}-{interval}-{y:04d}-{m:02d}.parquet", index=False)

    last_after = db.latest_ohlcv_ts(slug)
    db.mark_collector_state(COLLECTOR, slug, last_after, "ok")
    log.info(f"{slug}: wrote {total:,} rows; latest now {last_after.isoformat() if last_after else 'none'}")
    return total


def main() -> None:
    cfg = load_config()
    bcfg = cfg["data"]["binance"]
    storage_dir = resolve_path(bcfg["storage_dir"])
    archive_host = bcfg.get("archive_host", ARCHIVE_HOST_DEFAULT)

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="single symbol e.g. BTC/USDT")
    parser.add_argument("--years", type=int, default=bcfg.get("history_years", 3))
    parser.add_argument("--interval", default=TIMEFRAME)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else bcfg["symbols"]

    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=args.years * 365)

    db.ensure_schema(["ohlcv_30m", "collector_state_30m"])
    info = db.ping()
    log.info(f"DB ok: {info.get('user')}@{info.get('database')} (timescaledb {info.get('timescaledb')})")
    log.info(f"Fetching {symbols} × {args.interval} from {start_dt.isoformat()} to {end_dt.isoformat()}")

    for sym in symbols:
        try:
            fetch_symbol(
                sym, start_dt, end_dt, storage_dir,
                archive_host=archive_host, interval=args.interval,
                resume=not args.no_resume,
            )
        except Exception as exc:
            log.exception(f"{sym}: failed — {exc}")
            db.mark_collector_state(COLLECTOR, symbol_slug(sym), None, "error", str(exc)[:500])


if __name__ == "__main__":
    main()
