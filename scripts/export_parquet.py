"""Snapshot Postgres ohlcv_30m → parquet for feature building / model training.

Spec: Project Spec 30min.md §6.3; fresh v2.0 glue per Decision v2.36.

Why parquet snapshot when Postgres already has the data:
  * features/builder.py reads parquet (stable across DB schema evolution)
  * Reproducibility: a snapshot at train_val_end is immutable; reruns months
    later yield identical training inputs even as Postgres keeps growing

Usage:
    python -m scripts.export_parquet                       # all configured symbols, up to now
    python -m scripts.export_parquet --symbol BTC/USDT
    python -m scripts.export_parquet --until 2026-02-28
    python -m scripts.export_parquet --train-snapshot      # cap at splits.train_val_end
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data import db
from data.collectors.storage import OHLCV_COLS, ohlcv_path
from utils.config import load_config, resolve_path
from utils.logging_setup import get_logger

log = get_logger("export_parquet")

TIMEFRAME = "30m"


def _parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def export_one(
    symbol_ccxt: str, storage_dir: Path,
    start: datetime | None, end: datetime | None,
) -> int:
    df = db.fetch_ohlcv(_slug_ccxt(symbol_ccxt), start=start, end=end)
    if df.empty:
        log.warning(f"{symbol_ccxt}: no rows in ohlcv_30m — skipping")
        return 0

    out = pd.DataFrame({
        "timestamp": (df["ts"].astype("int64") // 1_000_000),  # ns → ms
        "open":   df["open"].astype(float),
        "high":   df["high"].astype(float),
        "low":    df["low"].astype(float),
        "close":  df["close"].astype(float),
        "volume": df["volume"].astype(float),
    })[OHLCV_COLS]

    path = ohlcv_path(storage_dir, symbol_ccxt, TIMEFRAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    first = pd.to_datetime(out["timestamp"].iloc[0], unit="ms", utc=True)
    last = pd.to_datetime(out["timestamp"].iloc[-1], unit="ms", utc=True)
    log.info(f"{symbol_ccxt}: {len(out):,} rows ({first.isoformat()} → {last.isoformat()}) → {path}")
    return len(out)


def _slug_ccxt(symbol_ccxt: str) -> str:
    return symbol_ccxt.replace("/", "").upper()


def main() -> None:
    cfg = load_config()
    bcfg = cfg["data"]["binance"]
    storage_dir = resolve_path(bcfg["storage_dir"]) / TIMEFRAME

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="single symbol e.g. BTC/USDT")
    parser.add_argument("--since", help="start ISO date")
    parser.add_argument("--until", help="end ISO date (inclusive)")
    parser.add_argument(
        "--train-snapshot", action="store_true",
        help="cap at splits.train_val_end (reproducible training slice)",
    )
    args = parser.parse_args()

    if args.train_snapshot and args.until:
        parser.error("--train-snapshot and --until are mutually exclusive")

    end: datetime | None = None
    if args.train_snapshot:
        end = _parse_iso(cfg["splits"]["train_val_end"])
        log.info(f"--train-snapshot: capping at splits.train_val_end = {end.isoformat()}")
    elif args.until:
        end = _parse_iso(args.until)
    start: datetime | None = _parse_iso(args.since) if args.since else None

    symbols = [args.symbol] if args.symbol else bcfg["symbols"]
    info = db.ping()
    log.info(f"DB ok: {info.get('user')}@{info.get('database')} (timescaledb {info.get('timescaledb')})")
    log.info(f"Exporting {symbols} → {storage_dir}")

    total = 0
    for sym in symbols:
        total += export_one(sym, storage_dir, start, end)
    log.info(f"Done — {total:,} rows across {len(symbols)} files.")


if __name__ == "__main__":
    main()
