"""Postgres + TimescaleDB helpers — v2.0 tables only.

Spec: Project Spec 30min.md §6.1.1 + Decision v2.27 (DB reuse, new tables only)
+ Decision v2.36 (clean glue, ~120 LOC, no 5m/1h baggage).

v2.0 tables (isolated from v1.0 — never read/write v1.0 tables):
  ohlcv_30m            — Binance 30m OHLCV (Phase 1.1)
  ohlcv_30m_hl         — Hyperliquid 30m rollup (Phase 3+)
  features_30m         — feature matrix (Phase 1.14 freeze target)   [created at 1.14]
  labels_30m           — triple-barrier labels                        [created at 1.14]
  wf_folds_30m         — walk-forward fold metadata                   [created at Phase 2]
  models_30m           — model artifact registry                      [created at Phase 2.9]
  decay_metrics_30m    — Phase 5 decay monitor (§17.2)                [created at Phase 5.0]
  collector_state_30m  — collector cursor + status
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from utils.logging_setup import get_logger

log = get_logger("db")

_pool: ConnectionPool | None = None


def _conninfo() -> str:
    """Build psycopg conninfo from PG_* env vars (v1.0 .env convention, per Decision v2.27)."""
    # Trigger .env load via utils.config.
    from utils.config import load_config  # local import avoids circular at module load
    load_config()

    required = ["PG_HOST", "PG_PORT", "PG_DB", "PG_USER", "PG_PASSWORD"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"missing env vars: {missing} (.env not loaded or incomplete)")
    return (
        f"host={os.environ['PG_HOST']} port={os.environ['PG_PORT']} "
        f"dbname={os.environ['PG_DB']} user={os.environ['PG_USER']} "
        f"password={os.environ['PG_PASSWORD']}"
    )


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        min_size = int(os.environ.get("PG_POOL_MIN", "1"))
        max_size = int(os.environ.get("PG_POOL_MAX", "5"))
        _pool = ConnectionPool(_conninfo(), min_size=min_size, max_size=max_size, open=True)
    return _pool


# ─── DDL — only Phase 1 tables created up front; later phases call ensure_schema with their list ───

DDL: dict[str, str] = {
    "ohlcv_30m": """
        CREATE TABLE IF NOT EXISTS ohlcv_30m (
            symbol TEXT NOT NULL,
            ts     TIMESTAMPTZ NOT NULL,
            open   DOUBLE PRECISION,
            high   DOUBLE PRECISION,
            low    DOUBLE PRECISION,
            close  DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            PRIMARY KEY (symbol, ts)
        );
        SELECT create_hypertable('ohlcv_30m', 'ts',
            chunk_time_interval => INTERVAL '90 days', if_not_exists => TRUE);
        CREATE INDEX IF NOT EXISTS ix_ohlcv_30m_symbol_ts
            ON ohlcv_30m (symbol, ts DESC);
    """,
    "ohlcv_30m_hl": """
        CREATE TABLE IF NOT EXISTS ohlcv_30m_hl (
            symbol TEXT NOT NULL,
            ts     TIMESTAMPTZ NOT NULL,
            open   DOUBLE PRECISION, high DOUBLE PRECISION,
            low    DOUBLE PRECISION, close DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            PRIMARY KEY (symbol, ts)
        );
        SELECT create_hypertable('ohlcv_30m_hl', 'ts',
            chunk_time_interval => INTERVAL '90 days', if_not_exists => TRUE);
    """,
    "collector_state_30m": """
        CREATE TABLE IF NOT EXISTS collector_state_30m (
            collector  TEXT NOT NULL,
            key        TEXT NOT NULL,
            last_ts    TIMESTAMPTZ,
            status     TEXT,
            error      TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (collector, key)
        );
    """,
}


def ensure_schema(tables: list[str] | None = None) -> None:
    """CREATE TABLE IF NOT EXISTS for the named tables (default: all in DDL)."""
    targets = tables or list(DDL.keys())
    with pool().connection() as conn, conn.cursor() as cur:
        for t in targets:
            log.info(f"ensure_schema: {t}")
            cur.execute(DDL[t])
        conn.commit()


def ping() -> dict[str, Any]:
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT current_user AS \"user\", current_database() AS database, "
            "(SELECT extversion FROM pg_extension WHERE extname='timescaledb') AS timescaledb"
        )
        return cur.fetchone() or {}


# ─── ohlcv_30m ─────────────────────────────────────────────────────────────────

def upsert_ohlcv(df: pd.DataFrame, symbol: str) -> int:
    """Upsert 30m OHLCV rows. df cols: timestamp(ms), open, high, low, close, volume."""
    if df.empty:
        return 0
    rows = [
        (symbol,
         datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc),
         float(o), float(h), float(l), float(c), float(v))
        for ts, o, h, l, c, v in df[["timestamp", "open", "high", "low", "close", "volume"]].itertuples(index=False)
    ]
    sql = """INSERT INTO ohlcv_30m (symbol, ts, open, high, low, close, volume)
             VALUES (%s, %s, %s, %s, %s, %s, %s)
             ON CONFLICT (symbol, ts) DO UPDATE SET
               open = EXCLUDED.open, high = EXCLUDED.high,
               low = EXCLUDED.low, close = EXCLUDED.close,
               volume = EXCLUDED.volume"""
    with pool().connection() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def latest_ohlcv_ts(symbol: str) -> datetime | None:
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT max(ts) FROM ohlcv_30m WHERE symbol = %s", (symbol,))
        row = cur.fetchone()
    return row[0] if row and row[0] else None


def fetch_ohlcv(
    symbol: str, start: datetime | None = None, end: datetime | None = None,
) -> pd.DataFrame:
    sql = "SELECT ts, open, high, low, close, volume FROM ohlcv_30m WHERE symbol = %s"
    args: list = [symbol]
    if start is not None:
        sql += " AND ts >= %s"
        args.append(start)
    if end is not None:
        sql += " AND ts <= %s"
        args.append(end)
    sql += " ORDER BY ts"
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(sql, args)
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


# ─── collector state ──────────────────────────────────────────────────────────

def mark_collector_state(
    collector: str, key: str, last_ts: datetime | None,
    status: str, error: str | None = None,
) -> None:
    sql = """INSERT INTO collector_state_30m (collector, key, last_ts, status, error, updated_at)
             VALUES (%s, %s, %s, %s, %s, now())
             ON CONFLICT (collector, key) DO UPDATE SET
               last_ts = EXCLUDED.last_ts, status = EXCLUDED.status,
               error = EXCLUDED.error, updated_at = now()"""
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (collector, key, last_ts, status, error))
        conn.commit()
