"""Cat 8 — Price Action / Candle (9 features).

v2.0 verification (2026-04-27, Decision v2.37 audit + v2.38 pilot):
    Math is timeframe-agnostic OHLC pattern recognition (body %, wicks,
    consecutive runs, engulfing, pin bar). Per Project Spec 30min §7.2 Cat 8
    "9 → 9, unchanged" — no parameter changes from v1.0 required for 30m.
    30m candles carry MORE signal than 5m (less microstructure noise),
    so the existing pattern thresholds are preserved.

    Reviewed by local-Claude pre-Phase-1.10 audit; no edits to function
    bodies needed. This docstring update is the workflow pilot per
    Decision v2.38 — exercises the local-edit → user-pull → VPS-validate
    cycle on a no-logic-change file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def consecutive_count(cond: pd.Series) -> pd.Series:
    """Count of consecutive True up to and including current row; resets on False."""
    grp = (~cond).cumsum()
    return cond.astype(int).groupby(grp).cumsum()


def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    body_pct = body / rng
    upper_wick = (h - np.maximum(o, c)) / rng
    lower_wick = (np.minimum(o, c) - l) / rng

    is_bull = (c > o).astype(int)
    is_bear = (c < o).astype(int)
    consec_bull = consecutive_count(is_bull == 1)
    consec_bear = consecutive_count(is_bear == 1)

    body_vs_prev = body / body.shift(1).replace(0, np.nan)

    # Engulfing: body fully covers prior body.
    prev_o, prev_c = o.shift(1), c.shift(1)
    bull_eng = ((c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c)).astype(int)
    bear_eng = ((c < o) & (prev_c > prev_o) & (c <= prev_o) & (o >= prev_c)).astype(int)
    engulfing = bull_eng - bear_eng

    # Pin bar: long wick on one side > 2x body, opposite wick small.
    bull_pin = ((lower_wick > 2 * body_pct) & (upper_wick < body_pct)).astype(int)
    bear_pin = ((upper_wick > 2 * body_pct) & (lower_wick < body_pct)).astype(int)
    pin = bull_pin - bear_pin

    return pd.DataFrame(
        {
            "body_pct": body_pct,
            "upper_wick_pct": upper_wick,
            "lower_wick_pct": lower_wick,
            "is_bullish": is_bull,
            "consecutive_bull": consec_bull,
            "consecutive_bear": consec_bear,
            "body_vs_prev_body": body_vs_prev,
            "engulfing": engulfing,
            "pin_bar": pin,
        }
    )
