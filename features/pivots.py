"""Cat 6 — S/R Structure (30 features) — v2.0.

Per Project Spec 30min §7.2 Cat 6 + Decisions v2.17–v2.19 (Fib expansions)
+ Decision v2.45 (Q14.1–Q14.4 implementation choices locked).

30 features = 9 (6.1 daily) + 3 (6.2 daily NEW) + 9 (6.3 weekly)
            + 2 (6.4 weekly NEW) + 7 (6.5 swing-Fib retracements).

DAILY FIB-PIVOTS (Standard Fibonacci pivot math):
    P  = (prev_H + prev_L + prev_C) / 3
    R1 = P + 0.382 × (prev_H − prev_L)
    R2 = P + 0.618 × (prev_H − prev_L)
    R3 = P + 1.000 × (prev_H − prev_L)
    S1 = P − 0.382 × range
    S2 = P − 0.618 × range
    S3 = P − 1.000 × range

WEEKLY FIB-PIVOTS: same math, weekly OHLC, anchor Monday 00:00 UTC.

SWING FIB RETRACEMENTS (Cat 6.5):
    swing_high, swing_low = forward-filled most recent confirmed fractal
                            pivots (lookback=5, 2-left-2-right rule)
    swing_range = swing_high − swing_low
    fib_retracement_pct = (close − swing_low) / swing_range

§7.5 TAGGING: All 30 features are `static` — pivot LEVELS are fixed for the
day/week (don't update intrabar); swing Fib levels only update on new
confirmed pivot. Position features (dist_pct, zone) depend on close but
are re-derivable each tick from cached static levels — intrabar-safe.

CALLER-SUPPLIED INPUTS:
    df    — DataFrame with open/high/low/close/volume + DatetimeIndex
    atr_14 — ATR(14) Series (from volatility.py output)
    cfg   — feature config dict

ROLLBACK: see PROJECT_LOG Decision v2.45 entry for v1.0→v2.0 transformation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import pct, safe_div
from .divergence import fractal_pivots


PIVOT_NAMES = ["S3", "S2", "S1", "P", "R1", "R2", "R3"]
FIB_LEVELS = [0.382, 0.5, 0.618, 0.786]


# ─── Public helper: pivot LEVELS only (used by event_memory.py) ──────────
def compute_pivot_levels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute daily and weekly pivot LEVELS (no derived features).

    Used by `event_memory.py` to detect pivot touches without re-running
    the full Cat 6 `pivot_features` pipeline. Same look-ahead safety:
    levels reflect PRIOR-period OHLC via `.shift(1)` upstream.

    Parameters
    ----------
    df : DataFrame with open/high/low/close + DatetimeIndex.

    Returns
    -------
    (daily_levels, weekly_levels) : tuple of DataFrames, each with 7
    columns (S3, S2, S1, P, R1, R2, R3) indexed like df.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "compute_pivot_levels requires df.index to be a DatetimeIndex."
        )
    day_id = pd.Series(df.index.floor("1D"), index=df.index)
    week_id = pd.Series(
        df.index.floor("1D") - pd.to_timedelta(df.index.dayofweek, unit="D"),
        index=df.index,
    )
    daily_levels = _compute_pivot_levels_mapped(df, day_id)
    weekly_levels = _compute_pivot_levels_mapped(df, week_id)
    return daily_levels, weekly_levels


# ─── Pivot level math ────────────────────────────────────────────────────
def _fib_pivots_from_ohlc(prev_high: pd.Series, prev_low: pd.Series, prev_close: pd.Series) -> pd.DataFrame:
    """Standard Fibonacci pivots from prior-period OHLC.

    Returns DataFrame with columns S3, S2, S1, P, R1, R2, R3 indexed
    same as inputs.
    """
    rng = prev_high - prev_low
    p = (prev_high + prev_low + prev_close) / 3.0
    return pd.DataFrame(
        {
            "S3": p - 1.000 * rng,
            "S2": p - 0.618 * rng,
            "S1": p - 0.382 * rng,
            "P":  p,
            "R1": p + 0.382 * rng,
            "R2": p + 0.618 * rng,
            "R3": p + 1.000 * rng,
        }
    )


def _aggregate_period(df: pd.DataFrame, group_id: pd.Series) -> pd.DataFrame:
    """Group OHLC by period (day or week), return per-period OHLC frame."""
    return df.groupby(group_id).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )


def _compute_pivot_levels_mapped(
    df: pd.DataFrame, group_id: pd.Series
) -> pd.DataFrame:
    """Compute prior-period pivots and broadcast onto df.index.

    Returns DataFrame with 7 columns (named per PIVOT_NAMES) indexed like df.
    For each row, the values are the pivots computed from THE PREVIOUS
    completed period's OHLC (no look-ahead — `.shift(1)` on the aggregated
    period frame).

    Implementation note (per Decision v2.45 CORRECTION 2026-04-29): use
    `period_pivots.reindex(group_id)` instead of `.loc[group_id.values]`.
    `group_id.values` returns a tz-naive numpy array which cannot match
    the tz-aware DatetimeIndex produced by groupby on tz-aware input —
    yields KeyError. `reindex(group_id)` preserves tz on both sides.
    """
    period_ohlc = _aggregate_period(df, group_id)
    period_pivots = _fib_pivots_from_ohlc(
        period_ohlc["high"].shift(1),
        period_ohlc["low"].shift(1),
        period_ohlc["close"].shift(1),
    )
    # Map each row's pivots based on which period it belongs to.
    # reindex preserves tz; .loc[.values] strips tz and breaks lookup.
    mapped = period_pivots.reindex(group_id).set_axis(df.index)
    mapped.columns = PIVOT_NAMES
    return mapped


# ─── Per-level dist_pct + zone + times_tested helpers ────────────────────
def _level_dist_pcts(close: pd.Series, levels: pd.DataFrame) -> pd.DataFrame:
    """Signed % distance from close to each of 7 levels."""
    return levels.apply(lambda lvl: pct(close - lvl, close))


def _zone_categorical(close: pd.Series, levels: pd.DataFrame) -> pd.Series:
    """Categorical 0..5 zone (v1.0 convention preserved):
        0 if close ≤ S3, 5 if close ≥ R3, else interval index.
    """
    arr = levels.values  # (n, 7) sorted S3<S2<S1<P<R1<R2<R3
    closes = close.values
    n = len(close)
    zone = np.full(n, np.nan)
    for i in range(n):
        row = arr[i]
        if np.isnan(row).any():
            continue
        c = closes[i]
        if c <= row[0]:
            zone[i] = 0
        elif c >= row[-1]:
            zone[i] = 5
        else:
            for z in range(6):
                if row[z] <= c < row[z + 1]:
                    zone[i] = z
                    break
    return pd.Series(zone, index=close.index)


def _times_tested_in_period(
    close: pd.Series, levels: pd.DataFrame, period_id: pd.Series, tolerance_pct: float
) -> pd.Series:
    """Cumulative count of bars per period that touched ANY pivot level
    within `tolerance_pct` (as fraction of price).
    """
    tol_abs = (tolerance_pct / 100.0)
    abs_dist = levels.sub(close, axis=0).abs()
    # bar is "touching" if any level is within tolerance × close
    touched_any = (abs_dist.divide(close, axis=0) <= tol_abs).any(axis=1).astype(int)
    return touched_any.groupby(period_id).cumsum()


# ─── Confluence helpers ──────────────────────────────────────────────────
def _daily_weekly_confluence(
    daily_levels: pd.DataFrame, weekly_levels: pd.DataFrame, atr_14: pd.Series, atr_mult: float
) -> pd.Series:
    """Binary flag: any of {daily_S1, daily_P, daily_R1} within atr_mult×ATR
    of any of 7 weekly levels.

    Per Decision v2.45 Q14.3.
    """
    daily_core = daily_levels[["S1", "P", "R1"]]
    weekly_all = weekly_levels  # all 7 levels
    threshold = atr_mult * atr_14  # per-bar absolute threshold

    # For each daily-core × weekly pair, check |d − w| ≤ threshold
    flag = pd.Series(False, index=daily_levels.index)
    for d_col in daily_core.columns:
        for w_col in weekly_all.columns:
            close_enough = (daily_core[d_col] - weekly_all[w_col]).abs() <= threshold
            flag = flag | close_enough.fillna(False)
    return flag.astype(int)


def _swing_fib_pivot_confluence(
    close: pd.Series,
    swing_low: pd.Series,
    swing_range: pd.Series,
    daily_levels: pd.DataFrame,
    weekly_levels: pd.DataFrame,
    atr_14: pd.Series,
    atr_mult: float,
) -> pd.Series:
    """Binary flag: nearest Fib retracement level (in price terms) is within
    atr_mult × ATR of any of 14 pivots (7 daily + 7 weekly).

    Per Decision v2.45 Q14.4.
    """
    # Compute the price at each Fib level
    fib_level_prices = {
        f: swing_low + f * swing_range for f in FIB_LEVELS
    }
    fib_prices_df = pd.DataFrame(fib_level_prices, index=close.index)

    # For each bar, find the Fib level closest to close (nearest by absolute distance)
    fib_dists = fib_prices_df.sub(close, axis=0).abs()
    nearest_fib_col = fib_dists.fillna(np.inf).idxmin(axis=1)
    nearest_fib_price = pd.Series(np.nan, index=close.index)
    for f in FIB_LEVELS:
        mask = nearest_fib_col == f
        nearest_fib_price.loc[mask] = fib_prices_df[f].loc[mask]

    threshold = atr_mult * atr_14

    # Check if any pivot (14 = 7 daily + 7 weekly) is within threshold of nearest_fib_price
    flag = pd.Series(False, index=close.index)
    for col in PIVOT_NAMES:
        flag = flag | ((nearest_fib_price - daily_levels[col]).abs() <= threshold).fillna(False)
        flag = flag | ((nearest_fib_price - weekly_levels[col]).abs() <= threshold).fillna(False)
    return flag.astype(int)


# ─── Public entry point ──────────────────────────────────────────────────
def pivot_features(df: pd.DataFrame, atr_14: pd.Series, cfg: dict) -> pd.DataFrame:
    """Compute Cat 6 = 30 pivot/Fib features.

    Parameters
    ----------
    df : DataFrame with open, high, low, close, volume; DatetimeIndex required
         (used for daily/weekly grouping).
    atr_14 : ATR(14) Series aligned to df.index.
    cfg : feature config dict. Uses (defaults baked in):
        cfg.get('pivots', {}).get('tolerance_pct', 0.05)        # times_tested touch threshold
        cfg.get('pivots', {}).get('swing_lookback', 5)          # fractal_pivots lookback
        cfg.get('pivots', {}).get('confluence_atr_mult', 0.25)  # for confluence flags
        cfg.get('pivots', {}).get('fib_touch_pct', 0.001)       # 0.1% — for fib_touches_*

    Returns
    -------
    DataFrame of 30 columns indexed like df.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "pivot_features requires df.index to be a DatetimeIndex (used "
            "for daily/weekly anchor grouping). Caller must set this before "
            "calling."
        )

    high, low, close = df["high"], df["low"], df["close"]
    p_cfg = cfg.get("pivots", {})
    tolerance_pct = p_cfg.get("tolerance_pct", 0.05)
    swing_lookback = p_cfg.get("swing_lookback", 5)
    confluence_atr_mult = p_cfg.get("confluence_atr_mult", 0.25)
    fib_touch_pct = p_cfg.get("fib_touch_pct", 0.001)

    # ── Daily and Weekly group IDs ──────────────────────────────────────
    day_id = pd.Series(df.index.floor("1D"), index=df.index)
    week_id = pd.Series(
        df.index.floor("1D") - pd.to_timedelta(df.index.dayofweek, unit="D"),
        index=df.index,
    )

    # ── Daily pivots (computed from prior-day OHLC, mapped onto df.index) ─
    daily_levels = _compute_pivot_levels_mapped(df, day_id)

    # ── Weekly pivots ────────────────────────────────────────────────────
    weekly_levels = _compute_pivot_levels_mapped(df, week_id)

    # ─────────────────────────────────────────────────────────────────────
    # Cat 6.1 — Daily Fib-pivots (9 features per Q14.1)
    # ─────────────────────────────────────────────────────────────────────
    daily_dist_pcts = _level_dist_pcts(close, daily_levels)
    daily_dist_pcts.columns = [f"pivot_{n}_dist_pct" for n in PIVOT_NAMES]
    pivot_zone = _zone_categorical(close, daily_levels)
    pivot_times_tested_today = _times_tested_in_period(
        close, daily_levels, day_id, tolerance_pct
    )

    # ─────────────────────────────────────────────────────────────────────
    # Cat 6.2 — Daily NEW encodings (3 features)
    # ─────────────────────────────────────────────────────────────────────
    pivot_position_daily_01 = safe_div(
        close - daily_levels["S1"],
        daily_levels["R1"] - daily_levels["S1"],
    )

    # min absolute distance to any of 7 daily levels, normalized by ATR
    daily_abs_dists = daily_levels.sub(close, axis=0).abs()
    nearest_daily_dist_abs = daily_abs_dists.min(axis=1)
    dist_to_nearest_pivot_atr = safe_div(nearest_daily_dist_abs, atr_14)

    daily_pivot_weekly_pivot_confluence = _daily_weekly_confluence(
        daily_levels, weekly_levels, atr_14, confluence_atr_mult
    )

    # ─────────────────────────────────────────────────────────────────────
    # Cat 6.3 — Weekly Fib-pivots (9 features, mirror of 6.1)
    # ─────────────────────────────────────────────────────────────────────
    weekly_dist_pcts = _level_dist_pcts(close, weekly_levels)
    weekly_dist_pcts.columns = [f"weekly_pivot_{n}_dist_pct" for n in PIVOT_NAMES]
    weekly_pivot_zone = _zone_categorical(close, weekly_levels)
    weekly_pivot_times_tested_this_week = _times_tested_in_period(
        close, weekly_levels, week_id, tolerance_pct
    )

    # ─────────────────────────────────────────────────────────────────────
    # Cat 6.4 — Weekly NEW encodings (2 features)
    # ─────────────────────────────────────────────────────────────────────
    pivot_position_weekly_01 = safe_div(
        close - weekly_levels["S1"],
        weekly_levels["R1"] - weekly_levels["S1"],
    )
    weekly_abs_dists = weekly_levels.sub(close, axis=0).abs()
    nearest_weekly_dist_abs = weekly_abs_dists.min(axis=1)
    dist_to_nearest_weekly_pivot_atr = safe_div(nearest_weekly_dist_abs, atr_14)

    # ─────────────────────────────────────────────────────────────────────
    # Cat 6.5 — Swing-based Fib retracements (7 features)
    # ─────────────────────────────────────────────────────────────────────
    # Use confirmed fractal pivots from divergence.py (lookback=5 → 2-left-2-right)
    pivot_high_series, _ = fractal_pivots(high, lookback=swing_lookback)
    _, pivot_low_series = fractal_pivots(low, lookback=swing_lookback)

    # Forward-fill: at any bar, swing_high = most recent confirmed pivot high value
    swing_high = pivot_high_series.ffill()
    swing_low = pivot_low_series.ffill()
    swing_range = (swing_high - swing_low).replace(0, np.nan)

    fib_retracement_pct = safe_div(close - swing_low, swing_range)

    in_golden_pocket = (
        (fib_retracement_pct >= 0.618) & (fib_retracement_pct <= 0.65)
    ).fillna(False).astype(int)

    # Fib level prices for nearest-distance computation
    fib_level_prices = pd.DataFrame(
        {f: swing_low + f * swing_range for f in FIB_LEVELS},
        index=close.index,
    )
    fib_abs_dists = fib_level_prices.sub(close, axis=0).abs()
    nearest_fib_dist_abs = fib_abs_dists.min(axis=1)
    nearest_fib_level_dist = safe_div(nearest_fib_dist_abs, atr_14)

    # Touches at 0.382 and 0.618 specifically — count last 20 bars within 0.1% × close
    fib_382_price = swing_low + 0.382 * swing_range
    fib_618_price = swing_low + 0.618 * swing_range
    touched_382 = ((close - fib_382_price).abs() <= fib_touch_pct * close).astype(int)
    touched_618 = ((close - fib_618_price).abs() <= fib_touch_pct * close).astype(int)
    fib_touches_382 = touched_382.rolling(20, min_periods=20).sum()
    fib_touches_618 = touched_618.rolling(20, min_periods=20).sum()

    # extension_progress_1272 per Q14.2 option (a)
    extension_progress_1272 = safe_div(close - swing_low, 1.272 * swing_range)

    swing_fib_pivot_confluence = _swing_fib_pivot_confluence(
        close, swing_low, swing_range,
        daily_levels, weekly_levels,
        atr_14, confluence_atr_mult,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Assemble all 30 features
    # ─────────────────────────────────────────────────────────────────────
    out = pd.DataFrame(index=df.index)

    # Cat 6.1 (9)
    for col in daily_dist_pcts.columns:
        out[col] = daily_dist_pcts[col]
    out["pivot_zone"] = pivot_zone
    out["pivot_times_tested_today"] = pivot_times_tested_today

    # Cat 6.2 (3)
    out["pivot_position_daily_01"] = pivot_position_daily_01
    out["dist_to_nearest_pivot_atr"] = dist_to_nearest_pivot_atr
    out["daily_pivot_weekly_pivot_confluence"] = daily_pivot_weekly_pivot_confluence

    # Cat 6.3 (9)
    for col in weekly_dist_pcts.columns:
        out[col] = weekly_dist_pcts[col]
    out["weekly_pivot_zone"] = weekly_pivot_zone
    out["weekly_pivot_times_tested_this_week"] = weekly_pivot_times_tested_this_week

    # Cat 6.4 (2)
    out["pivot_position_weekly_01"] = pivot_position_weekly_01
    out["dist_to_nearest_weekly_pivot_atr"] = dist_to_nearest_weekly_pivot_atr

    # Cat 6.5 (7)
    out["fib_retracement_pct"] = fib_retracement_pct
    out["in_golden_pocket"] = in_golden_pocket
    out["nearest_fib_level_dist"] = nearest_fib_level_dist
    out["fib_touches_382"] = fib_touches_382
    out["fib_touches_618"] = fib_touches_618
    out["extension_progress_1272"] = extension_progress_1272
    out["swing_fib_pivot_confluence"] = swing_fib_pivot_confluence

    return out
