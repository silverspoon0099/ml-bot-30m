"""Cat 11 — Previous Context (6 features) + Cat 12 — Lagged Dynamics (5 features) — v2.0.

Per Project Spec 30min §7.2 Cat 11 + Cat 12 + Decision v2.37 Q2 (literal
prev-bar interpretation) + Decision v2.51 Q20.

Two functions per Q20.9 (a) split per spec category:

  - prev_context_features(df, cfg=None) -> DataFrame[6]
      Cat 11 — self-contained; prev-bar via .shift(1) + today-running
              via groupby UTC day.

  - lagged_dynamics_features(df, rsi, adx, ema21_dist_pct, cfg=None) -> DataFrame[5]
      Cat 12 — caller-supplied 3 series (rsi from Cat 1 momentum_core,
              adx + ema21_dist_pct from Cat 2 trend); volume from df.

Cat 11 features (6 — MIXED-BLOCK SPLIT per Decision v2.51 Q20.8):

  STATIC (4 — `.shift(1)` lookups; intrabar-safe):
    - prev_bar_close_vs_close_pct  — (close.shift(1) − close) / close × 100
                                      Sign: positive when prev close > current
    - prev_bar_high_vs_close_pct   — (high.shift(1) − close) / close × 100
    - prev_bar_low_vs_close_pct    — (low.shift(1) − close) / close × 100
    - prev_bar_volume_ratio        — volume.shift(1) / volume
                                      > 1 when prev had higher volume

  DYNAMIC (2 — depend on current close):
    - today_open_to_now_pct        — (close − today_open) / today_open × 100
                                      where today_open = first 30m bar's
                                      OPEN per UTC day
    - today_high_low_distance_from_current_pct — SIGNED distance to whichever
                                      of today's running high/low is closer.
                                      Positive ⟺ today_low closer (price near
                                      support); negative ⟺ today_high closer
                                      (price near resistance). Magnitude =
                                      min(dist_high, dist_low) / close × 100.

Cat 12 features (5, all DYNAMIC — deltas of dynamic series):
    - delta_rsi_1               — rsi − rsi.shift(1) (raw point diff)
    - delta_rsi_3               — rsi − rsi.shift(3)
    - delta_adx_3               — adx − adx.shift(3) (raw point diff;
                                   ADX is bounded so raw delta is interpretable)
    - delta_volume_3            — (volume − volume.shift(3)) / volume.shift(3) × 100
                                   (% change; volume is unbounded so ratio
                                   is more interpretable than raw diff)
    - delta_close_vs_ema21_3    — ema21_dist_pct − ema21_dist_pct.shift(3)
                                   (raw pp difference; ema21_dist_pct is
                                   already in % so pp delta is interpretable)

DROPPED from v1.0 (per Decision v2.37 Q2 + Decision v2.51):
  - All v1.0 prev-DAY Cat 11 features (8): prev_day_range_pct,
    prev_day_close_vs_pivot, gap_pct, dist_to_prev_day_high_pct,
    dist_to_prev_day_low_pct, prev_session_direction,
    prev_session_volume_rank, daily_open_dist_pct.
    All duplicates of Cat 2a HTF1D context per Decision v2.37 Q2.
  - All v1.0 5bar-slope Cat 12 features (8): rsi_5bar_ago,
    wt1_slope_5bar, adx_slope_5bar, atr_slope_5bar, volume_slope_5bar,
    vwap_slope_5bar, di_spread_change_5bar, squeeze_mom_slope.
    All overlap Cat 1/2/5 outputs or use longer lag windows that
    overlap HTF context (Cat 2a) per spec.

§7.5 LIST EDIT (Decision v2.51 Q20.8): Cat 11 moved from STATIC list to
MIXED list with explicit per-feature 4-static / 2-dynamic split. Cat 11 +
Cat 16 = two mixed-split blocks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import pct, safe_div


# ─── Cat 11 — Previous Context (6 features) ─────────────────────────────
def prev_context_features(
    df: pd.DataFrame, cfg: dict | None = None
) -> pd.DataFrame:
    """Compute Cat 11 = 6 prev-bar + today-running context features.

    Self-contained — math derives from OHLC + volume + UTC-day boundary
    (df.index.floor('1D')) alone.

    Parameters
    ----------
    df : DataFrame with open, high, low, close, volume columns AND a
         DatetimeIndex (required for UTC-day grouping).
    cfg : optional config dict. Currently no tunables (windows + UTC-
          day boundary spec-locked).

    Returns
    -------
    DataFrame of 6 columns indexed like df. MIXED-BLOCK split per §7.5
    Q20.8: 4 static (prev_bar_*) + 2 dynamic (today_*).
    """
    _ = cfg  # reserved for future tunables

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "prev_context_features requires a DatetimeIndex on df for "
            "UTC-day boundary computation"
        )

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df["volume"]

    # ── Prev-bar lookups (4 STATIC) ───────────────────────────────────
    # `.shift(1)` lookups — locked once prev bar closed; intrabar-safe.
    # Sign convention per Q20.1 (a): positive when prev reference > close.
    prev_bar_close_vs_close_pct = pct(c.shift(1) - c, c)
    prev_bar_high_vs_close_pct = pct(h.shift(1) - c, c)
    prev_bar_low_vs_close_pct = pct(l.shift(1) - c, c)
    # Q20.2 (a) per spec literal numerator/denominator: prev / current.
    prev_bar_volume_ratio = safe_div(v.shift(1), v)

    # ── Today-running (2 DYNAMIC, UTC-day grouped) ────────────────────
    day_id = df.index.floor("1D")

    # Q20.5 (a) today_open = first 30m bar's OPEN per UTC day.
    today_open = o.groupby(day_id).transform("first")
    today_open_to_now_pct = pct(c - today_open, today_open)

    # Q20.3 (b) signed distance to whichever of today's running high/low
    # is closer. Positive ⟺ today_low closer (near support); negative ⟺
    # today_high closer (near resistance). Magnitude = min(...).
    today_high = h.groupby(day_id).cummax()
    today_low = l.groupby(day_id).cummin()
    dist_high = (today_high - c).abs()
    dist_low = (c - today_low).abs()
    nearer_is_low = dist_low < dist_high
    min_dist = pd.concat([dist_high, dist_low], axis=1).min(axis=1)
    signed_dist = np.where(nearer_is_low, min_dist, -min_dist)
    today_high_low_distance_from_current_pct = pd.Series(
        signed_dist, index=df.index
    ) / c.replace(0, np.nan) * 100.0

    return pd.DataFrame(
        {
            "prev_bar_close_vs_close_pct": prev_bar_close_vs_close_pct,
            "prev_bar_high_vs_close_pct": prev_bar_high_vs_close_pct,
            "prev_bar_low_vs_close_pct": prev_bar_low_vs_close_pct,
            "prev_bar_volume_ratio": prev_bar_volume_ratio,
            "today_open_to_now_pct": today_open_to_now_pct,
            "today_high_low_distance_from_current_pct": today_high_low_distance_from_current_pct,
        },
        index=df.index,
    )


# ─── Cat 12 — Lagged Dynamics (5 features) ──────────────────────────────
def lagged_dynamics_features(
    df: pd.DataFrame,
    rsi: pd.Series,
    adx: pd.Series,
    ema21_dist_pct: pd.Series,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Compute Cat 12 = 5 lagged-dynamics delta features.

    Parameters
    ----------
    df : DataFrame with volume column (only).
    rsi : caller-supplied RSI series from Cat 1 momentum_core output.
    adx : caller-supplied ADX series from Cat 2 trend output.
    ema21_dist_pct : caller-supplied (close − ema21) / close × 100 series
                     from Cat 2 trend output.
    cfg : optional config dict. Currently no tunables (lag windows
          spec-locked at 1, 3, 3, 3, 3).

    Returns
    -------
    DataFrame of 5 columns indexed like df. All DYNAMIC per §7.5 (deltas
    of dynamic series).

    Notes
    -----
    Lag-window choices per spec (Decision v2.51 Q20.10):
      - delta_rsi_1: lag=1 (most-recent change in RSI; momentum trigger)
      - delta_rsi_3: lag=3 (3-bar RSI trajectory; medium-term momentum)
      - delta_adx_3: lag=3 (ADX trend strength delta over 3 bars)
      - delta_volume_3: lag=3 (% volume change over 3 bars)
      - delta_close_vs_ema21_3: lag=3 (price-vs-EMA21 distance delta)

    Delta formula per-feature (Q20.7 (a)):
      - Raw point-difference for bounded series (RSI/ADX/ema21_dist_pct)
      - Percentage change for unbounded volume (volume.shift(3) ratio)
    """
    _ = cfg  # reserved for future tunable lag windows

    volume = df["volume"]

    return pd.DataFrame(
        {
            "delta_rsi_1": rsi - rsi.shift(1),
            "delta_rsi_3": rsi - rsi.shift(3),
            "delta_adx_3": adx - adx.shift(3),
            "delta_volume_3": pct(volume - volume.shift(3), volume.shift(3)),
            "delta_close_vs_ema21_3": ema21_dist_pct - ema21_dist_pct.shift(3),
        },
        index=df.index,
    )
