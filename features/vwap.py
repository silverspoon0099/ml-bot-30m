"""Cat 5 — VWAP (14 features, multi-anchor) — v2.0.

Per Project Spec 30min §7.2 Cat 5 + Decision v2.4 (multi-anchor architecture)
+ Decision v2.44 (Q13 implementation choices locked).

Cat 5 = 14 features = 5 daily-anchored + 9 NEW multi-anchor:

Daily-anchored (5, per Q13.1):
  - daily_vwap                       — raw cumulative VWAP (resets at UTC day)
  - daily_vwap_upper_band_1sig       — daily_vwap + 1×stdev_band
  - daily_vwap_lower_band_1sig       — daily_vwap − 1×stdev_band
  - daily_vwap_dist_pct              — (close − daily_vwap) / close × 100
  - daily_vwap_zone                  — categorical {0..4} using ±1σ / ±2σ thresholds:
                                        0 = <-2σ, 1 = [-2σ,-1σ),
                                        2 = [-1σ,+1σ], 3 = (+1σ,+2σ], 4 = >+2σ

NEW multi-anchor (9):
  - swing_high_vwap_pos              — pct dist from VWAP anchored at last
                                        confirmed fractal pivot HIGH
                                        (uses divergence.fractal_pivots)
  - swing_low_vwap_pos               — same for fractal pivot LOWS
  - htf_pivot_vwap_pos               — pct dist from VWAP anchored at each
                                        daily-pivot-P close-cross event
                                        (per Q13.5 option a)
  - weekly_vwap_pos                  — pct dist from cumulative VWAP, anchor
                                        at Monday 00:00 UTC (resets weekly)
  - multi_anchor_confluence_signed_count
                                     — (# above) − (# below) across 5 anchors;
                                        range -5..+5 (Q13.2 option a)
  - vwap_of_vwaps_mean_reversion_dist_pct
                                     — synthetic VWAP-of-VWAPs (mean of 5
                                        anchors), rolling 20-bar mean of it,
                                        signed pct dist from close
  - dist_to_nearest_anchored_vwap_atr
                                     — min(|close − vwap_i|) / atr_14
                                        across all 5 anchors
  - vwap_cross_events_count_10       — count of bars in last 10 with at least
                                        one cross of any of 5 VWAPs;
                                        range 0..10 (Q13.4 option b)
  - close_above_below_heavy_vwap_flag
                                     — +1 if close > heavy_vwap, -1 if below,
                                        0 if no leader. Heavy VWAP = anchored
                                        VWAP with most touches in last 100
                                        bars. Touch = |close − vwap| ≤
                                        0.1 × atr_14 (Q13.3).

INPUTS (caller-supplied):
  df            — DataFrame with high, low, close, volume; DatetimeIndex.
  atr_14        — ATR(14) series (from volatility.py output).
  daily_pivot_p — Daily pivot P series (from pivots.py output, mapped on 30m).
  cfg           — feature config dict.

DROPPED from v1.0 (per Decision v2.44):
  vwap_session, vwap_session_dist_pct, vwap_slope (3 dropped). Session
  features moved out (or dropped); slope replaced by multi-anchor confluence
  which carries strictly more information.

ROLLBACK: see PROJECT_LOG Decision v2.44 entry for full transformation map.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import pct, safe_div
from .divergence import fractal_pivots


# ─── Anchored VWAP helper ────────────────────────────────────────────────
def _anchored_vwap(
    typical: pd.Series, volume: pd.Series, anchor_id: pd.Series
) -> pd.Series:
    """VWAP that resets at each anchor change (groupby cumulative).

    anchor_id: integer or timestamp series; cumulative sums reset when the
    value changes. e.g., for daily anchor: anchor_id = ts.floor("1D");
    for swing-pivot anchor: anchor_id increments at each new confirmed pivot.
    """
    pv = typical * volume
    cum_pv = pv.groupby(anchor_id).cumsum()
    cum_v = volume.groupby(anchor_id).cumsum()
    return safe_div(cum_pv, cum_v)


def vwap_features(
    df: pd.DataFrame,
    atr_14: pd.Series,
    daily_pivot_p: pd.Series,
    cfg: dict,
) -> pd.DataFrame:
    """Compute Cat 5 = 14 multi-anchor VWAP features.

    Parameters
    ----------
    df : DataFrame with high, low, close, volume; DatetimeIndex required
         (used for daily/weekly anchor grouping via .floor / .dayofweek).
    atr_14 : ATR(14) series aligned to df.index.
    daily_pivot_p : daily pivot P series aligned to df.index (from pivots.py).
                    May contain NaN during warmup; cross-detection handles NaN.
    cfg : feature config dict. Uses (all defaults baked in via .get):
        cfg['vwap']['bands_window']               (default 20)
        cfg['vwap']['swing_lookback']             (default 5; for fractal_pivots)
        cfg['vwap']['heavy_lookback_bars']        (default 100)
        cfg['vwap']['heavy_touch_atr_mult']       (default 0.1)

    Returns
    -------
    DataFrame of 14 columns indexed like df.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "vwap_features requires df.index to be a DatetimeIndex (used "
            "for daily/weekly anchor grouping). Caller must set this before "
            "calling."
        )

    high, low, close, volume = (
        df["high"], df["low"], df["close"], df["volume"]
    )
    typical = (high + low + close) / 3.0

    vwap_cfg = cfg.get("vwap", {})
    bands_window = vwap_cfg.get("bands_window", 20)
    swing_lookback = vwap_cfg.get("swing_lookback", 5)
    heavy_lookback = vwap_cfg.get("heavy_lookback_bars", 100)
    heavy_touch_mult = vwap_cfg.get("heavy_touch_atr_mult", 0.1)

    # ── Anchor IDs ───────────────────────────────────────────────────────
    # Daily: floor to UTC day
    day_id = pd.Series(df.index.floor("1D"), index=df.index)

    # Weekly: floor to UTC day, then subtract dayofweek to get Monday
    week_id = pd.Series(
        df.index.floor("1D") - pd.to_timedelta(df.index.dayofweek, unit="D"),
        index=df.index,
    )

    # Swing-high: anchor increments at each new confirmed fractal pivot HIGH
    pivot_high_series, _ = fractal_pivots(high, lookback=swing_lookback)
    new_swing_high = (~pivot_high_series.isna()).astype(int)
    swing_high_anchor_id = new_swing_high.cumsum()

    # Swing-low: anchor increments at each new confirmed fractal pivot LOW
    _, pivot_low_series = fractal_pivots(low, lookback=swing_lookback)
    new_swing_low = (~pivot_low_series.isna()).astype(int)
    swing_low_anchor_id = new_swing_low.cumsum()

    # HTF pivot break: anchor increments when close crosses daily_pivot_p
    crossed_above_p = (
        (close.shift(1) <= daily_pivot_p) & (close > daily_pivot_p)
    ).fillna(False)
    crossed_below_p = (
        (close.shift(1) >= daily_pivot_p) & (close < daily_pivot_p)
    ).fillna(False)
    htf_pivot_break = (crossed_above_p | crossed_below_p).astype(int)
    htf_pivot_anchor_id = htf_pivot_break.cumsum()

    # ── Anchored VWAPs (5) ───────────────────────────────────────────────
    daily_vwap = _anchored_vwap(typical, volume, day_id)
    weekly_vwap = _anchored_vwap(typical, volume, week_id)
    swing_high_vwap = _anchored_vwap(typical, volume, swing_high_anchor_id)
    swing_low_vwap = _anchored_vwap(typical, volume, swing_low_anchor_id)
    htf_pivot_vwap = _anchored_vwap(typical, volume, htf_pivot_anchor_id)

    # ── Daily-anchored bands + zone (Q13.1) ──────────────────────────────
    # Per-day rolling stdev of close (within UTC day group)
    stdev_band = close.groupby(day_id).transform(
        lambda s: s.rolling(bands_window, min_periods=bands_window).std(ddof=0)
    )
    upper_1sig = daily_vwap + stdev_band
    lower_1sig = daily_vwap - stdev_band
    upper_2sig = daily_vwap + 2.0 * stdev_band
    lower_2sig = daily_vwap - 2.0 * stdev_band

    daily_vwap_dist_pct = pct(close - daily_vwap, close)

    # Categorical zone (default mid = 2; assigned per threshold)
    zone = pd.Series(2.0, index=close.index, dtype=float)
    zone[close < lower_2sig] = 0
    zone[(close >= lower_2sig) & (close < lower_1sig)] = 1
    zone[(close > upper_1sig) & (close <= upper_2sig)] = 3
    zone[close > upper_2sig] = 4
    # NaN propagation: if stdev_band is NaN (warmup), zone is undefined
    zone[stdev_band.isna()] = np.nan

    # ── Multi-anchor pos features (4) ────────────────────────────────────
    swing_high_vwap_pos = pct(close - swing_high_vwap, close)
    swing_low_vwap_pos = pct(close - swing_low_vwap, close)
    htf_pivot_vwap_pos = pct(close - htf_pivot_vwap, close)
    weekly_vwap_pos = pct(close - weekly_vwap, close)

    # ── Multi-anchor analytics (5) ───────────────────────────────────────
    # All 5 anchored VWAPs as a list/dict for vectorized cross-anchor calcs
    vwap_dict = {
        "daily": daily_vwap,
        "weekly": weekly_vwap,
        "swing_high": swing_high_vwap,
        "swing_low": swing_low_vwap,
        "htf_pivot": htf_pivot_vwap,
    }
    vwap_names = list(vwap_dict.keys())
    vwap_series = list(vwap_dict.values())

    # 1. Multi-anchor confluence: signed count of (above − below) (Q13.2)
    above_count = pd.concat(
        [(close > v).astype(int) for v in vwap_series], axis=1
    ).sum(axis=1)
    below_count = pd.concat(
        [(close < v).astype(int) for v in vwap_series], axis=1
    ).sum(axis=1)
    multi_anchor_confluence_signed_count = (above_count - below_count).astype(float)

    # 2. VWAP-of-VWAPs mean reversion distance
    vwap_of_vwaps = pd.concat(vwap_series, axis=1).mean(axis=1)
    rolling_vw_of_vw = vwap_of_vwaps.rolling(20, min_periods=20).mean()
    vwap_of_vwaps_mean_reversion_dist_pct = pct(close - rolling_vw_of_vw, close)

    # 3. Distance to nearest anchored VWAP in ATR units
    dists = pd.concat([(close - v).abs() for v in vwap_series], axis=1)
    nearest_dist = dists.min(axis=1)
    dist_to_nearest_anchored_vwap_atr = safe_div(nearest_dist, atr_14)

    # 4. VWAP cross-events count last 10 bars (Q13.4 option b)
    any_cross = pd.Series(False, index=close.index)
    for v in vwap_series:
        crossed_up = (
            (close.shift(1) <= v.shift(1)) & (close > v)
        ).fillna(False)
        crossed_dn = (
            (close.shift(1) >= v.shift(1)) & (close < v)
        ).fillna(False)
        any_cross = any_cross | crossed_up | crossed_dn
    vwap_cross_events_count_10 = (
        any_cross.astype(int).rolling(10, min_periods=10).sum()
    )

    # 5. Heavy VWAP flag (Q13.3): +1 above, -1 below, 0 if no leader
    touches_dict = {}
    for name, v in vwap_dict.items():
        touched = ((close - v).abs() <= heavy_touch_mult * atr_14).astype(int)
        touches_dict[name] = touched.rolling(
            heavy_lookback, min_periods=heavy_lookback
        ).sum()
    touches_df = pd.DataFrame(touches_dict)

    # Identify which VWAP has max touches at each bar.
    # fillna(0) before idxmax: pandas 3.x will raise ValueError on all-NaN
    # rows (warmup before heavy_lookback bars). Filling with 0 means idxmax
    # returns the first column for warmup rows; the no_leader mask below
    # then sets heavy_vwap_value to NaN, so the flag becomes 0 for warmup —
    # behavior preserved across pandas versions, no FutureWarning.
    heavy_name = touches_df.fillna(0).idxmax(axis=1)  # ties: first column wins

    # Map heavy name → corresponding VWAP value
    heavy_vwap_value = pd.Series(np.nan, index=close.index)
    for name, v in vwap_dict.items():
        mask = heavy_name == name
        heavy_vwap_value.loc[mask] = v.loc[mask]

    # No-leader case: when total touches across all VWAPs is 0 (warmup or
    # genuinely no touches in window), treat as no clear leader
    total_touches = touches_df.sum(axis=1)
    no_leader = (total_touches == 0) | total_touches.isna()
    heavy_vwap_value[no_leader] = np.nan

    close_above_below_heavy_vwap_flag = np.sign(close - heavy_vwap_value)
    close_above_below_heavy_vwap_flag = close_above_below_heavy_vwap_flag.fillna(0)

    return pd.DataFrame(
        {
            # Daily-anchored (5)
            "daily_vwap": daily_vwap,
            "daily_vwap_upper_band_1sig": upper_1sig,
            "daily_vwap_lower_band_1sig": lower_1sig,
            "daily_vwap_dist_pct": daily_vwap_dist_pct,
            "daily_vwap_zone": zone,
            # Multi-anchor pos (4)
            "swing_high_vwap_pos": swing_high_vwap_pos,
            "swing_low_vwap_pos": swing_low_vwap_pos,
            "htf_pivot_vwap_pos": htf_pivot_vwap_pos,
            "weekly_vwap_pos": weekly_vwap_pos,
            # Multi-anchor analytics (5)
            "multi_anchor_confluence_signed_count": multi_anchor_confluence_signed_count,
            "vwap_of_vwaps_mean_reversion_dist_pct": vwap_of_vwaps_mean_reversion_dist_pct,
            "dist_to_nearest_anchored_vwap_atr": dist_to_nearest_anchored_vwap_atr,
            "vwap_cross_events_count_10": vwap_cross_events_count_10,
            "close_above_below_heavy_vwap_flag": close_above_below_heavy_vwap_flag,
        }
    )
