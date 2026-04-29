"""Cat 20 — Event Memory (22 features) — v2.0.

Per Project Spec 30min §7.2 Cat 20 trim 41→22.

22 features (locked per spec; column names enumerated):

  RSI block (5):
    - bars_since_rsi_ob          — `bars_since(rsi > 70)`
    - bars_since_rsi_os          — `bars_since(rsi < 30)`
    - bars_since_rsi_mid_cross   — `bars_since(rsi crossed 50)`
    - bars_in_current_rsi_episode — consecutive bar count in OB or OS state
    - last_rsi_extreme_depth     — depth of most recent OB/OS episode

  Stoch block (2):
    - bars_since_stoch_ob        — `bars_since(stoch_k > 80)`
    - bars_since_stoch_os        — `bars_since(stoch_k < 20)`

  WaveTrend block (2):
    - bars_since_wt_ob           — `bars_since(wt1 > ob_level)`
    - bars_since_wt_os           — `bars_since(wt1 < os_level)`

  ADX block (2):
    - bars_since_adx_trend_start — `bars_since(adx just crossed above 25)`
    - bars_since_adx_weak_start  — `bars_since(adx just crossed below 20)`

  Squeeze block (3, depends on volatility.squeeze_state):
    - bars_since_squeeze_fire    — `bars_since(squeeze_state == +1)`
    - bars_since_squeeze_entry   — `bars_since(squeeze_state went 0→-1)`
    - squeeze_direction_at_fire  — sign(squeeze_value) at last fire bar (ffill)

  Volume block (1):
    - bars_since_volume_spike    — `bars_since(volume > 3 × SMA(20))`

  Structure block (2, uses fractal_pivots from divergence.py):
    - bars_since_last_hh         — bars since most recent confirmed Higher High
    - bars_since_last_ll         — bars since most recent confirmed Lower Low

  Pivot block (2, uses pivots.compute_pivot_levels):
    - bars_since_pivot_touch_daily   — `bars_since(close within tol×close of any daily pivot)`
    - bars_since_pivot_touch_weekly  — same for weekly

  EMA block (1):
    - bars_since_ema21_cross     — `bars_since(close crossed EMA21)`

  MACD block (2):
    - bars_since_macd_zero_cross    — `bars_since(macd_line crossed 0)`
    - bars_since_macd_signal_cross  — `bars_since(macd_line crossed macd_signal)`

DROPPED from v1.0 (~19 features): green/red bar counts, depth metrics,
rolling 50/100/200-bar count windows, recovery_speed, di_cross,
adx_peak, hist max/min, momentum_at_fire, etc.

§7.5: Cat 20 = dynamic (event triggers depend on dynamic oscillators
and current close; counters mutate as new bars open/close).

CALLER-SUPPLIED inputs (orchestrated by builder.py):
  df, rsi, stoch_k, wt1, adx, macd_line, macd_signal, squeeze_state,
  squeeze_value, daily_pivot_levels, weekly_pivot_levels, cfg.

DEFAULT THRESHOLDS (cfg.get with fallbacks; v1.0-aligned):
  rsi_ob=70, rsi_os=30, rsi_mid=50
  stoch_ob=80, stoch_os=20
  wt_ob=53, wt_os=-53 (LazyBear)
  adx_trend=25, adx_weak=20
  volume_spike_mult=3.0
  pivot tolerance: cfg['pivots']['tolerance_pct'] = 0.05 (0.05% × close)
  swing_lookback=5 (fractal pivots, 2-left-2-right)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import bars_since, crosses_above, crosses_below
from .divergence import fractal_pivots


# ─── Helpers ─────────────────────────────────────────────────────────────
def _bars_in_current_state(state: pd.Series) -> pd.Series:
    """Consecutive bar count where `state` is True; resets on False."""
    s = state.astype(bool)
    grp = (~s).cumsum()
    return s.astype(int).groupby(grp).cumsum()


def _last_extreme_depth(rsi: pd.Series, ob: float, os: float) -> pd.Series:
    """Depth of most recent OB/OS RSI episode, forward-filled.

    For OB episode: depth = max(rsi - ob) over the active episode.
    For OS episode: depth = max(os - rsi) over the active episode.
    Whichever was most recent dominates (tracks the most recent extreme).
    """
    arr = rsi.to_numpy()
    n = len(arr)
    out = np.full(n, np.nan)
    cur_kind: str | None = None
    cur_depth = np.nan
    last_depth = np.nan
    for i in range(n):
        v = arr[i]
        if np.isnan(v):
            out[i] = last_depth
            continue
        if v > ob:
            depth = v - ob
            if cur_kind != "ob":
                cur_depth = depth
                cur_kind = "ob"
            else:
                cur_depth = max(cur_depth, depth)
            last_depth = cur_depth
        elif v < os:
            depth = os - v
            if cur_kind != "os":
                cur_depth = depth
                cur_kind = "os"
            else:
                cur_depth = max(cur_depth, depth)
            last_depth = cur_depth
        else:
            cur_kind = None
            cur_depth = np.nan
        out[i] = last_depth
    return pd.Series(out, index=rsi.index)


def _hh_ll_events(
    high: pd.Series, low: pd.Series, lookback: int = 5
) -> tuple[pd.Series, pd.Series]:
    """Return (hh_event, ll_event) — boolean Series, True at bars where
    a new Higher High or Lower Low is confirmed.

    HH = current confirmed pivot_high > previous confirmed pivot_high.
    LL = current confirmed pivot_low < previous confirmed pivot_low.
    Uses fractal_pivots(lookback=5) — 2-left-2-right rule (no look-ahead).
    """
    pivot_high, _ = fractal_pivots(high, lookback)
    _, pivot_low = fractal_pivots(low, lookback)

    hh_event = pd.Series(False, index=high.index)
    ll_event = pd.Series(False, index=low.index)

    last_high = np.nan
    last_low = np.nan

    p_h_arr = pivot_high.to_numpy()
    p_l_arr = pivot_low.to_numpy()
    for i in range(len(high)):
        if not np.isnan(p_h_arr[i]):
            if not np.isnan(last_high) and p_h_arr[i] > last_high:
                hh_event.iloc[i] = True
            last_high = p_h_arr[i]
        if not np.isnan(p_l_arr[i]):
            if not np.isnan(last_low) and p_l_arr[i] < last_low:
                ll_event.iloc[i] = True
            last_low = p_l_arr[i]

    return hh_event, ll_event


# ─── Public entry point ──────────────────────────────────────────────────
def event_memory_features(
    df: pd.DataFrame,
    rsi: pd.Series,
    stoch_k: pd.Series,
    wt1: pd.Series,
    adx: pd.Series,
    macd_line: pd.Series,
    macd_signal: pd.Series,
    squeeze_state: pd.Series,
    squeeze_value: pd.Series,
    daily_pivot_levels: pd.DataFrame,
    weekly_pivot_levels: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Compute Cat 20 = 22 event memory features.

    Parameters
    ----------
    df : DataFrame with open/high/low/close/volume.
    rsi, stoch_k, wt1, adx, macd_line, macd_signal :
        Oscillator series aligned to df.index.
    squeeze_state : -1/0/+1 state machine from volatility.py output.
    squeeze_value : raw LazyBear squeeze momentum from indicators.py.
    daily_pivot_levels : 7-col DataFrame from pivots.compute_pivot_levels.
    weekly_pivot_levels : 7-col DataFrame from pivots.compute_pivot_levels.
    cfg : feature config dict.

    Returns
    -------
    DataFrame of 22 columns indexed like df.
    """
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    em_cfg = cfg.get("event_memory", {})
    rsi_ob = em_cfg.get("rsi_ob", 70)
    rsi_os = em_cfg.get("rsi_os", 30)
    rsi_mid = em_cfg.get("rsi_mid", 50)
    stoch_ob = em_cfg.get("stoch_ob", 80)
    stoch_os = em_cfg.get("stoch_os", 20)
    wt_ob = em_cfg.get("wt_ob", 53)
    wt_os = em_cfg.get("wt_os", -53)
    adx_trend = em_cfg.get("adx_trend", 25)
    adx_weak = em_cfg.get("adx_weak", 20)
    volume_spike_mult = em_cfg.get("volume_spike_mult", 3.0)
    swing_lookback = em_cfg.get("swing_lookback", 5)
    pivot_tolerance_pct = cfg.get("pivots", {}).get("tolerance_pct", 0.05)

    # ── RSI block (5) ────────────────────────────────────────────────────
    in_rsi_ob = rsi > rsi_ob
    in_rsi_os = rsi < rsi_os
    rsi_mid_cross = (
        crosses_above(rsi, rsi_mid).astype(bool)
        | crosses_below(rsi, rsi_mid).astype(bool)
    )

    bars_since_rsi_ob = bars_since(in_rsi_ob)
    bars_since_rsi_os = bars_since(in_rsi_os)
    bars_since_rsi_mid_cross = bars_since(rsi_mid_cross)
    bars_in_current_rsi_episode = _bars_in_current_state(in_rsi_ob | in_rsi_os)
    last_rsi_extreme_depth = _last_extreme_depth(rsi, rsi_ob, rsi_os)

    # ── Stoch block (2) ──────────────────────────────────────────────────
    bars_since_stoch_ob = bars_since(stoch_k > stoch_ob)
    bars_since_stoch_os = bars_since(stoch_k < stoch_os)

    # ── WT block (2) ─────────────────────────────────────────────────────
    bars_since_wt_ob = bars_since(wt1 > wt_ob)
    bars_since_wt_os = bars_since(wt1 < wt_os)

    # ── ADX block (2) ────────────────────────────────────────────────────
    just_started_trending = (
        (adx.shift(1) <= adx_trend) & (adx > adx_trend)
    ).fillna(False)
    just_started_weak = (
        (adx.shift(1) >= adx_weak) & (adx < adx_weak)
    ).fillna(False)
    bars_since_adx_trend_start = bars_since(just_started_trending)
    bars_since_adx_weak_start = bars_since(just_started_weak)

    # ── Squeeze block (3) ────────────────────────────────────────────────
    fire = (squeeze_state == 1)
    entered = (
        (squeeze_state.shift(1) != -1) & (squeeze_state == -1)
    ).fillna(False)
    bars_since_squeeze_fire = bars_since(fire)
    bars_since_squeeze_entry = bars_since(entered)
    # Direction at fire: sign(squeeze_value) at the bar where fire fired;
    # forward-filled to give "what was the direction at the last release"
    squeeze_direction_at_fire = np.sign(squeeze_value).where(fire).ffill()

    # ── Volume block (1) ─────────────────────────────────────────────────
    sma_vol20 = volume.rolling(20, min_periods=20).mean()
    volume_spike = volume > volume_spike_mult * sma_vol20
    bars_since_volume_spike = bars_since(volume_spike)

    # ── Structure block (2) ──────────────────────────────────────────────
    hh_event, ll_event = _hh_ll_events(high, low, lookback=swing_lookback)
    bars_since_last_hh = bars_since(hh_event)
    bars_since_last_ll = bars_since(ll_event)

    # ── Pivot touch block (2) ────────────────────────────────────────────
    tol_frac = pivot_tolerance_pct / 100.0
    daily_abs_dist = daily_pivot_levels.sub(close, axis=0).abs()
    daily_touched = (
        daily_abs_dist.divide(close, axis=0) <= tol_frac
    ).any(axis=1)
    bars_since_pivot_touch_daily = bars_since(daily_touched)

    weekly_abs_dist = weekly_pivot_levels.sub(close, axis=0).abs()
    weekly_touched = (
        weekly_abs_dist.divide(close, axis=0) <= tol_frac
    ).any(axis=1)
    bars_since_pivot_touch_weekly = bars_since(weekly_touched)

    # ── EMA cross block (1) ──────────────────────────────────────────────
    ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    ema21_crossed = (
        crosses_above(close, ema21).astype(bool)
        | crosses_below(close, ema21).astype(bool)
    )
    bars_since_ema21_cross = bars_since(ema21_crossed)

    # ── MACD block (2) ───────────────────────────────────────────────────
    macd_zero_crossed = (
        crosses_above(macd_line, 0).astype(bool)
        | crosses_below(macd_line, 0).astype(bool)
    )
    macd_signal_crossed = (
        crosses_above(macd_line, macd_signal).astype(bool)
        | crosses_below(macd_line, macd_signal).astype(bool)
    )
    bars_since_macd_zero_cross = bars_since(macd_zero_crossed)
    bars_since_macd_signal_cross = bars_since(macd_signal_crossed)

    return pd.DataFrame(
        {
            # RSI (5)
            "bars_since_rsi_ob": bars_since_rsi_ob,
            "bars_since_rsi_os": bars_since_rsi_os,
            "bars_since_rsi_mid_cross": bars_since_rsi_mid_cross,
            "bars_in_current_rsi_episode": bars_in_current_rsi_episode,
            "last_rsi_extreme_depth": last_rsi_extreme_depth,
            # Stoch (2)
            "bars_since_stoch_ob": bars_since_stoch_ob,
            "bars_since_stoch_os": bars_since_stoch_os,
            # WT (2)
            "bars_since_wt_ob": bars_since_wt_ob,
            "bars_since_wt_os": bars_since_wt_os,
            # ADX (2)
            "bars_since_adx_trend_start": bars_since_adx_trend_start,
            "bars_since_adx_weak_start": bars_since_adx_weak_start,
            # Squeeze (3)
            "bars_since_squeeze_fire": bars_since_squeeze_fire,
            "bars_since_squeeze_entry": bars_since_squeeze_entry,
            "squeeze_direction_at_fire": squeeze_direction_at_fire,
            # Volume (1)
            "bars_since_volume_spike": bars_since_volume_spike,
            # Structure (2)
            "bars_since_last_hh": bars_since_last_hh,
            "bars_since_last_ll": bars_since_last_ll,
            # Pivot (2)
            "bars_since_pivot_touch_daily": bars_since_pivot_touch_daily,
            "bars_since_pivot_touch_weekly": bars_since_pivot_touch_weekly,
            # EMA (1)
            "bars_since_ema21_cross": bars_since_ema21_cross,
            # MACD (2)
            "bars_since_macd_zero_cross": bars_since_macd_zero_cross,
            "bars_since_macd_signal_cross": bars_since_macd_signal_cross,
        },
        index=df.index,
    )
