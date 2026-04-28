"""Cat 4 — Volume / Buy-Sell Pressure (12 features) +
Cat 14 — Money Flow (6 features) — v2.0.

Per Project Spec 30min §7.2 Cat 4 (17→12) + Cat 14 (8→6).

Cat 4 = 12 features (volume_features):
  Volume statistics (3):
    - volume_ratio_20         — volume / SMA(volume, 20)
    - volume_zscore_20        — (volume − SMA20) / std20
    - volume_spike_flag       — 1 if volume > 3.0 × SMA20 (cfg-tunable)
  VFI block (3):
    - vfi                     — LazyBear Volume Flow Indicator
    - vfi_signal              — EMA(vfi, signal_period); momentum line
    - vfi_hist_slope          — (vfi − vfi_signal).diff(); 1-bar slope of histogram
  OBV block (2):
    - obv_slope               — (obv − obv.shift(10)) / 10; 10-bar slope
    - obv_divergence_flag     — signed -1/0/+1 from fractal-based detect_divergence
                                (price vs OBV regular divergence; reuses Cat 13 math)
  Buy/sell estimator (1):
    - candle_buy_sell_signed  — ((close-low)/(high-low)) × sign(close-open).
                                Spec: "(close-low)/(high-low), signed by body direction".
                                Range [-1, +1]; near 0 = doji or weak.
  Volume-weighted momentum (1):
    - volume_weighted_momentum_10 — Σ(return × volume) / Σ(volume) last 10 bars.
                                    pct_change-based; vol-weighted average return.
  High-volume-candle location (2 binary):
    - high_vol_close_top_tercile     — 1 if (volume > 1.5×SMA20) AND
                                       ((close-low)/(high-low) ≥ 2/3)
    - high_vol_close_bottom_tercile  — 1 if (volume > 1.5×SMA20) AND
                                       ((close-low)/(high-low) ≤ 1/3)
    Mid tercile is implicit when both flags = 0.

Cat 14 = 6 features (money_flow_features):
  CMF block (2):
    - cmf_20                  — Chaikin Money Flow, 20-bar
    - cmf_slope               — cmf_20.diff(); 1-bar slope (renamed from
                                v1.0 `cmf_direction`)
  MFI block (2):
    - mfi_14                  — Money Flow Index, 14-bar (Wilder)
    - mfi_zone                — +1 OB (MFI > 80) / -1 OS (MFI < 20) / 0 mid
                                (replaces v1.0 `mfi_direction`)
  A/D block (2):
    - ad_slope_10             — A/D line 10-bar slope
    - ad_price_divergence_flag — signed -1/0/+1 from fractal-based
                                 detect_divergence (price vs A/D)

DROPPED from v1.0 (per strict spec):
  Cat 4: volume_trend, buy_volume_pct, sell_volume_pct, buy_sell_ratio_sma5,
         cum_delta_5, cum_delta_20, vfi_direction, vfi_hist_direction,
         vfi_cross_zero, vfi_range_pct, vfi_hist (raw — replaced with slope),
         vfi_features_1h (entire function — no 1h in v2.0).
  Cat 14: cmf_direction (renamed → cmf_slope), mfi_direction (replaced by
          mfi_zone), ad_line (raw — kept slope only), mfi_rsi_divergence
          (cross-feature not in spec), cmf_vfi_agreement (cross-feature not
          in spec).

ROLLBACK: if SHAP shows dropped features useful, restore via git revert
on this commit + restore v1.0 column names in feature_stability.py.

SIGNATURE CHANGES vs v1.0:
  - vfi_features_5m + volume_features merged into single `volume_features(df, cfg)`
  - vfi_features_1h DELETED entirely
  - money_flow_features no longer takes `vfi_series` and `rsi_series` (cross-feature
    columns dropped per spec)

Math `vfi()` kept at module level for reusability if needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import safe_div
from .divergence import detect_divergence


# ─── VFI math helper (LazyBear, kept from v1.0) ──────────────────────────
def vfi(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    period: int,
    coef: float,
    vcoef: float,
) -> pd.Series:
    """LazyBear Volume Flow Indicator.

    Reference: LazyBear (Chris Moody) 2009 Pine implementation.
        typical = (high + low + close) / 3
        inter = log(typical) − log(typical[1])
        vinter = stdev(inter, 30)
        cutoff = coef × vinter × close
        vave = sma(volume, period)[1]      # PREVIOUS bar's avg (no look-ahead)
        vmax = vave × vcoef
        vc = min(volume, vmax)              # cap volume at vmax
        mf = typical − typical[1]
        vcp = vc × sign(mf if abs(mf) > cutoff else 0)
        vfi = sum(vcp, period) / vave
        # Final smoothing: sma(vfi, 3) per LazyBear's Pine.

    Default cfg from v1.0: period=130, coef=0.2, vcoef=2.5.
    """
    typical = (high + low + close) / 3.0
    inter = np.log(typical) - np.log(typical.shift(1))
    vinter = inter.rolling(30, min_periods=30).std(ddof=0)
    cutoff = coef * vinter * close
    vave = volume.rolling(period, min_periods=period).mean().shift(1)
    vmax = vave * vcoef
    vc = volume.where(volume <= vmax, vmax)
    mf = typical - typical.shift(1)
    direction = np.where(mf.abs() > cutoff, np.sign(mf), 0.0)
    vcp = vc * direction
    raw = vcp.rolling(period, min_periods=period).sum() / vave
    return raw.rolling(3, min_periods=3).mean()  # LazyBear final SMA(3)


# ─── Cat 4 — Volume / Buy-Sell Pressure (12 features) ────────────────────
def volume_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute Cat 4 = 12 volume/buy-sell features.

    Parameters
    ----------
    df : DataFrame with open, high, low, close, volume.
    cfg : feature config dict. Uses:
          - cfg['vfi']['period']           (default 130 — LazyBear)
          - cfg['vfi']['coef']             (default 0.2)
          - cfg['vfi']['vcoef']            (default 2.5)
          - cfg['vfi']['signal_period']    (default 5 — for vfi_signal EMA)
          - cfg.get('volume', {}).get('spike_mult', 3.0)
          - cfg.get('volume', {}).get('high_vol_mult', 1.5)
          - cfg.get('volume', {}).get('divergence_lookback_bars', 14)

    Returns
    -------
    DataFrame of 12 columns indexed like df.
    """
    open_, high, low, close, volume = (
        df["open"], df["high"], df["low"], df["close"], df["volume"]
    )
    rng = (high - low).replace(0, np.nan)

    vfi_cfg = cfg["vfi"]
    vol_cfg = cfg.get("volume", {})
    spike_mult = vol_cfg.get("spike_mult", 3.0)
    high_vol_mult = vol_cfg.get("high_vol_mult", 1.5)
    div_lookback = vol_cfg.get("divergence_lookback_bars", 14)

    # ── Volume statistics (3) ────────────────────────────────────────────
    sma20 = volume.rolling(20, min_periods=20).mean()
    std20 = volume.rolling(20, min_periods=20).std(ddof=0)

    volume_ratio_20 = safe_div(volume, sma20)
    volume_zscore_20 = safe_div(volume - sma20, std20)
    volume_spike_flag = (volume > spike_mult * sma20).astype(int)

    # ── VFI block (3) ────────────────────────────────────────────────────
    vfi_v = vfi(
        close, high, low, volume,
        vfi_cfg["period"], vfi_cfg["coef"], vfi_cfg["vcoef"],
    )
    sig_period = vfi_cfg.get("signal_period", 5)
    vfi_signal = vfi_v.ewm(
        span=sig_period, adjust=False, min_periods=sig_period
    ).mean()
    vfi_hist_slope = (vfi_v - vfi_signal).diff()

    # ── OBV block (2) ────────────────────────────────────────────────────
    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    obv_slope = (obv - obv.shift(10)) / 10
    # OBV divergence: reuse Cat 13's fractal-based detect_divergence math
    # for rigor (not a simpler sign-comparison heuristic). Returns signed
    # series -1/0/+1 (regular only; hidden discarded per spec single-flag).
    obv_regular_div, _ = detect_divergence(close, obv, lookback_bars=div_lookback)
    obv_divergence_flag = obv_regular_div

    # ── Buy/sell estimator (1) ───────────────────────────────────────────
    # Spec: "(close-low)/(high-low), signed by body direction"
    close_position_in_range = (close - low) / rng  # [0, 1]
    body_sign = np.sign(close - open_)
    candle_buy_sell_signed = close_position_in_range * body_sign

    # ── Volume-weighted momentum (1) ─────────────────────────────────────
    # Σ(return × volume) / Σ(volume) over last 10 bars
    returns = close.pct_change()
    rv = returns * volume
    volume_weighted_momentum_10 = safe_div(
        rv.rolling(10, min_periods=10).sum(),
        volume.rolling(10, min_periods=10).sum(),
    )

    # ── High-volume-candle tercile location (2 binary) ───────────────────
    high_vol = volume > high_vol_mult * sma20
    in_top_tercile = close_position_in_range >= (2.0 / 3.0)
    in_bottom_tercile = close_position_in_range <= (1.0 / 3.0)
    high_vol_close_top_tercile = (high_vol & in_top_tercile).astype(int)
    high_vol_close_bottom_tercile = (high_vol & in_bottom_tercile).astype(int)

    return pd.DataFrame(
        {
            # Volume statistics (3)
            "volume_ratio_20": volume_ratio_20,
            "volume_zscore_20": volume_zscore_20,
            "volume_spike_flag": volume_spike_flag,
            # VFI block (3)
            "vfi": vfi_v,
            "vfi_signal": vfi_signal,
            "vfi_hist_slope": vfi_hist_slope,
            # OBV block (2)
            "obv_slope": obv_slope,
            "obv_divergence_flag": obv_divergence_flag,
            # Buy/sell estimator (1)
            "candle_buy_sell_signed": candle_buy_sell_signed,
            # Volume-weighted momentum (1)
            "volume_weighted_momentum_10": volume_weighted_momentum_10,
            # High-volume-candle location (2)
            "high_vol_close_top_tercile": high_vol_close_top_tercile,
            "high_vol_close_bottom_tercile": high_vol_close_bottom_tercile,
        }
    )


# ─── Cat 14 — Money Flow (6 features) ────────────────────────────────────
def money_flow_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute Cat 14 = 6 money-flow features.

    Parameters
    ----------
    df : DataFrame with high, low, close, volume.
    cfg : feature config dict. Uses:
          - cfg.get('money_flow', {}).get('mfi_ob_level', 80)
          - cfg.get('money_flow', {}).get('mfi_os_level', 20)
          - cfg.get('money_flow', {}).get('divergence_lookback_bars', 14)

    Returns
    -------
    DataFrame of 6 columns indexed like df.
    """
    high, low, close, volume = (
        df["high"], df["low"], df["close"], df["volume"]
    )
    rng = (high - low).replace(0, np.nan)

    mf_cfg = cfg.get("money_flow", {})
    ob_level = mf_cfg.get("mfi_ob_level", 80)
    os_level = mf_cfg.get("mfi_os_level", 20)
    div_lookback = mf_cfg.get("divergence_lookback_bars", 14)

    # ── CMF block (2) ────────────────────────────────────────────────────
    # CMF money flow multiplier: -1..+1 based on close position
    mf_mult = ((close - low) - (high - close)) / rng
    mf_vol = mf_mult * volume
    cmf_20 = (
        mf_vol.rolling(20, min_periods=20).sum()
        / volume.rolling(20, min_periods=20).sum()
    )
    cmf_slope = cmf_20.diff()

    # ── MFI block (2) ────────────────────────────────────────────────────
    typical = (high + low + close) / 3.0
    raw_mf = typical * volume
    pos_mf = raw_mf.where(typical > typical.shift(1), 0.0)
    neg_mf = raw_mf.where(typical < typical.shift(1), 0.0)
    pos_sum = pos_mf.rolling(14, min_periods=14).sum()
    neg_sum = neg_mf.rolling(14, min_periods=14).sum()
    mfr = safe_div(pos_sum, neg_sum)
    mfi_14 = 100 - 100 / (1 + mfr)

    mfi_zone = pd.Series(0, index=close.index, dtype=float)
    mfi_zone[mfi_14 > ob_level] = 1
    mfi_zone[mfi_14 < os_level] = -1

    # ── A/D block (2) ────────────────────────────────────────────────────
    # A/D = cumulative sum of mf_vol (uses CMF money-flow multiplier × volume)
    ad_line = mf_vol.cumsum()
    ad_slope_10 = (ad_line - ad_line.shift(10)) / 10

    # A/D vs price divergence: reuse Cat 13's fractal-based detect_divergence
    # Returns signed -1/0/+1 (regular only).
    ad_regular_div, _ = detect_divergence(close, ad_line, lookback_bars=div_lookback)
    ad_price_divergence_flag = ad_regular_div

    return pd.DataFrame(
        {
            "cmf_20": cmf_20,
            "cmf_slope": cmf_slope,
            "mfi_14": mfi_14,
            "mfi_zone": mfi_zone,
            "ad_slope_10": ad_slope_10,
            "ad_price_divergence_flag": ad_price_divergence_flag,
        }
    )
