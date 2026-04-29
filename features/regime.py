"""Cat 10 — Market Regime (7 features) — v2.0.

Per Project Spec 30min §7.2 Cat 10 + Decision v2.37 Q3 (rewrite to spec).

7 features (locked per spec):

  - trending_regime    — (adx > 25) & (|di_plus − di_minus| > 10); binary 0/1
  - ranging_regime     — (adx < 20) & (bb_width_percentile < 30); binary 0/1
  - volatility_regime  — tercile 0/1/2 from atr_percentile (low/normal/high)
  - volume_regime      — tercile 0/1/2 from rolling 100-bar volume rank
  - trend_direction    — +1 if EMA9>EMA21>EMA50 (bull stack), −1 if reverse, 0 mixed
  - regime_change_bar  — bars since last regime label flip
                         (label = 1 trending / 4 ranging / 0 neither)
  - vol_adjusted_momentum_regime
                       — roc_3bar / atr_pct (signed continuous, vol-normalized
                         momentum strength)

DROPPED from v1.0 per Decision v2.37 Q3:
  - regime_volatile, regime_quiet (binary) — replaced by volatility_regime tercile
  - efficiency_ratio (UT Bot concept) — not in v2.0 spec keep list
  - choppiness_index — not in v2.0 spec keep list
  - bars_in_current_regime — RENAMED to regime_change_bar

§7.5 TAGGING: Cat 10 = dynamic (regime classifications depend on current
ADX/ATR/volume/EMA values which mutate intrabar; tercile percentiles are
rolling-window evaluations on the latest bar's value).

CALLER-SUPPLIED INPUTS (orchestrated by builder.py):
  df          — DataFrame with close, volume
  adx, di_plus, di_minus  — from indicators.adx_di output (or trend.py)
  atr_14      — ATR(14) from volatility.py output
  atr_percentile — rolling 100-bar percentile of atr_14 (from volatility.py)
  bb_width_percentile — rolling 100-bar percentile of BB width
                        (NOT in v2.0 volatility.py output per Cat 3 trim,
                        but easily derived inline if not supplied: see code)

If `bb_width_percentile` is not supplied, computed inline from `bb_width_pct`
which is in volatility.py output (Cat 3 feature).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import rolling_percentile, safe_div


def _tercile(percentile_series: pd.Series, low: float = 33.33, high: float = 66.67) -> pd.Series:
    """Bin a percentile series (0..100) into tercile labels 0/1/2.

    0 = low (< low_threshold), 2 = high (> high_threshold), 1 = normal.
    NaN where input is NaN.
    """
    out = pd.Series(1.0, index=percentile_series.index)
    out[percentile_series < low] = 0
    out[percentile_series > high] = 2
    out[percentile_series.isna()] = np.nan
    return out


def regime_features(
    df: pd.DataFrame,
    adx: pd.Series,
    di_plus: pd.Series,
    di_minus: pd.Series,
    atr_14: pd.Series,
    atr_percentile: pd.Series,
    bb_width_percentile: pd.Series,
    cfg: dict,
) -> pd.DataFrame:
    """Compute Cat 10 = 7 market regime features.

    Parameters
    ----------
    df : DataFrame with close, volume.
    adx, di_plus, di_minus : ADX/DI series from indicators.adx_di or trend.py.
    atr_14 : ATR(14) Series from volatility.py output.
    atr_percentile : rolling 100-bar percentile of atr_14 (0..100 scale)
                     from volatility.py output.
    bb_width_percentile : rolling 100-bar percentile of BB width (0..100).
                          v2.0 volatility.py dropped this column per Cat 3 trim;
                          caller can compute inline as
                          rolling_percentile(bb_width_pct, 100).
    cfg : feature config dict (currently no tunables; reserved for future).

    Returns
    -------
    DataFrame of 7 columns indexed like df.
    """
    close = df["close"]
    volume = df["volume"]

    # ── trending_regime + ranging_regime (2) ──────────────────────────────
    trending_regime = (
        (adx > 25) & ((di_plus - di_minus).abs() > 10)
    ).astype(int)
    ranging_regime = ((adx < 20) & (bb_width_percentile < 30)).astype(int)

    # ── volatility_regime (1, tercile from atr_percentile) ─────────────────
    volatility_regime = _tercile(atr_percentile, 33.33, 66.67)

    # ── volume_regime (1, tercile from rolling 100-bar volume rank) ────────
    volume_percentile = volume.rolling(100, min_periods=100).rank(pct=True) * 100
    volume_regime = _tercile(volume_percentile, 33.33, 66.67)

    # ── trend_direction (1, EMA stack) ────────────────────────────────────
    ema9 = close.ewm(span=9, adjust=False, min_periods=9).mean()
    ema21 = close.ewm(span=21, adjust=False, min_periods=21).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    bull_stack = (ema9 > ema21) & (ema21 > ema50)
    bear_stack = (ema9 < ema21) & (ema21 < ema50)
    trend_direction = pd.Series(0.0, index=close.index)
    trend_direction[bull_stack] = 1
    trend_direction[bear_stack] = -1

    # ── regime_change_bar (1) ─────────────────────────────────────────────
    # Label: 1=trending, 4=ranging, 0=neither. Then count bars since flip.
    regime_label = pd.Series(0, index=close.index, dtype=int)
    regime_label[trending_regime == 1] = 1
    regime_label[ranging_regime == 1] = 4
    # cumcount within consecutive equal-label runs
    regime_change_bar = (
        regime_label.groupby((regime_label != regime_label.shift()).cumsum()).cumcount() + 1
    )

    # ── vol_adjusted_momentum_regime (1) ──────────────────────────────────
    # 3-bar % return normalized by ATR%
    roc_3bar = (close / close.shift(3) - 1.0) * 100.0
    atr_pct = (atr_14 / close) * 100.0
    vol_adjusted_momentum_regime = safe_div(roc_3bar, atr_pct)

    return pd.DataFrame(
        {
            "trending_regime": trending_regime,
            "ranging_regime": ranging_regime,
            "volatility_regime": volatility_regime,
            "volume_regime": volume_regime,
            "trend_direction": trend_direction,
            "regime_change_bar": regime_change_bar,
            "vol_adjusted_momentum_regime": vol_adjusted_momentum_regime,
        },
        index=df.index,
    )
