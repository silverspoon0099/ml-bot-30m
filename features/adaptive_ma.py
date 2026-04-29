"""Cat 18 — Adaptive MAs (4 features) — v2.0.

Per Project Spec 30min §7.2 Cat 18 + Decision v2.48 Q17 (trim 5→4 with
psar_direction + psar_dist_pct combined into single signed feature).

4 features locked per Decision v2.48 Q17.7 — all close-relative, all
DYNAMIC per Q17.6:

  - kama_dist_pct        — (close − kama) / close × 100; KAMA(10, 2, 30)
  - dema_dist_pct        — (close − dema) / close × 100; DEMA(21)
  - tema_dist_pct        — (close − tema) / close × 100; TEMA(21)
  - psar_state_dist_pct  — (close − sar)  / close × 100; SIGNED feature
                            combining direction + magnitude (Q17.1 (a)).
                            Sign carries trend state by Wilder construction:
                            positive ⟺ long trend (SAR below close),
                            negative ⟺ short trend (SAR above close).

DROPPED from v1.0:
  - psar_direction (+1/−1 trend state) — redundant with sign of combined
    psar_state_dist_pct. Same drop pattern as Cat 19 tk_cross.

CHANGED from v1.0:
  - psar_dist_pct → psar_state_dist_pct (rename per §15 directive; matches
    spec wording "PSAR state, PSAR distance % (1 aggregated field)").

Function signature change vs v1.0:
  - was `adaptive_ma_features(df, cfg)` with REQUIRED nested cfg keys;
  - now `adaptive_ma_features(df, cfg=None) -> DataFrame[4]` with all
    keys optional + canonical defaults. Keeps v1.0 NESTED structure
    (cfg["kama"]["period"], cfg["psar"]["af_start"], etc.) for v1.0
    config.yaml backward-compat.

§7.5 TAGGING per Q17.6: all 4 features = DYNAMIC. KAMA/DEMA/TEMA values
include current bar's close in their rolling/cascading EMA computations;
PSAR state/distance is evaluated at the current bar; all four
`*_dist_pct` features are close-relative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import pct, safe_div


def kama(close: pd.Series, period: int, fast_ema: int, slow_ema: int) -> pd.Series:
    """Kaufman Adaptive Moving Average — efficiency-ratio-modulated EMA.

    Smoothing constant interpolates between fast (≈ EMA(fast_ema)) and
    slow (≈ EMA(slow_ema)) based on price efficiency. Period determines
    the lookback over which efficiency ratio is computed.
    """
    change = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period, min_periods=period).sum()
    er = safe_div(change, volatility)
    fast_sc = 2.0 / (fast_ema + 1)
    slow_sc = 2.0 / (slow_ema + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = np.full(len(close), np.nan)
    arr = close.to_numpy()
    sc_arr = sc.to_numpy()
    for i in range(period, len(close)):
        if np.isnan(out[i - 1]):
            out[i] = arr[i]
        else:
            out[i] = out[i - 1] + sc_arr[i] * (arr[i] - out[i - 1])
    return pd.Series(out, index=close.index)


def dema(close: pd.Series, period: int) -> pd.Series:
    """Double Exponential MA — `2×EMA(close, p) − EMA(EMA(close, p), p)`."""
    e1 = close.ewm(span=period, adjust=False, min_periods=period).mean()
    e2 = e1.ewm(span=period, adjust=False, min_periods=period).mean()
    return 2 * e1 - e2


def tema(close: pd.Series, period: int) -> pd.Series:
    """Triple Exponential MA — `3×EMA1 − 3×EMA2 + EMA3` (cascading EMAs)."""
    e1 = close.ewm(span=period, adjust=False, min_periods=period).mean()
    e2 = e1.ewm(span=period, adjust=False, min_periods=period).mean()
    e3 = e2.ewm(span=period, adjust=False, min_periods=period).mean()
    return 3 * e1 - 3 * e2 + e3


def parabolic_sar(
    high: pd.Series, low: pd.Series, af_start: float, af_step: float, af_max: float
) -> tuple[pd.Series, pd.Series]:
    """Wilder Parabolic SAR — returns (sar, trend) where trend is +1 long, −1 short.

    Long trend: SAR is BELOW close (close − sar > 0).
    Short trend: SAR is ABOVE close (close − sar < 0).
    Flip bars: SAR repositions to prior extreme `ep` (on opposite side
    of close), maintaining sign-trend alignment.

    The `trend` Series is kept as an internal helper for backward
    compatibility / debugging; the public Cat 18 feature uses only the
    signed dist via `psar_state_dist_pct`.
    """
    n = len(high)
    sar = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)
    h = high.to_numpy()
    l = low.to_numpy()
    if n < 2:
        return pd.Series(sar, index=high.index), pd.Series(trend, index=high.index)
    trend[1] = 1 if h[1] >= h[0] else -1
    sar[1] = l[0] if trend[1] == 1 else h[0]
    ep = h[1] if trend[1] == 1 else l[1]
    af = af_start

    for i in range(2, n):
        prev_sar = sar[i - 1]
        if trend[i - 1] == 1:
            new_sar = prev_sar + af * (ep - prev_sar)
            # Wilder clamp: SAR cannot exceed the lows of the prior two bars.
            new_sar = min(new_sar, l[i - 1], l[i - 2])
            if l[i] < new_sar:
                trend[i] = -1
                sar[i] = ep
                ep = l[i]
                af = af_start
            else:
                trend[i] = 1
                sar[i] = new_sar
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_step, af_max)
        else:
            new_sar = prev_sar + af * (ep - prev_sar)
            # Wilder clamp: SAR cannot fall below the highs of the prior two bars.
            new_sar = max(new_sar, h[i - 1], h[i - 2])
            if h[i] > new_sar:
                trend[i] = 1
                sar[i] = ep
                ep = h[i]
                af = af_start
            else:
                trend[i] = -1
                sar[i] = new_sar
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_step, af_max)

    return pd.Series(sar, index=high.index), pd.Series(trend, index=high.index)


def adaptive_ma_features(
    df: pd.DataFrame, cfg: dict | None = None
) -> pd.DataFrame:
    """Compute Cat 18 = 4 adaptive-MA features.

    Self-contained — math derives from OHLC alone; no caller-supplied
    dependencies.

    Parameters
    ----------
    df : DataFrame with high, low, close columns.
    cfg : optional config dict with v1.0-style nested structure (Q17.5):
            cfg["kama"] = {"period": 10, "fast_ema": 2, "slow_ema": 30}
            cfg["dema"] = {"period": 21}
            cfg["tema"] = {"period": 21}
            cfg["psar"] = {"af_start": 0.02, "af_step": 0.02, "af_max": 0.20}
          All keys optional; canonical Wilder/Kaufman defaults applied
          when absent. Backward-compat with v1.0 config.yaml.

    Returns
    -------
    DataFrame of 4 columns indexed like df. All DYNAMIC per Q17.6.
    """
    cfg = cfg or {}
    kama_cfg = cfg.get("kama", {}) if isinstance(cfg.get("kama"), dict) else {}
    dema_cfg = cfg.get("dema", {}) if isinstance(cfg.get("dema"), dict) else {}
    tema_cfg = cfg.get("tema", {}) if isinstance(cfg.get("tema"), dict) else {}
    psar_cfg = cfg.get("psar", {}) if isinstance(cfg.get("psar"), dict) else {}

    kama_period = int(kama_cfg.get("period", 10))
    kama_fast = int(kama_cfg.get("fast_ema", 2))
    kama_slow = int(kama_cfg.get("slow_ema", 30))
    dema_period = int(dema_cfg.get("period", 21))
    tema_period = int(tema_cfg.get("period", 21))
    psar_af_start = float(psar_cfg.get("af_start", 0.02))
    psar_af_step = float(psar_cfg.get("af_step", 0.02))
    psar_af_max = float(psar_cfg.get("af_max", 0.20))

    close = df["close"]
    high = df["high"]
    low = df["low"]

    kama_v = kama(close, kama_period, kama_fast, kama_slow)
    dema_v = dema(close, dema_period)
    tema_v = tema(close, tema_period)
    sar, _trend = parabolic_sar(high, low, psar_af_start, psar_af_step, psar_af_max)

    return pd.DataFrame(
        {
            "kama_dist_pct": pct(close - kama_v, close),
            "dema_dist_pct": pct(close - dema_v, close),
            "tema_dist_pct": pct(close - tema_v, close),
            # Signed close-vs-SAR distance: sign carries trend state by
            # Wilder construction (long ⟺ SAR below close ⟺ positive;
            # short ⟺ SAR above close ⟺ negative). Replaces v1.0 separate
            # psar_direction + psar_dist_pct features (Q17.1 (a)).
            "psar_state_dist_pct": pct(close - sar, close),
        },
        index=df.index,
    )
