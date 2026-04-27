"""Cat 2 — Trend / Direction (14 features) — v2.0.

Per Project Spec 30min §7.2 Cat 2 + Decision v2.37 Q5 + Decision v2.39.

Cat 2 = ADX/DI block (9) + EMA block (5) = 14 features:

ADX/DI block (9):
  - di_plus, di_minus, adx, di_spread, adx_slope         (5 continuous)
  - adx_trending      (binary, ADX > 25)
  - adx_weak          (binary, ADX < 20)
  - adx_accelerating  (binary, ADX rising — slope > +threshold)
  - adx_decelerating  (binary, ADX falling — slope < -threshold; per Decision v2.37 Q5)

EMA block (5):
  - ema9_dist_pct, ema21_dist_pct, ema50_dist_pct        (3 continuous)
  - ema_stack_30m     (signed score; +1/-1 per condition; max ±3)
  - ema21_dist_atr    (price vs EMA21 in ATR units)

Per Decision v2.39 (sub of v2.37 Q4): selection lives in this NEW file;
math (`adx_di`) imported from `indicators.py`. Symmetric with `momentum_core.py`
for Cat 1.

ATR(14) is supplied by caller (computed once in `volatility.py`); avoids
duplicate computation across modules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import pct, safe_div
from .indicators import adx_di


def trend_features(
    df: pd.DataFrame,
    atr_14: pd.Series,
    cfg: dict,
) -> pd.DataFrame:
    """Compute Cat 2 = 14 trend/direction features.

    Parameters
    ----------
    df : DataFrame with high, low, close columns.
    atr_14 : ATR(14) series — supplied by caller (typically the output of
             `volatility.volatility_features` to avoid duplicate computation).
    cfg : feature config dict; uses ``cfg['adx']['period']``,
          ``cfg['adx'].get('accel_threshold', 0.5)``, ``cfg['ema_periods']``.

    Returns
    -------
    DataFrame of 14 columns indexed like ``df``.
    """
    high, low, close = df["high"], df["low"], df["close"]
    adx_cfg = cfg["adx"]
    ema_periods = cfg["ema_periods"]
    accel_thresh = adx_cfg.get("accel_threshold", 0.5)

    # ── ADX/DI block (9 features) ────────────────────────────────────────
    di_plus, di_minus, adx = adx_di(high, low, close, adx_cfg["period"])
    di_spread = di_plus - di_minus
    adx_slope = adx.diff()

    adx_trending = (adx > 25).astype(int)
    adx_weak = (adx < 20).astype(int)
    adx_accelerating = (adx_slope > accel_thresh).astype(int)
    adx_decelerating = (adx_slope < -accel_thresh).astype(int)

    # ── EMA block (5 features) ───────────────────────────────────────────
    # Compute the EMAs we need (9, 21, 50) — `ema_periods` may include extras
    # but Cat 2 stack scoring requires {9, 21, 50}.
    emas = {
        p: close.ewm(span=p, adjust=False, min_periods=p).mean()
        for p in ema_periods
    }
    ema9 = emas.get(9)
    ema21 = emas.get(21)
    ema50 = emas.get(50)

    out: dict[str, pd.Series] = {
        # ADX/DI (9)
        "di_plus": di_plus,
        "di_minus": di_minus,
        "adx": adx,
        "di_spread": di_spread,
        "adx_slope": adx_slope,
        "adx_trending": adx_trending,
        "adx_weak": adx_weak,
        "adx_accelerating": adx_accelerating,
        "adx_decelerating": adx_decelerating,
    }

    # EMA dist (3)
    for p in ema_periods:
        if p in (9, 21, 50):
            out[f"ema{p}_dist_pct"] = pct(close - emas[p], close)

    # EMA stack (1) — signed score, max ±3.
    # +1 each for: EMA9 > EMA21, EMA21 > EMA50, close > EMA9 (and inverses → -1).
    if ema9 is not None and ema21 is not None and ema50 is not None:
        score = (
            np.sign(ema9 - ema21).fillna(0)
            + np.sign(ema21 - ema50).fillna(0)
            + np.sign(close - ema9).fillna(0)
        )
        out["ema_stack_30m"] = score
    else:
        out["ema_stack_30m"] = pd.Series(np.nan, index=close.index)

    # Price vs EMA21 in ATR units (1) — volatility-normalized distance.
    if ema21 is not None:
        out["ema21_dist_atr"] = safe_div(close - ema21, atr_14)
    else:
        out["ema21_dist_atr"] = pd.Series(np.nan, index=close.index)

    return pd.DataFrame(out, index=df.index)
