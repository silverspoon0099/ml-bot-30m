"""Cat 15 — Additional Momentum (7 features) — v2.0.

Per Project Spec 30min §7.2 Cat 15 trim 9→7 + Decision v2.42 (Q11 keep both
TSI cross states).

Cat 15 = 7 features:
  - williams_r              — Williams %R(14) raw value
  - williams_r_zone         — +1 OB (WR > -20) / -1 OS (WR < -80) / 0 mid
  - cci_20                  — CCI(20) raw value
  - cci_zero_cross_state    — sign(cci_20); +1 above zero, -1 below
  - tsi_signal_cross_state  — sign(tsi - tsi_signal); momentum acceleration trigger
                              (TSI vs its EMA(7) signal line, like MACD pattern)
  - tsi_zero_cross_state    — sign(tsi); directional regime (long vs short bias)
  - cmo_zero_cross_state    — sign(cmo_14)

Per Decision v2.42 Q11: BOTH TSI cross states kept because they carry
orthogonal information — signal-line cross = momentum acceleration trigger,
zero cross = directional regime. SHAP trim in Phase 2.6 retains whichever
is more predictive.

DROPPED from v1.0 (cat 15 was 9 features):
  - williams_r_direction, cci_direction, cci_extreme — redundant slope/extreme
  - cmo_direction, cmo_14 raw value — spec keeps only zero-cross state
  - raw tsi value — spec keeps only cross states
  - roc_10 — overlaps Cat 1 multi-period momentum (roc_1/3/6/12bar in
             features/momentum_core.py)

Math functions kept at module level (williams_r, cci, cmo, tsi) for
reusability if other modules need them later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import safe_div


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    hh = high.rolling(period, min_periods=period).max()
    ll = low.rolling(period, min_periods=period).min()
    return -100 * safe_div(hh - close, hh - ll)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return safe_div(tp - sma, 0.015 * mad)


def cmo(close: pd.Series, period: int) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).rolling(period, min_periods=period).sum()
    dn = (-diff.clip(upper=0)).rolling(period, min_periods=period).sum()
    return safe_div(up - dn, up + dn) * 100


def tsi(close: pd.Series, long: int, short: int) -> pd.Series:
    """Trade Strength Index (Blau).

    TSI = 100 × EMA_short(EMA_long(price_diff)) / EMA_short(EMA_long(|price_diff|))
    """
    diff = close.diff()
    ema1 = diff.ewm(span=long, adjust=False, min_periods=long).mean()
    ema2 = ema1.ewm(span=short, adjust=False, min_periods=short).mean()
    abs_ema1 = diff.abs().ewm(span=long, adjust=False, min_periods=long).mean()
    abs_ema2 = abs_ema1.ewm(span=short, adjust=False, min_periods=short).mean()
    return 100 * safe_div(ema2, abs_ema2)


def extra_momentum_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute Cat 15 = 7 additional-momentum features.

    Parameters
    ----------
    df : DataFrame with high, low, close columns.
    cfg : feature config dict. Uses:
          - cfg['williams_r']['period']                       (default 14)
          - cfg['williams_r'].get('ob_level', -20)
          - cfg['williams_r'].get('os_level', -80)
          - cfg['cci']['period']                              (default 20)
          - cfg['cmo']['period']                              (default 14)
          - cfg['tsi']['long'], cfg['tsi']['short']           (TSI periods)
          - cfg['tsi'].get('signal_period', 7)                (signal-line EMA)

    Returns
    -------
    DataFrame of 7 columns indexed like df.
    """
    h, l, c = df["high"], df["low"], df["close"]
    wr_cfg = cfg["williams_r"]
    cci_cfg = cfg["cci"]
    cmo_cfg = cfg["cmo"]
    tsi_cfg = cfg["tsi"]

    # Williams %R (2)
    wr = williams_r(h, l, c, wr_cfg["period"])
    wr_ob = wr_cfg.get("ob_level", -20)
    wr_os = wr_cfg.get("os_level", -80)
    wr_zone = pd.Series(0, index=c.index, dtype=float)
    wr_zone[wr > wr_ob] = 1
    wr_zone[wr < wr_os] = -1

    # CCI (2)
    cci_v = cci(h, l, c, cci_cfg["period"])
    cci_zero_cross_state = np.sign(cci_v)

    # CMO (1)
    cmo_v = cmo(c, cmo_cfg["period"])
    cmo_zero_cross_state = np.sign(cmo_v)

    # TSI (2 — both cross states kept per Decision v2.42 Q11)
    tsi_v = tsi(c, tsi_cfg["long"], tsi_cfg["short"])
    signal_period = tsi_cfg.get("signal_period", 7)
    tsi_signal = tsi_v.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    tsi_signal_cross_state = np.sign(tsi_v - tsi_signal)
    tsi_zero_cross_state = np.sign(tsi_v)

    return pd.DataFrame(
        {
            "williams_r": wr,
            "williams_r_zone": wr_zone,
            "cci_20": cci_v,
            "cci_zero_cross_state": cci_zero_cross_state,
            "tsi_signal_cross_state": tsi_signal_cross_state,
            "tsi_zero_cross_state": tsi_zero_cross_state,
            "cmo_zero_cross_state": cmo_zero_cross_state,
        },
        index=df.index,
    )
