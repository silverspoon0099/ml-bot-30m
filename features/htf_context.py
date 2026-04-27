"""Cat 2a — HTF Context (18 features) — v2.0.

Per Project Spec 30min §7.2 Cat 2a + Decision v2.3 (HTF as first-class
category, dominant SHAP per 2026 research) + Decision v2.37 Q4 architecture
(math from indicators.py; selection in this file).

18 features split:

4H block (9):
  - htf4h_bb_position           — BB(20, 2.0) position on 4H close
  - htf4h_rsi                   — RSI(14) on 4H close
  - htf4h_macd_hist             — MACD(12,26,9) histogram on 4H close
  - htf4h_adx                   — ADX(14) on 4H high/low/close
  - htf4h_ema21_pos             — (close - EMA21) / close × 100 on 4H
  - htf4h_atr_ratio             — ATR(5) / ATR(14) on 4H (volatility regime)
  - htf4h_close_vs_ema50_pct    — (close - EMA50) / close × 100 on 4H
  - htf4h_return_1bar           — 4H return over last 1 bar (4 hours)
  - htf4h_return_3bar           — 4H return over last 3 bars (12 hours)

1D block (9):
  - htf1d_ema20_pos             — (close - EMA20) / close × 100 on 1D
  - htf1d_ema50_pos             — (close - EMA50) / close × 100 on 1D
  - htf1d_ema200_pos            — (close - EMA200) / close × 100 on 1D
  - htf1d_rsi                   — RSI(14) on 1D close
  - htf1d_atr_pct               — ATR(14) / close × 100 on 1D
  - htf1d_return_1bar           — Daily return (1 bar)
  - htf1d_return_5bar           — 5-day return
  - htf1d_close_vs_20d_high_pct — (close - rolling_max(close, 20)) / close × 100
  - htf1d_close_vs_20d_low_pct  — (close - rolling_min(close, 20)) / close × 100

INPUTS: pre-aggregated 4H and 1D dataframes (caller produces these via
pandas resample per §6.4 Step A — NOT this module's responsibility).
Each input df must have columns ['high', 'low', 'close']; 'volume' optional.

OUTPUT: two separate DataFrames — one 4H-indexed, one 1D-indexed (different
time grids, can't be a single DataFrame). Caller (typically builder.py)
is responsible for prev-closed-bar shift + merge onto the 30m frame
per §6.4 Step C — that's NOT done here. This file does NOT shift, merge,
or look-ahead-protect; it just emits per-TF features at their native index.

ARCHITECTURE NOTE: function signature returns `tuple[DataFrame, DataFrame]`,
not `DataFrame` (as the stub originally suggested). 4H and 1D have different
indices and cannot share a single DataFrame without synthetic alignment.
The wrapper `htf_context_features()` returns the tuple; callers can also
invoke `htf_4h_features()` and `htf_1d_features()` independently if they
already have only one HTF frame.
"""
from __future__ import annotations

import pandas as pd

from ._common import pct, safe_div, true_range, wilder_ema
from .indicators import adx_di, macd, rsi


# Defaults match v2.0 spec; cfg overrides individual keys.
_DEFAULTS = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "adx_period": 14,
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_period": 14,
    "atr_short_period": 5,
    "rolling_window_1d": 20,  # for 20d high/low features
}


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """ATR via Wilder smoothing of true range."""
    return wilder_ema(true_range(high, low, close), period)


def _bb_position(close: pd.Series, period: int, std: float) -> pd.Series:
    """BB position: (close - bb_lo) / (bb_up - bb_lo). 0..1 in band, ±beyond."""
    basis = close.rolling(period, min_periods=period).mean()
    dev = close.rolling(period, min_periods=period).std(ddof=0)
    upper = basis + std * dev
    lower = basis - std * dev
    return safe_div(close - lower, upper - lower)


def _ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def htf_4h_features(df_4h: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Compute 9 4H context features. Indexed by df_4h.index.

    Caller merges to lower TF (30m) via prev-closed-bar shift in builder.py
    per §6.4 Step C — this function does NOT shift.
    """
    c = {**_DEFAULTS, **(cfg or {})}
    high, low, close = df_4h["high"], df_4h["low"], df_4h["close"]

    bb_position = _bb_position(close, c["bb_period"], c["bb_std"])
    rsi_v = rsi(close, c["rsi_period"])
    _line, _sig, macd_hist = macd(close, c["macd_fast"], c["macd_slow"], c["macd_signal"])
    _dp, _dm, adx = adx_di(high, low, close, c["adx_period"])

    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)

    atr_long = _atr(high, low, close, c["atr_period"])
    atr_short = _atr(high, low, close, c["atr_short_period"])
    atr_ratio = safe_div(atr_short, atr_long)

    return_1bar = (close / close.shift(1) - 1.0) * 100.0
    return_3bar = (close / close.shift(3) - 1.0) * 100.0

    return pd.DataFrame(
        {
            "htf4h_bb_position": bb_position,
            "htf4h_rsi": rsi_v,
            "htf4h_macd_hist": macd_hist,
            "htf4h_adx": adx,
            "htf4h_ema21_pos": pct(close - ema21, close),
            "htf4h_atr_ratio": atr_ratio,
            "htf4h_close_vs_ema50_pct": pct(close - ema50, close),
            "htf4h_return_1bar": return_1bar,
            "htf4h_return_3bar": return_3bar,
        },
        index=df_4h.index,
    )


def htf_1d_features(df_1d: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Compute 9 1D context features. Indexed by df_1d.index.

    Caller merges to lower TF (30m) via prev-closed-bar shift in builder.py
    per §6.4 Step C — this function does NOT shift.
    """
    c = {**_DEFAULTS, **(cfg or {})}
    high, low, close = df_1d["high"], df_1d["low"], df_1d["close"]

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)

    rsi_v = rsi(close, c["rsi_period"])

    atr_v = _atr(high, low, close, c["atr_period"])
    atr_pct_series = pct(atr_v, close)

    return_1bar = (close / close.shift(1) - 1.0) * 100.0
    return_5bar = (close / close.shift(5) - 1.0) * 100.0

    win = c["rolling_window_1d"]
    high_w = close.rolling(win, min_periods=win).max()
    low_w = close.rolling(win, min_periods=win).min()
    close_vs_20d_high = pct(close - high_w, close)
    close_vs_20d_low = pct(close - low_w, close)

    return pd.DataFrame(
        {
            "htf1d_ema20_pos": pct(close - ema20, close),
            "htf1d_ema50_pos": pct(close - ema50, close),
            "htf1d_ema200_pos": pct(close - ema200, close),
            "htf1d_rsi": rsi_v,
            "htf1d_atr_pct": atr_pct_series,
            "htf1d_return_1bar": return_1bar,
            "htf1d_return_5bar": return_5bar,
            "htf1d_close_vs_20d_high_pct": close_vs_20d_high,
            "htf1d_close_vs_20d_low_pct": close_vs_20d_low,
        },
        index=df_1d.index,
    )


def htf_context_features(
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the full Cat 2a HTF context block (18 features = 9 4H + 9 1D).

    Returns
    -------
    (df_4h_features, df_1d_features) — each DataFrame indexed by its input df.

    The two DataFrames have different time grids and CANNOT be combined into a
    single DataFrame without synthetic alignment. Caller (builder.py) uses
    prev-closed-bar shift on each before merging onto the 30m frame per
    §6.4 Step C.
    """
    return htf_4h_features(df_4h, cfg), htf_1d_features(df_1d, cfg)
