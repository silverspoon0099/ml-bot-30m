"""Indicator math primitives — base layer for v2.0 feature pipeline.

Per Decision v2.37 Q4 + Decision v2.52 Q21 (refactor): MATH-ONLY base
layer. Pure calculation functions consumed by momentum_core.py (Cat 1) +
trend.py (Cat 2) + htf_context.py (Cat 2a) + volatility.py (Cat 3, where
applicable).

7 math primitives (Q21.1):
  - rsi(close, period)                                 -> Series
  - wavetrend(high, low, close, n1, n2, signal)        -> tuple(wt1, wt2)
  - stochastic(high, low, close, k_period, k_smooth, d_smooth)
                                                        -> tuple(k, d)
  - squeeze_momentum_value(high, low, close, length)   -> Series
  - macd(close, fast, slow, signal)                    -> tuple(line, signal, hist)
  - adx_di(high, low, close, period)                   -> tuple(di_plus, di_minus, adx)
  - ema(close, period)                                 -> Series  [NEW per Q21.5]

Calculations match TradingView Pine reference where possible:
- RSI: Wilder's smoothing
- WaveTrend (LazyBear): EMA(EMA-distance / 0.015 × EMA(abs))
- Stochastic: standard fast %K then SMA smoothing
- Squeeze Momentum (LazyBear): linreg of close − midprice basis
- MACD: EMA(fast) − EMA(slow), signal = EMA(line, signal_period)
- ADX/DI: Wilder DM/TR smoothing → DX → ADX
- EMA: pandas ewm(span=period, adjust=False, min_periods=period)

DROPPED in v2.52 refactor (11 v1.0 helpers — see Decision v2.52 Q21.8):
  - rsi_features, wavetrend_features (suffix=5min|1h), stoch_features,
    squeeze_features, macd_features_5m → moved to momentum_core.py
  - adx_features_5m, ema_features_5m → moved to trend.py
  - macd_features_1h, adx_features_1h, ema_features_1h, ema_features_1d
    → DELETED entirely (HTF flows through htf_context.py per §6.4
    in-pipeline aggregation)

Math primitives return tuples or single Series; only feature-selection
functions (in category files) return DataFrames. Clean separation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import safe_div, true_range, wilder_ema


# ─── RSI ────────────────────────────────────────────────────────────────
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing.

    Formula: RSI = 100 − 100/(1 + RS) where RS = avg_up / avg_down,
    avg_* are Wilder-smoothed (alpha = 1/period).
    """
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = wilder_ema(up, period)
    avg_dn = wilder_ema(down, period)
    rs = safe_div(avg_up, avg_dn)
    return 100 - 100 / (1 + rs)


# ─── EMA (NEW per Q21.5) ────────────────────────────────────────────────
def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average with span=period.

    Equivalent to pandas `close.ewm(span=period, adjust=False,
    min_periods=period).mean()`. Provided as a clean math primitive per
    v2.37 Q4 inventory; consumers (trend.py, htf_context.py) may use
    this helper or keep the inline ewm() pattern.
    """
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


# ─── WaveTrend (LazyBear) ───────────────────────────────────────────────
def wavetrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n1: int,
    n2: int,
    signal: int,
) -> tuple[pd.Series, pd.Series]:
    """LazyBear's WaveTrend oscillator. Returns (wt1, wt2).

    hlc3 = (high + low + close) / 3
    esa  = EMA(hlc3, n1)
    d    = EMA(|hlc3 − esa|, n1)
    ci   = (hlc3 − esa) / (0.015 × d)
    wt1  = EMA(ci, n2)
    wt2  = SMA(wt1, signal)
    """
    hlc3 = (high + low + close) / 3.0
    esa = hlc3.ewm(span=n1, adjust=False, min_periods=n1).mean()
    d = (hlc3 - esa).abs().ewm(span=n1, adjust=False, min_periods=n1).mean()
    ci = safe_div(hlc3 - esa, 0.015 * d)
    wt1 = ci.ewm(span=n2, adjust=False, min_periods=n2).mean()
    wt2 = wt1.rolling(signal, min_periods=signal).mean()
    return wt1, wt2


# ─── Stochastic ─────────────────────────────────────────────────────────
def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int,
    k_smooth: int,
    d_smooth: int,
) -> tuple[pd.Series, pd.Series]:
    """Standard stochastic oscillator. Returns (k, d).

    fast_k = (close − lowest_low(k_period)) / (highest_high(k_period) − lowest_low(k_period)) × 100
    k      = SMA(fast_k, k_smooth)
    d      = SMA(k, d_smooth)
    """
    ll = low.rolling(k_period, min_periods=k_period).min()
    hh = high.rolling(k_period, min_periods=k_period).max()
    fast_k = safe_div(close - ll, hh - ll) * 100
    k = fast_k.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_smooth, min_periods=d_smooth).mean()
    return k, d


# ─── Squeeze Momentum (LazyBear) ────────────────────────────────────────
def squeeze_momentum_value(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int
) -> pd.Series:
    """LazyBear's Squeeze Momentum value (oscillator only — separate from
    BB-vs-KC compression state in volatility.py).

    val = linreg(close − basis, length, offset=0)
    where basis = avg(avg(highest(length), lowest(length)), SMA(close, length)).

    Returns the oscillator value Series. To classify squeeze entry/release
    state (TTM Squeeze concept), use volatility.py which derives squeeze_state
    from BB-inside-KC compression — different concept.
    """
    hh = high.rolling(length, min_periods=length).max()
    ll = low.rolling(length, min_periods=length).min()
    midprice = (hh + ll) / 2
    sma_close = close.rolling(length, min_periods=length).mean()
    basis = (midprice + sma_close) / 2
    diff = close - basis
    arr = diff.to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    if n < length:
        return pd.Series(out, index=close.index)
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    for i in range(length - 1, n):
        window = arr[i - length + 1 : i + 1]
        if np.isnan(window).any():
            continue
        y_mean = window.mean()
        slope = ((x - x_mean) * (window - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        out[i] = intercept + slope * (length - 1)
    return pd.Series(out, index=close.index)


# ─── MACD ───────────────────────────────────────────────────────────────
def macd(
    close: pd.Series, fast: int, slow: int, signal: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Standard MACD. Returns (line, signal_line, hist).

    line   = EMA(close, fast) − EMA(close, slow)
    signal = EMA(line, signal)
    hist   = line − signal
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


# ─── ADX / DI ───────────────────────────────────────────────────────────
def adx_di(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's ADX/DI. Returns (di_plus, di_minus, adx).

    Up/down moves: up = high − prev_high, dn = prev_low − low
    +DM fires when up > dn AND up > 0; −DM fires when dn > up AND dn > 0
    ATR = Wilder-smoothed TR
    +DI = 100 × Wilder(+DM) / ATR
    −DI = 100 × Wilder(−DM) / ATR
    DX  = 100 × |+DI − −DI| / (+DI + −DI)
    ADX = Wilder-smoothed DX
    """
    up_move = high.diff()
    dn_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0),
        index=high.index,
    )
    tr = true_range(high, low, close)
    atr = wilder_ema(tr, period)
    di_plus = 100 * safe_div(wilder_ema(plus_dm, period), atr)
    di_minus = 100 * safe_div(wilder_ema(minus_dm, period), atr)
    dx = 100 * safe_div((di_plus - di_minus).abs(), di_plus + di_minus)
    adx_val = wilder_ema(dx, period)
    return di_plus, di_minus, adx_val
