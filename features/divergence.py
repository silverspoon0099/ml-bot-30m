"""Cat 13 — Divergence Detection (7 features) — v2.0.

Per Project Spec 30min §7.2 Cat 13 + Decision v2.43 (Q12 strict spec).

Cat 13 = 7 features (count unchanged from v1.0; content RESHAPED per
Decision v2.43):

KEPT (binary 0/1, except recency which is int):
  - regular_bullish_div_rsi   — price LL + RSI HL (bullish regular div)
  - regular_bearish_div_rsi   — price HH + RSI LH (bearish regular div)
  - regular_bullish_div_macd  — same on MACD histogram
  - regular_bearish_div_macd  — same on MACD histogram
  - hidden_bullish_div_rsi    — price HL + RSI LL
  - hidden_bearish_div_rsi    — price LH + RSI HH
  - divergence_recency        — int: bars_since(any of above)

v1.0 → v2.0 TRANSFORMATION (per Decision v2.43, recorded for rollback):
  - v1.0 `rsi_price_divergence` (signed -1/0/+1) split into
    `regular_bullish_div_rsi` + `regular_bearish_div_rsi` (binary 0/1)
  - v1.0 `macd_price_divergence` (signed) split into
    `regular_bullish_div_macd` + `regular_bearish_div_macd` (binary)
  - v1.0 `hidden_divergence` (signed, RSI-only) split into
    `hidden_bullish_div_rsi` + `hidden_bearish_div_rsi` (binary)
  - v1.0 `divergence_freshness` renamed → `divergence_recency`

DROPPED from v1.0 (per Decision v2.43 Q12 strict spec):
  - `wt_price_divergence` — WT correlated with RSI by construction; redundant
  - `stoch_price_divergence` — Stoch is a slower-period RSI variant; redundant
  - `divergence_count` — derivable as `sum(6 flags)` if downstream needs it

If Phase 2.5 SHAP analysis shows the dropped features would have been
useful, see PROJECT_LOG Decision v2.43 entry for rollback procedure.

Detection method (unchanged): fractal pivots (default 5-bar window =
2 left + center + 2 right, confirmed via shift); compare current confirmed
pivot to most recent prior confirmed pivot of same type within
`lookback_bars` window.

Math functions `fractal_pivots` and `detect_divergence` are kept at
module level — they're imported by `structure.py` (Cat 16) too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import bars_since


def fractal_pivots(series: pd.Series, lookback: int = 5) -> tuple[pd.Series, pd.Series]:
    """Return (pivot_high, pivot_low) — value at the pivot bar, NaN elsewhere.

    A bar is a pivot high if it's the max of `lookback` bars centered on it.
    Pivots confirm `half = lookback // 2` bars after the center bar (no
    look-ahead by definition).
    """
    half = lookback // 2
    rolling = series.rolling(lookback, center=True, min_periods=lookback)
    is_high = series == rolling.max()
    is_low = series == rolling.min()
    pivot_high = series.where(is_high)
    pivot_low = series.where(is_low)
    # Confirm only after `half` bars passed (causal)
    pivot_high = pivot_high.shift(half)
    pivot_low = pivot_low.shift(half)
    return pivot_high, pivot_low


def detect_divergence(
    price: pd.Series, oscillator: pd.Series, lookback_bars: int = 14
) -> tuple[pd.Series, pd.Series]:
    """Return (regular_div, hidden_div) — both signed: +1 bullish, -1 bearish, 0 none.

    Sign convention: +1 = bullish/long-favoring, -1 = bearish/short-favoring.
    Caller (divergence_features) splits the signed series into separate
    bullish and bearish binary flags per Decision v2.43.
    """
    p_high, p_low = fractal_pivots(price)
    o_high, o_low = fractal_pivots(oscillator)

    regular = pd.Series(0.0, index=price.index)
    hidden = pd.Series(0.0, index=price.index)

    last_p_high_idx = -1
    last_p_high_val = np.nan
    last_o_high_val = np.nan
    last_p_low_idx = -1
    last_p_low_val = np.nan
    last_o_low_val = np.nan

    p_h_arr = p_high.to_numpy()
    p_l_arr = p_low.to_numpy()
    o_h_arr = o_high.to_numpy()
    o_l_arr = o_low.to_numpy()

    for i in range(len(price)):
        # Update most-recent pivots with the values confirmed at bar i
        if not np.isnan(p_h_arr[i]):
            if last_p_high_idx >= 0 and (i - last_p_high_idx) <= lookback_bars:
                # Compare current high pivot with previous high pivot
                if p_h_arr[i] > last_p_high_val and o_h_arr[i] < last_o_high_val:
                    regular.iloc[i] = -1  # regular bearish: price HH + osc LH
                elif p_h_arr[i] < last_p_high_val and o_h_arr[i] > last_o_high_val:
                    hidden.iloc[i] = -1  # hidden bearish: price LH + osc HH
            last_p_high_idx = i
            last_p_high_val = p_h_arr[i]
            last_o_high_val = o_h_arr[i]

        if not np.isnan(p_l_arr[i]):
            if last_p_low_idx >= 0 and (i - last_p_low_idx) <= lookback_bars:
                if p_l_arr[i] < last_p_low_val and o_l_arr[i] > last_o_low_val:
                    regular.iloc[i] = 1  # regular bullish: price LL + osc HL
                elif p_l_arr[i] > last_p_low_val and o_l_arr[i] < last_o_low_val:
                    hidden.iloc[i] = 1  # hidden bullish: price HL + osc LL
            last_p_low_idx = i
            last_p_low_val = p_l_arr[i]
            last_o_low_val = o_l_arr[i]

    return regular, hidden


def divergence_features(
    df: pd.DataFrame,
    rsi_series: pd.Series,
    macd_hist: pd.Series,
    lookback_bars: int = 14,
) -> pd.DataFrame:
    """Compute Cat 13 = 7 divergence features.

    Parameters
    ----------
    df : DataFrame with 'close' column.
    rsi_series : RSI(14) series aligned to df.index — typically caller
                 supplies from indicators.rsi(close, 14) or from
                 momentum_core's emitted rsi_14 column.
    macd_hist : MACD histogram series aligned to df.index.
    lookback_bars : max bar gap between consecutive pivots to compare for
                    divergence (default 14 = 7 hours pivot-to-pivot at 30m).
                    Tunable in cfg if SHAP shows underfiring.

    Returns
    -------
    DataFrame of 7 columns indexed like df:
        regular_bullish_div_rsi, regular_bearish_div_rsi   (binary 0/1)
        regular_bullish_div_macd, regular_bearish_div_macd (binary 0/1)
        hidden_bullish_div_rsi, hidden_bearish_div_rsi     (binary 0/1)
        divergence_recency                                  (int, bars)

    SIGNATURE CHANGE vs v1.0: dropped wt1 and stoch_k parameters (WT and
    Stoch divergences dropped per Decision v2.43 Q12 strict spec). This
    will break v1.0 builder.py call:
        div_df = divergence.divergence_features(df_5m, rsi_series=...,
                    macd_hist=..., wt1=..., stoch_k=...)
    Known transitional state until builder.py rewrite in Phase 1.10c.
    """
    close = df["close"]

    # Detect divergences on RSI and MACD only (per Decision v2.43)
    rsi_regular_signed, rsi_hidden_signed = detect_divergence(close, rsi_series, lookback_bars)
    macd_regular_signed, _macd_hidden_unused = detect_divergence(close, macd_hist, lookback_bars)

    # Split signed series into separate binary flags (strict spec format)
    regular_bullish_div_rsi = (rsi_regular_signed > 0).astype(int)
    regular_bearish_div_rsi = (rsi_regular_signed < 0).astype(int)
    regular_bullish_div_macd = (macd_regular_signed > 0).astype(int)
    regular_bearish_div_macd = (macd_regular_signed < 0).astype(int)
    hidden_bullish_div_rsi = (rsi_hidden_signed > 0).astype(int)
    hidden_bearish_div_rsi = (rsi_hidden_signed < 0).astype(int)

    # Recency: bars since any of the 6 flags fired
    any_div = (
        regular_bullish_div_rsi
        | regular_bearish_div_rsi
        | regular_bullish_div_macd
        | regular_bearish_div_macd
        | hidden_bullish_div_rsi
        | hidden_bearish_div_rsi
    ).astype(bool)
    divergence_recency = bars_since(any_div)

    return pd.DataFrame(
        {
            "regular_bullish_div_rsi": regular_bullish_div_rsi,
            "regular_bearish_div_rsi": regular_bearish_div_rsi,
            "regular_bullish_div_macd": regular_bullish_div_macd,
            "regular_bearish_div_macd": regular_bearish_div_macd,
            "hidden_bullish_div_rsi": hidden_bullish_div_rsi,
            "hidden_bearish_div_rsi": hidden_bearish_div_rsi,
            "divergence_recency": divergence_recency,
        },
        index=df.index,
    )
