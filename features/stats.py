"""Cat 9 — Mean Reversion / Statistics (7 features) + Cat 17 — Statistical / Fractal (6 features) — v2.0.

Per Project Spec 30min §7.2 Cat 9 + Cat 17 + Decision v2.50 Q19.

Two functions per Q19.11 (a) split per spec category (semantic naming):

  - mean_reversion_features(df, bb_position, cfg=None) -> DataFrame[7]
      Cat 9 — caller-supplied bb_position from Cat 3 volatility.py
              (renamed in output to bb_pct_b for spec wording match).

  - fractal_stats_features(df, cfg=None) -> DataFrame[6]
      Cat 17 — self-contained; derives log_returns internally.

Cat 9 features (7, all DYNAMIC per Q19.10):
  - bb_pct_b              — caller-supplied (rename of Cat 3 bb_position)
  - bb_dist_mid_sigma     — (close − sma20) / std20
                            ≡ zscore_20 by construction (redundancy
                            accepted per Q19.2 (a))
  - zscore_20             — same calc as bb_dist_mid_sigma
  - zscore_50             — (close − sma50) / std50
  - skewness_20           — log_returns.rolling(20).skew()
  - kurtosis_20           — log_returns.rolling(20).kurt()  (Fisher-adjusted)
  - autocorr_1            — log_returns.rolling(50).apply(autocorr, lag=1)

Cat 17 features (6, all DYNAMIC per Q19.10):
  - hurst_exponent        — simplified single-pass R/S, 100-bar window
                            on log returns
  - fractal_dimension     — box-counting unit-square, 50-bar window on
                            close LEVELS (geometric path complexity)
  - autocorr_5            — log_returns.rolling(50).apply(autocorr, lag=5)
  - autocorr_20           — log_returns.rolling(100).apply(autocorr, lag=20)
                            NEW vs v1.0 (Cat 17 expansion per spec)
  - entropy_20            — Shannon entropy on log returns, 10 bins,
                            20-bar window (renamed from v1.0 price_entropy;
                            window 50→20 per spec)
  - realized_vol_of_realized_vol
                          — log_returns.rolling(20).std().rolling(20).std()
                            NEW vs v1.0 (two-pass rolling std, W1=W2=20)

DROPPED from v1.0 (per Decision v2.50):
  - mean_reversion_score (per §15 — composite z-score average; redundant)
  - rsi_zscore (orphan — not in §7.2 Cat 9; derivable if needed)
  - return_5bar, return_20bar, return_60bar (orphans — overlap with Cat 1
    multi-period momentum which has roc_1/3/6/12_bar)
  - parkinson_vol (per §15 — moved to Cat 3 removal list)
  - variance_ratio (per §15 — multi-scale variance ratio not in spec keep
    list; redundant with Hurst as serial-correlation measure)
  - autocorrelation_1 → moved Cat 17 → Cat 9 with rename autocorr_1
  - autocorrelation_5 → autocorr_5 (rename only)
  - price_entropy → entropy_20 (rename + window 50→20)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import safe_div


# ─── Helper functions for Cat 17 ─────────────────────────────────────────
def hurst_exponent_window(arr: np.ndarray) -> float:
    """Simplified single-pass R/S Hurst estimator on 1D array.

    Formula: H = log(R/S) / log(N) where R = max(cumsum(deviations)) −
    min(cumsum(deviations)), S = std(arr), N = window length.

    Returns NaN when S=0 (constant series) or R=0 (degenerate) or input
    has any NaN.
    """
    if len(arr) < 20 or np.isnan(arr).any():
        return np.nan
    arr = arr - arr.mean()
    z = arr.cumsum()
    r = z.max() - z.min()
    s = arr.std(ddof=0)
    if s == 0 or r == 0:
        return np.nan
    return float(np.log(r / s) / np.log(len(arr)))


def shannon_entropy_window(arr: np.ndarray, bins: int = 10) -> float:
    """Shannon entropy of arr, binned into `bins` equal-width bins.

    Returns NaN if arr has any NaN or length < 10. Probabilities computed
    from histogram counts; zero-bins excluded from −sum(p × log(p)).
    """
    if len(arr) < 10 or np.isnan(arr).any():
        return np.nan
    hist, _ = np.histogram(arr, bins=bins)
    total = hist.sum()
    if total == 0:
        return np.nan
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def fractal_dim_window(arr: np.ndarray) -> float:
    """Box-counting fractal dimension of price path on unit square.

    Normalize (t, price) to [0,1]² unit square, cover with dyadic grids
    of size eps = 1/2^k for k = 1..floor(log2(N)), count occupied boxes
    including vertical spans between consecutive samples.
    D = −slope of log(N(eps)) vs log(eps).

    Note: operates on close LEVELS (not log returns) — fractal dim
    measures geometric complexity of price PATH.
    """
    if len(arr) < 10 or np.isnan(arr).any():
        return np.nan
    n = len(arr)
    rng = arr.max() - arr.min()
    if rng == 0:
        return np.nan
    x = np.arange(n, dtype=float) / (n - 1)
    y = (arr - arr.min()) / rng
    scales: list[float] = []
    counts: list[int] = []
    max_k = int(np.floor(np.log2(n)))
    for k in range(1, max_k + 1):
        eps = 1.0 / (2 ** k)
        bx = np.floor(x / eps).astype(int)
        by = np.floor(y / eps).astype(int)
        boxes: set[tuple[int, int]] = set()
        for i in range(n):
            boxes.add((bx[i], by[i]))
            if i > 0 and bx[i] != bx[i - 1]:
                lo, hi = (
                    (by[i - 1], by[i]) if by[i - 1] <= by[i] else (by[i], by[i - 1])
                )
                for v in range(lo, hi + 1):
                    boxes.add((bx[i], v))
        scales.append(eps)
        counts.append(len(boxes))
    if len(scales) < 3:
        return np.nan
    slope = np.polyfit(np.log(scales), np.log(counts), 1)[0]
    return float(-slope)


def _autocorr_lag(lag: int):
    """Return a closure suitable for rolling.apply that computes lag-N autocorrelation."""

    def fn(x):
        return pd.Series(x).autocorr(lag=lag)

    return fn


# ─── Cat 9 — Mean Reversion / Statistics (7 features) ───────────────────
def mean_reversion_features(
    df: pd.DataFrame,
    bb_position: pd.Series,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Compute Cat 9 = 7 mean-reversion / statistics features.

    Parameters
    ----------
    df : DataFrame with close column.
    bb_position : caller-supplied Series from Cat 3 volatility.py output;
                  (close − bb_lower) / (bb_upper − bb_lower). Renamed
                  in output to bb_pct_b for spec wording match.
    cfg : optional config dict. Currently no tunables; canonical 20/50
          windows + 50-bar autocorr_1 window baked in.

    Returns
    -------
    DataFrame of 7 columns indexed like df. All DYNAMIC per Q19.10.

    Notes
    -----
    `bb_dist_mid_sigma` and `zscore_20` are mathematically identical when
    BB period = 20 (canonical Cat 3). Both kept as separate columns per
    spec count of 7; redundancy accepted per Decision v2.50 Q19.2 (a).
    """
    _ = cfg  # reserved for future tunables

    close = df["close"]

    # 20-bar rolling stats (used for both bb_dist_mid_sigma + zscore_20)
    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=0)

    # 50-bar rolling stats (zscore_50)
    sma50 = close.rolling(50, min_periods=50).mean()
    std50 = close.rolling(50, min_periods=50).std(ddof=0)

    # Log returns for higher-moment + autocorr stats
    log_ret = np.log(close / close.shift(1))

    # bb_dist_mid_sigma ≡ zscore_20 by construction (BB period = 20).
    # Both emitted per spec count of 7 (Q19.2 (a) redundancy accepted).
    z20 = safe_div(close - sma20, std20)
    z50 = safe_div(close - sma50, std50)

    # autocorr_1 — log returns, lag=1, 50-bar rolling window (Q19.4 (a))
    autocorr_1 = log_ret.rolling(50, min_periods=50).apply(
        _autocorr_lag(1), raw=False
    )

    return pd.DataFrame(
        {
            "bb_pct_b": bb_position,
            "bb_dist_mid_sigma": z20,
            "zscore_20": z20,
            "zscore_50": z50,
            "skewness_20": log_ret.rolling(20, min_periods=20).skew(),
            "kurtosis_20": log_ret.rolling(20, min_periods=20).kurt(),
            "autocorr_1": autocorr_1,
        },
        index=df.index,
    )


# ─── Cat 17 — Statistical / Fractal (6 features) ────────────────────────
def fractal_stats_features(
    df: pd.DataFrame, cfg: dict | None = None
) -> pd.DataFrame:
    """Compute Cat 17 = 6 fractal / statistical features.

    Self-contained — derives log_returns from close internally.

    Parameters
    ----------
    df : DataFrame with close column.
    cfg : optional config dict. Currently canonical windows baked in
          (100/50/50/100/20/20 for hurst/fd/autocorr_5/autocorr_20/
          entropy/rvol_rvol).

    Returns
    -------
    DataFrame of 6 columns indexed like df. All DYNAMIC per Q19.10.
    """
    _ = cfg  # reserved for future tunable windows

    close = df["close"]
    log_ret = np.log(close / close.shift(1))

    # Hurst — simplified R/S on log returns, 100-bar window (Q19.5 (a))
    hurst = log_ret.rolling(100, min_periods=100).apply(
        hurst_exponent_window, raw=True
    )

    # Fractal dimension — box-counting on close LEVELS, 50-bar window
    # (Q19.6 (a)) — measures geometric complexity of price path.
    fract_dim = close.rolling(50, min_periods=50).apply(
        fractal_dim_window, raw=True
    )

    # autocorr_5 — log returns, lag=5, 50-bar window (Q19.8 (a))
    autocorr_5 = log_ret.rolling(50, min_periods=50).apply(
        _autocorr_lag(5), raw=False
    )

    # autocorr_20 — log returns, lag=20, 100-bar window scaled with lag
    # (Q19.8 (a)). NEW vs v1.0.
    autocorr_20 = log_ret.rolling(100, min_periods=100).apply(
        _autocorr_lag(20), raw=False
    )

    # entropy_20 — Shannon on log returns, 10 bins, 20-bar window
    # (Q19.7 (a)). Window 50→20 per spec.
    entropy_20 = log_ret.rolling(20, min_periods=20).apply(
        shannon_entropy_window, raw=True
    )

    # realized_vol_of_realized_vol — two-pass rolling std on log returns,
    # W1=W2=20 (Q19.9 (a)). NEW vs v1.0.
    inner_vol = log_ret.rolling(20, min_periods=20).std(ddof=0)
    rvol_rvol = inner_vol.rolling(20, min_periods=20).std(ddof=0)

    return pd.DataFrame(
        {
            "hurst_exponent": hurst,
            "fractal_dimension": fract_dim,
            "autocorr_5": autocorr_5,
            "autocorr_20": autocorr_20,
            "entropy_20": entropy_20,
            "realized_vol_of_realized_vol": rvol_rvol,
        },
        index=df.index,
    )
