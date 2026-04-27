"""Cat 1 — Momentum core (32 features) — v2.0.

Per Project Spec 30min §7.2 Cat 1 + Decision v2.37 Q4 + Decision v2.40 (Q6 + Q8).

Cat 1 = 32 features across 8 blocks:

RSI block (4):
  - rsi_14, rsi_slope, rsi_zone (-1 OS / 0 mid / +1 OB), rsi_dist_from_50

MACD block (6):
  - macd_line, macd_signal, macd_hist
  - macd_hist_slope (= d(macd_hist)/dt)
  - macd_zero_cross_state (sign of macd_line: +1 above zero / -1 below)
  - macd_hist_acceleration (= d²(macd_hist)/dt²) — distinct from `d2_macd_line`
    in vel-of-vel group, see Decision v2.40 Q6

WaveTrend block (4):
  - wt1, wt2, wt_cross_signal (+1 wt1 just crossed above wt2 / -1 below / 0 none)
  - wt_ob_os_zone (-1 OS / 0 mid / +1 OB)

Stochastic block (4):
  - stoch_k, stoch_d, stoch_cross_signal (+1/-1/0)
  - stoch_ob_os_zone (-1 OS / 0 mid / +1 OB)

Squeeze block (4):
  - squeeze_value (LazyBear momentum)
  - squeeze_signal (5-bar slope of squeeze_value)
  - squeeze_release_state (-1 in squeeze / 0 normal / +1 just released)
  - bars_in_squeeze (running count while in squeeze)
  NOTE: squeeze_release_state and bars_in_squeeze conceptually overlap with
  `volatility.py:volatility_features` outputs `squeeze_state` and
  `bars_since_squeeze_release`. To avoid drift, momentum_core consumes those
  columns as caller-supplied inputs (pattern: volatility owns the squeeze
  state machine; momentum_core wraps + relabels).

Multi-period momentum block (4):
  - roc_1bar, roc_3bar, roc_6bar, roc_12bar (price rate-of-change over N bars,
    in %; on 30m: 1bar=30min, 3bar=90min, 6bar=3hr, 12bar=6hr)

Cross-feature block (2 — formulas locked per Decision v2.40 Q8):
  - rsi_wt_divergence_flag = int(sign(rsi-50) != sign(wt1))
    Binary 0/1: 1 means oscillators disagree on regime
  - macd_rsi_alignment = sign(macd_hist) * sign(rsi-50)
    Signed -1/0/+1: +1 aligned, -1 misaligned

Velocity-of-velocity block (4 — features locked per Decision v2.40 Q6 option a):
  - d2_rsi        = rsi.diff().diff()
  - d2_macd_line  = macd_line.diff().diff()  (NOT macd_hist; that is in MACD block)
  - d2_wt1        = wt1.diff().diff()
  - d2_stoch_k    = stoch_k.diff().diff()

Per Decision v2.37 Q4 architecture: this file is the Cat 1 *selection* layer.
All math comes from `indicators.py` (rsi, macd, wavetrend, stochastic,
squeeze_momentum_value); momentum_core picks the 32 features per §7.2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import bars_since, crosses_above, crosses_below
from .indicators import (
    macd,
    rsi,
    squeeze_momentum_value,
    stochastic,
    wavetrend,
)


def _zone(value: pd.Series, ob_level: float, os_level: float) -> pd.Series:
    """Discrete OB/OS zone: +1 if above ob_level, -1 if below os_level, 0 else."""
    out = pd.Series(0, index=value.index, dtype=float)
    out[value > ob_level] = 1
    out[value < os_level] = -1
    return out


def _cross_signal(line: pd.Series, signal: pd.Series) -> pd.Series:
    """+1 line just crossed above signal, -1 below, 0 no cross."""
    up = crosses_above(line, signal).astype(int)
    dn = crosses_below(line, signal).astype(int)
    return up - dn


def momentum_core_features(
    df: pd.DataFrame,
    cfg: dict,
    squeeze_state: pd.Series | None = None,
    bars_since_squeeze_release: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute Cat 1 = 32 momentum-core features.

    Parameters
    ----------
    df : DataFrame with high, low, close columns.
    cfg : feature config dict; uses cfg['rsi'], cfg['macd'], cfg['wavetrend'],
          cfg['stochastic'], cfg['squeeze'], cfg.get('rsi_ob_level', 70),
          cfg.get('rsi_os_level', 30).
    squeeze_state : optional. Caller-supplied squeeze state series from
                    `volatility.volatility_features` output (`squeeze_state`
                    column). If None, momentum_core re-derives it via a thin
                    BB-vs-KC check using existing math (slower; documents the
                    coupling for future refactor).
    bars_since_squeeze_release : optional. Caller-supplied bars-since series
                    from volatility output. If None, derived inline from
                    squeeze_state.

    Returns
    -------
    DataFrame of 32 columns indexed like df.
    """
    high, low, close = df["high"], df["low"], df["close"]
    rsi_cfg = cfg["rsi"]
    macd_cfg = cfg["macd"]
    wt_cfg = cfg["wavetrend"]
    stoch_cfg = cfg["stochastic"]
    squeeze_cfg = cfg["squeeze"]

    rsi_ob = cfg.get("rsi_ob_level", 70)
    rsi_os = cfg.get("rsi_os_level", 30)
    wt_ob = wt_cfg["ob_level"]
    wt_os = wt_cfg["os_level"]
    stoch_ob = stoch_cfg.get("ob_level", 80)
    stoch_os = stoch_cfg.get("os_level", 20)

    # ── Compute base oscillators (math from indicators.py) ───────────────
    rsi_v = rsi(close, rsi_cfg["period"])
    macd_line, macd_signal_line, macd_hist = macd(
        close, macd_cfg["fast"], macd_cfg["slow"], macd_cfg["signal"]
    )
    wt1, wt2 = wavetrend(high, low, close, wt_cfg["n1"], wt_cfg["n2"], wt_cfg["signal"])
    stoch_k, stoch_d = stochastic(
        high, low, close,
        stoch_cfg["k_period"], stoch_cfg["k_smooth"], stoch_cfg["d_smooth"],
    )
    squeeze_value = squeeze_momentum_value(high, low, close, squeeze_cfg["kc_length"])

    # Squeeze release_state + bars_in_squeeze: prefer caller-supplied to avoid
    # drift with volatility.py's authoritative squeeze state machine. If not
    # supplied, fall back to inline derivation (kept simple — full state
    # machine lives in volatility.py per architecture).
    if squeeze_state is None or bars_since_squeeze_release is None:
        # Inline fallback: BB vs KC squeeze detection (mirrors volatility.py).
        sq_bb_basis = close.rolling(squeeze_cfg["bb_length"], min_periods=squeeze_cfg["bb_length"]).mean()
        sq_bb_dev = close.rolling(squeeze_cfg["bb_length"], min_periods=squeeze_cfg["bb_length"]).std(ddof=0)
        sq_bb_up = sq_bb_basis + squeeze_cfg["bb_mult"] * sq_bb_dev
        sq_bb_lo = sq_bb_basis - squeeze_cfg["bb_mult"] * sq_bb_dev
        kc_basis = close.rolling(squeeze_cfg["kc_length"], min_periods=squeeze_cfg["kc_length"]).mean()
        from ._common import true_range
        kc_rng = true_range(high, low, close).rolling(
            squeeze_cfg["kc_length"], min_periods=squeeze_cfg["kc_length"]
        ).mean()
        kc_up = kc_basis + squeeze_cfg["kc_mult"] * kc_rng
        kc_lo = kc_basis - squeeze_cfg["kc_mult"] * kc_rng
        in_squeeze = ((sq_bb_up < kc_up) & (sq_bb_lo > kc_lo)).astype(int)
        just_released = ((in_squeeze.shift(1) == 1) & (in_squeeze == 0)).astype(int)
        squeeze_state = pd.Series(0, index=close.index, dtype=float)
        squeeze_state[in_squeeze == 1] = -1
        squeeze_state[just_released == 1] = 1

    # bars_in_squeeze: running count while squeeze_state == -1, resets when out
    in_sq = (squeeze_state == -1).astype(bool)
    grp = (~in_sq).cumsum()
    bars_in_squeeze = in_sq.astype(int).groupby(grp).cumsum()

    # ── Assemble 32 features ─────────────────────────────────────────────
    # RSI block (4)
    rsi_slope = rsi_v.diff()
    rsi_zone = _zone(rsi_v, rsi_ob, rsi_os)
    rsi_dist_from_50 = rsi_v - 50.0

    # MACD block (6)
    macd_hist_slope = macd_hist.diff()
    macd_hist_acceleration = macd_hist_slope.diff()
    macd_zero_cross_state = np.sign(macd_line)

    # WaveTrend block (4)
    wt_cross_signal = _cross_signal(wt1, wt2)
    wt_ob_os_zone = _zone(wt1, wt_ob, wt_os)

    # Stochastic block (4)
    stoch_cross_signal = _cross_signal(stoch_k, stoch_d)
    stoch_ob_os_zone = _zone(stoch_k, stoch_ob, stoch_os)

    # Squeeze block (4)
    squeeze_signal = (squeeze_value - squeeze_value.shift(5)) / 5.0

    # Multi-period momentum block (4)
    roc_1bar = (close / close.shift(1) - 1.0) * 100.0
    roc_3bar = (close / close.shift(3) - 1.0) * 100.0
    roc_6bar = (close / close.shift(6) - 1.0) * 100.0
    roc_12bar = (close / close.shift(12) - 1.0) * 100.0

    # Cross-feature block (2 — locked formulas per Decision v2.40 Q8)
    rsi_sign = np.sign(rsi_v - 50.0)
    wt1_sign = np.sign(wt1)
    macd_hist_sign = np.sign(macd_hist)
    rsi_wt_divergence_flag = (rsi_sign != wt1_sign).astype(int)
    macd_rsi_alignment = macd_hist_sign * rsi_sign

    # Velocity-of-velocity block (4 — locked features per Decision v2.40 Q6 option a)
    d2_rsi = rsi_slope.diff()
    d2_macd_line = macd_line.diff().diff()
    d2_wt1 = wt1.diff().diff()
    d2_stoch_k = stoch_k.diff().diff()

    return pd.DataFrame(
        {
            # RSI block (4)
            "rsi_14": rsi_v,
            "rsi_slope": rsi_slope,
            "rsi_zone": rsi_zone,
            "rsi_dist_from_50": rsi_dist_from_50,
            # MACD block (6)
            "macd_line": macd_line,
            "macd_signal": macd_signal_line,
            "macd_hist": macd_hist,
            "macd_hist_slope": macd_hist_slope,
            "macd_zero_cross_state": macd_zero_cross_state,
            "macd_hist_acceleration": macd_hist_acceleration,
            # WaveTrend block (4)
            "wt1": wt1,
            "wt2": wt2,
            "wt_cross_signal": wt_cross_signal,
            "wt_ob_os_zone": wt_ob_os_zone,
            # Stochastic block (4)
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "stoch_cross_signal": stoch_cross_signal,
            "stoch_ob_os_zone": stoch_ob_os_zone,
            # Squeeze block (4)
            "squeeze_value": squeeze_value,
            "squeeze_signal": squeeze_signal,
            "squeeze_release_state": squeeze_state,
            "bars_in_squeeze": bars_in_squeeze,
            # Multi-period momentum block (4)
            "roc_1bar": roc_1bar,
            "roc_3bar": roc_3bar,
            "roc_6bar": roc_6bar,
            "roc_12bar": roc_12bar,
            # Cross-feature block (2)
            "rsi_wt_divergence_flag": rsi_wt_divergence_flag,
            "macd_rsi_alignment": macd_rsi_alignment,
            # Velocity-of-velocity block (4)
            "d2_rsi": d2_rsi,
            "d2_macd_line": d2_macd_line,
            "d2_wt1": d2_wt1,
            "d2_stoch_k": d2_stoch_k,
        },
        index=df.index,
    )
