"""Cat 16 — Market Structure (10 features) — v2.0.

Per Project Spec 30min §7.2 Cat 16 + Decision v2.46 Q15 (rewrite to spec).

10 features locked per Decision v2.46 Q15.6 — first MIXED-tagged category
in §7.5 with explicit per-feature static/dynamic split:

  STATIC (8) — only update when a new fractal pivot confirms (deterministic
  at bar = pivot_bar + lookback; intrabar-safe for cached lookup):
    - higher_highs_count_20    — sum of HH events in last 20 BARS
    - higher_lows_count_20     — sum of HL events in last 20 BARS
    - lower_highs_count_20     — sum of LH events in last 20 BARS
    - lower_lows_count_20      — sum of LL events in last 20 BARS
    - structure_type           — +1 (HH+HL bull) / −1 (LL+LH bear) / 0 mixed
    - swing_length_ratio       — |swing_n| / |swing_(n-1)| over H/L chain
    - fractal_pivot_count_20   — total pivot density in last 20 BARS
    - break_of_structure       — signed binary: +1 −1→+1, −1 +1→−1, else 0

  DYNAMIC (2) — depend on current close, mutate intrabar:
    - swing_high_dist_pct      — (close − last_pivot_high) / close × 100
    - swing_low_dist_pct       — (close − last_pivot_low) / close × 100

HH/HL/LH/LL event definitions (Q15.3 — bar-anchored, NOT pivot-anchored):
  - HH event at bar i: p_high[i].notna() & (p_high[i] > previous_non_nan(p_high))
  - HL event at bar i: p_low[i].notna()  & (p_low[i]  > previous_non_nan(p_low))
  - LH event at bar i: p_high[i].notna() & (p_high[i] < previous_non_nan(p_high))
  - LL event at bar i: p_low[i].notna()  & (p_low[i]  < previous_non_nan(p_low))

Pivot detection: divergence.fractal_pivots(series, lookback=5) — 2-left-2-
right rule, confirms `lookback // 2 = 2` bars after the actual swing
(causal shift built in — no look-ahead).

DROPPED from v1.0 per Decision v2.46:
  - swing_range_pct (low SHAP; derivable from swing_high_dist - swing_low_dist)
  - retrace_depth (overlaps Cat 6.5 `fib_retracement_pct`)
  - range_position (overlaps Cat 6 `pivot_position_daily_01` + Cat 3 `bb_position`)
  - bars_since_structure_break (replaced by signed binary `break_of_structure`;
    counter pattern belongs in Cat 20 event_memory which is locked at 22
    features without it)

CHANGED from v1.0:
  - Count window: "last 20 PIVOTS" (v1.0 _pivot_running_count — pivot-density
    anchored, ~200-300 bars at 30m with lookback=5) → "last 20 BARS"
    (v2.0 spec literal). Symmetric with pivot_times_tested_today (Cat 6)
    and bars_since_last_hh (Cat 20) which are bar-anchored.
  - swing_ratio → swing_length_ratio (rename per Q15.2 to disambiguate from
    spec narrative wording "(HH count vs LL count)" which is degenerate
    given separate count features in the same set).

NEW vs v1.0:
  - higher_lows_count_20 (v1.0 had only HH/LL counts)
  - lower_highs_count_20
  - fractal_pivot_count_20 (`(p_high.notna() | p_low.notna()).rolling(20).sum()`
    — boolean-OR collapses simultaneous high+low pivot to count-of-1)
  - break_of_structure (signed binary; replaces v1.0 bars_since_structure_break)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import pct
from .divergence import fractal_pivots


def _swing_length_ratio(p_high: pd.Series, p_low: pd.Series) -> pd.Series:
    """|swing_n| / |swing_(n-1)| over alternating H/L pivot chain.

    Walks confirmed pivots in chronological order; consecutive same-kind
    pivots collapse to the more extreme one (a higher H replaces a prior H,
    a lower L replaces a prior L). Each bar after the chain reaches 3
    elements carries the ratio of the two most recent completed swings,
    forward-filled between confirmations.

    Returns NaN before the third chain element confirms.
    """
    n = len(p_high)
    ph = p_high.to_numpy()
    pl = p_low.to_numpy()
    out = np.full(n, np.nan)
    chain_prices: list[float] = []
    chain_kinds: list[str] = []
    for i in range(n):
        if not np.isnan(ph[i]):
            if chain_kinds and chain_kinds[-1] == "H":
                if ph[i] > chain_prices[-1]:
                    chain_prices[-1] = ph[i]
            else:
                chain_prices.append(float(ph[i]))
                chain_kinds.append("H")
        if not np.isnan(pl[i]):
            if chain_kinds and chain_kinds[-1] == "L":
                if pl[i] < chain_prices[-1]:
                    chain_prices[-1] = pl[i]
            else:
                chain_prices.append(float(pl[i]))
                chain_kinds.append("L")
        if len(chain_prices) >= 3:
            cur = abs(chain_prices[-1] - chain_prices[-2])
            prev = abs(chain_prices[-2] - chain_prices[-3])
            if prev > 0:
                out[i] = cur / prev
    return pd.Series(out, index=p_high.index).ffill()


def structure_features(
    df: pd.DataFrame, cfg: dict | None = None
) -> pd.DataFrame:
    """Compute Cat 16 = 10 market structure features.

    Self-contained — derives fractal pivots internally from df['high'] and
    df['low'] via divergence.fractal_pivots. No caller-supplied dependencies.

    Parameters
    ----------
    df : DataFrame with high, low, close columns.
    cfg : optional config dict. Tunable keys:
            - structure.fractal_lookback (default 5; 2-left-2-right rule)
            - structure.count_window (default 20; rolling-bar window)

    Returns
    -------
    DataFrame of 10 columns indexed like df. Per §7.5 Q15.5:
      8 STATIC (intrabar-safe; only update on new pivot confirm)
      2 DYNAMIC (swing_*_dist_pct depend on close, mutate intrabar).
    """
    cfg = cfg or {}
    lookback = int(cfg.get("structure.fractal_lookback", 5))
    window = int(cfg.get("structure.count_window", 20))

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # ── Fractal pivots (causal shift built into fractal_pivots) ───────
    p_high, _ = fractal_pivots(high, lookback)
    _, p_low = fractal_pivots(low, lookback)

    # Most recent prior confirmed pivot value (excludes current bar).
    # ffill carries the last non-NaN forward; shift(1) takes the value as
    # of the previous bar. At a confirmation bar i, this is the value of
    # the pivot confirmed strictly BEFORE bar i.
    prev_high = p_high.ffill().shift(1)
    prev_low = p_low.ffill().shift(1)

    # ── HH/HL/LH/LL events — boolean at confirmation bars ─────────────
    hh_event = p_high.notna() & (p_high > prev_high)
    hl_event = p_low.notna() & (p_low > prev_low)
    lh_event = p_high.notna() & (p_high < prev_high)
    ll_event = p_low.notna() & (p_low < prev_low)

    # ── Counts in last `window` BARS (Q15.3 (a) — bar-anchored) ───────
    higher_highs_count_20 = (
        hh_event.astype(int).rolling(window, min_periods=window).sum()
    )
    higher_lows_count_20 = (
        hl_event.astype(int).rolling(window, min_periods=window).sum()
    )
    lower_highs_count_20 = (
        lh_event.astype(int).rolling(window, min_periods=window).sum()
    )
    lower_lows_count_20 = (
        ll_event.astype(int).rolling(window, min_periods=window).sum()
    )
    fractal_pivot_count_20 = (
        (p_high.notna() | p_low.notna())
        .astype(int)
        .rolling(window, min_periods=window)
        .sum()
    )

    # ── structure_type — +1 (HH+HL bull) / −1 (LL+LH bear) / 0 mixed ──
    # Track at each bar: was the most recent CONFIRMED H-pivot HH or LH?
    # Was the most recent L-pivot HL or LL? Forward-fill the boolean
    # state; NaN-guard with fillna(0). Need ≥2 H AND ≥2 L pivots to
    # have any HH/LH or HL/LL classification at all (1st pivot has no
    # prior to compare to).
    h_is_hh = pd.Series(np.nan, index=df.index, dtype=float)
    h_is_hh[hh_event] = 1.0
    h_is_hh[lh_event] = 0.0
    last_h_was_hh = h_is_hh.ffill().fillna(0).astype(bool)

    l_is_hl = pd.Series(np.nan, index=df.index, dtype=float)
    l_is_hl[hl_event] = 1.0
    l_is_hl[ll_event] = 0.0
    last_l_was_hl = l_is_hl.ffill().fillna(0).astype(bool)

    h_pivot_count = p_high.notna().cumsum()
    l_pivot_count = p_low.notna().cumsum()
    ready = (h_pivot_count >= 2) & (l_pivot_count >= 2)

    structure_type = pd.Series(0, index=df.index, dtype=int)
    structure_type[ready & last_h_was_hh & last_l_was_hl] = 1
    structure_type[ready & (~last_h_was_hh) & (~last_l_was_hl)] = -1

    # ── break_of_structure — signed binary (Q15.1 (a) strict) ─────────
    # +1 only on bar where structure_type flips from −1 → +1
    # −1 only on bar where structure_type flips from +1 → −1
    # Transitions via 0 (mixed/ranging) do not fire — only direct
    # directional reversals (CHoCH-style). Replaces v1.0
    # bars_since_structure_break.
    prev_structure_type = structure_type.shift(1).fillna(0).astype(int)
    break_of_structure = pd.Series(0, index=df.index, dtype=int)
    break_of_structure[(structure_type == 1) & (prev_structure_type == -1)] = 1
    break_of_structure[(structure_type == -1) & (prev_structure_type == 1)] = -1

    # ── swing_length_ratio (Q15.2 — last-two-swing-lengths ratio) ─────
    swing_length_ratio = _swing_length_ratio(p_high, p_low)

    # ── DYNAMIC: distances of close from last confirmed pivots ────────
    last_high = p_high.ffill()
    last_low = p_low.ffill()
    swing_high_dist_pct = pct(close - last_high, close)
    swing_low_dist_pct = pct(close - last_low, close)

    return pd.DataFrame(
        {
            "swing_high_dist_pct": swing_high_dist_pct,
            "swing_low_dist_pct": swing_low_dist_pct,
            "structure_type": structure_type,
            "swing_length_ratio": swing_length_ratio,
            "higher_highs_count_20": higher_highs_count_20,
            "higher_lows_count_20": higher_lows_count_20,
            "lower_highs_count_20": lower_highs_count_20,
            "lower_lows_count_20": lower_lows_count_20,
            "fractal_pivot_count_20": fractal_pivot_count_20,
            "break_of_structure": break_of_structure,
        },
        index=df.index,
    )
