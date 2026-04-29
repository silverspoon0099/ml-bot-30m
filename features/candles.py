"""Cat 8 — Price Action / Candle (9 features) — v2.0.

Per Project Spec 30min §7.2 Cat 8 + Decision v2.49 Q18 (hybrid spec
amendment — drop binary doji + hammer flags as redundant with continuous
body+wicks; preserve v1.0 is_bullish + body_vs_prev_body for distinct
continuous signal; keep spec's range_pct + inside_bar_flag).

Research-grounded rationale (Decision v2.49):
  Tree-based models (LightGBM) trivially learn threshold splits from
  continuous body_pct + wick features at SHAP-optimal thresholds.
  Encoding `(body_pct ≤ 0.10)` as a hardcoded binary `doji_flag` adds
  NO signal and may displace the model's learned splits. Same for
  hammer/shooting-star — derivable from body_pct + lower_wick + upper_wick
  combination.

  The candle-pattern ALPHA lives in CONTEXT (overbought/oversold zones,
  momentum fading, structural breaks), not in the pattern flag itself.
  Cat 8 contributes the candle-shape signal that COMBINES with Cat 1
  (RSI/Stoch/WT zones) + Cat 16 (structure breaks) + Cat 6 (S/R levels)
  via tree splits.

  Multi-bar conditional patterns (engulfing, pin_bar, inside_bar) ARE
  kept as explicit features because they are NOT derivable from
  single-bar continuous features without explicit prev-bar comparison.

9 features locked per Decision v2.49 Q18.10 — all DYNAMIC per Q18.8:

  Continuous candle-shape (4):
    - body_pct          — |close − open| / range
    - upper_wick_pct    — (high − max(open, close)) / range
    - lower_wick_pct    — (min(open, close) − low) / range
    - range_pct         — (high − low) / close × 100  [NEW]

  Quality / direction (2):
    - is_bullish        — int(close > open)               [v1.0 KEPT]
    - body_vs_prev_body — body_n / body_(n-1)             [v1.0 KEPT]

  Multi-bar patterns (3):
    - engulfing_signal  — signed +1/−1/0 (rename of v1.0 engulfing)
    - pin_bar_signal    — signed +1/−1/0 (rename of v1.0 pin_bar)
    - inside_bar_flag   — binary 0/1                       [NEW]

DROPPED from v1.0 (per Decision v2.49):
  - consecutive_bull, consecutive_bear (weak signal at 30m; counter
    pattern belongs to Cat 20 event_memory if needed later — currently
    Cat 20 locked at 22 features without these)

DROPPED from original §7.2 Cat 8 narrative (per Decision v2.49 amendment):
  - doji_flag — derivable from body_pct continuous via tree split
  - hammer_or_shooting_star_flag — derivable from body_pct + lower_wick
    + upper_wick combination via tree splits
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def candle_features(
    df: pd.DataFrame, cfg: dict | None = None
) -> pd.DataFrame:
    """Compute Cat 8 = 9 candle-shape + multi-bar pattern features.

    Self-contained — math derives from OHLC alone; no caller-supplied
    dependencies.

    Parameters
    ----------
    df : DataFrame with open, high, low, close columns.
    cfg : optional config dict. Currently no tunables — defaults baked
          in. Reserved for future tunable thresholds (e.g., pin_bar
          wick-to-body ratio, inside_bar tolerance).

    Returns
    -------
    DataFrame of 9 columns indexed like df. All DYNAMIC per Q18.8 (each
    feature evaluates on current bar's OHLC; close/high/low mutate
    intrabar).
    """
    _ = cfg  # reserved for future tunables

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    # ── Continuous candle-shape metrics (4) ───────────────────────────
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    body_pct = body / rng
    upper_wick = (h - np.maximum(o, c)) / rng
    lower_wick = (np.minimum(o, c) - l) / rng
    range_pct = (h - l) / c.replace(0, np.nan) * 100.0

    # ── Quality / direction (2 — v1.0 kept per Decision v2.49) ────────
    is_bullish = (c > o).astype(int)
    body_vs_prev_body = body / body.shift(1).replace(0, np.nan)

    # ── Multi-bar pattern signals (3 — NOT derivable from single-bar) ─
    # Engulfing: current body fully covers prior body (opposite color).
    # v1.0 formula verbatim, renamed to engulfing_signal for spec match.
    prev_o = o.shift(1)
    prev_c = c.shift(1)
    bull_engulf = (
        (c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c)
    ).astype(int)
    bear_engulf = (
        (c < o) & (prev_c > prev_o) & (c <= prev_o) & (o >= prev_c)
    ).astype(int)
    engulfing_signal = bull_engulf - bear_engulf

    # Pin bar: long wick on one side > 2× body, opposite wick < body.
    # v1.0 formula verbatim, renamed to pin_bar_signal for spec match.
    bull_pin = (
        (lower_wick > 2 * body_pct) & (upper_wick < body_pct)
    ).astype(int)
    bear_pin = (
        (upper_wick > 2 * body_pct) & (lower_wick < body_pct)
    ).astype(int)
    pin_bar_signal = bull_pin - bear_pin

    # Inside bar: current bar's range fully contained within prior bar's
    # range (consolidation/compression pattern). True 2-bar conditional
    # NOT derivable from single-bar continuous features.
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    inside_bar_flag = ((h < prev_h) & (l > prev_l)).astype(int)

    return pd.DataFrame(
        {
            "body_pct": body_pct,
            "upper_wick_pct": upper_wick,
            "lower_wick_pct": lower_wick,
            "range_pct": range_pct,
            "is_bullish": is_bullish,
            "body_vs_prev_body": body_vs_prev_body,
            "engulfing_signal": engulfing_signal,
            "pin_bar_signal": pin_bar_signal,
            "inside_bar_flag": inside_bar_flag,
        },
        index=df.index,
    )
