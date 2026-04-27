"""Feature stability taxonomy — static / dynamic / mixed tagging — v2.0.

Per Project Spec 30min §7.5.

Tags every feature column produced by builder.py as one of:

  - `static`   — stable within a 30m bar; safe for intrabar inference on
                 5m/1m WITHOUT recomputation. Examples: Cat 6 daily/weekly
                 pivots (fixed for the day/week), Cat 6.5 swing Fib levels
                 (only update on new fractal pivot confirm), Cat 7
                 session/time (fixed per bar boundary), Cat 2a HTF context
                 (fixed until next 4H/1D bar closes).

  - `dynamic`  — mutates intrabar; should NOT drive intrabar entry directly.
                 Examples: Cat 1 momentum (RSI/WT/Stoch/MACD/Squeeze, all
                 update as price moves), Cat 2 EMA-based trend, Cat 3
                 volatility (ATR/BB/Keltner roll over the bar), Cat 4
                 volume (current-bar volume incomplete intrabar), Cat 13
                 divergence (depends on current close), Cat 20 event memory
                 (bars_since_* tick forward dynamically as triggers reset).

  - `mixed`    — stable after their signal fires, dynamic before. Examples:
                 Cat 16 market structure (HH/HL/LH/LL confirmations are
                 static once confirmed; the "current move so far" is
                 dynamic), Cat 19 Ichimoku (spans displaced to past are
                 static; Tenkan/Kijun are dynamic).

This taxonomy pays off in Phase 4 intrabar scout: reads only `static` +
confirmed-`mixed` features plus fresh 5m bar-close values of selected
`dynamic` features.

CURRENT TAGGING SCOPE (Phase 1.10a):
  Tags are LOCKED for the 4 NEW algorithm files implemented per
  Decisions v2.37/v2.39/v2.40/v2.41:
    Cat 1   (32 features) — momentum_core.py
    Cat 2   (14 features) — trend.py
    Cat 2a  (18 features) — htf_context.py
    Cat 22  ( 7 features, 1 override) — cross_asset.py
  Total currently tagged: 71 features.

PENDING TAGGING (Phase 1.10b — to be added when each modify-in-place file
lands in Phase 1.10b):
    Cat 3   volatility, Cat 4 volume, Cat 5 vwap, Cat 6 pivots (+ 6.2/6.4),
    Cat 6.5 swing fib retracements, Cat 7 sessions, Cat 8 candles,
    Cat 9 stats, Cat 10 regime, Cat 11 context, Cat 12 lagged, Cat 13
    divergence, Cat 14 money flow, Cat 15 extra momentum, Cat 16 structure,
    Cat 17 fractal, Cat 18 adaptive MA, Cat 19 ichimoku, Cat 20 event memory.
  Each file edit will append to FEATURE_STABILITY at the same time as the
  feature names lock.
"""
from __future__ import annotations

from typing import Literal

Stability = Literal["static", "dynamic", "mixed"]


# ─── Cat 1 — Momentum (32 features, all dynamic) ─────────────────────────
# Per Decision v2.37 Q4 + v2.40 (Q6/Q8). Implemented in features/momentum_core.py.
_CAT_1_DYNAMIC = [
    # RSI block (4)
    "rsi_14", "rsi_slope", "rsi_zone", "rsi_dist_from_50",
    # MACD block (6)
    "macd_line", "macd_signal", "macd_hist", "macd_hist_slope",
    "macd_zero_cross_state", "macd_hist_acceleration",
    # WaveTrend block (4)
    "wt1", "wt2", "wt_cross_signal", "wt_ob_os_zone",
    # Stochastic block (4)
    "stoch_k", "stoch_d", "stoch_cross_signal", "stoch_ob_os_zone",
    # Squeeze block (4)
    "squeeze_value", "squeeze_signal", "squeeze_release_state", "bars_in_squeeze",
    # Multi-period momentum (4)
    "roc_1bar", "roc_3bar", "roc_6bar", "roc_12bar",
    # Cross-feature (2 per Decision v2.40 Q8)
    "rsi_wt_divergence_flag", "macd_rsi_alignment",
    # Velocity-of-velocity (4 per Decision v2.40 Q6 option a)
    "d2_rsi", "d2_macd_line", "d2_wt1", "d2_stoch_k",
]


# ─── Cat 2 — Trend / Direction (14 features, all dynamic) ────────────────
# Per Decision v2.37 Q5 + v2.39. Implemented in features/trend.py.
_CAT_2_DYNAMIC = [
    # ADX/DI block (9)
    "di_plus", "di_minus", "adx", "di_spread", "adx_slope",
    "adx_trending", "adx_weak", "adx_accelerating", "adx_decelerating",
    # EMA block (5)
    "ema9_dist_pct", "ema21_dist_pct", "ema50_dist_pct",
    "ema_stack_30m", "ema21_dist_atr",
]


# ─── Cat 2a — HTF Context (18 features, all static) ──────────────────────
# Fixed until next 4H/1D bar closes. Implemented in features/htf_context.py.
_CAT_2A_STATIC = [
    # 4H block (9)
    "htf4h_bb_position", "htf4h_rsi", "htf4h_macd_hist", "htf4h_adx",
    "htf4h_ema21_pos", "htf4h_atr_ratio", "htf4h_close_vs_ema50_pct",
    "htf4h_return_1bar", "htf4h_return_3bar",
    # 1D block (9)
    "htf1d_ema20_pos", "htf1d_ema50_pos", "htf1d_ema200_pos",
    "htf1d_rsi", "htf1d_atr_pct",
    "htf1d_return_1bar", "htf1d_return_5bar",
    "htf1d_close_vs_20d_high_pct", "htf1d_close_vs_20d_low_pct",
]


# ─── Cat 22 — Cross-Asset Correlation (7 features) ───────────────────────
# Per Decision v2.41 (ETH dropped, ATR-norm difference). Implemented in
# features/cross_asset.py. Mostly dynamic, with 1 static exception.
_CAT_22_DYNAMIC = [
    "btc_corr_20bar",
    "btc_return_1bar", "btc_return_3bar", "btc_return_12bar",
    "btc_vs_asset_atr_norm_diff",
    # Phase 3+ feature (only fires when caller supplies btc_funding):
    "btc_funding_rate",
]
# 1 override: btc_above_ema200_daily uses prev-day shift, so its value at
# any 30m bar within day D is yesterday's daily-EMA200 sign — fixed for the
# entire UTC day. Therefore static within a 30m bar.
_CAT_22_STATIC_OVERRIDE = ["btc_above_ema200_daily"]


# ─── Build flat dict ─────────────────────────────────────────────────────
FEATURE_STABILITY: dict[str, Stability] = {}

for f in _CAT_1_DYNAMIC + _CAT_2_DYNAMIC + _CAT_22_DYNAMIC:
    FEATURE_STABILITY[f] = "dynamic"

for f in _CAT_2A_STATIC + _CAT_22_STATIC_OVERRIDE:
    FEATURE_STABILITY[f] = "static"


# ─── Public API ──────────────────────────────────────────────────────────
def get_stability(feature_name: str) -> Stability:
    """Return stability class for a feature column.

    Raises
    ------
    KeyError if the feature name is not yet tagged (Phase 1.10b will add
    remaining categories as their feature files land).
    """
    if feature_name not in FEATURE_STABILITY:
        raise KeyError(
            f"Feature '{feature_name}' not tagged in FEATURE_STABILITY. "
            f"Phase 1.10a tagged Cat 1/2/2a/22 only ({len(FEATURE_STABILITY)} "
            f"features). Other categories will be added as their files are "
            f"rewritten in Phase 1.10b."
        )
    return FEATURE_STABILITY[feature_name]


def stability_summary() -> dict[Stability, int]:
    """Return count of features per stability class. Diagnostic helper."""
    summary: dict[Stability, int] = {"static": 0, "dynamic": 0, "mixed": 0}
    for tag in FEATURE_STABILITY.values():
        summary[tag] += 1
    return summary
