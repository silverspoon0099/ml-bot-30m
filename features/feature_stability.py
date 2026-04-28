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

CURRENT TAGGING SCOPE:
  Phase 1.10a (NEW algorithm files, Decisions v2.37/v2.39/v2.40/v2.41):
    Cat 1   (32 features) — momentum_core.py
    Cat 2   (14 features) — trend.py
    Cat 2a  (18 features) — htf_context.py
    Cat 22  ( 7 features, 1 override) — cross_asset.py
  Phase 1.10b (modify-in-place trims, lands per file):
    Cat 3   (12 features) — volatility.py    [v2.0 trim 15→12]
    Cat 15  ( 7 features) — extra_momentum.py [v2.0 trim 9→7 per Decision v2.42 Q11]
    Cat 13  ( 7 features) — divergence.py    [v2.0 reshape 7→7 strict spec per Decision v2.43 Q12]
    Cat 4   (12 features) — volume.py         [v2.0 trim 17→12]
    Cat 14  ( 6 features) — volume.py (`money_flow_features`) [v2.0 trim 8→6]
    Cat 5   (14 features) — vwap.py           [v2.0 multi-anchor expand 8→14 per Decision v2.44]
  Total currently tagged: 129 features.

PENDING TAGGING (Phase 1.10b — to be added when each modify-in-place file
lands):
    Cat 6 pivots (+ 6.2/6.4), Cat 6.5 swing fib retracements,
    Cat 7 sessions, Cat 8 candles, Cat 9 stats, Cat 10 regime,
    Cat 11 context, Cat 12 lagged, Cat 16 structure, Cat 17 fractal,
    Cat 18 adaptive MA, Cat 19 ichimoku, Cat 20 event memory.
  Each file edit appends to FEATURE_STABILITY when the feature names lock.
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


# ─── Cat 3 — Volatility (12 features, all dynamic) ───────────────────────
# Per §7.2 Cat 3 trim 14→12. Implemented in features/volatility.py.
# All 12 features mutate intrabar:
#   - ATR (Wilder smoothing) updates on every new high/low/close
#   - BB rolling stats (mean, std) update with current close
#   - bb_position depends on current close
#   - Keltner channels update with TR
#   - squeeze_state (-1/0/+1) toggles intrabar based on BB-vs-KC at current bar
_CAT_3_DYNAMIC = [
    # ATR family (4)
    "atr_14", "atr_5", "atr_ratio", "atr_percentile",
    # BB family (5)
    "bb_basis", "bb_upper", "bb_lower", "bb_width_pct", "bb_position",
    # KC + squeeze (3)
    "kc_upper", "kc_lower", "squeeze_state",
]


# ─── Cat 15 — Additional Momentum (7 features, all dynamic) ──────────────
# Per §7.2 Cat 15 trim 9→7 + Decision v2.42 Q11 (both TSI cross states kept).
# Implemented in features/extra_momentum.py. All 7 mutate intrabar (oscillator
# values + sign flips that follow them).
_CAT_15_DYNAMIC = [
    "williams_r", "williams_r_zone",
    "cci_20", "cci_zero_cross_state",
    "tsi_signal_cross_state", "tsi_zero_cross_state",
    "cmo_zero_cross_state",
]


# ─── Cat 13 — Divergence Detection (7 features, all dynamic) ─────────────
# Per §7.2 Cat 13 + Decision v2.43 Q12 (strict spec: split bullish/bearish
# binary flags, drop WT/Stoch divs). Implemented in features/divergence.py.
# All 7 dynamic — divergence flags depend on the most recent confirmed
# fractal pivot, which can fire/cancel as new bars close. Recency is also
# dynamic (counter ticks forward each bar, resets on new divergence).
_CAT_13_DYNAMIC = [
    "regular_bullish_div_rsi", "regular_bearish_div_rsi",
    "regular_bullish_div_macd", "regular_bearish_div_macd",
    "hidden_bullish_div_rsi", "hidden_bearish_div_rsi",
    "divergence_recency",
]


# ─── Cat 4 — Volume / Buy-Sell Pressure (12 features, all dynamic) ───────
# Per §7.2 Cat 4 trim 17→12. Implemented in features/volume.py.
# All 12 mutate intrabar:
#   - volume rolling stats update with current bar's volume
#   - VFI uses current bar in cumulative + rolling
#   - OBV is cumulative (updates each bar); slope mutates with new bars
#   - obv_divergence_flag depends on most recent fractal pivot
#   - candle_buy_sell_signed depends on current close vs current bar's range
#   - volume_weighted_momentum_10 includes current bar's return × volume
#   - high_vol_close_*_tercile depends on current bar's volume + close position
_CAT_4_DYNAMIC = [
    # Volume statistics (3)
    "volume_ratio_20", "volume_zscore_20", "volume_spike_flag",
    # VFI block (3)
    "vfi", "vfi_signal", "vfi_hist_slope",
    # OBV block (2)
    "obv_slope", "obv_divergence_flag",
    # Buy/sell estimator (1)
    "candle_buy_sell_signed",
    # Volume-weighted momentum (1)
    "volume_weighted_momentum_10",
    # High-volume-candle location (2)
    "high_vol_close_top_tercile", "high_vol_close_bottom_tercile",
]


# ─── Cat 14 — Money Flow (6 features, all dynamic) ───────────────────────
# Per §7.2 Cat 14 trim 8→6. Implemented in features/volume.py
# (`money_flow_features` function). All 6 mutate intrabar:
#   - CMF rolling sum updates each bar
#   - MFI rolling sum updates; zone derives from current MFI value
#   - A/D is cumulative; slope mutates; divergence flag depends on pivots
_CAT_14_DYNAMIC = [
    "cmf_20", "cmf_slope",
    "mfi_14", "mfi_zone",
    "ad_slope_10", "ad_price_divergence_flag",
]


# ─── Cat 5 — VWAP (14 features, all dynamic) ─────────────────────────────
# Per §7.2 Cat 5 multi-anchor expansion 8→14 + Decision v2.4 + Decision v2.44
# (Q13 implementation choices). Implemented in features/vwap.py.
# All 14 mutate intrabar:
#   - VWAP cumulative numerator/denominator update each tick
#   - Bands derive from intrabar stdev rolling
#   - Zone depends on close vs current bands
#   - Multi-anchor analytics depend on current close vs all 5 VWAP values
#   - Heavy VWAP flag depends on current close (touches counted in rolling
#     window which itself slides)
_CAT_5_DYNAMIC = [
    # Daily-anchored (5)
    "daily_vwap", "daily_vwap_upper_band_1sig", "daily_vwap_lower_band_1sig",
    "daily_vwap_dist_pct", "daily_vwap_zone",
    # Multi-anchor pos (4)
    "swing_high_vwap_pos", "swing_low_vwap_pos",
    "htf_pivot_vwap_pos", "weekly_vwap_pos",
    # Multi-anchor analytics (5)
    "multi_anchor_confluence_signed_count",
    "vwap_of_vwaps_mean_reversion_dist_pct",
    "dist_to_nearest_anchored_vwap_atr",
    "vwap_cross_events_count_10",
    "close_above_below_heavy_vwap_flag",
]


# ─── Build flat dict ─────────────────────────────────────────────────────
FEATURE_STABILITY: dict[str, Stability] = {}

for f in (
    _CAT_1_DYNAMIC + _CAT_2_DYNAMIC + _CAT_22_DYNAMIC + _CAT_3_DYNAMIC
    + _CAT_15_DYNAMIC + _CAT_13_DYNAMIC + _CAT_4_DYNAMIC + _CAT_14_DYNAMIC
    + _CAT_5_DYNAMIC
):
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
            f"{len(FEATURE_STABILITY)} features tagged so far (Cat "
            f"1/2/2a/3/4/5/13/14/15/22 implemented; remaining categories in Phase 1.10b)."
        )
    return FEATURE_STABILITY[feature_name]


def stability_summary() -> dict[Stability, int]:
    """Return count of features per stability class. Diagnostic helper."""
    summary: dict[Stability, int] = {"static": 0, "dynamic": 0, "mixed": 0}
    for tag in FEATURE_STABILITY.values():
        summary[tag] += 1
    return summary
