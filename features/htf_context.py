"""Category 2a — HTF Context features (18 features).

Spec: Project Spec 30min.md §7.2 Category 2a.

Computes 4H and 1D context features (htf4h_bb_position, htf4h_rsi,
htf4h_macd_hist, htf4h_adx, htf4h_ema21_pos, htf4h_atr_ratio,
htf4h_close_vs_ema50_pct, htf4h_return_1bar, htf4h_return_3bar,
htf1d_ema20_pos, htf1d_ema50_pos, htf1d_ema200_pos, htf1d_rsi,
htf1d_atr_pct, htf1d_return_1bar, htf1d_return_5bar,
htf1d_close_vs_20d_high_pct, htf1d_close_vs_20d_low_pct).

Inputs: aggregated 4H and 1D frames produced by features/builder.py
        merge_htf_into_30m() (§6.4 Steps A–B).
Output: feature columns merged into the 30m frame via prev-closed-bar
        lookup (§6.4 Step C). No look-ahead.

Implementation deferred to Phase 1.4 (Appendix C row 1.4).
"""
from __future__ import annotations

import pandas as pd


def htf_context_features(df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> pd.DataFrame:
    """Compute 18-feature HTF context block. Phase 1.4 implementation."""
    raise NotImplementedError("Phase 1.4 — Project Spec 30min.md §7.2 Cat 2a")
