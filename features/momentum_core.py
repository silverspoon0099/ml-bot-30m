"""Category 1 — Momentum core (32 features, refactored from v1.0 47).

Spec: Project Spec 30min.md §7.2 Category 1.

NEW in v2.0: consolidates v1.0 Cat 1 momentum logic that was split across
indicators.py and extra_momentum.py. v1.0 had 47 momentum features; v2.0
trims to 32 by dropping redundant period variants and sub-1m-equivalent
oscillator variants that have no analog at 30m.

Keep set per §7.2 Cat 1:
- RSI(14) value, slope, zone (OB/OS/neutral), distance from 50 (4)
- MACD line, signal, histogram, hist slope, zero-cross state, hist accel (6)
- WaveTrend wt1, wt2, wt_cross signal, OB/OS zone (4)
- Stochastic %K, %D, cross signal, OB/OS zone (4)
- Squeeze Momentum value, signal, release_state, bars_in_squeeze (4)
- Multi-period momentum: roc_1bar, roc_3bar, roc_6bar, roc_12bar (4)
- Cross-feature: rsi vs wt divergence, macd vs rsi alignment (2)
- Velocity-of-velocity: d²rsi/dt², d²macd/dt² (4)

Implementation deferred to Phase 1.10 (per-category trim per §7.2 deltas).
"""
from __future__ import annotations

import pandas as pd


def momentum_core_features(df_30m: pd.DataFrame) -> pd.DataFrame:
    """Compute 32-feature Cat 1 momentum block. Phase 1.10 implementation."""
    raise NotImplementedError("Phase 1.10 — Project Spec 30min.md §7.2 Cat 1")
