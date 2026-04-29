"""Cat 19 — Ichimoku (6 features) — v2.0.

Per Project Spec 30min §7.2 Cat 19 + Decision v2.47 Q16.

6 features locked per Decision v2.47 Q16.7 — all close-relative percent
distances; per Q16.4 (a) all tagged DYNAMIC and Cat 19 was REMOVED from
the §7.5 mixed-block list (locked feature shapes per §15 emit `*_dist_pct`
which mask the original static-span intent at the feature level).

Canonical Hosoda math (TradingView ta.ichimoku-equivalent):

    tenkan      = (max_high(9)  + min_low(9))  / 2
    kijun       = (max_high(26) + min_low(26)) / 2
    span_a_raw  = (tenkan + kijun) / 2                  plotted +26 bars FORWARD
    span_b_raw  = (max_high(52) + min_low(52)) / 2      plotted +26 bars FORWARD

"Cloud projected over current bar t" = span values computed at t - kijun.
At current bar we read span_*_raw.shift(kijun) — no look-ahead.

6 features (all close-relative % per Q16.1 (a)):
  - tenkan_dist_pct      — (close − tenkan)   / close × 100
  - kijun_dist_pct       — (close − kijun)    / close × 100
  - tk_diff_pct          — (tenkan − kijun)   / close × 100   (rename from
                                                v1.0 `tk_spread`; carries
                                                magnitude AND sign)
  - senkou_a_dist_pct    — (close − senkou_a) / close × 100   NEW per §15
  - senkou_b_dist_pct    — (close − senkou_b) / close × 100   NEW per §15
  - cloud_dist_pct       — (close − cloud_mid) / close × 100   where
                                                cloud_mid = (senkou_a + senkou_b)/2

DROPPED from v1.0:
  - tk_cross — sign of (tenkan − kijun); fully derivable from sign of
    `tk_diff_pct`. Redundant feature per Q16.2 (a).

§7.5 TAGGING per Q16.4 (a): All 6 features = DYNAMIC. Cat 19 is no longer
on the §7.5 mixed-block list (Cat 16 remains the only mixed-split block).
The original §7.5 narrative ("spans displaced to past = static") was
intent-based; the locked feature shapes per §15 are all close-relative,
so per-feature tagging is uniformly dynamic. Phase 4 intrabar scout can
still optimize `senkou_*_dist_pct` recomputation by caching the displaced
span values (an implementation detail in scout code, not a tagging
concern).
"""
from __future__ import annotations

import pandas as pd

from ._common import pct


def ichimoku_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Compute Cat 19 = 6 Ichimoku features.

    Self-contained — math derives from OHLC alone; no caller-supplied
    dependencies.

    Parameters
    ----------
    df : DataFrame with high, low, close columns.
    cfg : optional config dict. Tunable keys (with canonical Hosoda
          defaults; legacy v1.0 keys also supported for config.yaml compat):
            - "ichimoku.tenkan"    (default 9; legacy: "tenkan")
            - "ichimoku.kijun"     (default 26; legacy: "kijun")
            - "ichimoku.senkou_b"  (default 52; legacy: "senkou_b")

    Returns
    -------
    DataFrame of 6 columns indexed like df. All DYNAMIC per Q16.4 (a).
    """
    cfg = cfg or {}
    # Accept both v2.0 namespaced keys (preferred) and legacy v1.0 keys.
    t = int(cfg.get("ichimoku.tenkan", cfg.get("tenkan", 9)))
    k = int(cfg.get("ichimoku.kijun", cfg.get("kijun", 26)))
    sb = int(cfg.get("ichimoku.senkou_b", cfg.get("senkou_b", 52)))

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # ── Tenkan / Kijun (current-bar moving midlines) ──────────────────
    tenkan = (
        high.rolling(t, min_periods=t).max() + low.rolling(t, min_periods=t).min()
    ) / 2
    kijun = (
        high.rolling(k, min_periods=k).max() + low.rolling(k, min_periods=k).min()
    ) / 2

    # ── Raw leading spans (computed at current bar from current data) ─
    span_a_raw = (tenkan + kijun) / 2
    span_b_raw = (
        high.rolling(sb, min_periods=sb).max() + low.rolling(sb, min_periods=sb).min()
    ) / 2

    # ── Cloud projected over current bar = spans computed `kijun` bars
    # ago (causal shift; no look-ahead). v1.0 correctness fix retained.
    senkou_a = span_a_raw.shift(k)
    senkou_b = span_b_raw.shift(k)
    cloud_mid = (senkou_a + senkou_b) / 2

    return pd.DataFrame(
        {
            "tenkan_dist_pct": pct(close - tenkan, close),
            "kijun_dist_pct": pct(close - kijun, close),
            "tk_diff_pct": pct(tenkan - kijun, close),
            "senkou_a_dist_pct": pct(close - senkou_a, close),
            "senkou_b_dist_pct": pct(close - senkou_b, close),
            "cloud_dist_pct": pct(close - cloud_mid, close),
        },
        index=df.index,
    )
