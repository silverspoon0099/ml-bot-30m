"""Category 22 — Cross-Asset Correlation (8 features for altcoins).

Spec: Project Spec 30min.md §7.2 Category 22 (promoted from Phase 4 to
Phase 1 for altcoins per Decision v2.11).

Per Decision v2.35: v2.0 asset universe is BTC, SOL, LINK. Cross-asset
features fire for SOL and LINK (BTC has nothing to correlate to within
this universe). TAO deferred to Phase 4 candidate.

Features per §7.2 Cat 22:
- BTC 20-bar correlation with this asset (1)
- BTC return last 1 bar, 3 bar, 12 bar (3)
- BTC ATR-normalized move vs this asset's ATR-normalized move (1)
- BTC above/below its EMA200 daily (macro regime) (1)
- BTC funding rate (Phase 3+ only — gated by Cat 21 Hyperliquid microstructure online)
- ETH correlation with this asset (SOL/LINK only per Decision v2.35) (1)

User memory: BTC correlation is a *booster*, not a *filter* — independent
altcoin moves remain tradeable.

Implementation deferred to Phase 1.10 (per-category build per §7.2).
"""
from __future__ import annotations

import pandas as pd


def cross_asset_features(
    df_asset: pd.DataFrame,
    df_btc: pd.DataFrame,
    df_eth: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute Cat 22 cross-asset features. Phase 1.10 implementation."""
    raise NotImplementedError("Phase 1.10 — Project Spec 30min.md §7.2 Cat 22")
