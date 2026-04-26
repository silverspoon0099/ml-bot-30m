"""Verify multi-anchor VWAP feature correctness.

Spec: Project Spec 30min.md §7.2 Category 5 (VWAP, 8 → 14 features).

Anchors to validate (§7.2 Cat 5):
- daily_vwap (5: value, ±1σ, ±2σ, close vs VWAP %, zone)
- swing_high_vwap_pos / swing_low_vwap_pos (anchor at last confirmed swing)
- htf_pivot_vwap_pos (anchor at daily pivot line break)
- weekly_vwap_pos (Monday 00:00 UTC anchored)
- multi-anchor confluence flag, rolling VWAP-of-VWAPs reversion,
  distance to nearest anchored VWAP in ATR units, cross events count,
  close above/below "heavy" VWAP flag.

Implementation deferred to Phase 1.5 (Appendix C row 1.5).
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 1.5 — Project Spec 30min.md §7.2 Cat 5")
def test_daily_vwap_correctness() -> None:
    """daily_vwap matches manual cumulative VWAP over the trading day."""


@pytest.mark.skip(reason="Phase 1.5 — Project Spec 30min.md §7.2 Cat 5")
def test_swing_anchored_vwap_resets_on_new_swing() -> None:
    """swing_high/low VWAP re-anchors at confirmed fractal pivots."""


@pytest.mark.skip(reason="Phase 1.5 — Project Spec 30min.md §7.2 Cat 5")
def test_multi_anchor_confluence_flag() -> None:
    """Confluence flag counts how many anchored VWAPs agree on direction."""
