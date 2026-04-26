"""Unit tests for triple-barrier labeler.

Spec: Project Spec 30min.md §8 (Labeling).

v1.0 had no tests/ directory — written fresh in v2.0 per Decision v2.34.
Validates the v1.0-reused triple-barrier labeler against v2.0 parameters
(tp_atr_mult=3.0, sl_atr_mult=3.0, max_holding_bars=8 (4h), per-asset
min_profit_pct timeout gate from §8.1).

Acceptance:
- TP hit returns LONG (0).
- SL hit returns SHORT (1).
- Timeout with drift > min_profit_pct returns LONG/SHORT depending on direction.
- Timeout with drift ≤ min_profit_pct returns NEUTRAL (2).
- Pessimistic tie-break (§8.3): if TP and SL hit on the same bar, the tighter
  side wins (conservative).
- Symmetric barriers (3.0/3.0) — no directional bias from label asymmetry.

Implementation deferred to Phase 1.12 (label generation) /
Phase 1.13 (distribution sanity check).
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 1.12 — Project Spec 30min.md §8.1")
def test_tp_hit_first_returns_long() -> None:
    """TP barrier hit before SL → label = LONG (0)."""


@pytest.mark.skip(reason="Phase 1.12 — Project Spec 30min.md §8.1")
def test_sl_hit_first_returns_short() -> None:
    """SL barrier hit before TP → label = SHORT (1)."""


@pytest.mark.skip(reason="Phase 1.12 — Project Spec 30min.md §8.1")
def test_timeout_above_min_profit_returns_directional() -> None:
    """Timeout drift > min_profit_pct → LONG or SHORT per sign."""


@pytest.mark.skip(reason="Phase 1.12 — Project Spec 30min.md §8.1")
def test_timeout_below_min_profit_returns_neutral() -> None:
    """Timeout drift ≤ min_profit_pct → NEUTRAL (2)."""


@pytest.mark.skip(reason="Phase 1.12 — Project Spec 30min.md §8.3")
def test_pessimistic_tie_break_sl_wins_when_same_bar() -> None:
    """When TP and SL both hit on the same bar, the tighter side wins."""


@pytest.mark.skip(reason="Phase 1.13 — Project Spec 30min.md §8.2")
def test_label_distribution_within_expected_band() -> None:
    """LONG/SHORT each 35–45%, NEUTRAL 15–25% on real 30m fixture."""
