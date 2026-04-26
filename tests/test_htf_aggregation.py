"""Verify lossless 30m → 4H / 30m → 1D aggregation.

Spec: Project Spec 30min.md §6.4 (Step A — aggregate).

Acceptance per §6.4:
- 8 consecutive 30m bars aggregate to exactly 1 × 4H bar.
- 48 consecutive 30m bars aggregate to exactly 1 × 1D bar.
- Open = first, High = max, Low = min, Close = last, Volume = sum.
- resample uses label="left", closed="left" (so the 4H bar at 12:00 UTC
  represents 12:00–15:59:59; close = the 30m bar that closed at 15:30).
- No look-ahead: the 4H bar is "complete" only once the 15:30 30m bar closes.

Implementation deferred to Phase 1.3 (Appendix C row 1.3).
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 1.3 — Project Spec 30min.md §6.4")
def test_30m_to_4h_aggregation_lossless() -> None:
    """8 × 30m bars → 1 × 4H bar with first/max/min/last/sum semantics."""


@pytest.mark.skip(reason="Phase 1.3 — Project Spec 30min.md §6.4")
def test_30m_to_1d_aggregation_lossless() -> None:
    """48 × 30m bars → 1 × 1D bar with first/max/min/last/sum semantics."""


@pytest.mark.skip(reason="Phase 1.3 — Project Spec 30min.md §6.4")
def test_resample_no_lookahead() -> None:
    """4H/1D bars use only completed 30m bars (label=left, closed=left)."""
