"""Unit tests for purged walk-forward CV with embargo.

Spec: Project Spec 30min.md §9.2 (Walk-forward).

v1.0 had no tests/ directory — written fresh in v2.0 per Decision v2.34.

Walk-forward parameters per §9.2:
  train_months=9, val_months=1, step_months=1
  purge_bars=8, embargo_bars=8 (= max_holding_bars per §8.1)

Acceptance:
- Each fold's val window starts ≥ purge_bars after train end (no leakage from
  label horizon overlap).
- Each fold's next train window starts ≥ embargo_bars after val end.
- 3 years of data + 1-month step → ~14 folds (was 8 on v1.0 with 18mo).
- Fold-local normalization: scaler fit on train only; val/OOT use transform-only.
- OOT slice is strictly outside the WF range (never folded).

Implementation deferred to Phase 2.3 (walk-forward on BTC, 14 folds).
"""
from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 2.3 — Project Spec 30min.md §9.2")
def test_fold_count_at_3yr_with_1mo_step() -> None:
    """3-year window with train=9mo / val=1mo / step=1mo → ~14 folds."""


@pytest.mark.skip(reason="Phase 2.3 — Project Spec 30min.md §9.2")
def test_purge_separates_train_and_val_by_max_holding_bars() -> None:
    """val_start ≥ train_end + 8 bars (no label-horizon leakage)."""


@pytest.mark.skip(reason="Phase 2.3 — Project Spec 30min.md §9.2")
def test_embargo_separates_val_and_next_train() -> None:
    """next_train_start ≥ val_end + 8 bars."""


@pytest.mark.skip(reason="Phase 2.3 — Project Spec 30min.md §9.2")
def test_oot_slice_excluded_from_all_folds() -> None:
    """No fold (train or val) overlaps the OOT month (§10.1)."""


@pytest.mark.skip(reason="Phase 2.3 — Project Spec 30min.md §9.2")
def test_scaler_fit_on_train_only() -> None:
    """val/OOT must transform-only against the fold's train-fitted scaler."""
