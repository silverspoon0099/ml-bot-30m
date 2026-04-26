"""Two-stage baseline gate — pre-gate (5 features) + full-gate (~202 features).

Spec: Project Spec 30min.md §10.3 (Baseline gates, two stages).

Pre-gate (§10.3.1): exactly 5 hand-picked features
  close_vs_ema21_pct, rsi_14, adx_14, htf4h_ema21_pos, volume_ratio_20.
  Single fold (months [1..9] train, month 10 val).
  PASS = val log-loss beats empirical prior by ≥1%.
  FAIL = halt the project (premise-level signal — do not tune your way out).

Full-gate (§10.3.2): full v2.0 ~202 feature set with default LightGBM.
  Single fold, default params (no Optuna yet).
  PASS = val log-loss beats empirical prior by ≥2%.
  FAIL = SHAP-trim aggressively to ~30–50 features and re-run before Optuna.

Empirical prior for 3-class label = -Σ pᵢ log pᵢ.
For expected 40/40/20 split, prior ≈ 1.055 → gate threshold ≈ 1.034.

Phase 2.1 / 2.2 in Appendix C. Implementation when entering Phase 2.
"""
from __future__ import annotations

import sys


def main() -> int:
    """Phase 2.1/2.2 entry point. Implementation deferred."""
    raise NotImplementedError(
        "Phase 2.1 / 2.2 — Project Spec 30min.md §10.3 (baseline gates)"
    )


if __name__ == "__main__":
    sys.exit(main())
