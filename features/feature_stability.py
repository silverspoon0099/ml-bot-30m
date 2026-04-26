"""Feature stability taxonomy — static / dynamic / mixed tagging.

Spec: Project Spec 30min.md §7.5.

Tags every feature column produced by builder.py as one of:
- `static`  — stable within a 30m bar; safe for intrabar inference on 5m/1m
              without recomputation. (Cat 6 pivots, Cat 6.5 swing Fib levels,
              Cat 7 session/time, Cat 2a HTF context, Cat 11 prev context.)
- `dynamic` — mutates intrabar; should NOT drive intrabar entry directly.
              (Cat 1 momentum, Cat 2 EMA trend, Cat 3 volatility, Cat 4
              volume, Cat 13 divergence, Cat 20 event memory.)
- `mixed`   — stable after their signal fires, dynamic before. (Cat 16
              market structure, Cat 19 Ichimoku.)

Pays off in Phase 4 intrabar scout: reads only `static` + confirmed-`mixed`
features plus fresh 5m bar-close values of selected `dynamic` features.

Implementation deferred to Phase 1.11 (Appendix C row 1.11).
"""
from __future__ import annotations

from typing import Literal

Stability = Literal["static", "dynamic", "mixed"]

FEATURE_STABILITY: dict[str, Stability] = {
    # Phase 1.11 populates this dict during feature catalog build per §7.5.
}


def get_stability(feature_name: str) -> Stability:
    """Return stability class for a feature column. Phase 1.11 implementation."""
    raise NotImplementedError("Phase 1.11 — Project Spec 30min.md §7.5")
