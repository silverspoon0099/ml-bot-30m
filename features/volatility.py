"""Cat 3 — Volatility (12 features) — v2.0.

Per Project Spec 30min §7.2 Cat 3 (15→12). Trim from v1.0:

  KEEP (12 features = 4 ATR + 5 BB + 3 KC/squeeze):
    ATR family   (4): atr_14, atr_5, atr_ratio, atr_percentile
    BB family    (5): bb_basis, bb_upper, bb_lower, bb_width_pct, bb_position
    KC + squeeze (3): kc_upper, kc_lower, squeeze_state

  DROP from v1.0: atr_roc, bb_width_percentile, kc_width, realized_vol_5,
                  realized_vol_20, vol_ratio_5_20, high_low_range_pct,
                  session_range_pct, bars_since_squeeze_release.

squeeze_state is the v1.0 state machine output: -1 (in squeeze) / 0
(normal) / +1 (just released). Spec wording is "KC-BB squeeze flag" but
the richer 3-state encoding carries strictly more information; downstream
modules already depend on the signed encoding (v1.0 carry-over). Treated
as Cat 3's 12th feature.

bars_since_squeeze_release: dropped here. Cat 20 event_memory.py (when
rewritten in Phase 1.10b) will derive its own bars_since_squeeze_fire and
bars_since_squeeze_entry from squeeze_state directly.

OWNED BY THIS MODULE (consumed by other selection modules):
  - atr_14         → trend.py (price vs EMA21 in ATR units)
                   → cross_asset.py (ATR-norm-diff feature)
  - squeeze_state  → momentum_core.py (Cat 1 squeeze block)
                   → event_memory.py (Cat 20 squeeze events, when rewritten)

SIGNATURE CHANGE vs v1.0: dropped session_id parameter (no longer
needed since session_range_pct is dropped from output). Caller (builder.py
when rewritten) calls `volatility_features(df, cfg)` directly.
"""
from __future__ import annotations

import pandas as pd

from ._common import pct, rolling_percentile, safe_div, true_range, wilder_ema


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """ATR via Wilder smoothing of true range."""
    return wilder_ema(true_range(high, low, close), period)


def bbands(
    close: pd.Series, period: int, std: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (basis, upper, lower) for Bollinger Bands."""
    basis = close.rolling(period, min_periods=period).mean()
    dev = close.rolling(period, min_periods=period).std(ddof=0)
    return basis, basis + std * dev, basis - std * dev


def keltner(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int, mult: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (basis, upper, lower) for Keltner Channels.

    LazyBear Squeeze default: useTrueRange=true (matches reference Pine).
    """
    basis = close.rolling(length, min_periods=length).mean()
    rng = true_range(high, low, close).rolling(length, min_periods=length).mean()
    return basis, basis + mult * rng, basis - mult * rng


def volatility_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute Cat 3 = 12 volatility features.

    Parameters
    ----------
    df : DataFrame with high, low, close columns.
    cfg : feature config dict. Uses:
          - cfg['atr']['period']            (default 14 — for atr_14)
          - cfg['atr']['short_period']      (default 5  — for atr_5)
          - cfg['atr']['percentile_window'] (default 100)
          - cfg['bb']['period']             (default 20)
          - cfg['bb']['std']                (default 2.0)
          - cfg['squeeze']['bb_length']     (default 20)
          - cfg['squeeze']['bb_mult']       (default 2.0)
          - cfg['squeeze']['kc_length']     (default 20)
          - cfg['squeeze']['kc_mult']       (default 1.5)

    Returns
    -------
    DataFrame of 12 columns indexed like df.
    """
    high, low, close = df["high"], df["low"], df["close"]
    a_cfg = cfg["atr"]
    bb_cfg = cfg["bb"]
    sq_cfg = cfg["squeeze"]

    # ── ATR family (4) ───────────────────────────────────────────────────
    atr_14 = atr(high, low, close, a_cfg["period"])
    atr_5 = atr(high, low, close, a_cfg["short_period"])
    atr_ratio = safe_div(atr_5, atr_14)
    atr_percentile = rolling_percentile(atr_14, a_cfg["percentile_window"])

    # ── Bollinger Bands family (5) ───────────────────────────────────────
    bb_basis, bb_upper, bb_lower = bbands(close, bb_cfg["period"], bb_cfg["std"])
    bb_width_pct = pct(bb_upper - bb_lower, bb_basis)
    bb_position = safe_div(close - bb_lower, bb_upper - bb_lower)

    # ── Keltner + squeeze state machine (3) ──────────────────────────────
    # Squeeze uses its own BB params (sq_cfg) — independent of the bb_cfg
    # used for the BB-family features above. This matches v1.0 LazyBear
    # Squeeze convention: separate BB(20, 2.0) for the band features and
    # BB(20, 2.0) inside KC for the squeeze trigger. They happen to share
    # the period but are configured independently for flexibility.
    sq_bb_basis, sq_bb_up, sq_bb_lo = bbands(close, sq_cfg["bb_length"], sq_cfg["bb_mult"])
    kc_basis, kc_upper, kc_lower = keltner(
        high, low, close, sq_cfg["kc_length"], sq_cfg["kc_mult"]
    )

    in_squeeze = ((sq_bb_up < kc_upper) & (sq_bb_lo > kc_lower)).astype(int)
    just_released = ((in_squeeze.shift(1) == 1) & (in_squeeze == 0)).astype(int)
    squeeze_state = pd.Series(0, index=close.index, dtype=float)
    squeeze_state[in_squeeze == 1] = -1
    squeeze_state[just_released == 1] = 1

    return pd.DataFrame(
        {
            # ATR family (4)
            "atr_14": atr_14,
            "atr_5": atr_5,
            "atr_ratio": atr_ratio,
            "atr_percentile": atr_percentile,
            # BB family (5)
            "bb_basis": bb_basis,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_width_pct": bb_width_pct,
            "bb_position": bb_position,
            # KC + squeeze (3)
            "kc_upper": kc_upper,
            "kc_lower": kc_lower,
            "squeeze_state": squeeze_state,
        }
    )
