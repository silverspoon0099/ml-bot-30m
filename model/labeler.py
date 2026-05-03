"""Triple-barrier labeling with chop filter (López de Prado + Decision v2.56 / DR-007).

For each 30m candle, simulate a forward trade with three exit conditions:
  1. Upper barrier (close + tp_atr_mult × ATR) hit       → LONG (0)
  2. Lower barrier (close − sl_atr_mult × ATR) hit       → SHORT (1)
  3. Time barrier (max_holding_bars) reached             → check net P&L vs min_profit_pct

NEW per Decision v2.56 (DR-007): CHOP FILTER applied BEFORE barrier simulation.
At the entry bar, compute `atr_pct = atr_14 / close × 100`. If
`atr_pct < min_atr_pct_threshold`, the bar is in a low-volatility chop regime
where directional moves in 4 hours are essentially noise. Such bars are
labeled `-2` (CHOP_FILTERED) and excluded from training and inference per the
labeler's tradeable-regime contract.

Sentinel labels (preserved through Phase 1.10d → 1.11):
  -2 = CHOP_FILTERED  (NEW per Decision v2.56) — low-vol regime; excluded from training + inference
  -1 = UNLABELABLE    (preserved from v1.0) — tail rows where forward window incomplete
   0 = LONG           (per spec §8.1 classes mapping)
   1 = SHORT
   2 = NEUTRAL

Builder.py training-X filter: `label.isin({0, 1, 2})` drops both sentinels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    atr_col: str = "atr_14",
    tp_atr_mult: float = 2.5,
    sl_atr_mult: float = 2.5,
    max_holding_bars: int = 8,
    min_profit_pct: float = 0.3,
    min_atr_pct_threshold: float = 0.0,
    classes: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Returns DataFrame indexed like `df` with columns:
        label          0=LONG, 1=SHORT, 2=NEUTRAL, -2=CHOP_FILTERED, -1=UNLABELABLE
        holding_bars   bars to barrier hit (or max_holding_bars at time-out)
        exit_price     realized exit price (NaN for chop-filtered + unlabelable)
        pnl_pct        realized % return (NaN for chop-filtered + unlabelable)

    Parameters
    ----------
    df : DataFrame with high, low, close, and `atr_col` columns.
    atr_col : column name containing ATR(14) per spec §8.1; default "atr_14".
    tp_atr_mult : take-profit barrier in ATR units. Default 2.5 per Decision v2.56.
    sl_atr_mult : stop-loss barrier in ATR units. Default 2.5 (symmetric).
    max_holding_bars : forward window in bars. Default 8 (= 4 hours at 30m).
    min_profit_pct : timeout-gate threshold (% of entry); applied when neither
                     TP nor SL fires within max_holding_bars. If |move| <
                     min_profit_pct → NEUTRAL.
    min_atr_pct_threshold : CHOP FILTER threshold per Decision v2.56 (DR-007).
                     If `atr_14 / close × 100 < min_atr_pct_threshold` at the
                     entry bar, label = -2 (CHOP_FILTERED). Default 0.0 (no
                     chop filter; preserves v1.0 behavior when caller doesn't
                     supply asset-specific threshold).
    classes : optional override of class index mapping; default
              {LONG: 0, SHORT: 1, NEUTRAL: 2}.
    """
    classes = classes or {"LONG": 0, "SHORT": 1, "NEUTRAL": 2}
    chop_label = -2  # per Decision v2.56 §8.1 amendment

    n = len(df)
    labels = np.full(n, classes["NEUTRAL"], dtype=np.int8)
    holding = np.full(n, max_holding_bars, dtype=np.int32)
    exit_price = np.full(n, np.nan)
    pnl_pct = np.full(n, np.nan)

    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = df[atr_col].to_numpy()

    for i in range(n - max_holding_bars):
        entry = close[i]
        a = atr[i]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(entry):
            continue

        # ── CHOP FILTER (Decision v2.56 / DR-007) ─────────────────────
        # Excludes low-volatility regime bars from training + inference.
        # Applied BEFORE barrier simulation so chop bars don't consume
        # the simulation budget AND are clearly distinguished from
        # NEUTRAL (which means "tradeable regime, no clear direction").
        if min_atr_pct_threshold > 0:
            atr_pct = a / entry * 100.0
            if atr_pct < min_atr_pct_threshold:
                labels[i] = chop_label
                # exit_price + pnl_pct + holding_bars left as defaults (NaN/max_holding)
                # — chop bars don't simulate trades; metadata not meaningful
                continue

        # ── Triple-barrier simulation ─────────────────────────────────
        upper = entry + tp_atr_mult * a
        lower = entry - sl_atr_mult * a

        hit = False
        for j in range(1, max_holding_bars + 1):
            hi = high[i + j]
            lo = low[i + j]
            up_hit = hi >= upper
            dn_hit = lo <= lower
            if up_hit and dn_hit:
                # Pessimistic tie-break (§8.3): tighter side wins (conservative).
                if sl_atr_mult <= tp_atr_mult:
                    labels[i] = classes["SHORT"]
                    exit_price[i] = lower
                    pnl_pct[i] = (lower - entry) / entry * 100
                else:
                    labels[i] = classes["LONG"]
                    exit_price[i] = upper
                    pnl_pct[i] = (upper - entry) / entry * 100
                holding[i] = j
                hit = True
                break
            if up_hit:
                labels[i] = classes["LONG"]
                exit_price[i] = upper
                pnl_pct[i] = (upper - entry) / entry * 100
                holding[i] = j
                hit = True
                break
            if dn_hit:
                labels[i] = classes["SHORT"]
                exit_price[i] = lower
                pnl_pct[i] = (lower - entry) / entry * 100
                holding[i] = j
                hit = True
                break

        if not hit:
            # Timeout: classify by realized move vs min_profit_pct gate.
            final = close[i + max_holding_bars]
            move_pct = (final - entry) / entry * 100
            exit_price[i] = final
            pnl_pct[i] = move_pct
            holding[i] = max_holding_bars
            if move_pct > min_profit_pct:
                labels[i] = classes["LONG"]
            elif move_pct < -min_profit_pct:
                labels[i] = classes["SHORT"]
            else:
                labels[i] = classes["NEUTRAL"]

    # The last `max_holding_bars` rows can't be labeled (no forward window).
    labels[n - max_holding_bars :] = -1  # UNLABELABLE sentinel; drop before training

    return pd.DataFrame(
        {
            "label": labels,
            "holding_bars": holding,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
        },
        index=df.index,
    )
