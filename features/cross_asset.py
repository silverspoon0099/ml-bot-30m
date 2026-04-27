"""Cat 22 — Cross-Asset Correlation (7 features) — v2.0.

Per Project Spec 30min §7.2 Cat 22 + Decision v2.11 (BTC corr as booster, not
filter) + Decision v2.35 (universe = BTC/SOL/LINK; TAO deferred) + Decision
v2.41 (ETH correlation DROPPED; ATR-norm formula = difference).

Promoted from v1.0 Phase 4 to Phase 1 for altcoin models (still Phase 3+ for
BTC — BTC has nothing to correlate to within this universe). For SOL/LINK
only; do not call this for BTC.

Phase 1 fires 6 features:
  - btc_corr_20bar              — 20-bar return correlation with BTC
  - btc_return_1bar             — BTC % return over last 1 bar (30 min)
  - btc_return_3bar             — BTC % return over last 3 bars (90 min)
  - btc_return_12bar            — BTC % return over last 12 bars (6 hours)
  - btc_vs_asset_atr_norm_diff  — (btc_move/atr_btc) − (asset_move/atr_asset)
                                  per Decision v2.41 Q10. Signed:
                                  positive = BTC led in vol-units,
                                  negative = asset led / decoupled alt strength,
                                  near-zero = synchronized.
  - btc_above_ema200_daily      — sign(BTC_close_daily − BTC_EMA200_daily),
                                  prev-day-shifted (no look-ahead).
                                  +1 macro-bull, −1 macro-bear.

Phase 3+ adds 1 feature when wired:
  - btc_funding_rate            — gated on Hyperliquid microstructure online.

ETH correlation: PERMANENTLY DROPPED per Decision v2.41 Q9. ETH is not a v2.0
trade asset; correlation feature would have served only as signal-source for
SOL/LINK and added cost > benefit. If reconsidered, requires DR.

Per user's memory: BTC correlation is a *booster, not filter*. Independent
altcoin moves remain tradeable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import safe_div


def cross_asset_features(
    df_asset: pd.DataFrame,
    df_btc: pd.DataFrame,
    atr_asset: pd.Series,
    atr_btc: pd.Series,
    btc_above_ema200_daily: pd.Series | None = None,
    btc_funding: pd.Series | None = None,
    correlation_window: int = 20,
) -> pd.DataFrame:
    """Compute Cat 22 cross-asset features for SOL or LINK.

    Parameters
    ----------
    df_asset : 30m DataFrame for the alt (SOL or LINK). Must have 'close';
               'high'/'low' optional. Index aligned to df_btc.index.
    df_btc : 30m DataFrame for BTC. Must have 'close'. Index aligned to
             df_asset.index. (Caller pre-aligns; this function asserts.)
    atr_asset : ATR(14) series for the alt, aligned to df_asset.index.
    atr_btc : ATR(14) series for BTC, aligned to df_btc.index.
    btc_above_ema200_daily : optional. Caller-supplied series (sign of
        BTC_daily_close − BTC_daily_EMA200, prev-day-shifted) aligned to
        df_asset.index. If None, computed inline from df_btc (requires
        df_btc.index to be a DatetimeIndex).
    btc_funding : optional. Phase 3+ — when supplied, adds 1 feature
        `btc_funding_rate`. Phase 1 leaves this absent.
    correlation_window : window for rolling Pearson correlation (default 20).

    Returns
    -------
    DataFrame indexed like df_asset. Phase 1 = 6 columns; +1 if
    `btc_funding` supplied.

    Raises
    ------
    ValueError if df_asset and df_btc indices do not match.
    """
    if not df_asset.index.equals(df_btc.index):
        raise ValueError(
            "df_asset and df_btc must share the same index (aligned 30m "
            "timestamps). Caller is responsible for alignment."
        )

    asset_close = df_asset["close"]
    btc_close = df_btc["close"]

    # 1. BTC 20-bar return correlation
    asset_returns = asset_close.pct_change()
    btc_returns = btc_close.pct_change()
    btc_corr_20bar = asset_returns.rolling(
        correlation_window, min_periods=correlation_window
    ).corr(btc_returns)

    # 2-4. BTC % returns at 1/3/12 bar lags
    btc_return_1bar = (btc_close / btc_close.shift(1) - 1.0) * 100.0
    btc_return_3bar = (btc_close / btc_close.shift(3) - 1.0) * 100.0
    btc_return_12bar = (btc_close / btc_close.shift(12) - 1.0) * 100.0

    # 5. BTC ATR-normalized move vs asset ATR-normalized move (Decision v2.41 Q10)
    btc_move = btc_close.diff()
    asset_move = asset_close.diff()
    btc_move_atr = safe_div(btc_move, atr_btc)
    asset_move_atr = safe_div(asset_move, atr_asset)
    btc_vs_asset_atr_norm_diff = btc_move_atr - asset_move_atr

    # 6. BTC above/below EMA200 daily — caller-supplied or inline-computed
    if btc_above_ema200_daily is None:
        if not isinstance(df_btc.index, pd.DatetimeIndex):
            # Cannot resample without datetime index; emit NaN sentinel
            btc_above_ema200_daily = pd.Series(np.nan, index=df_asset.index)
        else:
            btc_close_daily = (
                df_btc["close"]
                .resample("1D", label="left", closed="left")
                .last()
            )
            btc_ema200_daily = btc_close_daily.ewm(
                span=200, adjust=False, min_periods=200
            ).mean()
            # prev-day shift to avoid look-ahead (today's 30m bars use yesterday's daily close)
            above_signed = np.sign(btc_close_daily - btc_ema200_daily).shift(1)
            # Map daily values back onto the 30m frame
            day_keys = df_asset.index.floor("1D")
            mapped = above_signed.reindex(day_keys).reset_index(drop=True)
            btc_above_ema200_daily = pd.Series(mapped.values, index=df_asset.index)

    out: dict[str, pd.Series] = {
        "btc_corr_20bar": btc_corr_20bar,
        "btc_return_1bar": btc_return_1bar,
        "btc_return_3bar": btc_return_3bar,
        "btc_return_12bar": btc_return_12bar,
        "btc_vs_asset_atr_norm_diff": btc_vs_asset_atr_norm_diff,
        "btc_above_ema200_daily": btc_above_ema200_daily,
    }

    # Optional Phase 3+ feature
    if btc_funding is not None:
        out["btc_funding_rate"] = btc_funding

    return pd.DataFrame(out, index=df_asset.index)
