"""Master feature builder — Phase 1.10d v2.0 orchestration (Decision v2.54).

Loads 30m OHLCV (single-timeframe per Decision v2.28), aggregates 4H + 1D
in-pipeline (per §6.4), computes all 250 v2.0 features across 22 categories
using the Phase 1.10b-locked caller-supplied dependency signatures, applies
the warmup trim, and writes the feature matrix.

Per-symbol feature count:
  - BTC:        243 features (Cat 22 cross-asset SKIPPED — nothing to
                correlate to within v2.0 universe per Decision v2.41)
  - SOL/LINK:   250 features (full 22-category set; Cat 22 = 6 Phase 1
                features; btc_funding_rate Phase 3+)

24-phase dependency-ordered orchestration (Decision v2.54 Q22.8):

  Phase A: Load 30m + aggregate 4H/1D + per-symbol BTC sidecar load
  Phase B: Cat 3  volatility       (atr_14 etc)
  Phase C: Cat 1  momentum         (uses Cat 3 squeeze_state)
  Phase D: Cat 2  trend            (uses Cat 3 atr_14)
  Phase E: Cat 2a HTF context      (4H + 1D shifted-by-1)
  Phase F: Cat 22 cross-asset      (alts only)
  Phase G: Cat 4  volume
  Phase H: Cat 14 money flow
  Phase I: Cat 6  pivots           (uses Cat 3 atr_14)
  Phase J: Cat 5  vwap             (uses Cat 3 atr_14 + Cat 6 daily_pivot_p)
  Phase K: Cat 7  sessions
  Phase L: Cat 8  candles
  Phase M: Cat 9  mean reversion   (uses Cat 3 bb_position)
  Phase N: Cat 10 regime           (uses Cat 2 adx/di + Cat 3 atr_14/atr_pct
                                    + inline bb_width_percentile)
  Phase O: Cat 11 prev context
  Phase P: Cat 12 lagged dynamics  (uses Cat 1 rsi + Cat 2 adx + ema21_dist_pct)
  Phase Q: Cat 13 divergence       (uses Cat 1 rsi + macd_hist)
  Phase R: Cat 15 extra momentum
  Phase S: Cat 16 structure
  Phase T: Cat 17 fractal stats
  Phase U: Cat 18 adaptive MA
  Phase V: Cat 19 ichimoku
  Phase W: Cat 20 event memory     (full dependency consumer — Cat 1 rsi,
                                    stoch_k, wt1, squeeze_value, macd_line,
                                    macd_signal; Cat 2 adx; Cat 3 squeeze_state;
                                    Cat 6 daily_pivot_levels, weekly_pivot_levels)
  Phase X: Concat + insert OHLC + warmup trim + (optional) labels

Usage:
    python -m features.builder --symbol BTC/USDT
    python -m features.builder --symbol SOL/USDT
    python -m features.builder --symbol LINK/USDT --no-label
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.collectors.storage import ohlcv_path, read_ohlcv
from model.labeler import triple_barrier_labels
from utils.config import load_config, resolve_path
from utils.logging_setup import get_logger

from . import (
    adaptive_ma,
    candles,
    context,
    cross_asset,
    divergence,
    event_memory,
    extra_momentum,
    htf_context,
    ichimoku,
    momentum_core,
    pivots,
    regime,
    sessions,
    stats,
    structure,
    trend,
    volatility,
    volume,
    vwap,
)
from ._common import rolling_percentile

log = get_logger("builder")

V20_HTF_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def coin_from_symbol(symbol: str) -> str:
    return symbol.split("/")[0].upper()


def load_ohlcv_30m(symbol: str, storage_dir: Path) -> pd.DataFrame:
    """Load 30m OHLCV per Decision v2.28 (sole fetched timeframe)."""
    p = ohlcv_path(storage_dir, symbol, "30m")
    df = read_ohlcv(p)
    if df.empty:
        raise FileNotFoundError(f"No 30m data at {p} — run the fetcher first.")
    df["timestamp"] = pd.to_numeric(df["timestamp"], downcast="integer")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def to_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with DatetimeIndex (tz-aware UTC) from `timestamp` (ms).

    Most v2.0 feature modules require a DatetimeIndex (UTC-day grouping,
    week boundaries, etc.).
    """
    out = df.copy()
    out.index = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    out.index.name = "timestamp_dt"
    return out


def aggregate_htf(df_30m_indexed: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate 30m → 4H or 1D per §6.4 (label='left', closed='left').

    A 4H bar = 8 consecutive 30m bars; a 1D bar = 48 consecutive 30m bars.
    """
    return (
        df_30m_indexed[["open", "high", "low", "close", "volume"]]
        .resample(freq, label="left", closed="left")
        .agg(V20_HTF_AGG)
    )


def merge_htf_into_30m(
    df_30m_indexed: pd.DataFrame,
    htf_features: pd.DataFrame,
    htf_freq: str,
) -> pd.DataFrame:
    """Map HTF features onto 30m grid via floor-key reindex with shift(1).

    shift(1) ensures look-ahead safety: a 30m bar at hour H sees the HTF
    bar that closed at H, not at H + period.
    """
    lookup = htf_features.shift(1)
    floor_key = df_30m_indexed.index.floor(htf_freq)
    mapped = lookup.reindex(floor_key)
    mapped.index = df_30m_indexed.index
    return mapped


def build_features(symbol: str, cfg: dict) -> pd.DataFrame:
    bcfg = cfg["data"]["binance"]
    fcfg = cfg["features"]
    # CORRECTION (Decision v2.54 Q22.1 + VPS Claude validation 2026-05-01):
    # cfg points at data/storage/binance/; export_parquet.py writes parquet
    # snapshots one level deeper at data/storage/binance/30m/<SYM>_30m.parquet
    # (per spec §15 canonical layout). storage.ohlcv_path() appends only the
    # filename, not the timeframe subdir. Prepend "/30m" here so both the
    # primary asset load and the BTC sidecar load (alts only) hit the right
    # snapshot directory.
    storage_dir = resolve_path(bcfg["storage_dir"]) / "30m"
    coin = coin_from_symbol(symbol)
    is_btc = coin == "BTC"

    # ── Phase A: Load 30m + aggregate 4H/1D + (alts) BTC sidecar ──────
    log.info(f"Phase A: Loading 30m OHLCV for {symbol}")
    df_30m_raw = load_ohlcv_30m(symbol, storage_dir)
    log.info(f"  30m bars: {len(df_30m_raw):,}")
    df = to_dt_index(df_30m_raw)

    log.info("Phase A: Aggregating 30m → 4H + 1D in-pipeline (§6.4)")
    df_4h = aggregate_htf(df, "4h")
    df_1d = aggregate_htf(df, "1D")

    df_btc_aligned: pd.DataFrame | None = None
    if not is_btc:
        log.info(f"Phase A: Loading BTC/USDT 30m sidecar for Cat 22 cross-asset ({coin})")
        df_btc_raw = load_ohlcv_30m("BTC/USDT", storage_dir)
        df_btc = to_dt_index(df_btc_raw)
        # Inner-align BTC to asset index — drop bars where either is missing.
        df_btc_aligned = df_btc.reindex(df.index)
        if df_btc_aligned["close"].isna().any():
            n_missing = df_btc_aligned["close"].isna().sum()
            log.warning(f"  BTC alignment: {n_missing} bars missing — Cat 22 will produce NaN for those rows")

    parts: list[pd.DataFrame] = []

    # ── Phase B: Cat 3 Volatility (12 features) ───────────────────────
    log.info("Phase B: Cat 3 volatility (12 features)")
    vol_df = volatility.volatility_features(df, fcfg)
    parts.append(vol_df)

    # ── Phase C: Cat 1 Momentum (32 features) ─────────────────────────
    log.info("Phase C: Cat 1 momentum_core (32 features)")
    mom_df = momentum_core.momentum_core_features(
        df,
        fcfg,
        squeeze_state=vol_df["squeeze_state"],
    )
    parts.append(mom_df)

    # ── Phase D: Cat 2 Trend (14 features) ────────────────────────────
    log.info("Phase D: Cat 2 trend (14 features)")
    trend_df = trend.trend_features(df, vol_df["atr_14"], fcfg)
    parts.append(trend_df)

    # ── Phase E: Cat 2a HTF Context (18 features) ─────────────────────
    log.info("Phase E: Cat 2a HTF context (18 features = 9 4H + 9 1D)")
    htf_4h_feats, htf_1d_feats = htf_context.htf_context_features(df_4h, df_1d, fcfg)
    htf_4h_mapped = merge_htf_into_30m(df, htf_4h_feats, "4h")
    htf_1d_mapped = merge_htf_into_30m(df, htf_1d_feats, "1D")
    parts.append(htf_4h_mapped)
    parts.append(htf_1d_mapped)

    # ── Phase F: Cat 22 Cross-Asset (6 features, alts only) ───────────
    if is_btc:
        log.info("Phase F: Cat 22 cross-asset SKIPPED for BTC (no self-correlation)")
    else:
        log.info("Phase F: Cat 22 cross-asset (6 features)")
        atr_btc = volatility.atr(
            df_btc_aligned["high"],
            df_btc_aligned["low"],
            df_btc_aligned["close"],
            14,
        )
        cross_df = cross_asset.cross_asset_features(
            df_asset=df,
            df_btc=df_btc_aligned,
            atr_asset=vol_df["atr_14"],
            atr_btc=atr_btc,
        )
        parts.append(cross_df)

    # ── Phase G: Cat 4 Volume (12 features) ───────────────────────────
    log.info("Phase G: Cat 4 volume (12 features)")
    vol4_df = volume.volume_features(df, fcfg)
    parts.append(vol4_df)

    # ── Phase H: Cat 14 Money Flow (6 features) ───────────────────────
    log.info("Phase H: Cat 14 money flow (6 features)")
    mf_df = volume.money_flow_features(df, fcfg)
    parts.append(mf_df)

    # ── Phase I: Cat 6 Pivots (30 features) ───────────────────────────
    log.info("Phase I: Cat 6 pivots (30 features)")
    pivots_df = pivots.pivot_features(df, vol_df["atr_14"], fcfg)
    parts.append(pivots_df)

    # Pivot levels for downstream consumers (vwap + event_memory).
    daily_levels, weekly_levels = pivots.compute_pivot_levels(df)
    daily_pivot_p = daily_levels["P"]

    # ── Phase J: Cat 5 VWAP (14 features) ─────────────────────────────
    log.info("Phase J: Cat 5 vwap (14 features, multi-anchor)")
    vwap_df = vwap.vwap_features(df, vol_df["atr_14"], daily_pivot_p, fcfg)
    parts.append(vwap_df)

    # ── Phase K: Cat 7 Sessions (9 features) ──────────────────────────
    log.info("Phase K: Cat 7 sessions (9 features)")
    sess_df = sessions.session_features(df)
    parts.append(sess_df)

    # ── Phase L: Cat 8 Candles (9 features) ───────────────────────────
    log.info("Phase L: Cat 8 candles (9 features)")
    candle_df = candles.candle_features(df, fcfg)
    parts.append(candle_df)

    # ── Phase M: Cat 9 Mean Reversion (7 features) ────────────────────
    log.info("Phase M: Cat 9 mean reversion (7 features)")
    mr_df = stats.mean_reversion_features(df, vol_df["bb_position"], fcfg)
    parts.append(mr_df)

    # ── Phase N: Cat 10 Regime (7 features) ───────────────────────────
    log.info("Phase N: Cat 10 regime (7 features)")
    bb_width_percentile = rolling_percentile(vol_df["bb_width_pct"], 100)
    reg_df = regime.regime_features(
        df,
        adx=trend_df["adx"],
        di_plus=trend_df["di_plus"],
        di_minus=trend_df["di_minus"],
        atr_14=vol_df["atr_14"],
        atr_percentile=vol_df["atr_percentile"],
        bb_width_percentile=bb_width_percentile,
        cfg=fcfg,
    )
    parts.append(reg_df)

    # ── Phase O: Cat 11 Previous Context (6 features) ────────────────
    log.info("Phase O: Cat 11 prev context (6 features, MIXED 4 static + 2 dynamic)")
    prev_df = context.prev_context_features(df, fcfg)
    parts.append(prev_df)

    # ── Phase P: Cat 12 Lagged Dynamics (5 features) ──────────────────
    log.info("Phase P: Cat 12 lagged dynamics (5 features)")
    lag_df = context.lagged_dynamics_features(
        df,
        rsi=mom_df["rsi_14"],
        adx=trend_df["adx"],
        ema21_dist_pct=trend_df["ema21_dist_pct"],
        cfg=fcfg,
    )
    parts.append(lag_df)

    # ── Phase Q: Cat 13 Divergence (7 features) ───────────────────────
    log.info("Phase Q: Cat 13 divergence (7 features)")
    div_df = divergence.divergence_features(
        df,
        rsi_series=mom_df["rsi_14"],
        macd_hist=mom_df["macd_hist"],
    )
    parts.append(div_df)

    # ── Phase R: Cat 15 Extra Momentum (7 features) ───────────────────
    log.info("Phase R: Cat 15 extra momentum (7 features)")
    em_df = extra_momentum.extra_momentum_features(df, fcfg)
    parts.append(em_df)

    # ── Phase S: Cat 16 Structure (10 features, MIXED 8 static + 2 dynamic) ─
    log.info("Phase S: Cat 16 structure (10 features)")
    struct_df = structure.structure_features(df, fcfg)
    parts.append(struct_df)

    # ── Phase T: Cat 17 Fractal Stats (6 features) ────────────────────
    log.info("Phase T: Cat 17 fractal stats (6 features)")
    frac_df = stats.fractal_stats_features(df, fcfg)
    parts.append(frac_df)

    # ── Phase U: Cat 18 Adaptive MA (4 features) ──────────────────────
    log.info("Phase U: Cat 18 adaptive MA (4 features)")
    am_df = adaptive_ma.adaptive_ma_features(df, fcfg)
    parts.append(am_df)

    # ── Phase V: Cat 19 Ichimoku (6 features) ─────────────────────────
    log.info("Phase V: Cat 19 ichimoku (6 features)")
    ich_df = ichimoku.ichimoku_features(df, fcfg)
    parts.append(ich_df)

    # ── Phase W: Cat 20 Event Memory (22 features — full dep consumer) ─
    log.info("Phase W: Cat 20 event memory (22 features)")
    ev_df = event_memory.event_memory_features(
        df,
        rsi=mom_df["rsi_14"],
        stoch_k=mom_df["stoch_k"],
        wt1=mom_df["wt1"],
        adx=trend_df["adx"],
        macd_line=mom_df["macd_line"],
        macd_signal=mom_df["macd_signal"],
        squeeze_state=vol_df["squeeze_state"],
        squeeze_value=mom_df["squeeze_value"],
        daily_pivot_levels=daily_levels,
        weekly_pivot_levels=weekly_levels,
        cfg=fcfg,
    )
    parts.append(ev_df)

    # ── Phase X: Concat + insert OHLC ─────────────────────────────────
    log.info("Phase X: Assembling feature matrix")
    feature_matrix = pd.concat(parts, axis=1)
    feature_matrix = feature_matrix.reset_index(drop=True)
    feature_matrix.insert(0, "timestamp", df_30m_raw["timestamp"].values)
    feature_matrix.insert(1, "open", df_30m_raw["open"].values)
    feature_matrix.insert(2, "high", df_30m_raw["high"].values)
    feature_matrix.insert(3, "low", df_30m_raw["low"].values)
    feature_matrix.insert(4, "close", df_30m_raw["close"].values)
    feature_matrix.insert(5, "volume", df_30m_raw["volume"].values)

    n_features = feature_matrix.shape[1] - 6
    expected = 243 if is_btc else 250
    log.info(
        f"Feature matrix shape: {feature_matrix.shape}  "
        f"(features: {n_features}, expected: {expected})"
    )
    if n_features != expected:
        log.warning(
            f"Feature count {n_features} differs from expected {expected} — "
            f"audit category outputs"
        )
    return feature_matrix


def add_labels(feature_matrix: pd.DataFrame, symbol: str, cfg: dict) -> pd.DataFrame:
    lcfg = cfg["labeling"]
    coin = coin_from_symbol(symbol)
    min_profit = lcfg["min_profit_pct"].get(coin, 0.5)
    log.info(
        f"Triple-barrier labels: tp={lcfg['tp_atr_mult']}x, sl={lcfg['sl_atr_mult']}x, "
        f"hold={lcfg['max_holding_bars']}, min_profit_pct={min_profit}"
    )
    labels = triple_barrier_labels(
        feature_matrix,
        atr_col="atr_14",
        tp_atr_mult=lcfg["tp_atr_mult"],
        sl_atr_mult=lcfg["sl_atr_mult"],
        max_holding_bars=lcfg["max_holding_bars"],
        min_profit_pct=min_profit,
        classes=lcfg["classes"],
    )
    out = pd.concat([feature_matrix, labels], axis=1)
    dist = out["label"].value_counts().to_dict()
    log.info(f"Label distribution: {dist}")
    return out


def trim_warmup(df: pd.DataFrame, warmup: int) -> pd.DataFrame:
    """Drop the first `warmup` rows (longest indicator lookback per §7.6)."""
    return df.iloc[warmup:].reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--no-label", action="store_true")
    parser.add_argument("--output-dir", default=cfg["features"]["output_dir"])
    args = parser.parse_args()

    matrix = build_features(args.symbol, cfg)
    matrix = trim_warmup(matrix, cfg["features"]["warmup_bars"])

    if not args.no_label:
        matrix = add_labels(matrix, args.symbol, cfg)
        # Drop unlabelable tail rows (where the labeler ran out of forward-bars).
        matrix = matrix[matrix["label"] != -1].reset_index(drop=True)

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coin = coin_from_symbol(args.symbol)
    out_path = out_dir / f"{coin}_features.parquet"
    matrix.to_parquet(out_path, index=False)
    log.info(f"Saved {len(matrix):,} rows × {matrix.shape[1]} cols -> {out_path}")


if __name__ == "__main__":
    main()
