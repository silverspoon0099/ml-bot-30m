"""Two-stage baseline gate — pre-gate (5 features) + full-gate (~202 features).

Spec: Project Spec 30min.md §10.3 (Baseline gates, two stages).

Pre-gate (§10.3.1): exactly 5 hand-picked features (v2.0 column-name mapping)
  ema21_dist_pct, rsi_14, adx, htf4h_ema21_pos, volume_ratio_20.
  Single fold (months [1..9] train, month 10 val).
  PASS = val log-loss / empirical prior ≤ 0.99 (beats prior by ≥1%).
  FAIL = halt the project (premise-level signal — do not tune your way out).

Full-gate (§10.3.2): full v2.0 ~250 feature set with default LightGBM.
  PASS = val_logloss / prior ≤ 0.98 (beats prior by ≥2%).

Empirical prior for 3-class label = -Σ pᵢ log pᵢ where pᵢ = val class frequencies.

Walk-forward purge per §9.2: drop last 12 train bars (= max_holding_bars) so
their label-evaluation windows do not overlap val.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger


REPO = Path(__file__).resolve().parents[1]


def load_btc() -> pd.DataFrame:
    fp = REPO / "data" / "storage" / "features" / "BTC_features.parquet"
    df = pd.read_parquet(fp)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def empirical_prior(y: pd.Series) -> float:
    """Entropy of the empirical class distribution = best constant predictor."""
    p = y.value_counts(normalize=True).sort_index().values
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def pre_gate(
    df: pd.DataFrame,
    features: list[str],
    purge_bars: int = 12,
    seed: int = 42,
) -> dict:
    """Single-fold pre-gate: train months [1..9], val month 10."""
    import lightgbm as lgb
    from sklearn.metrics import log_loss

    # Filter to labelled bars (LONG/SHORT/NEUTRAL)
    df = df[df["label"].isin([0, 1, 2])].copy()

    # Compute calendar-month boundaries from data start
    start = df["dt"].min().normalize()  # midnight UTC of first bar's date
    # Anchor at first day of next full month for clean month boundaries
    if start.day != 1:
        anchor = (start + pd.offsets.MonthBegin(1)).tz_localize(None).tz_localize("UTC")
    else:
        anchor = start
    train_start = anchor
    train_end = train_start + pd.DateOffset(months=9)
    val_end = train_end + pd.DateOffset(months=1)

    train_mask = (df["dt"] >= train_start) & (df["dt"] < train_end)
    val_mask = (df["dt"] >= train_end) & (df["dt"] < val_end)

    train = df[train_mask].copy()
    val = df[val_mask].copy()

    # Purge: drop last `purge_bars` of train (label-evaluation overlaps val)
    train = train.iloc[:-purge_bars] if len(train) > purge_bars else train

    logger.info(
        f"Train window: {train_start.isoformat()} → {train_end.isoformat()} "
        f"(n={len(train):,} after purge -{purge_bars})"
    )
    logger.info(
        f"Val window:   {train_end.isoformat()} → {val_end.isoformat()} "
        f"(n={len(val):,})"
    )

    X_train, y_train = train[features], train["label"]
    X_val, y_val = val[features], val["label"]

    train_dist = y_train.value_counts(normalize=True).sort_index()
    val_dist = y_val.value_counts(normalize=True).sort_index()
    logger.info(f"Train class dist: " + ", ".join(f"{int(k)}={v*100:.2f}%" for k, v in train_dist.items()))
    logger.info(f"Val   class dist: " + ", ".join(f"{int(k)}={v*100:.2f}%" for k, v in val_dist.items()))

    prior = empirical_prior(y_val)
    logger.info(f"Empirical prior (val entropy): {prior:.6f}")

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "verbose": -1,
        "seed": seed,
        "n_jobs": -1,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    proba = model.predict(X_val, num_iteration=model.best_iteration)
    val_logloss = log_loss(y_val, proba, labels=[0, 1, 2])
    ratio = val_logloss / prior
    delta_pct = (1.0 - ratio) * 100.0

    return {
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "val_end": val_end.isoformat(),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "best_iteration": int(model.best_iteration or 0),
        "val_logloss": float(val_logloss),
        "prior": float(prior),
        "ratio": float(ratio),
        "delta_pct": float(delta_pct),
        "pass": bool(ratio <= 0.99),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pre", "full"], default="pre")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg_path = REPO / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    purge_bars = cfg["walk_forward"]["purge_bars"]

    if args.stage == "pre":
        features = [
            "ema21_dist_pct",
            "rsi_14",
            "adx",
            "htf4h_ema21_pos",
            "volume_ratio_20",
        ]
        logger.info(f"Phase 2.1 pre-gate — 5 features: {features}")
        df = load_btc()
        logger.info(f"BTC parquet shape={df.shape}")
        result = pre_gate(df, features, purge_bars=purge_bars, seed=args.seed)
    else:
        raise NotImplementedError("Phase 2.2 full-gate — TODO after pre-gate PASSES")

    logger.info("─" * 70)
    logger.info(f"  best_iteration : {result['best_iteration']}")
    logger.info(f"  n_train        : {result['n_train']:,}")
    logger.info(f"  n_val          : {result['n_val']:,}")
    logger.info(f"  val_logloss    : {result['val_logloss']:.6f}")
    logger.info(f"  empirical_prior: {result['prior']:.6f}")
    logger.info(f"  ratio          : {result['ratio']:.6f}  (gate ≤ 0.99 to PASS)")
    logger.info(f"  delta vs prior : {result['delta_pct']:+.2f}%")
    logger.info("─" * 70)
    if result["pass"]:
        logger.success("Phase 2.1 pre-gate: PASS — proceed to Phase 2.2 full-feature gate")
        return 0
    else:
        logger.error("Phase 2.1 pre-gate: FAIL — halt project per §10.3.1")
        return 1


if __name__ == "__main__":
    sys.exit(main())
