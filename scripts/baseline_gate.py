"""Two-stage baseline gate — pre-gate (5 features) + full-gate (~250 features).

Spec: Project Spec 30min.md §10.3 (Baseline gates, two stages) — pre-gate
amended per Decision v2.67 (DR-014) to walk-forward CV across all available
folds (was single-fold; first-fold Feb 2024 ETF rally was a regime-anomalous
outlier).

Pre-gate (§10.3.1, amended): exactly 5 hand-picked features (v2.0 mapping)
  ema21_dist_pct, rsi_14, adx, htf4h_ema21_pos, volume_ratio_20.
  Walk-forward per §9.2 (train_months=9, val_months=1, step_months=1,
  purge=embargo=12). Per fold: fit LightGBM, compute val_logloss + per-fold
  prior. Gate: mean_val_logloss / mean_prior ≤ 0.99.

Full-gate (§10.3.2): full v2.0 feature set with default LightGBM.

Empirical prior for 3-class label = -Σ pᵢ log pᵢ where pᵢ = val class freqs.

Walk-forward purge per §9.2: drop last 12 train bars (= max_holding_bars) so
their label-evaluation windows do not overlap val. Embargo: drop first 12
val bars to preserve symmetry across rolling folds.
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
PRE_GATE_FEATURES = [
    "ema21_dist_pct",
    "rsi_14",
    "adx",
    "htf4h_ema21_pos",
    "volume_ratio_20",
]


NON_FEATURE_COLS = {
    "timestamp", "dt",
    "open", "high", "low", "close", "volume", "quote_volume",
    "label", "exit_price", "holding_bars", "pnl_pct",
}


def load_asset(asset: str) -> pd.DataFrame:
    fp = REPO / "data" / "storage" / "features" / f"{asset}_features.parquet"
    df = pd.read_parquet(fp)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def empirical_prior(y: pd.Series) -> float:
    """Entropy of empirical class distribution = best constant predictor."""
    p = y.value_counts(normalize=True).sort_index().values
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def first_month_start(df: pd.DataFrame) -> pd.Timestamp:
    """First day of the first calendar month after the warmup-trimmed start."""
    start = df["dt"].min().normalize()
    if start.day == 1 and start.hour == 0:
        return start
    return (start.tz_convert("UTC") + pd.offsets.MonthBegin(1)).tz_convert("UTC")


def fit_one_fold(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    seed: int,
    return_importance: bool = False,
) -> dict:
    import lightgbm as lgb
    from sklearn.metrics import log_loss

    X_train, y_train = train[features], train["label"]
    X_val, y_val = val[features], val["label"]

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
    prior = empirical_prior(y_val)
    train_dist = y_train.value_counts(normalize=True).sort_index().to_dict()
    val_dist = y_val.value_counts(normalize=True).sort_index().to_dict()
    out = {
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "best_iteration": int(model.best_iteration or 0),
        "val_logloss": float(val_logloss),
        "prior": float(prior),
        "ratio": float(val_logloss / prior) if prior > 0 else float("nan"),
        "train_dist": {int(k): float(v) for k, v in train_dist.items()},
        "val_dist": {int(k): float(v) for k, v in val_dist.items()},
    }
    if return_importance:
        gain = model.feature_importance(importance_type="gain")
        out["importance"] = dict(zip(features, [float(g) for g in gain]))
    return out


def walk_forward_gate(
    df: pd.DataFrame,
    features: list[str],
    train_months: int = 9,
    val_months: int = 1,
    step_months: int = 1,
    purge_bars: int = 12,
    embargo_bars: int = 12,
    oot_start: pd.Timestamp | None = None,
    seed: int = 42,
    gate_threshold: float = 0.99,
    collect_importance: bool = False,
    label: str = "",
) -> dict:
    """Walk-forward pre-gate per §10.3.1 (amended Decision v2.67).

    Iterates folds while val_end ≤ oot_start. Purge: drop last `purge_bars`
    train rows. Embargo: drop first `embargo_bars` val rows.
    """
    df = df[df["label"].isin([0, 1, 2])].copy()
    base = first_month_start(df)
    folds = []
    k = 0
    while True:
        train_start = base + pd.DateOffset(months=k * step_months)
        train_end = train_start + pd.DateOffset(months=train_months)
        val_end = train_end + pd.DateOffset(months=val_months)
        if oot_start is not None and val_end > oot_start:
            break
        if val_end > df["dt"].max():
            break
        train = df[(df["dt"] >= train_start) & (df["dt"] < train_end)].copy()
        val = df[(df["dt"] >= train_end) & (df["dt"] < val_end)].copy()
        if len(train) <= purge_bars or len(val) <= embargo_bars:
            break
        train = train.iloc[:-purge_bars]
        val = val.iloc[embargo_bars:]
        result = fit_one_fold(train, val, features, seed=seed, return_importance=collect_importance)
        result["fold"] = k
        result["train_start"] = train_start.isoformat()
        result["train_end"] = train_end.isoformat()
        result["val_end"] = val_end.isoformat()
        folds.append(result)
        prefix = f"[{label}] " if label else ""
        logger.info(
            f"{prefix}fold {k:02d} | train {train_start.date()}→{train_end.date()} "
            f"(n={result['n_train']:>5,})  val {train_end.date()}→{val_end.date()} "
            f"(n={result['n_val']:>4,})  best={result['best_iteration']:>3}  "
            f"logloss={result['val_logloss']:.4f}  prior={result['prior']:.4f}  "
            f"ratio={result['ratio']:.4f}"
        )
        k += 1

    if not folds:
        raise RuntimeError("No folds generated — check date config")

    val_loglosses = np.array([f["val_logloss"] for f in folds])
    priors = np.array([f["prior"] for f in folds])
    mean_val_logloss = float(val_loglosses.mean())
    mean_prior = float(priors.mean())
    mean_ratio = float(mean_val_logloss / mean_prior)
    median_ratio = float(np.median(val_loglosses / priors))
    n_pass_folds = int(np.sum(val_loglosses / priors <= gate_threshold))
    n_total_folds = len(folds)

    aggregated_importance: dict[str, float] = {}
    if collect_importance:
        for f in folds:
            for name, gain in f.get("importance", {}).items():
                aggregated_importance[name] = aggregated_importance.get(name, 0.0) + gain

    return {
        "n_folds": n_total_folds,
        "folds": folds,
        "mean_val_logloss": mean_val_logloss,
        "mean_prior": mean_prior,
        "mean_ratio": mean_ratio,
        "median_ratio": median_ratio,
        "n_pass_folds": n_pass_folds,
        "delta_pct": (1.0 - mean_ratio) * 100.0,
        "pass": bool(mean_ratio <= gate_threshold),
        "gate_threshold": gate_threshold,
        "aggregated_importance": aggregated_importance,
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
    embargo_bars = cfg["walk_forward"]["embargo_bars"]
    train_months = cfg["walk_forward"]["train_months"]
    val_months = cfg["walk_forward"]["val_months"]
    step_months = cfg["walk_forward"]["step_months"]
    oot_start = pd.Timestamp(cfg["splits"]["oot_start"]).tz_localize("UTC")

    if args.stage == "pre":
        logger.info(f"Phase 2.1 pre-gate (walk-forward per Decision v2.67) — 5 features")
        df = load_asset("BTC")
        logger.info(f"BTC parquet shape={df.shape}")
        logger.info(
            f"Walk-forward: train={train_months}mo / val={val_months}mo / "
            f"step={step_months}mo / purge={purge_bars} / embargo={embargo_bars} / "
            f"oot_start={oot_start.date()}"
        )
        result = walk_forward_gate(
            df,
            PRE_GATE_FEATURES,
            train_months=train_months,
            val_months=val_months,
            step_months=step_months,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            oot_start=oot_start,
            seed=args.seed,
            gate_threshold=0.99,
            label="BTC",
        )
        logger.info("─" * 78)
        logger.info(f"  n_folds            : {result['n_folds']}")
        logger.info(f"  mean_val_logloss   : {result['mean_val_logloss']:.6f}")
        logger.info(f"  mean_prior         : {result['mean_prior']:.6f}")
        logger.info(f"  mean_ratio         : {result['mean_ratio']:.6f}  (gate ≤ 0.99)")
        logger.info(f"  median_ratio       : {result['median_ratio']:.6f}")
        logger.info(f"  per-fold PASS count: {result['n_pass_folds']} / {result['n_folds']}")
        logger.info(f"  delta vs prior     : {result['delta_pct']:+.2f}%")
        logger.info("─" * 78)
        if result["pass"]:
            logger.success("Phase 2.1 walk-forward pre-gate: PASS")
            return 0
        else:
            logger.error("Phase 2.1 walk-forward pre-gate: FAIL — halt v2.0 per §10.3.1 / Decision v2.67")
            return 1

    # stage == "full" — Phase 2.2 multi-asset full-feature gate per Decision v2.69
    logger.info("Phase 2.2 full-gate (walk-forward per Decision v2.68 + multi-asset per v2.69) — full feature set, gate ≤ 0.98")
    assets = ["BTC", "SOL", "LINK"]
    per_asset: dict[str, dict] = {}
    for asset in assets:
        df = load_asset(asset)
        feats = feature_columns(df)
        logger.info("═" * 78)
        logger.info(f"[{asset}] parquet shape={df.shape}; n_features={len(feats)}")
        logger.info(
            f"[{asset}] Walk-forward: train={train_months}mo / val={val_months}mo / "
            f"step={step_months}mo / purge={purge_bars} / embargo={embargo_bars} / "
            f"oot_start={oot_start.date()}"
        )
        per_asset[asset] = walk_forward_gate(
            df,
            feats,
            train_months=train_months,
            val_months=val_months,
            step_months=step_months,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            oot_start=oot_start,
            seed=args.seed,
            gate_threshold=0.98,
            collect_importance=True,
            label=asset,
        )
        r = per_asset[asset]
        logger.info(
            f"[{asset}] mean_ratio={r['mean_ratio']:.4f} (gate ≤ 0.98)  "
            f"per-fold PASS={r['n_pass_folds']}/{r['n_folds']}  "
            f"delta={r['delta_pct']:+.2f}%  {'PASS' if r['pass'] else 'FAIL'}"
        )

    logger.info("═" * 78)
    logger.info("Phase 2.2 cross-asset summary")
    logger.info(f"  {'asset':<5}  {'n_folds':>7}  {'mean_logloss':>13}  {'mean_prior':>11}  {'mean_ratio':>11}  {'PASS_folds':>10}  {'delta':>7}  {'gate':>5}")
    for a in assets:
        r = per_asset[a]
        verdict = "PASS" if r["pass"] else "FAIL"
        logger.info(
            f"  {a:<5}  {r['n_folds']:>7d}  {r['mean_val_logloss']:>13.6f}  "
            f"{r['mean_prior']:>11.6f}  {r['mean_ratio']:>11.6f}  "
            f"{r['n_pass_folds']:>3}/{r['n_folds']:<6d}  {r['delta_pct']:>+6.2f}%  {verdict:>5}"
        )

    # Per-asset top-20 feature importance
    for a in assets:
        imp = per_asset[a]["aggregated_importance"]
        if not imp:
            continue
        ranked = sorted(imp.items(), key=lambda kv: -kv[1])[:20]
        total = sum(imp.values()) or 1.0
        logger.info(f"\n[{a}] top-20 features by aggregated gain:")
        for i, (name, gain) in enumerate(ranked, 1):
            logger.info(f"  {i:>2}. {name:<35}  gain={gain:>14,.1f}  ({gain/total*100:>5.2f}%)")

    # 4-tier decision matrix per Decision v2.69
    btc_pass = per_asset["BTC"]["pass"]
    alt_passes = [a for a in ("SOL", "LINK") if per_asset[a]["pass"]]
    n_alt_pass = len(alt_passes)

    logger.info("═" * 78)
    if btc_pass and n_alt_pass == 2:
        verdict = "(a) ALL 3 PASS — premise alive; proceed Phase 2.3 with full universe per Decision v2.35"
        rc = 0
    elif not btc_pass and n_alt_pass >= 1:
        verdict = f"(b) BTC FAIL, {n_alt_pass} alt PASS ({','.join(alt_passes)}) — DR-017 candidate: tiered universe (drop BTC)"
        rc = 2
    elif not btc_pass and n_alt_pass == 0:
        verdict = "(c) ALL 3 FAIL — HALT v2.0 with very high confidence; fresh DR for major rework"
        rc = 1
    else:  # BTC pass, ≥1 alt fail
        failing = [a for a in ("SOL", "LINK") if not per_asset[a]["pass"]]
        verdict = f"(d) BTC PASS, alt FAIL ({','.join(failing)}) — diagnose per-asset divergence before halt"
        rc = 3
    logger.info(f"4-tier decision matrix outcome: {verdict}")
    logger.info("═" * 78)
    return rc


if __name__ == "__main__":
    sys.exit(main())
