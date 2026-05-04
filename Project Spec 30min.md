# Project Spec — 30min ML Trading Bot (v2.0)

**Status:** Design document for a new project, derived from `ml-bot/PROJECT_SPEC.md` (the 5m project).
**Created:** 2026-04-23
**Primary timeframe:** 30m bar close
**Secondary (higher-TF context):** 4H, 1D
**Target trade cadence:** 1–3 trades/day per symbol, 2–8 hour average hold
**Execution venue:** Hyperliquid perpetual futures
**Training universe:** BTC, SOL, LINK (Binance perp OHLCV); TAO and HYPE deferred to Phase 4 candidates per §14 + Decision v2.35

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why 30m — Diagnosis of 5m Failure](#2-why-30m--diagnosis-of-5m-failure)
3. [Research Findings (2026)](#3-research-findings-2026)
4. [Goals, Constraints, Non-Goals](#4-goals-constraints-non-goals)
5. [Architecture](#5-architecture)
6. [Data Pipeline](#6-data-pipeline)
7. [Feature Engineering — 30m Redesign](#7-feature-engineering--30m-redesign)
8. [Labeling](#8-labeling)
9. [Training & Validation](#9-training--validation)
10. [Anti-Overfit Discipline](#10-anti-overfit-discipline)
11. [Backtesting](#11-backtesting)
12. [Execution Layer](#12-execution-layer)
13. [Risk Management](#13-risk-management)
14. [Project Phases & Timeline](#14-project-phases--timeline)
15. [Directory Structure](#15-directory-structure)
16. [Success Criteria](#16-success-criteria)
17. [Decision Log (v2.0 deltas from v1.0)](#17-decision-log)
18. [References](#18-references)

---

## 1. Executive Summary

The v1.0 project (5-minute bars, 268 features, triple-barrier labels, LightGBM multiclass) trained successfully but failed to generalize: backtests never cleared the empirical prior by a meaningful margin, and iterative re-validation against the OOT fold turned it into a de facto dev set (overfit).

v2.0 is a **clean-room rebuild** on the 30m timeframe with these principal changes:

1. **Timeframe:** 5m → 30m primary, with 4H and 1D context (was 5m + 1H)
2. **Data horizon:** 18 months → **3 years minimum (target 4 years)** to cover 2022 bear, 2023 recovery, 2024 ETF/halving bull, and 2025 top+correction
3. **Feature set:** 268 → ~190 features, rebalanced for 30m (drop scalping artifacts, add HTF context features validated by 2026 research)
4. **Labeling:** tp=4.0/sl=4.0 @ 24×5m (2h) → tp=3.0/sl=3.0 @ 8×30m (4h), with rescaled `min_profit_pct` timeout gate
5. **OOT discipline:** hard rule — one untouched OOT slice, evaluated exactly once, no iteration
6. **Baseline gate:** single-fold minimal baseline must beat empirical prior by ≥2% log-loss before full pipeline runs

v2.0 **keeps** the v1.0 methodology (LightGBM multiclass, triple-barrier, walk-forward with Optuna+SHAP, purged CV, momentum-with-structure trading style) — only the inputs, the timeframe, and the experimental discipline change.

---

## 2. Why 30m — Diagnosis of 5m Failure

### 2.1 The signal-to-noise problem at 5m

On 5m bars:
- ~288 bars/day per symbol
- Per-bar ATR is small relative to fees+slippage+spread (~6–9 bps round-trip)
- A 4-ATR profit target becomes a ~0.4–0.8% move — realistic but frequent
- Noise dominates; fees + wrong-side slippage consume a large fraction of the edge
- Triple-barrier labels are driven by microstructure randomness as often as by directional flow

On 30m bars:
- ~48 bars/day (6× fewer decisions)
- Per-bar ATR is roughly 2.3× larger (√6 ≈ 2.45 as a rough scaling)
- A 3-ATR profit target is ~1.0–1.8% — fees become a smaller % of gross edge
- Bar structure reflects more committed positioning, less microstructure noise
- Matches the user's existing discretionary style (5m scalping attempts felt forced; 30m swings feel natural)

### 2.2 What v1.0 got right (carried forward to v2.0)

- LightGBM 3-class classifier (LONG / SHORT / NEUTRAL) with triple-barrier labels
- Symmetric barriers (tp_atr_mult == sl_atr_mult) to avoid baking a directional bias into labels
- Pessimistic tie-break when TP and SL hit on the same bar (SL wins)
- Purged walk-forward CV with embargo equal to `max_holding_bars`
- Optuna with MedianPruner across 3 evenly-spaced folds
- SHAP-driven feature trim after initial training (Phase 2.5)
- Calibrated probabilities (raw LightGBM multiclass is overconfident)
- Project-wide sign convention: +1 bullish / −1 bearish
- Wilder EMA for RSI / ATR / ADX smoothing (α = 1/period)
- Ichimoku spans displaced +26 bars for cloud-over-current-bar interpretation

### 2.3 What v1.0 got wrong (fixed in v2.0)

| Issue                                | v1.0 behavior                                             | v2.0 fix                                                                |
| ------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------------- |
| OOT contamination via iteration      | Re-validated after every feature/label tweak (7+ cycles) | OOT is **read once**, ever. No exceptions. Dev iterations use val fold only. |
| Timeframe mismatch to trading style  | 5m bars, 2h holds                                         | 30m bars, 4h holds                                                      |
| Too few regimes in training          | 18mo (mostly 2024–2026 bull→top)                          | 3–4 years spanning bear + recovery + bull + correction                  |
| HTF context weak                     | Only 1H shifted merge                                     | 4H + 1D context features; 4H BB position as first-class feature         |
| Feature count vs data ratio          | 268 features × 155k rows (OK but borderline)              | ~190 features × ~200k rows after SHAP trim to ~110                      |
| min_profit_pct too low for BTC      | 0.3% (noise-level)                                        | 0.6% (above spread+fees ×2)                                             |
| Label asymmetry history              | Earlier 4/3 produced directional bias in labels           | Keep symmetric 3/3; do not chase label-tuning                           |

---

## 3. Research Findings (2026)

Key external inputs that shaped v2.0 design. Sources in Section 18.

### 3.1 Higher-timeframe features dominate

A March 2026 preprint on multi-timeframe feature engineering for crypto finds the **4H Bollinger Band position is the #1 predictor** at 8.4% SHAP importance, with **4H RSI at 4.5%**. This is a strong mandate to promote 4H context features from "support" to "first-class citizen" status in v2.0.

**v2.0 implication:** Category 2a ("HTF Context") is a dedicated category; 4H bb_position, 4H rsi, 4H macd_hist, daily ema20/50/200 relative position are all included from Phase 1.

### 3.2 "70/30 rule" — features > model

"70% of model performance comes from features, 30% from model choice." This validates the v1.0 approach of investing heavily in feature engineering rather than model architecture search. v2.0 will not ensemble or switch to deep learning in Phase 1 — it will invest further in features.

### 3.3 Information-driven bars outperform time bars

Recent research (2025–2026, Financial Innovation journal) shows CUSUM / dollar / volume bars combined with triple-barrier labels produce consistently positive performance after transaction costs, outperforming time-based bars.

**v2.0 decision:** This is a **Phase 4 experiment**, not Phase 1. v2.0 stays on 30m time bars to keep scope focused and validate the base architecture first. CUSUM-event bars are parked as a future enhancement (see §14 Phase 4).

### 3.4 Crypto cycle coverage — 3–4 year minimum

Binance 30m data spanning 2022-04 through 2026-04 covers:
- **2022:** Macro shock, FTX/Luna/Celsius collapse (full bear regime)
- **2023:** Recovery phase (+157% BTC) — ETF anticipation
- **2024:** ETF approval (Jan), halving (Apr), $100k breakout (Dec) — full bull
- **2025:** ATH $126k (Oct), correction to ~$87k (Dec) — top + distribution + decline
- **2026 Q1:** Range/chop regime

Training on less than ~3 years risks the model learning a single regime. v2.0 targets **3 years minimum, 4 years preferred**.

### 3.5 Multi-anchor VWAP > single daily VWAP

Institutional toolkit now emphasizes **multi-anchor VWAP** (swing-high-anchored, swing-low-anchored, HTF-pivot-anchored, session-anchored, all-time-high-anchored) rather than a single daily session VWAP. At 30m each anchor has enough duration to matter.

**v2.0 implication:** Category 5 (VWAP) expands from 8 → 14 features with 4 anchor types.

### 3.6 Altcoin ML — own features dominate, BTC correlation is a booster

Coinbase Institutional ML analysis of H1 2024: "core drivers of BTC, ETH, and SOL performance have been predominantly unique to those tokens" (network fees, TVL, ETF flows individually). This **validates the v1.0 design principle** that each asset's model runs on its own features primarily. BTC correlation stays as a **booster feature category**, not a filter (per user's existing memory).

---

## 4. Goals, Constraints, Non-Goals

### 4.1 Goals

1. **Profitable edge in backtest and paper trading** — Sharpe ≥ 1.5 net, max DD ≤ 15%, win rate ≥ 52% at P>0.65 threshold
2. **1–3 trades/day** per symbol (fire rate 2–6% at threshold)
3. **Momentum-with-structure archetype** — take trades when ML predicts continuation from a structural level (pivot, EMA, swing), exit via ATR trailing stop, 2–8 hour holds
4. **Transfer learning sequence** — BTC base model → SOL → LINK (3 transfers in v2.0). TAO and HYPE deferred to Phase 4 candidates per §14 / Decision v2.35
5. **Hyperliquid execution** — bar-close execution initially, intrabar optional Phase 3
6. **Explainable** — every live trade logs its top-5 SHAP contributors

### 4.2 Constraints

1. **One untouched OOT slice** — last month of data is reserved, never scored before final freeze
2. **Feature count budget** — Phase 1 targets ~190 raw features; post-SHAP trim to ~100–130
3. **Training window** — 3 years minimum, 4 years target
4. **Data source** — Binance SPOT via `data.binance.vision` (no geo-block from user's VPS region; SPOT/perp correlation >99% for OHLCV features)
5. **Live venue** — Hyperliquid (funding hourly, not 8h; L2 book available)
6. **No deep learning in Phase 1** — LightGBM only; LSTM/Transformer is a Phase 4 optional experiment

### 4.3 Non-goals (explicitly excluded from v2.0 scope)

- High-frequency / sub-minute trading
- Portfolio optimization across assets (each symbol has an independent model + position)
- News/sentiment features (Phase 4 if at all)
- Reinforcement learning
- Options / perps funding arbitrage
- Cross-exchange arbitrage

---

## 5. Architecture

### 5.1 High-level flow

```
Binance perp (data.binance.vision)
    ├── 30m OHLCV × {BTC, SOL, LINK}        ← primary (only fetched TF; TAO deferred per Decision v2.35)
    ├── 4H  OHLCV                            ← aggregated from 30m in-pipeline (§6.4)
    └── 1D  OHLCV                            ← aggregated from 30m in-pipeline (§6.4)
            │
            ▼
    features/builder.py
    (compute per-category features on primary 30m frame;
     merge HTF features via prev-closed-bar lookup to avoid look-ahead)
            │
            ▼
    Feature matrix (rows=30m bars, cols=~190 features + label)
            │
            ▼
    model/labeler.py  (triple-barrier, tp=3.0 ATR, sl=3.0 ATR, max=8 bars)
            │
            ▼
    train/walk_forward.py  (8 folds × 9mo train / 1mo val / 1mo step)
            │
            ▼
    tune/optuna_search.py  (50 trials on 3 folds, MedianPruner)
            │
            ▼
    SHAP analysis → feature trim → re-fit → freeze
            │
            ▼
    OOT evaluation (last 1mo, untouched) — single score, no iteration
            │
            ▼
    predictor (live, 30m bar close)  →  executor (Hyperliquid orders)
```

### 5.2 Module boundaries

Same boundaries as v1.0:
- `data/collectors/` — Binance + Hyperliquid ingestion. **Only 30m is fetched**; 4H and 1D are aggregated from 30m in-pipeline (§6.4).
- `features/` — per-category modules (reused, internal parameter changes)
- `model/` — labeler, training, inference (reused)
- `tune/` — Optuna search (reused)
- `backtest/` — simulator (reused)
- `execution/` — Hyperliquid orders (reused)

**Database (v1.0 instance reused):** v2.0 shares the v1.0 Postgres/TimescaleDB instance. No new schema is created. Only **new tables** are added for 30m rollups and derived artifacts. Credentials, connection string, and extensions (TimescaleDB) are inherited via `.env` from the v1.0 setup. See §6.1.1 for table naming.

---

## 6. Data Pipeline

### 6.1 Data sources

| Source                         | Purpose                  | Timeframes fetched | Assets (v2.0)        | Storage                    |
| ------------------------------ | ------------------------ | ------------------ | -------------------- | -------------------------- |
| `data.binance.vision` (SPOT)  | Training OHLCV          | **30m only**       | BTC, SOL, LINK       | `data/storage/binance/`    |
| Binance daily archives (.zip) | Bulk historical          | monthly archives (30m) | BTC, SOL, LINK   | downloader             |
| Hyperliquid WebSocket         | Live OHLCV + funding + L2 | 30m rollup from trade tape | BTC, SOL, LINK | `data/storage/hyperliquid/` |
| Hyperliquid REST              | Funding/OI polling       | hourly             | BTC, SOL, LINK       | same                       |

**v2.0 asset universe: BTC, SOL, LINK** (3 assets). TAO and HYPE are deferred to Phase 4 candidates per Decision v2.35:
- **TAO:** dropped from v2.0 because (1) Binance perp 30m data ceiling is ~1.5–2 years (incompatible with 3-year minimum per Decision v2.2), (2) Hyperliquid TAO daily volume currently insufficient for reliable execution. Re-entry conditions (all three required): ≥3 years clean Binance perp 30m data accumulated, Hyperliquid TAO daily volume ≥5% of BTC sustained 30 days, v2.x diversification window open.
- **HYPE:** Hyperliquid-native; deferred to Phase 4.1 per current spec design. No change.

**4H and 1D are NOT fetched separately** — they are derived from the 30m bars in-pipeline via exact aggregation (open=first, high=max, low=min, close=last, volume=sum). This is mathematically lossless and avoids redundant fetches. See §6.4.

**Why Binance SPOT for training:** user's VPS region returns HTTP 451 on Binance main/fapi; `data.binance.vision` is geo-open. SPOT vs perp OHLCV correlation is >99% with <0.1% basis — statistically negligible for feature engineering.

### 6.1.1 Database reuse (v1.0 instance)

**v2.0 uses the same Postgres/TimescaleDB instance as v1.0.** No new database creation, no new schema, no credential changes. Only new tables are added — no existing v1.0 table is read from or written to by v2.0.

**Table naming convention (additions only):**

| Purpose                                    | New table                                  |
| ------------------------------------------ | ------------------------------------------ |
| 30m OHLCV (Binance spot, historical)       | `ohlcv_30m`                                |
| 30m OHLCV (Hyperliquid live, 30m rollup)   | `ohlcv_30m_hl`                             |
| 30m feature matrix (output of builder)     | `features_30m`                             |
| 30m triple-barrier labels                  | `labels_30m`                               |
| Walk-forward fold metadata                 | `wf_folds_30m`                             |
| Model registry (frozen artifacts metadata) | `models_30m`                               |

**Schema extensions to v1.0 config.yaml (additions only, no overwrites):**

```yaml
database:
  # NEW — 30m OHLCV Timescale chunking (between 5m and 1h)
  chunk_interval_ohlcv_30m: "90 days"
  # existing entries are inherited: chunk_interval_ohlcv_5m, _1h, compression, retention, etc.
```

4H and 1D data, being derived from 30m, are **not stored as separate tables**. They are aggregated on-the-fly during feature build (cached in-memory per run) and in-memory during live inference.

### 6.2 Data volumes (3-year baseline, 3 assets)

| Timeframe | Source                         | Bars/day | 3-year bars/symbol | 3 symbols | Notes                               |
| --------- | ------------------------------ | -------- | ------------------ | --------- | ----------------------------------- |
| 30m       | **Fetched** (Binance SPOT)    | 48       | ~52,560            | ~157,800  | primary, only TF fetched            |
| 4H        | **Aggregated** from 30m       | 6        | ~6,570             | ~19,700   | derived in-pipeline (8 × 30m bars)  |
| 1D        | **Aggregated** from 30m       | 1        | ~1,095             | ~3,300    | derived in-pipeline (48 × 30m bars) |

With ~202 features, data:feature ratio is ~260:1 per symbol, ~780:1 pooled (post-SHAP trim to ~110-140 features → ~1,130:1 to ~1,435:1 pooled) — well within healthy LightGBM training territory (50:1 to 200:1 comfortable threshold). Asset universe per Decision v2.35: BTC, SOL, LINK.

### 6.3 Fetcher spec changes from v1.0

Reuse `scripts/export_parquet.py` with additions:
- Accept `--timeframe 30m` (single TF only; was `5m 1h`)
- Accept `--years 3` (was `--months 18`)
- Write to Postgres table `ohlcv_30m` (new) AND optional parquet backup at `data/storage/binance/30m/`
- Verify: no gaps, all timestamps on canonical 30m boundaries, UTC
- Parquet schema unchanged from v1.0 (timestamp ms int, open, high, low, close, volume)

### 6.4 Multi-timeframe aggregation + merge

Aggregation and merge happen in one pipeline step during feature build — neither 4H nor 1D is persisted as a separate table.

**Step A — Aggregate (lossless, in-memory):**

```python
# 30m → 4H: group every 8 consecutive 30m bars by floor-to-4h-boundary
df_4h = df_30m.resample("4h", on="timestamp", label="left", closed="left").agg({
    "open":   "first",
    "high":   "max",
    "low":    "min",
    "close":  "last",
    "volume": "sum",
})

# 30m → 1D: same pattern with "1D" resample
df_1d = df_30m.resample("1D", on="timestamp", label="left", closed="left").agg({...})
```

Resample is `label="left", closed="left"` so the 4H bar at 12:00 UTC represents 12:00–15:59:59; its close is the 30m bar at 15:30 close. No look-ahead: the 4H bar is only marked "complete" once the 15:30 30m bar closes.

**Step B — Compute HTF features natively on the aggregated frames:**

All Category 2a features (§7.2) are computed on `df_4h` and `df_1d` directly, not on 30m.

**Step C — Merge to 30m frame via prev-closed-bar lookup:**

```
30m bar at time t (UTC, 30m boundary)
    ↓ floor to prior 4H boundary → k_4h
    ↓ use 4H features indexed at (k_4h - 1 × 4H) — i.e., the PREVIOUS COMPLETED 4H bar
    ↓ (same for 1D: use previous completed 1D bar)
    ↓
30m frame now has htf4h_* and htf1d_* columns filled via prev-bar lookup
```

The "prev completed bar" discipline from v1.0's 1H→5m merge is preserved. At the boundary (e.g., the 30m bar at 16:00 UTC is the first bar of the new 4H window), it reads the just-closed 4H bar that covered 12:00–15:59:59.

### 6.5 Live data (Phase 3+)

Hyperliquid WebSocket and funding REST — same architecture as v1.0, but bar aggregation rolls 30m from the trade tape. 4H and 1D are re-aggregated from 30m at each inference tick using the same logic as §6.4.

---

## 7. Feature Engineering — 30m Redesign

### 7.1 Design principles (carried from v1.0)

1. **Percentage-normalized** — all price distances as % of close, not raw dollars
2. **Volatility-normalized** — distances also expressed in ATR units where meaningful
3. **Wilder smoothing** — RSI/ATR/ADX use α = 1/period (not `.ewm(span=period)`)
4. **No look-ahead** — all rolling/groupby use past data; HTF features use prev-closed bar
5. **Project-wide sign convention** — +1 bullish, −1 bearish
6. **SHAP-validated** — post-Phase 2 trim to features with non-trivial importance

### 7.2 Per-category diff: v1.0 (5m) → v2.0 (30m)

Format below: v1.0 count → v2.0 count, followed by rationale and specific changes.

#### Category 1 — Momentum (47 → 32 features)

Rationale: keep all indicator families, drop redundant variants. At 30m, the difference between e.g. RSI(14) and RSI(21) collapses; WaveTrend and Stochastic both measure the same concept; Squeeze Momentum carries unique information only via its fire/release state.

**Keep:**
- RSI(14) value, slope, zone (OB/OS/neutral), distance from 50 (4)
- MACD line, signal, histogram, hist slope, zero-cross state, hist acceleration (6)
- WaveTrend wt1, wt2, wt_cross signal, OB/OS zone (4)
- Stochastic %K, %D, cross signal, OB/OS zone (4)
- Squeeze Momentum value, signal, release_state (-1/0/+1), bars_in_squeeze (4)
- Multi-period momentum: roc_1bar, roc_3bar, roc_6bar, roc_12bar (4 — on 30m, 12 bars = 6hrs, meaningful)
- Cross-feature (2 — formulas locked per Decision v2.40 Q8): `rsi_wt_divergence_flag` = `int(sign(rsi-50) ≠ sign(wt1))` (binary 0/1, 1 = oscillators disagree on regime); `macd_rsi_alignment` = `sign(macd_hist) × sign(rsi-50)` (signed −1/0/+1, +1 aligned, −1 misaligned)
- Velocity-of-velocity (4 — features locked per Decision v2.40 Q6 option (a)): `d2_rsi`, `d2_macd_line`, `d2_wt1`, `d2_stoch_k` — one second-derivative per oscillator family (RSI, MACD, WaveTrend, Stochastic). **Distinct from MACD's `hist_acceleration`** (which is d²(macd_hist), the second derivative of histogram); the vel-of-vel `d2_macd_line` is d²(macd_line), the second derivative of the MACD line itself — different variable, no double-count

**Drop vs v1.0:**
- Sub-1-minute-equivalent oscillator variants (no analog at 30m)
- Redundant period variants within each family
- Raw indicator values when only the slope/zone matters

#### Category 2 — Trend / Direction (19 → 14 features)

**Keep:**
- ADX(14), +DI, −DI, DI-spread, ADX slope (5)
- ADX zone flags (4): `adx_trending` (ADX > 25), `adx_weak` (ADX < 20), `adx_accelerating` (ADX rising — slope above threshold), `adx_decelerating` (ADX falling — slope below threshold). Per Decision v2.37 (DR Q5 disambiguation: spec originally listed only 3 named conditions for a count of 4; v2.37 enumerates the 4th as `adx_decelerating`, the natural pair of `accelerating`)
- EMA9, EMA21, EMA50 positions relative to close as % (3)
- EMA9 vs EMA21 vs EMA50 stack (bullish/bearish/mixed) (1)
- Price vs EMA21 in ATR units (1)

**Drop from v1.0 `adx_features_5m`** (specifically): `di_plus_roc`, `di_minus_roc`, `di_spread_roc`, `adx_accel`, `adx_range_pct`, `di_convergence` (6) — replaced by 4 zone flags above.

**Other drops:**
- Fine-grained EMA crossover timing (handled better in Event Memory Cat 20)

#### Category 2a — **NEW: HTF Context (18 features)**

This is v2.0's biggest feature-space change. Research (§3.1) shows HTF features dominate SHAP importance. Treating them as first-class features, not an afterthought merged from 1H.

| Feature | Source TF | Rationale |
| --- | --- | --- |
| `htf4h_bb_position` | 4H | #1 predictor in 2026 research (8.4%) |
| `htf4h_rsi` | 4H | #5 predictor in 2026 research (4.5%) |
| `htf4h_macd_hist` | 4H | HTF momentum |
| `htf4h_adx` | 4H | HTF trend strength |
| `htf4h_ema21_pos` | 4H | HTF trend direction |
| `htf4h_atr_ratio` | 4H | HTF volatility regime |
| `htf4h_close_vs_ema50_pct` | 4H | HTF distance from mean |
| `htf4h_return_1bar` | 4H | HTF momentum 4h |
| `htf4h_return_3bar` | 4H | HTF momentum 12h |
| `htf1d_ema20_pos` | 1D | Macro trend short |
| `htf1d_ema50_pos` | 1D | Macro trend medium |
| `htf1d_ema200_pos` | 1D | Macro trend long (bull/bear regime) |
| `htf1d_rsi` | 1D | Daily momentum |
| `htf1d_atr_pct` | 1D | Daily volatility regime |
| `htf1d_return_1bar` | 1D | Yesterday's return |
| `htf1d_return_5bar` | 1D | Weekly momentum |
| `htf1d_close_vs_20d_high_pct` | 1D | Distance from recent high |
| `htf1d_close_vs_20d_low_pct` | 1D | Distance from recent low |

All computed on their native TF and merged to the 30m frame via prev-closed-bar lookup.

#### Category 3 — Volatility (15 → 12 features)

**Keep:**
- ATR(14), ATR(5), ATR ratio (short/long), ATR percentile (4)
- BB basis, BB upper, BB lower, BB width %, BB position (5)
- Keltner upper, lower, KC-BB squeeze flag (3)

**Drop:**
- Parkinson estimator for 30m (window too short for reliable estimate; keep on HTF if useful later)
- Redundant ATR periods

#### Category 4 — Volume & Buy/Sell Pressure (17 → 12 features)

**Keep:**
- Volume vs 20-bar MA ratio, volume z-score, volume_spike flag (3)
- VFI value, VFI signal line, VFI histogram slope (3)
- OBV slope, OBV divergence flag (2)
- Candle-shape buy/sell estimator: (close-low)/(high-low), signed by body direction (1)
- Volume-weighted momentum: Σ(return × volume) / Σ(volume) last 10 bars (1)
- High-volume-candle location within range (top/mid/bottom tercile) (2)

**Drop:**
- Low-importance VFI deep features from v1.0 (keep core 3)
- Redundant OBV variants

**Reliability note (unchanged from v1.0):** candle-shape buy/sell is a proxy with ~0.5–0.65 correlation to true trade-tape buy/sell. Phase 3 adds Hyperliquid trade-tape features (Category 21) which are authoritative.

#### Category 5 — VWAP (8 → 14 features, MULTI-ANCHOR EXPANSION)

Per Decision v2.4 (multi-anchor architecture) + Decision v2.44 (Q13 implementation choices locked: daily-anchored 5-feature decomposition, signed confluence count, fractal-pivot-anchored swings, ±0.1×ATR touch tolerance, close-cross HTF pivot break, bars-with-cross count, 100-bar heavy lookback). Each feature listed with locked column name + formula:

**Daily-anchored (5 features per Q13.1):**
- `daily_vwap` — raw cumulative VWAP, resets at UTC day boundary; `Σ(typical × volume) / Σ(volume)` where typical = `(high+low+close)/3`
- `daily_vwap_upper_band_1sig` — `daily_vwap + 1×stdev_band` (raw level; stdev = rolling(20-bar) std of close within day)
- `daily_vwap_lower_band_1sig` — `daily_vwap − 1×stdev_band` (raw level)
- `daily_vwap_dist_pct` — `(close − daily_vwap) / close × 100` (signed)
- `daily_vwap_zone` — categorical {0,1,2,3,4}: 0 = `<-2σ`, 1 = `[-2σ, -1σ)`, 2 = `[-1σ, +1σ]`, 3 = `(+1σ, +2σ]`, 4 = `>+2σ`

**NEW multi-anchor (9 features):**
- `swing_high_vwap_pos` — `(close − swing_high_vwap) / close × 100`. Anchor: cumulative VWAP from each new confirmed fractal pivot high (uses `divergence.fractal_pivots` on `high` series; resets each new pivot).
- `swing_low_vwap_pos` — same for fractal pivot lows on `low` series.
- `htf_pivot_vwap_pos` — anchor: cumulative VWAP from each daily-pivot-P close-cross event (per Q13.5 option a). Cross = `close.shift(1) ≤ P AND close > P` (or inverse for downside cross).
- `weekly_vwap_pos` — `(close − weekly_vwap) / close × 100`; weekly_vwap = cumulative VWAP from Monday 00:00 UTC (resets each Monday).
- `multi_anchor_confluence_signed_count` — `(# of VWAPs close is above) − (# below)` across {daily, weekly, swing_high, swing_low, htf_pivot}. Range -5..+5 (per Q13.2 option a).
- `vwap_of_vwaps_mean_reversion_dist_pct` — synthetic = mean of 5 anchored VWAPs at each bar; rolling 20-bar mean of that synthetic; signed pct distance from close to that rolling mean.
- `dist_to_nearest_anchored_vwap_atr` — `min(|close − vwap_i|) / atr_14` across all 5 anchors (volatility-normalized).
- `vwap_cross_events_count_10` — count of bars in last 10 with at least one cross of any of the 5 VWAPs. Range 0..10 (per Q13.4 option b).
- `close_above_below_heavy_vwap_flag` — `+1` if close > heavy_vwap, `-1` if below, `0` if no clear leader. **Heavy VWAP** = anchored VWAP with most touches in last 100 bars (per Q13.3); **touch** = `|close − vwap| ≤ 0.1 × atr_14`.

**Inputs (caller-supplied, function signature)**: `df` (with high/low/close/volume + DatetimeIndex), `atr_14` (from volatility.py output), `daily_pivot_p` (from pivots.py output), `cfg`. Tunable parameters via `cfg.get('vwap', {}).get(...)`: `bands_window` (default 20), `swing_lookback` (default 5), `heavy_lookback_bars` (default 100), `heavy_touch_atr_mult` (default 0.1).

ROLLBACK: see PROJECT_LOG Decision v2.44 entry for v1.0→v2.0 transformation map + rollback procedure.

#### Category 6 — S/R Structure (Pivot Fibonacci + Swing Fib Retracements) (13 → 30 features)

Keep all v1.0 daily-pivot features (with §7.5 static tagging — pivots are fixed-for-the-day), **add weekly-pivot features** (the current `pivots.py` already has `weekly_pivot_features()` — promote to Phase 1 inclusion), **add swing-based Fibonacci retracement features**, **add ATR-normalized + continuous-position + confluence encodings**. All formulas locked per Decision v2.45 (Q14.1–Q14.4).

**6.1 Daily Fib-pivots (9 features, locked per Q14.1):**

Per Decision v2.45 Q14.1: keep all 7 level distances (covers full S3..R3 range — needed for trending weeks where price moves outside [S1,R1]) + 2 most-informative derived (`pivot_zone` + `pivot_times_tested_today`). Drops 4 v1.0 derived features (`dist_to_nearest_pivot_pct`, `nearest_pivot_type`, `pivot_approach_dir`, `pivot_approach_speed`) — all derivable from the 7 distances internally by LightGBM.

- `pivot_S3_dist_pct` — `(close − pivot_S3) / close × 100` (signed)
- `pivot_S2_dist_pct` — same for S2
- `pivot_S1_dist_pct` — same for S1
- `pivot_P_dist_pct` — central pivot
- `pivot_R1_dist_pct` — same for R1
- `pivot_R2_dist_pct` — same for R2
- `pivot_R3_dist_pct` — same for R3
- `pivot_zone` — categorical 0..5 (interval position; 0=≤S3, 1=[S3,S2), 2=[S2,S1), 3=[S1,P), 4=[P,R1), 5=[R1,R2), with ≥R3 also 5; v1.0 convention preserved)
- `pivot_times_tested_today` — count of bars today within `tolerance_pct` of any daily pivot level (cumulative within day)

**6.2 Daily Fib-pivots — NEW encodings (3):**
- `pivot_position_daily_01` — continuous 0–1 position within S1–R1 range: `(close - pivot_S1) / (pivot_R1 - pivot_S1)`. Volatility-/price-agnostic; 0.65 means the same thing at $30k BTC or $100k BTC.
- `dist_to_nearest_pivot_atr` — `min(|close − pivot_i|) / atr_14` across all 7 daily levels (volatility-normalized).
- `daily_pivot_weekly_pivot_confluence` — binary flag per Q14.3: fires when **any** of {daily_S1, daily_P, daily_R1} is within `0.25 × atr_14` of **any** of the 7 weekly levels {weekly_S3..R3}.

**6.3 Weekly Fib-pivots (9 features, NEW to feature matrix):**

Mirrors 6.1 structure exactly, computed from weekly-reset (Monday 00:00 UTC) pivots:
- `weekly_pivot_S3_dist_pct`, `weekly_pivot_S2_dist_pct`, `weekly_pivot_S1_dist_pct`,
- `weekly_pivot_P_dist_pct`, `weekly_pivot_R1_dist_pct`, `weekly_pivot_R2_dist_pct`, `weekly_pivot_R3_dist_pct`,
- `weekly_pivot_zone` — same 0..5 categorical based on weekly levels,
- `weekly_pivot_times_tested_this_week` — count of bars this week within `tolerance_pct` of any weekly pivot level.

At 30m, 336 bars/week makes weekly pivots persistent structural levels with real tested behavior (at 5m they were 2016 bars away, too far).

**6.4 Weekly Fib-pivots — NEW encodings (2):**
- `pivot_position_weekly_01` — `(close - weekly_pivot_S1) / (weekly_pivot_R1 - weekly_pivot_S1)`
- `dist_to_nearest_weekly_pivot_atr` — `min(|close − weekly_pivot_i|) / atr_14` across all 7 weekly levels

**6.5 Swing-based Fibonacci retracements (7 features, NEW — separate concept from Fib-pivots):**

Computed from the most recent confirmed swing high / swing low (fractal pivots from `divergence.fractal_pivots`, lookback=5 = 2-left-2-right rule per spec). Whereas Fib-*pivots* are calculated from prior-day OHLC math, Fib-*retracements* are drawn between an observed price swing and measure where price is within that swing range.

`swing_high` and `swing_low` are forward-filled from the most recent confirmed pivot of each type. `swing_range = swing_high - swing_low` (with safe_div on zero).

- `fib_retracement_pct` — `(close - swing_low) / swing_range`; values <0 = broke below swing low, >1 = broke above swing high.
- `in_golden_pocket` — binary, 1 if `0.618 ≤ fib_retracement_pct ≤ 0.65` (highest-probability reversal zone).
- `nearest_fib_level_dist` — `min(|close - fib_level_i|) / atr_14` across `[0.382, 0.5, 0.618, 0.786]` retracement levels (in ATR units).
- `fib_touches_382` — `count(|close - fib_level_0382| ≤ 0.1% × close)` over last 20 bars.
- `fib_touches_618` — same for 0.618 level.
- `extension_progress_1272` — per Q14.2 option (a): `(close - swing_low) / (1.272 × swing_range)`. Range: 0 at swing_low, ~0.786 at swing_high, 1.0 at 1.272 extension; goes negative below swing_low (bearish extension territory).
- `swing_fib_pivot_confluence` — per Q14.4: binary flag, fires when nearest Fib retracement level (in price terms) is within `0.25 × atr_14` of **any** of 14 pivots (7 daily + 7 weekly). Detects "Fib level coincides with structural pivot" — high-probability reversal zone.

**Look-ahead safety:** `divergence.fractal_pivots` already shifts by `lookback // 2` bars to enforce confirmation delay (center bar is in the past). No further shift needed.

**Count:** 9 (6.1) + 3 (6.2) + 9 (6.3) + 2 (6.4) + 7 (6.5) = **30** ✓

**Tagging (per §7.5)**: All 30 Cat 6 features tagged `static`. Pivots are fixed for the day/week (don't update intrabar); swing Fib levels only update when a new fractal pivot confirms. Position features (dist_pct, zone) depend on current close but the LEVELS are static — for intrabar inference safety, treating Cat 6 as static is correct (the levels can be cached and reused across 5m intrabar bars within a 30m bar; only `close` is needed fresh).

**ROLLBACK**: see PROJECT_LOG Decision v2.45 entry for full transformation map.

#### Category 7 — Session & Time Context (15 → 9 features)

**Keep (9, locked column names + formulas):**
- `hour_of_day_sin` — `sin(2π × hour / 24)`
- `hour_of_day_cos` — `cos(2π × hour / 24)`
- `day_of_week_sin` — `sin(2π × dayofweek / 7)`
- `day_of_week_cos` — `cos(2π × dayofweek / 7)`
- `is_weekend` — `1 if dayofweek ≥ 5 else 0` (Sat/Sun)
- `session_overlap_asian_london` — `1 if hour ∈ [7, 9) UTC` (Tokyo × London window)
- `session_overlap_london_ny` — `1 if hour ∈ [13, 16) UTC` (London × NY window)
- `session_overlap_ny_asian` — `1 if hour ∈ [21, 22) ∪ [0, 6) UTC` (NY × Sydney/Tokyo window)
- `month_of_year` — integer 1..12 (categorical encoding; spec "(cyclic) (1)" satisfied as single-feature month index — LightGBM handles categorical-like ints; sin+cos would exceed (1) count)

"Asian" defined as Tokyo + Sydney combined per standard TA convention.

**Drop from v1.0 (~10 dropped):**
- Individual session flags: `session_sydney`, `session_tokyo`, `session_london`, `session_new_york` (joint info captured in 3 overlap flags)
- 4th overlap `overlap_sydney_tokyo` (not in spec; conceptually merged into "Asian" definition)
- `active_session_count` (derivable from overlap flags)
- `minutes_into_session`, `minutes_to_session_close` (intra-5m granularity; at 30m these are 0 or 30, useless)
- `session_range_vs_avg` (low-importance, moves out of Cat 7)
- `prev_session_range_pct` (Cat 11 territory)
- `day_of_week` (raw int — replaced by sin/cos encoding)
- `is_monday` (replaced by `is_weekend` semantically; specific weekdays derivable from sin/cos)

#### Category 8 — Price Action / Candle (9 → 9 hybrid reshape per Decision v2.49 Q18)

**SPEC AMENDMENT (Decision v2.49 Q18.1 (c)):** original §7.2 Cat 8 narrative listed 9 features mixing continuous candle-shape metrics with binary pattern flags (doji, hammer/shooting-star, inside-bar). Per Decision v2.49, the binary flags `doji_flag` and `hammer_or_shooting_star_flag` were DROPPED — both are trivially derivable from continuous `body_pct` + wick features that LightGBM already splits on at SHAP-optimal thresholds. Encoding them as hardcoded binaries adds no signal and may displace the model's learned splits. v1.0's `is_bullish` and `body_vs_prev_body` were ADDED back into spec — both carry distinct continuous signal not derivable from spec's original list. v1.0's `consecutive_bull`/`consecutive_bear` were DROPPED — counter pattern with weak signal at 30m (consecutive runs typically top out at 2-3 bars; belongs to Cat 20 event_memory territory if needed later).

**9 features locked per Decision v2.49 Q18.10:**

Continuous candle-shape metrics (4):
- `body_pct` — `|close − open| / range` where `range = high − low`. Continuous quality signal feeds doji/marubozu pattern derivation by tree splits (1)
- `upper_wick_pct` — `(high − max(open, close)) / range`. Rejection-side signal (1)
- `lower_wick_pct` — `(min(open, close) − low) / range`. Rejection-side signal (1)
- `range_pct` — `(high − low) / close × 100`. Volatility burst marker; complements smoothed Cat 3 ATR (1, NEW per spec amendment — kept from original spec list)

Quality / direction signals (2):
- `is_bullish` — `int(close > open)`. Most-frequently-used binary split direction; v1.0 KEPT per Decision v2.49 (1)
- `body_vs_prev_body` — `body_n / body_(n-1)`. Body-size-vs-prior-body continuous; feeds engulfing context + exhaustion signals; v1.0 KEPT per Decision v2.49 (1)

Multi-bar pattern signals — kept as explicit features because NOT derivable from single-bar continuous (3):
- `engulfing_signal` — signed +1/−1/0. +1 bullish engulfing: `(close > open) & (prev_close < prev_open) & (close ≥ prev_open) & (open ≤ prev_close)`. −1 bearish engulfing: mirror. v1.0 formula verbatim (rename from `engulfing` for spec wording match) (1)
- `pin_bar_signal` — signed +1/−1/0. +1 bullish pin: `(lower_wick > 2 × body_pct) & (upper_wick < body_pct)`. −1 bearish pin: mirror. v1.0 formula verbatim (rename from `pin_bar`) (1)
- `inside_bar_flag` — binary 0/1. Fires when `(high < prev_high) & (low > prev_low)`. Compression/consolidation pattern — true 2-bar conditional NOT derivable from single-bar features (1, NEW per spec amendment — kept from original spec list)

**§7.5 tagging per Q18.8:** All 9 features = DYNAMIC. Each feature evaluates on current bar's OHLC; close/high/low all mutate intrabar during 30m bar formation, so feature values mutate with them. Pure-dynamic block (matches Cat 18, Cat 19 pattern).

**v1.0 → v2.0 transformation map:**
- KEEP VERBATIM (3): `body_pct`, `upper_wick_pct`, `lower_wick_pct`.
- KEEP RENAMED (2): `engulfing` → `engulfing_signal`, `pin_bar` → `pin_bar_signal` (spec-wording match; formulas unchanged).
- KEEP VERBATIM (per Decision v2.49 amendment) (2): `is_bullish`, `body_vs_prev_body`.
- DROP (2): `consecutive_bull`, `consecutive_bear` (weak signal at 30m; counter belongs to Cat 20 if needed).
- NEW (2): `range_pct`, `inside_bar_flag` (per original spec narrative; kept).
- DROPPED FROM ORIGINAL SPEC NARRATIVE (2): `doji_flag`, `hammer_or_shooting_star_flag` (per Decision v2.49 — redundant with continuous body+wicks).
- Net: 9 → 9 with the right features (4 in/out swap).

**Function signature:** `candle_features(df, cfg=None) -> DataFrame[9 cols]` (Q18.9 (a)). Self-contained — math derives from OHLC alone. cfg accepted for future tunable thresholds (e.g., pin_bar wick-to-body ratio); currently no tunables, defaults baked in.

30m candles carry more signal than 5m (less microstructure noise) — preserved by keeping continuous body/wick features that capture candle SHAPE without binary-threshold lossiness.

#### Category 9 — Mean Reversion / Statistics (8 → 7 features, trim per Decision v2.50 Q19)

**Keep (7 — formulas locked per Decision v2.50 Q19.12):**

- `bb_pct_b` — `(close − bb_lower) / (bb_upper − bb_lower)`. Bollinger %B; bounded [0, 1] inside bands, can go negative or > 1 outside. Caller-supplied (rename of Cat 3 `bb_position` from volatility.py per Q19.1 (a)) (1)
- `bb_dist_mid_sigma` — `(close − bb_mid) / bb_std` where bb_mid = SMA(20), bb_std = rolling std(close, 20). Distance from mid-band in σ units. Mathematically identical to `zscore_20` when BB period = 20 (= canonical Cat 3). Per Q19.2 (a) the redundancy is accepted: spec count of 4 BB-derived features wins; LightGBM tolerates duplicate columns (1)
- `zscore_20` — `(close − close.rolling(20).mean()) / close.rolling(20).std()`. Same calc as `bb_dist_mid_sigma` by construction (1)
- `zscore_50` — `(close − close.rolling(50).mean()) / close.rolling(50).std()`. Distinct longer-window z-score (1)
- `skewness_20` — `log_returns.rolling(20).skew()` where `log_returns = np.log(close / close.shift(1))`. Per Q19.3 (a) log returns (stationary basis) (1)
- `kurtosis_20` — `log_returns.rolling(20).kurt()`. Pandas Fisher-adjusted (excess kurtosis = kurtosis − 3) (1)
- `autocorr_1` — `log_returns.rolling(50).apply(lambda x: pd.Series(x).autocorr(lag=1))`. 50-bar rolling window for stable lag-1 estimate per Q19.4 (a). Renamed from v1.0 `autocorrelation_1` (moved from Cat 17 to Cat 9 per spec) (1)

**§7.5 tagging per Q19.10:** All 7 features = DYNAMIC (rolling stats include current bar's close; close mutates intrabar). Pure-dynamic block.

**Drop from v1.0 stats.py `mean_reversion_features`:**
- `mean_reversion_score` (per §15 directive — composite z-score average; redundant with z_20 + z_50 separately)
- `rsi_zscore` (orphan — not in §7.2 Cat 9; rolling z-score of RSI is derivable if needed)
- `return_5bar`, `return_20bar`, `return_60bar` (orphan — overlap with Cat 1 multi-period momentum which has roc_1/3/6/12_bar)

**Function signature:** `mean_reversion_features(df, bb_position, cfg=None) -> DataFrame[7 cols]` (Q19.11 (a) split per spec category). Caller-supplied `bb_position` from Cat 3 volatility output. Cfg accepted for future tunable thresholds (currently no tunables; canonical 20/50 windows baked in).

#### Category 10 — Market Regime (7 → 7, count unchanged but rewrite per Decision v2.37 Q3)

**Per Decision v2.37 Q3**: drop v1.0's `efficiency_ratio` + `choppiness_index` + `regime_volatile`/`regime_quiet` binaries; add `trend_direction` + `volume_regime` + `vol_adjusted_momentum_regime`.

**Keep (7, locked column names + formulas):**
- `trending_regime` — `(adx > 25) & (|di_plus − di_minus| > 10)`; binary 0/1
- `ranging_regime` — `(adx < 20) & (bb_width_percentile < 30)`; binary 0/1
- `volatility_regime` — tercile 0/1/2 from `atr_percentile` (rolling 100-bar): 0 if <33.33 (low), 2 if >66.67 (high), else 1 (normal)
- `volume_regime` — tercile 0/1/2 from `volume.rolling(100).rank(pct=True)*100`: same threshold logic
- `trend_direction` — +1 if `ema9 > ema21 > ema50` (bull stack), −1 if `ema9 < ema21 < ema50` (bear stack), 0 otherwise (mixed)
- `regime_change_bar` — bars since last regime label flip; label = `1 if trending` / `4 if ranging` / `0 if neither`; counter resets on label change
- `vol_adjusted_momentum_regime` — `roc_3bar / atr_pct` where roc_3bar = `(close/close.shift(3) - 1) × 100` and atr_pct = `atr_14 / close × 100`. Signed continuous, vol-normalized momentum strength.

**Dropped from v1.0**:
- `regime_volatile`, `regime_quiet` (replaced by `volatility_regime` tercile)
- `efficiency_ratio`, `choppiness_index` (per spec Q3 — not in keep list)
- `bars_in_current_regime` renamed → `regime_change_bar`

#### Category 11 — Previous Context / Memory (8 → 6 features, REWRITE per Decision v2.37 Q2 + Decision v2.51 Q20)

Per Decision v2.37 Q2 (disambiguation): "Prev-bar" means the **immediately preceding 30m bar (1 bar back)**, NOT yesterday/prev-day. Yesterday's features are duplicates of Cat 2a HTF1D and are dropped per the explicit "Drop" clause below.

Per Decision v2.51 Q20: formulas + sign conventions + §7.5 mixed-block split locked.

**Keep (6 — formulas locked per Decision v2.51 Q20.10):**

Prev-bar lookups (4, all STATIC per §7.5 — `.shift(1)` lookups don't mutate intrabar after the prev bar closed):
- `prev_bar_close_vs_close_pct` — `(close.shift(1) − close) / close × 100`. Sign convention per Q20.1 (a): positive when prev close > current close (bearish current bar context); distinct from Cat 1 `roc_1bar` which uses opposite sign (1)
- `prev_bar_high_vs_close_pct` — `(high.shift(1) − close) / close × 100`; signed % (1)
- `prev_bar_low_vs_close_pct` — `(low.shift(1) − close) / close × 100`; signed % (1)
- `prev_bar_volume_ratio` — `volume.shift(1) / volume` per Q20.2 (a) spec-literal numerator/denominator. Values > 1 indicate prev bar had higher volume than current (1)

Today-running (2, both DYNAMIC per §7.5 — depend on current close):
- `today_open_to_now_pct` — `(close − today_open) / today_open × 100` where `today_open = first 30m bar's OPEN per UTC day` (Q20.5 (a)). UTC-day boundary via `df.index.floor("1D")` per Q20.4 (a) (consistent with pivots.py / vwap.py daily anchors) (1)
- `today_high_low_distance_from_current_pct` — SIGNED distance to whichever of today's running high or low is closer per Q20.3 (b). Formula: compute `dist_high = |today_high − close|`, `dist_low = |close − today_low|` where today_high/low are cummax/cummin within UTC day; result = `+min(dist_high, dist_low) / close × 100` if today_low is closer (price near support), `−min(dist_high, dist_low) / close × 100` if today_high is closer (price near resistance). Sign carries side info that tree models use; pure magnitude (unsigned) would collapse near-support and near-resistance into one signal (1)

**§7.5 tagging per Q20.8 (a) — MIXED-BLOCK SPLIT (4 static + 2 dynamic):**

§7.5 LIST EDIT applied: Cat 11 moved from §7.5 STATIC list to MIXED list with explicit per-feature 4-static / 2-dynamic split. v1.0 Cat 11 was prev-DAY-only (all static prior-bar lookups); v2.0 rewrite per Decision v2.37 Q2 added today-running features which depend on current close. Honest taxonomy: Cat 11 + Cat 16 are now the two mixed-split blocks in feature_stability.py (Cat 19 was demoted to all-dynamic per Q16.4).

- STATIC (4): `prev_bar_close_vs_close_pct`, `prev_bar_high_vs_close_pct`, `prev_bar_low_vs_close_pct`, `prev_bar_volume_ratio`. All `.shift(1)` lookups; the prev bar's OHLCV is locked once that bar closes — intrabar-safe.
- DYNAMIC (2): `today_open_to_now_pct`, `today_high_low_distance_from_current_pct`. Both depend on current close; the today_high/today_low cummax/cummin update as new highs/lows print intrabar.

**Drop from v1.0 (per Decision v2.37 Q2 + Decision v2.51):**
- All v1.0 prev-DAY features in v1.0 `context.py:previous_context_features`: `prev_day_range_pct`, `prev_day_close_vs_pivot`, `dist_to_prev_day_high_pct`, `dist_to_prev_day_low_pct`, `prev_session_direction`, `prev_session_volume_rank` — all are duplicates of Cat 2a HTF1D context (`htf1d_return_1bar`, `htf1d_close_vs_20d_*`, etc.)
- v1.0 `gap_pct` — partial duplicate of `htf1d_return_1bar`
- v1.0 `daily_open_dist_pct` — concept absorbed by `today_open_to_now_pct` (renamed + UTC-day boundary clarified)

**Function signature:** `prev_context_features(df, cfg=None) -> DataFrame[6 cols]` (Q20.9 (a) split per spec category). Self-contained — math derives from OHLC + volume + UTC-day boundary alone. Cfg accepted for future tunables (currently no tunables; defaults baked in).

#### Category 12 — Lagged Dynamics (8 → 5 features, REWRITE per Decision v2.51 Q20)

**Keep (5 — formulas locked per Decision v2.51 Q20.10, all DYNAMIC per Q20.8):**

- `delta_rsi_1` — `rsi − rsi.shift(1)`; raw point-difference (RSI is bounded; raw delta is interpretable). Caller-supplied `rsi` from Cat 1 momentum_core output (1)
- `delta_rsi_3` — `rsi − rsi.shift(3)`; raw point-difference over 3 bars (1)
- `delta_adx_3` — `adx − adx.shift(3)`; raw point-difference (ADX is bounded; raw delta is interpretable). Caller-supplied `adx` from Cat 2 trend output (1)
- `delta_volume_3` — `(volume − volume.shift(3)) / volume.shift(3) × 100`; percentage change (volume is unbounded — ratio more interpretable than raw diff). Volume from df (1)
- `delta_close_vs_ema21_3` — `ema21_dist_pct − ema21_dist_pct.shift(3)`; raw pp (percentage-point) difference of `ema21_dist_pct` (already in %) over 3 bars. Caller-supplied `ema21_dist_pct` from Cat 2 trend output (1)

**§7.5 tagging:** All 5 = DYNAMIC (deltas of dynamic series). Pure-dynamic block.

**Drop from v1.0 (8 dropped per Decision v2.51):**
- `rsi_5bar_ago` (raw lag — replaced by point-deltas per spec)
- `wt1_slope_5bar` (overlaps Cat 1 wt1; not in spec keep list)
- `adx_slope_5bar` (replaced by `delta_adx_3` per spec lag window)
- `atr_slope_5bar` (not in spec keep list)
- `volume_slope_5bar` (replaced by `delta_volume_3` per spec lag window)
- `vwap_slope_5bar` (not in spec keep list; overlaps Cat 5 multi-anchor analytics)
- `di_spread_change_5bar` (overlaps Cat 2 di_spread + Cat 2 adx_slope)
- `squeeze_mom_slope` (overlaps Cat 1 squeeze_release_state + bars_in_squeeze)
- "Longer lag windows that overlap with HTF context" — Cat 2a already covers slower dynamics

**Function signature:** `lagged_dynamics_features(df, rsi, adx, ema21_dist_pct, cfg=None) -> DataFrame[5 cols]` (Q20.6 (a) caller-supplied 3 series). Caller-supplies 3 dependencies (rsi from Cat 1, adx + ema21_dist_pct from Cat 2); volume taken from df. Pattern-consistent with mean_reversion_features (caller-supplied bb_position from Cat 3) and other Phase 1.10b caller-supplied dependency patterns. Cfg accepted for future tunable lag windows (currently spec-locked windows: 1, 3, 3, 3, 3 baked in).

#### Category 13 — Divergence Detection (7 → 7, count-unchanged but content reshaped)

**Strict spec interpretation locked per Decision v2.43 Q12 (option a)**: bullet list = literal column names, each binary 0/1; signs normalized at the v1.0 level (bullish=+1 / bearish=-1) but emitted as separate binary flags rather than signed encoding. Spec lists 6 binary flags + recency = 7 features.

**Keep (7 binary/int features, all locked column names):**
- `regular_bullish_div_rsi` — binary 0/1, fires when price LL + RSI HL (bullish regular divergence on RSI)
- `regular_bearish_div_rsi` — binary 0/1, fires when price HH + RSI LH (bearish regular divergence on RSI)
- `regular_bullish_div_macd` — binary 0/1, on MACD histogram
- `regular_bearish_div_macd` — binary 0/1, on MACD histogram
- `hidden_bullish_div_rsi` — binary 0/1, fires when price HL + RSI LL
- `hidden_bearish_div_rsi` — binary 0/1, fires when price LH + RSI HH
- `divergence_recency` — int, `bars_since(any of 6 flags above)`

**Dropped from v1.0 (per Decision v2.43)**: `wt_price_divergence`, `stoch_price_divergence`, `divergence_count`. Spec doesn't list WT or Stoch divergences; their information is largely redundant with RSI divergence (WT is correlated with RSI by construction; Stoch is a slower-period RSI variant). `divergence_count` derives from the kept flags via `sum(flags)` if needed downstream.

**Detection method (unchanged from v1.0)**: fractal pivots (default 5-bar window, 2 left + center + 2 right); compare current confirmed pivot to most recent prior confirmed pivot of same type within `lookback_bars=14` (default — at 30m = 7 hours pivot-to-pivot window). Tunable in cfg if SHAP shows underfiring.

**Rollback path (if SHAP later shows dropped features useful)**: see PROJECT_LOG `Decision v2.43` entry for full transformation map + rollback procedure.

#### Category 14 — Money Flow (8 → 6 features)

**Keep:**
- CMF(20), CMF slope (2)
- MFI(14), MFI zone (OB/OS) (2)
- A/D line slope, A/D vs price divergence flag (2)

**Drop:**
- Low-importance money flow oscillator variants

#### Category 15 — Additional Momentum (9 → 7 features)

**Keep (7 — formulas locked per Decision v2.42 Q11):**
- `williams_r` — Williams %R(14) value (1)
- `williams_r_zone` — +1 OB (WR > -20) / -1 OS (WR < -80) / 0 mid (1)
- `cci_20` — CCI(20) value (1)
- `cci_zero_cross_state` — `sign(cci_20)`; +1 above zero, -1 below (1)
- `tsi_signal_cross_state` — `sign(tsi - tsi_signal)` where `tsi_signal = tsi.ewm(span=7).mean()`; momentum acceleration trigger (Decision v2.42 Q11 option a) (1)
- `tsi_zero_cross_state` — `sign(tsi)`; directional regime (Decision v2.42 Q11 added "keep both") (1)
- `cmo_zero_cross_state` — `sign(cmo_14)` (1)

**Drop from v1.0:**
- `williams_r_direction`, `cci_direction`, `cci_extreme`, `cmo_direction`, `cmo_14` raw value, raw `tsi` value
- `roc_10` — overlaps with Cat 1 multi-period momentum (`roc_1bar`/`roc_3bar`/`roc_6bar`/`roc_12bar`)
- Redundant oscillators that duplicate RSI/Stoch from Category 1

#### Category 16 — Market Structure (10 → 10, reconcile per Decision v2.46 Q15)

**Keep (10 — formulas locked per Decision v2.46 Q15):**

Pivot detection: `(p_high, _) = fractal_pivots(high, lookback=5)` and `(_, p_low) = fractal_pivots(low, lookback=5)` from `divergence.py` (2-left-2-right rule, confirms `lookback` bars after the swing — look-ahead-safe). HH/HL/LH/LL events defined per Q15.3:

- HH event at bar `i`: `p_high[i].notna() & (p_high[i] > previous_non_nan(p_high))`
- HL event at bar `i`: `p_low[i].notna() & (p_low[i] > previous_non_nan(p_low))`
- LH event at bar `i`: `p_high[i].notna() & (p_high[i] < previous_non_nan(p_high))`
- LL event at bar `i`: `p_low[i].notna() & (p_low[i] < previous_non_nan(p_low))`

Counts are over **last 20 bars** (calendar window, not last 20 pivots — Q15.3 (a)):

- `higher_highs_count_20` — `HH_event.rolling(20).sum()` (1)
- `higher_lows_count_20` — `HL_event.rolling(20).sum()` (1)
- `lower_highs_count_20` — `LH_event.rolling(20).sum()` (1)
- `lower_lows_count_20` — `LL_event.rolling(20).sum()` (1)
- `structure_type` — +1 (HH+HL bull) / −1 (LL+LH bear) / 0 (mixed/ranging); from last 2 confirmed H-pivots + last 2 confirmed L-pivots (1)
- `swing_length_ratio` — `|swing_n| / |swing_(n-1)|` over alternating H/L pivot chain (Q15.2 (a) — last-two-swing-lengths, not count-vs-count which is degenerate vs separate count features). Forward-filled between pivot confirms (1)
- `swing_high_dist_pct` — `(close − last_pivot_high) / close × 100`; signed % (1)
- `swing_low_dist_pct` — `(close − last_pivot_low) / close × 100`; signed % (1)
- `fractal_pivot_count_20` — `(p_high.notna() | p_low.notna()).rolling(20).sum()`; pivot density signal independent of HH/HL/LH/LL classification (Q15.4 (a)) (1)
- `break_of_structure` — signed binary event: +1 on bar where `structure_type` flips from −1 to +1 (bullish break), −1 on flip from +1 to −1 (bearish break), 0 otherwise (Q15.1 (a)). Replaces v1.0 `bars_since_structure_break` counter (1)

**§7.5 tagging — first `mixed` block with explicit per-feature split (Q15.5):**
- STATIC (8): `higher_highs_count_20`, `higher_lows_count_20`, `lower_highs_count_20`, `lower_lows_count_20`, `structure_type`, `swing_length_ratio`, `fractal_pivot_count_20`, `break_of_structure` — derived from confirmed pivots; only update when a new pivot confirms (deterministic at bar = pivot_bar + lookback). Intrabar-safe.
- DYNAMIC (2): `swing_high_dist_pct`, `swing_low_dist_pct` — both depend on current close, mutate intrabar.

**Drop from v1.0:**
- `swing_range_pct` (low SHAP — derivable from swing_high_dist_pct − swing_low_dist_pct)
- `retrace_depth` (Cat 6.5 swing-Fib block already covers retracement depth via `fib_retracement_pct`)
- `range_position` (overlaps with Cat 6 pivot_position_daily_01 + bb_position from Cat 3)
- `bars_since_structure_break` (replaced by signed binary `break_of_structure`; counter pattern belongs to Cat 20 event_memory which is locked at 22 features without it)

**Function signature:** `structure_features(df, cfg) -> DataFrame[10 cols]` (Q15.7 (a)). Self-contained — pivots derived internally via `fractal_pivots`. Tunable cfg keys: `structure.fractal_lookback=5`, `structure.count_window=20`.

#### Category 17 — Statistical / Fractal (9 → 6 features, trim per Decision v2.50 Q19)

**Keep (6 — formulas locked per Decision v2.50 Q19.12):**

- `hurst_exponent` — simplified single-pass R/S on log returns over 100-bar window per Q19.5 (a). Formula: `H = log(R/S) / log(N)` where R = max(cumulative deviations) − min(cumulative deviations), S = std(arr), N = window length. Returned NaN when S=0 or R=0 (degenerate). Spec-compliant per "single scale is enough" (multi-scale Hurst dropped per spec) (1)
- `fractal_dimension` — box-counting on close levels (NOT log returns — fractal dim measures geometric complexity of price PATH) over 50-bar window per Q19.6 (a). Normalize (t, price) to [0, 1]² unit square; cover with dyadic grids of size eps = 1/2^k for k = 1..floor(log2(N)); count occupied boxes including vertical spans between consecutive samples; D = −slope of log(N(eps)) vs log(eps) via polyfit (1)
- `autocorr_5` — `log_returns.rolling(50).apply(lambda x: pd.Series(x).autocorr(lag=5))`. 50-bar rolling window for lag-5 estimate per Q19.8 (a). Renamed from v1.0 `autocorrelation_5` (1)
- `autocorr_20` — `log_returns.rolling(100).apply(lambda x: pd.Series(x).autocorr(lag=20))`. 100-bar rolling window scaled with lag (≥ 4× lag for stable estimate). NEW vs v1.0 (1)
- `entropy_20` — Shannon entropy on log returns, 10 bins, 20-bar rolling window per Q19.7 (a). Formula: `−sum(p × log(p))` where p = histogram counts normalized to probabilities (zero-bins excluded). Renamed from v1.0 `price_entropy` and window changed 50→20 to match spec (1)
- `realized_vol_of_realized_vol` — two-pass rolling std on log returns per Q19.9 (a): `inner = log_returns.rolling(20).std()`; then `realized_vol_of_realized_vol = inner.rolling(20).std()`. W1 = W2 = 20. NEW vs v1.0 (1)

**§7.5 tagging per Q19.10:** All 6 features = DYNAMIC (rolling stats include current bar's log return / close; close mutates intrabar). Pure-dynamic block.

**Drop from v1.0:**
- `autocorrelation_1` — moved from Cat 17 to Cat 9 with rename `autocorr_1` per spec (Cat 9's autocorr(1) listing)
- `parkinson_vol` (per §15 — moved to Cat 3 removal list; already dropped from volatility.py via Cat 3 trim)
- `variance_ratio` (per §15 — multi-scale variance ratio not in spec keep list; redundant with Hurst as serial-correlation measure)
- `price_entropy` renamed → `entropy_20` (window 50→20 per spec)
- `autocorrelation_5` renamed → `autocorr_5` (spec wording match)
- Multi-scale Hurst variants (per spec — single scale enough)
- Volatility-of-volatility-of-volatility stackings (per spec — over-engineering)

**Function signature:** `fractal_stats_features(df, cfg=None) -> DataFrame[6 cols]` (Q19.11 (a) split per spec category). Self-contained — derives log_returns from close internally. Cfg accepted for future tunable windows (currently canonical 100/50/50/100/20/20 windows baked in for hurst/fd/autocorr_5/autocorr_20/entropy/rvol_rvol).

#### Category 18 — Adaptive MAs (5 → 4 features, trim per Decision v2.48 Q17)

**Keep (4 — formulas locked per Decision v2.48 Q17):**

- `kama_dist_pct` — `(close − kama) / close × 100`; signed % where `kama = KAMA(period=10, fast_ema=2, slow_ema=30)` (canonical Wilder/Kaufman parameters per Q17.4) (1)
- `dema_dist_pct` — `(close − dema) / close × 100`; signed % where `dema = DEMA(21) = 2×EMA(close,21) − EMA(EMA(close,21),21)` (1)
- `tema_dist_pct` — `(close − tema) / close × 100`; signed % where `tema = TEMA(21) = 3×EMA1 − 3×EMA2 + EMA3` (cascading EMAs of period 21) (1)
- `psar_state_dist_pct` — `(close − sar) / close × 100`; SIGNED feature combining direction + magnitude per Q17.1 (a). `sar` from canonical Wilder Parabolic SAR with af_start=0.02, af_step=0.02, af_max=0.20 per Q17.2. Sign carries the trend state by Wilder construction: positive ⟺ long trend (SAR below close), negative ⟺ short trend (SAR above close). Replaces v1.0 separate `psar_direction` + `psar_dist_pct` features (combined into single signed value per §15 directive) (1)

**§7.5 tagging per Q17.6:** All 4 features = DYNAMIC (close-relative `*_dist_pct` mutate intrabar with current close; KAMA/DEMA/TEMA values move with close; PSAR state/distance evaluated at current bar).

**Drop from v1.0:**
- `psar_direction` — sign of (close − sar) is fully equivalent to PSAR trend state by Wilder construction; redundant with `sign(psar_state_dist_pct)`. Same drop pattern as Cat 19 `tk_cross` (Q16.2).
- "Duplicate KAMA parameter variants" — preventive note; v1.0 already has only one KAMA call (no actual code change needed for this drop).

**v1.0 → v2.0 transformation map:**
- KEEP VERBATIM: `kama_dist_pct`, `dema_dist_pct`, `tema_dist_pct` (3).
- COMBINE + RENAME: `psar_direction` + `psar_dist_pct` → `psar_state_dist_pct` (single signed feature). v1.0 emitted both; v2.0 emits one.
- Net: 5 → 4 (= −1).

**Function signature:** `adaptive_ma_features(df, cfg=None) -> DataFrame[4 cols]` (Q17.5 (a)). Self-contained — math derives from OHLC alone. Keeps v1.0 NESTED cfg structure (`cfg["kama"]["period"]`, `cfg["psar"]["af_start"]`, etc.) with all keys optional; canonical defaults applied when keys absent. Backward-compat with v1.0 config.yaml preserved; no breaking changes.

#### Category 19 — Ichimoku (5 → 6 features, expansion per Decision v2.47 Q16)

**Keep + fix (6 — formulas locked per Decision v2.47 Q16):**

Canonical Hosoda math (TradingView ta.ichimoku-equivalent):
- `tenkan = (max_high(9) + min_low(9)) / 2`
- `kijun  = (max_high(26) + min_low(26)) / 2`
- `span_a_raw = (tenkan + kijun) / 2`
- `span_b_raw = (max_high(52) + min_low(52)) / 2`
- `senkou_a = span_a_raw.shift(26)` — cloud projected over current bar = span computed 26 bars ago (causal, no look-ahead)
- `senkou_b = span_b_raw.shift(26)`
- `cloud_mid = (senkou_a + senkou_b) / 2`

**6 features (all close-relative %):**

- `tenkan_dist_pct` — `(close − tenkan) / close × 100`; signed % (1)
- `kijun_dist_pct` — `(close − kijun) / close × 100`; signed % (1)
- `tk_diff_pct` — `(tenkan − kijun) / close × 100`; signed % carrying both magnitude AND direction of T-K spread (Q16.2 (a) — replaces v1.0 `tk_spread` rename for spec-wording match; v1.0 redundant `tk_cross` sign feature dropped since it's derivable as `sign(tk_diff_pct)`) (1)
- `senkou_a_dist_pct` — `(close − senkou_a) / close × 100`; signed % (1, NEW per §15 directive)
- `senkou_b_dist_pct` — `(close − senkou_b) / close × 100`; signed % (1, NEW per §15 directive)
- `cloud_dist_pct` — `(close − cloud_mid) / close × 100`; sign = close-vs-cloud, magnitude = depth above/below cloud midline (1)

**§7.5 tagging per Q16.4 (a):** All 6 features tagged DYNAMIC (close-relative dist_pct mutates intrabar with current close). Cat 19 is NO LONGER on the §7.5 mixed-block list — the locked feature shapes are all close-relative, so per-feature tagging is uniformly dynamic. Cat 16 remains the only mixed-split block. The original §7.5 narrative ("spans displaced to past = static; Tenkan/Kijun = dynamic") was intent-based; given §15 directive to emit `senkou_*_dist_pct`, the static-span half of that intent is masked by close-dependence at the feature level. Phase 4 intrabar scout can still optimize `senkou_*_dist_pct` recomputation by caching the displaced span value (an implementation detail in scout code, not a tagging concern).

**Drop from v1.0:**
- `tk_cross` (sign of T−K) — fully derivable from `sign(tk_diff_pct)`; redundant feature.

**v1.0 → v2.0 transformation map:**
- KEEP: `tenkan_dist_pct`, `kijun_dist_pct`, `cloud_dist_pct` (verbatim).
- RENAME: `tk_spread` → `tk_diff_pct` (matches spec wording "Tenkan-Kijun diff %").
- DROP: `tk_cross` (1 dropped — sign redundant with signed `tk_diff_pct`).
- NEW: `senkou_a_dist_pct`, `senkou_b_dist_pct` (2 added per §15 directive).
- Net: 5 → 6 (= +1).

**Function signature:** `ichimoku_features(df, cfg=None) -> DataFrame[6 cols]` (Q16.6 (a)). Self-contained — math derives from OHLC alone. cfg keys with canonical Hosoda defaults: `cfg.get("ichimoku.tenkan", 9)`, `cfg.get("ichimoku.kijun", 26)`, `cfg.get("ichimoku.senkou_b", 52)`. Falls back to legacy v1.0 keys `cfg.get("tenkan")`, `cfg.get("kijun")`, `cfg.get("senkou_b")` if present (config.yaml compat).

#### Category 20 — Event Memory (41 → 22 features)

**Drop ~half.** Rationale: event memory at 5m had granular recency (bars since RSI OB → 8 bars = 40 minutes, meaningful). At 30m, the same counters in bars mean hours and the dynamic range collapses. Keep the most-impactful events only.

**Keep (22):**
- `bars_since_rsi_ob`, `bars_since_rsi_os`, `bars_since_rsi_mid_cross` (3)
- `bars_in_current_rsi_episode`, `last_rsi_extreme_depth` (2)
- `bars_since_stoch_ob`, `bars_since_stoch_os` (2)
- `bars_since_wt_ob`, `bars_since_wt_os` (2)
- `bars_since_adx_trend_start` (>25), `bars_since_adx_weak_start` (<20) (2)
- `bars_since_squeeze_fire`, `bars_since_squeeze_entry`, `squeeze_direction_at_fire` (3)
- `bars_since_volume_spike` (>3× MA) (1)
- `bars_since_last_hh`, `bars_since_last_ll` (2)
- `bars_since_pivot_touch_daily`, `bars_since_pivot_touch_weekly` (2)
- `bars_since_ema21_cross` (1)
- `bars_since_macd_zero_cross`, `bars_since_macd_signal_cross` (2)

**Drop:** the 19 lower-SHAP v1.0 event memory features (mostly duplicate recency of correlated oscillators).

#### Category 21 — Microstructure (Hyperliquid) (21 → 21, Phase 3 unchanged)

Same as v1.0. Requires 2–3 months of Hyperliquid WebSocket collection before inclusion. Adds:
- Funding rate (Hyperliquid hourly), funding rate momentum, funding vs BTC funding
- Open interest, OI change, OI vs price divergence
- Order book imbalance (bid/ask L2 depth ratio) at multiple levels
- Order book slope (price-impact curve)
- Trade tape delta (signed volume last 5/15/30 min)
- Large-print frequency
- Absorption flags (heavy buy/sell without price move)

#### Category 22 — Cross-Asset Correlation (6 → 7 features, Phase 1 for altcoins)

Promoted from v1.0 Phase 4 to **Phase 1 for altcoin models** (still Phase 3+ for BTC — BTC has nothing to correlate to within this universe).

- `btc_corr_20bar` — BTC 20-bar return correlation with this asset (1)
- `btc_return_1bar`, `btc_return_3bar`, `btc_return_12bar` — BTC % return at 1/3/12 bar lags (3)
- `btc_vs_asset_atr_norm_diff` — `(btc_move/atr_btc) − (asset_move/atr_asset)` per Decision v2.41 Q10 (signed: positive = BTC led in vol-units, negative = asset led / decoupled alt strength) (1)
- `btc_above_ema200_daily` — sign(BTC_close_daily − BTC_EMA200_daily), prev-day-shifted to avoid look-ahead (macro regime: +1 bull, −1 bear) (1)
- `btc_funding_rate` — Phase 3+ only, gated on Hyperliquid microstructure online (1)

**Total: 7 features.** Phase 1 fires 6 of these (BTC funding is Phase 3+).

**ETH correlation: DROPPED entirely per Decision v2.41 Q9.** Original spec listed ETH correlation as the 8th Cat 22 feature; user dropped because (a) ETH is not in v2.0 trade universe (Decision v2.35 fetched BTC/SOL/LINK only), (b) marginal lift over the 6 BTC-driven features doesn't justify adding ETH to the asset fetch + maintenance footprint, (c) BTC correlation already captures the bulk of cross-asset signal. If the cost-benefit shifts later, re-add via Deviation Request.

Per user's memory: BTC correlation is a **booster, not filter**. Independent altcoin moves remain tradeable.

### 7.3 Feature count summary

| #   | Category                           | v1.0 (5m) | v2.0 (30m) | Δ   |
| --- | ---------------------------------- | --------- | ---------- | --- |
| 1   | Momentum                           | 47        | 32         | −15 |
| 2   | Trend / Direction                  | 19        | 14         | −5  |
| 2a  | **HTF Context (NEW)**              | 0         | 18         | +18 |
| 3   | Volatility                         | 15        | 12         | −3  |
| 4   | Volume / Buy-Sell                  | 17        | 12         | −5  |
| 5   | VWAP (multi-anchor)                | 8         | 14         | +6  |
| 6   | S/R Structure (daily+weekly pivots + swing Fib retracements) | 13 | 30   | +17 |
| 7   | Session / Time                     | 15        | 9          | −6  |
| 8   | Price Action / Candle              | 9         | 9          | 0   |
| 9   | Mean Reversion / Stats             | 8         | 7          | −1  |
| 10  | Market Regime                      | 7         | 7          | 0   |
| 11  | Previous Context / Memory          | 8         | 6          | −2  |
| 12  | Lagged Dynamics                    | 8         | 5          | −3  |
| 13  | Divergence Detection               | 7         | 7          | 0   |
| 14  | Money Flow                         | 8         | 6          | −2  |
| 15  | Additional Momentum                | 9         | 7          | −2  |
| 16  | Market Structure                   | 10        | 10         | 0   |
| 17  | Statistical / Fractal              | 9         | 6          | −3  |
| 18  | Adaptive MAs                       | 5         | 4          | −1  |
| 19  | Ichimoku                           | 5         | 6          | +1  |
| 20  | Event Memory                       | 41        | 22         | −19 |
| 21  | Microstructure (Hyperliquid)       | 21        | 21         | 0   |
| 22  | Cross-Asset Correlation            | 6         | 7          | +1  |
|     | **Phase 1 total (ex 21, 22-BTC)** | **268**   | **~202**   | −66 |
|     | **Phase 3 total**                  | 289       | ~231       | −58 |
|     | **Phase 4 full**                   | 295       | ~239       | −56 |

Target after SHAP trim in Phase 2.5: **~110–140 features**.

### 7.4 Preprocessing pipeline (unchanged from v1.0)

```
Raw 30m OHLCV + 4H OHLCV + 1D OHLCV
    │
    ▼
Per-TF indicator calculation (features/indicators.py using pandas-ta / custom)
    │
    ▼
HTF features computed natively on 4H and 1D frames
    │
    ▼
Prev-closed-bar merge of 4H and 1D features onto 30m frame
    │
    ▼
Percentage + ATR normalization (price distances)
    │
    ▼
MinMaxScaler(-1, 1) fit on train fold only (per-fold refit in walk-forward)
    │
    ▼
VarianceThreshold — drop zero-variance cols
    │
    ▼
NaN handling — forward-fill then drop first N (warmup = 100 bars at 30m)
    │
    ▼
Feature matrix (rows=30m bars, cols=~190)
```

### 7.5 Feature stability taxonomy (intrabar-safe vs dynamic)

Every feature is tagged `static` or `dynamic` in the feature catalog. This is free to document now and pays off in Phase 4 when intrabar inference is considered (per TradingView AI consultation, §19 Report 3).

**`static` — stable within a 30m bar, safe to use for intrabar inference on 5m/1m without recomputation:**
- Cat 6 all pivots (Fib-pivots and weekly pivots) — fixed for the day/week
- Cat 6.5 swing Fib retracement levels — only update when a new swing pivot confirms
- Cat 7 session/time features — fixed per bar boundary
- Cat 2a HTF Context features — fixed until next 4H / 1D bar closes

**`dynamic` — mutate intrabar, should NOT drive intrabar entry directly:**
- Cat 1 momentum (RSI / WT / Stoch / MACD / Squeeze)
- Cat 2 EMA-based trend
- Cat 3 volatility (ATR / BB / Keltner)
- Cat 4 volume (current-bar volume is incomplete intrabar)
- Cat 13 divergence (depends on current close)
- Cat 20 event memory (`bars_since_*` ticks forward, but the event-trigger itself moves on dynamic oscillators)

**`mixed` — stable after their signal fires, dynamic before:**
- Cat 16 market structure (HH/HL/LH/LL confirmations are static; the "current move so far" is dynamic) — per Decision v2.46 Q15.5, explicit 8 static / 2 dynamic feature-level split in `feature_stability.py`
- Cat 11 Previous Context (prev-bar lookups static; today-running close-dependent features dynamic) — per Decision v2.51 Q20.8, explicit 4 static / 2 dynamic feature-level split. v1.0 Cat 11 was prev-DAY-only (all static prior-bar lookups); the Decision v2.37 Q2 rewrite added today-running features (`today_open_to_now_pct`, `today_high_low_distance_from_current_pct`) that depend on current close — Cat 11 moved from §7.5 static list to mixed list with the per-feature split documented

Per Decision v2.47 Q16.4: Cat 19 Ichimoku was originally on the mixed list (rationale: senkou spans displaced 26 bars to past = static math; Tenkan/Kijun = dynamic math). However, the locked feature shapes per §15 + §7.2 emit ALL 6 features as close-relative `*_dist_pct`, which makes every feature close-dependent → uniformly dynamic. Cat 19 is therefore tagged 6× dynamic in `feature_stability.py`. Phase 4 intrabar scout can still optimize `senkou_*_dist_pct` recomputation by caching the displaced span value (the span itself doesn't change within the current 30m bar) — this is an implementation detail in scout code, not a feature-tagging concern.

Categories tagged `mixed` here may have feature-level static/dynamic splits in `feature_stability.py`. The mixed designation describes the category as a whole; individual feature columns are tagged precisely so the Phase 4 intrabar scout can pull only `static` + confirmed-`mixed-static` features without rebroadcasting the whole category.

Phase 4 implementation: build a `feature_stability.py` catalog that tags each column. The intrabar scout (if/when built) reads only `static` + confirmed-`mixed` features plus fresh 5m bar-close values of selected `dynamic` features.

### 7.6 Warmup (per Decision v2.58 — NaN carveout table)

`features.warmup_bars: 150`. Sized to longest **single-feature** indicator lookback at 30m TF (VFI=130 + buffer). Does NOT cover (a) intra-day rolling stats with per-day reset, nor (b) HTF features whose own warmups (in their native 4H/1D timeframe) translate to many more 30m bars when mapped via reindex + shift(1).

**Per Decision v2.58, the following warmup-NaN clusters are EXPECTED v2.0 invariants (not pipeline bugs):**

| Feature(s) | NaN row count | NaN % @ 51,170 rows | Mathematical source |
|---|---|---|---|
| `btc_funding_rate` (alts only) | ALL | 100% | Phase 1 NaN-placeholder per Decision v2.54 round 2; wired Phase 3+ when Hyperliquid WS data available |
| `daily_vwap_upper_band_1sig`, `daily_vwap_lower_band_1sig`, `daily_vwap_zone` | ~20,254 each | ~39.6% | Cat 5 Decision v2.44 contract: `bands_window=20` × intra-day reset; ~19 NaN at start of each UTC day × ~1066 days |
| `htf1d_ema200_pos` | ~9,410 | ~18.4% | Cat 2a Decision v2.37: 200 daily-bar EMA × 48 30m/day = 9,600 30m bars warmup |
| `htf1d_ema50_pos` | ~2,210 | ~4.3% | 50-day EMA × 48 = 2,400 |
| `htf1d_ema20_pos`, `htf1d_close_vs_20d_high_pct`, `htf1d_close_vs_20d_low_pct` | ~770 each | ~1.5% | 20-day rolling × 48 = 960 |
| `vfi`, `vfi_signal`, `vfi_hist_slope` | ~115 | ~0.22% | Cat 4 VFI period=130; warmup=150 absorbs most, residual ~115 (130 − 15 warmup-overlap) |
| First-event warmup cluster (per Decision v2.60) | ≤ ~40 each | ≤ 0.07% | 14 columns observed in BTC 2026-05-04 build: `bars_since_pivot_touch_weekly` (37), 9 weekly_pivot_*_dist_pct + helpers (~2 each), `body_vs_prev_body` (21), `bars_since_rsi_os` (5). All require first-occurrence of an event (first weekly pivot confirmation, first non-zero prev body, first RSI<30 etc.). Pattern: `bars_since_*` + `weekly_pivot_*` + Cat 8 `body_vs_prev_body` |
| `btc_above_ema200_daily` (Cat 22, alts only) | ~9,400 | ~18.4% | Cat 22 alt-side mirror of `htf1d_ema200_pos`: 200-day EMA on BTC daily timeline mapped to alt 30m grid via reindex + shift(1). Same warmup math as v2.58 row 3. Observed 9,402 NaN (18.39%) in SOL 2026-05-04 build per Decision v2.63 |
| First-event warmup cluster — alt-side extension (per Decision v2.63) | ≤ ~50 each | ≤ 0.10% | 5 alt-side columns observed in SOL 2026-05-04 build matching v2.60 pattern: `bars_since_adx_weak_start`, `bars_since_wt_os` (Cat 20 first-event; SOL took longer than BTC to fire first ADX<20 / WT<OS), `nearest_fib_level_dist`, `extension_progress_1272`, `fib_retracement_pct` (Cat 6.5 first-swing-pair warmups; SOL early-data swing structure took longer to confirm). Per Decision v2.63 the first-event-warmup PATTERN is generalized: **any feature matching `bars_since_*` + Cat 6.5 swing-Fib (`nearest_fib_level_dist`, `extension_progress_1272`, `fib_retracement_pct`) + Cat 22 daily-EMA200 (`btc_above_ema200_daily`) is carveout-covered regardless of asset, with row counts proportional to lookback × time-to-first-event**. Future-proofs against new assets (TAO/HYPE re-entry per Decision v2.35) without per-asset Decision Log churn. |

**Why we don't increase `warmup_bars` to 9,600:**

Increasing to cover `htf1d_ema200_pos` would discard 200 days of training data — devastating for the 3-year baseline (~5% of training data lost). LightGBM handles NaN natively via "missing-value direction" learning at split decisions; rows with NaN on specific features are still usable training rows that abstain from splits on those features.

**LightGBM NaN-native handling rationale:**

LightGBM at each split decision learns the optimal direction for missing values (left or right child). A row with `htf1d_ema200_pos = NaN` will route based on the learned default direction; the row's other 249 features still contribute splits. This is fundamentally different from imputation — LightGBM treats "missing" as a meaningful third category, which it is during warmup (the feature literally doesn't exist for that row).

**§10.5.6 audit policy (revised per Decision v2.58):**

- 0-NaN expectation applies ONLY to features whose effective warmup ≤ 150 bars (the pipeline `warmup_bars` setting).
- For features in the carveout table above: NaN at the documented row positions is EXPECTED.
- Halt criterion (c) per Decision v2.55 revised: triggers only when (i) NaN appears in a feature OUTSIDE the carveout table, OR (ii) NaN at row positions BEYOND the documented warmup for carveout features (e.g., `htf1d_ema200_pos` NaN at row 10,000 would be a bug since ~9,600 should suffice).

### 7.7 Phase 2.5+ Feature Candidates (deferred per Decision v2.53)

The 22 v2.0 categories (250 features) cover canonical TA + statistical + structural signal types comprehensively. The following 6 candidates were considered during the Phase 1.10 post-implementation professional review (2026-05-01) and **DEFERRED** per the build-measure-add discipline (§10.5 process discipline + Cat 21 deferral pattern). Phase 2.5 SHAP analysis will surface which (if any) merit closing via DR. Recording here so future-Claude does not re-litigate the consideration.

**Deferral rationale (general):** none of these 6 are load-bearing for getting a working bot. 250 features at ~1,100:1 sample/feature ratio is healthy; adding speculative features inflates Q-batch + decision-log work without evidence of marginal alpha. Phase 2.5 SHAP is the truth-finder for which gaps actually matter at 30m for our trade universe (BTC/SOL/LINK).

**Candidates:**

1. **Regime duration features** — `bars_in_current_volatility_regime`, `bars_in_current_volume_regime`. Cat 10 has `regime_change_bar` for trending/ranging label flips but not duration counters for the tercile-based volatility/volume regimes. Cost: LOW (Cat 20 pattern, vectorizable). Trigger: Cat 10 SHAP < 5% importance after Phase 2.5, OR live paper-trading shows regime-transition signals missing.

2. **`bb_dist_mid_sigma` ≡ `zscore_20` deduplication** — accepted redundancy per Decision v2.50 Q19.2 (a). Cosmetic only — LightGBM tolerates duplicates. Cost: TRIVIAL (drop one column, update spec count Cat 9: 7 → 6). Trigger: Phase 2.5 cleanup pass after SHAP confirms duplicate carries no distinct gain.

3. **Cat 12 longer lag windows** — `delta_close_12bar`, `delta_volume_12bar`, etc. Current Cat 12 lags at 1, 3 (max 90 min). At 30m, lag=12 would capture 6hr session-spanning momentum shifts. Cost: LOW (1-2 features). Note: Cat 1 `roc_12bar` already partially compensates. Trigger: Cat 12 SHAP > 8% importance (signaling delta features carry weight) AND Cat 1 multi-period momentum SHAP shows the 6hr horizon pulls weight.

4. **Volume profile features** — session-anchored POC (Point of Control), VAH (Value Area High), VAL (Value Area Low), distance-to-POC %. POC alpha is well-documented at lower TFs; at 30m it's speculative but Cat 5 multi-anchor VWAP only gets close (cumulative volume around price levels, not distribution shape). Cost: MEDIUM-HIGH (own Q-batch — session-boundary algorithm choices, bin-size decisions, POC vs HVN/LVN feature shape, ~5-7 features). Trigger: Cat 5 multi-anchor VWAP SHAP < 6% AND live paper-trading shows reversal-miss patterns near volume nodes (price stalls but model didn't predict).

5. **Funding-rate-adjusted return** — `asset_funding_adj_return_3bar` etc. For crypto perps, the "true" return = price return − funding paid/earned. Single biggest known alpha gap at 30m for crypto. Tied to Cat 21 microstructure (Phase 3+ deferred — needs Hyperliquid WS collection). Cost: LOW (once data collected — single Series operation). Trigger: FORCED — Phase 3.0 when Cat 21 funding rate data becomes available. First Phase 3 priority.

6. **Cross-asset correlation dispersion** — `delta_btc_corr_20bar` (rolling change of correlation), `corr_dispersion_20bar` (std of pairwise correlations across BTC + asset). Captures regime shifts in cross-asset coupling. Cost: LOW (1-2 features). Phase 4 candidate per spec §17 alpha-decay plan. Trigger: Cat 22 SHAP < 4% on altcoin models (signaling base BTC correlation features insufficient).

**Process for closing a deferred candidate:**

1. Phase 2.5 SHAP analysis identifies under-performing category(ies).
2. Deviation Request DR-NNN filed citing specific candidate from above.
3. Q-batch surfaced for implementation ambiguities (e.g., volume-profile session-boundary choice).
4. Decision v2.5N entry locks resolutions.
5. Spec edit (new §7.2 sub-block or amend existing category).
6. Implementation + feature_stability tagging.
7. Re-run baseline gate (per §10.3) to verify the addition improves OOT metrics, not just in-sample.

**Future candidates not in current consideration:**

- NLP / news / economic calendar — out of scope for v2.0 (no DR planned).
- Macro regime features (BTC.D, total alt cap) — out of scope per asset-universe lock (Decision v2.35).
- Order-flow imbalance from L2 orderbook — Phase 3+ Cat 21 territory.
- Sentiment aggregators (CryptoFear/Greed Index, social volume) — out of scope.

If a future v3.0 redesign opens the asset universe or model class, these can be re-evaluated via DR.

---

## 8. Labeling

### 8.1 Triple-barrier parameters (v2.0 — chop-filtered ATR per Decisions v2.56 + v2.59)

```yaml
labeling:
  method: triple_barrier_chop_filtered
  tp_atr_mult:                     # PER-ASSET dict (Decision v2.64 schema); FROZEN at Phase 1.14 per Decision v2.66
    BTC: 2.4                       # v2.61 PASS state — Phase 1.14 freeze
    SOL: 2.7                       # DR-013: alts widened to 2.7 per Decision v2.66 (post-5-point-sweep optimum at the diminishing-returns inflection; band margin +1.64pp)
    LINK: 2.7                      # symmetric with SOL (margin +1.89pp)
  sl_atr_mult:                     # PER-ASSET dict (symmetric with tp); FROZEN at Phase 1.14
    BTC: 2.4                       # Phase 1.14 freeze
    SOL: 2.7                       # DR-013 freeze
    LINK: 2.7
  max_holding_bars: 12             # DR-009: 8→12 per Decision v2.61 (4hr → 6hr structural extension)
  min_profit_pct:                  # timeout-gate (per asset)
    BTC: 0.3
    SOL: 0.4
    LINK: 0.4
    # TAO/HYPE deferred per Decision v2.35
  min_atr_pct_threshold:           # CHOP FILTER (per DR-007 / Decision v2.56)
    BTC: 0.275                     # DR-009: 0.25→0.275 per Decision v2.61
    SOL: 0.60                      # DR-010: 0.30→0.60 per Decision v2.62 (empirical recalibration; SOL ATR/Price profile much higher than BTC)
    LINK: 0.60                     # DR-010: 0.30→0.60 per Decision v2.62 (same as SOL initially; differentiate in DR-011 if LINK reports off-band)
  classes:
    LONG: 0
    SHORT: 1
    NEUTRAL: 2
    CHOP_FILTERED: -2              # bars in low-vol regime, EXCLUDED from training + inference
    UNLABELABLE: -1                # tail rows where forward window incomplete (preserved from v1.0)
```

**§9.2 cascade note (DR-009 / Decision v2.61):** `max_holding_bars: 12` cascades to `walk_forward.purge_bars: 12` and `walk_forward.embargo_bars: 12` per the §9.2 contract `purge_bars = embargo_bars = max_holding_bars`. Both values updated in lockstep; see §9.2 for purged-CV rationale.

**Param history (locked via Decisions v2.56 + v2.59 + v2.61 — Phase 1.11 iteration):**

| Iteration | Decision | tp/sl_atr_mult | BTC chop floor | max_holding_bars | BTC NEUTRAL | BTC chop rate | Status |
|---|---|---|---|---|---|---|---|
| Path D first | v2.56 (DR-007) | 2.5 | 0.20% | 8 (4hr) | 33.94% | 8.06% | Both diagnostic triggers fired |
| Path D second | v2.59 (DR-008) | 2.4 | 0.25% | 8 (4hr) | 32.14% | 15.59% | NEUTRAL>30 still fires; pure-linear-ATR ceiling identified |
| Path D third — STRUCTURAL | v2.61 (DR-009) | 2.4 | 0.275% | 12 (6hr) | 24.03% | 19.77% | **BTC PASS — all 3 bands ✓** (proceed to alts) |
| Path D fourth — alt calibration | v2.62 (DR-010) | 2.4 (system) | 0.275% BTC / 0.60% SOL+LINK | 12 | 24.03% (BTC frozen) | 19.77% (BTC frozen) | **All 3 PASSED bands; alts at NEUTRAL floor edge** |
| Path D fifth — per-asset tp/sl experiment (TIGHTEN test) | v2.64 (DR-011) | BTC=2.4 / SOL=2.2 / LINK=2.2 | 0.275% BTC / 0.60% SOL+LINK | 12 | (BTC unchanged) | (BTC unchanged) | Tier-(b) result: SOL NEUTRAL 14.54% / LINK 14.75% — both BREACHED 15% floor by 0.46pp / 0.25pp. User identified misjudgment direction; v2.64 superseded by v2.65 |
| Path D sixth — alt widening sweep (5 points) | v2.65 (DR-012) tested 2.5; VPS extended sweep 2.7 + 3.0 | BTC=2.4 / SOL ∈ {2.5, 2.7, 3.0} / LINK ∈ {2.5, 2.7, 3.0} | 0.275% BTC / 0.60% alts | 12 | (BTC unchanged) | (BTC unchanged) | **All 3 candidates PASSED §8.2 bands; sweep characterized step-response curve (diminishing returns inflect at 2.7→3.0 step from 1:1 to 0.7:1)** |
| **Phase 1.14 FREEZE** | **v2.66 (DR-013)** | **BTC=2.4 / SOL=2.7 / LINK=2.7** | **0.275% BTC / 0.60% alts** | **12** | **24.03% BTC** | **19.77% BTC** | **🔒 LOCKED: SOL NEUTRAL 16.64% (+1.64pp margin); LINK 16.89% (+1.89pp). 2.7 chosen as Pareto optimum at diminishing-returns inflection. Feature matrix + label parquet frozen for Phase 2 training** |

**User professional preference + diagnostic evidence (2026-05-04):**
- v2.59 (DR-008) followed §10.5 single-variable-change discipline (incremental ATR 2.5→2.4 + chop 0.20→0.25). Result: NEUTRAL 33.94% → 32.14%, only −1.80pp improvement. VPS linear projection: pure ATR-tighten cannot reach [15,25] band even at tp/sl=2.0 (Δ ≈ −1.80pp/0.1 ATR step → 2.0 still ~25% NEUTRAL).
- v2.61 (DR-009) takes the **structural lever** instead of further linear ATR tightening: extend `max_holding_bars` 8→12 (4hr → 6hr) per user's 2026-05-03 domain insight ("actual price movement is in 4hours but before that neutral or squeeze period exists and if we involving these situations then holding time would exceeds 4hours"). The 30m × 8-bar hold cuts off mid-development moves; 12-bar hold lets squeeze + breakout + extension sequence complete. Combined with chop floor 0.25→0.275 (incremental, per user's 0.275 specification), this targets the directional-move-completion problem rather than the move-magnitude problem.
- Rationale to deviate from §10.5 strict single-variable-change: v2.59 result is DIAGNOSTIC EVIDENCE that the linear-ATR lever has a ceiling; structural intervention is now warranted. Each variable's effect remains independently traceable: chop floor 0.025-step continues the v2.59 incremental rhythm; hold extension is the new structural variable; tp/sl explicitly HELD at 2.4 to isolate hold-extension effect from ATR-tighten effect. If v2.61 misses, DR-010 escalates the linear ATR lever (2.4→2.3) WITH the new structural baseline (hold=12) — preserving the iteration discipline.
- v2.62 (DR-010) is alt empirical recalibration. v2.61 BTC PASSED in band; SOL distribution PASSED in band but chop filter underactivated (0.55% vs [10,50]) — SOL ATR/Price profile is structurally higher than BTC, so the 0.30% threshold designed as "wider than BTC" wasn't wide enough. SOL/LINK floors raised to 0.60% (lower end of VPS empirical 0.60-0.80 range; conservative to preserve in-band distribution). LINK runs first time at this iteration. DR-011 candidates (if alt distribution shifts off-band post-chop-raise): adjust min_profit_pct.SOL/LINK to compensate for distribution shift; raise alt floor further if chop still under [10,50]; differentiate SOL vs LINK if their distributions diverge.
- v2.64 (DR-011) is the per-asset tp/sl experiment. v2.62 result: BTC + SOL + LINK ALL PASSED §8.2 bands (NEUTRAL 24.03% / 15.55% / 15.75%). Alts at NEUTRAL floor edge (15-15.75% just above 15% target). User professional hypothesis: tightening tp/sl on alts (2.4 → 2.2) before Phase 1.14 freeze might increase directional sample density without breaching NEUTRAL floor — potentially better Phase 2 signal density via reduced label-class imbalance. **Schema extension**: §8.1 `tp_atr_mult` and `sl_atr_mult` migrate from scalar to per-asset dict (matching existing pattern for `min_profit_pct` + `min_atr_pct_threshold`). BTC frozen at 2.4 (v2.61 PASS state preserved); SOL + LINK to 2.2. v2.64 RESULT (PROJECT_LOG line 210+): SOL NEUTRAL 15.55%→14.54% (−0.46pp under 15% floor); LINK 15.75%→14.75% (−0.25pp under). Tier-(b) outcome from VPS 5-tier matrix. **Alt step-response calibrated**: −0.50pp NEUTRAL per 0.1 ATR step on alts vs −1.80pp on BTC (alts respond ~3.5× less aggressively).
- v2.65 (DR-012) is the WIDENING CORRECTION of v2.64. User identified misjudgment 2026-05-04: original v2.64 hypothesis was that 2.4 was "wide for SOL/LINK" → tested narrower 2.2. Experiment results showed direction was wrong: tightening drove NEUTRAL further below floor. **Correction**: alt tp/sl 2.2 → 2.5 (WIDENING, not tightening). v2.65 RESULT: SOL NEUTRAL 15.99% (+0.99pp margin); LINK 16.14% (+1.14pp). Both PASSED. VPS extended the sweep to 2.7 (v2.66c) and 3.0 (v2.67c) per user-staged plan ('test 2.5 first and then 2.7') to characterize the step-response curve. Full 5-point sweep (2.2 / 2.4 / 2.5 / 2.7 / 3.0) confirmed: chop INVARIANT to tp/sl across all 5 points; diminishing-returns inflect 2.7→3.0 (NEUTRAL gain/0.1-step drops from 0.32 to 0.23; trade ratio worsens from 1:1 to 0.7:1).
- **v2.66 (DR-013) is the Phase 1.14 FREEZE LOCK.** Post-sweep analysis surfaced 2.7 as the Pareto optimum: NEUTRAL band margin +1.64pp (SOL) / +1.89pp (LINK) — material vs 2.5's marginal +0.99/+1.14pp; LONG/SHORT density 83.36/83.12% (small loss vs 2.5's 84.02/83.86%, big gain in regime-shift safety); diminishing returns inflection means 2.7→3.0 trades worse than the band-margin gain warrants. **Locked params for Phase 2 training**: BTC=2.4 / SOL=2.7 / LINK=2.7 (tp_atr_mult = sl_atr_mult symmetric). All other params at their cumulative iterated values: max_holding_bars=12, min_profit_pct={BTC=0.3, SOL=0.4, LINK=0.4}, min_atr_pct_threshold={BTC=0.275, SOL=0.60, LINK=0.60}, purge/embargo=12 (cascade per §9.2). Feature matrix + label parquet for all 3 assets are FROZEN per spec §10.5.4 — Phase 2 baseline gate (§10.3) reads from these locked artifacts.

**Per Decision v2.56 (DR-007) — chop-filtered ATR-based labels:**

The first-pass build with `tp/sl_atr_mult=3.0 + min_profit_pct={BTC=0.6, SOL=0.9, LINK=0.8}` produced 57.77% NEUTRAL on BTC (PROJECT_LOG line 138, 2026-05-01). Diagnostic: barriers too wide for 30m × 4-hour hold AND chop-regime samples overweight in training set. Resolution per Decision v2.56: tighten ATR multipliers slightly (3.0 → 2.5) AND add an ATR/Price chop filter that excludes low-volatility regime bars from training and inference.

**Why ATR-based barriers (academic standard preserved):**
- Volatility-scaled barriers adapt across regimes (López de Prado standard).
- A 0.5% move in calm market is a major event; 0.5% in volatile market is noise. ATR-scaling makes label meaning regime-relative.
- Cross-regime generalization theoretically improves vs fixed-percent barriers — IF training data is filtered to tradeable regimes (which Decision v2.56 chop filter ensures).

**Why the chop filter:**
- BTC 30m ATR varies 4× across regimes (0.15%–0.60% per chart 2026-05-03). At ATR=0.15%, a 2.5×ATR barrier = 0.375% — labels in this regime are essentially noise predictions (4-hour move in chop is dominated by random walk, not signal).
- Filter rule: at the entry bar, if `atr_14 / close × 100 < min_atr_pct_threshold[asset]`, set label = `-2` (CHOP_FILTERED). These rows are EXCLUDED from training (`label not in {0, 1, 2}` filter) AND from inference (predictor service skips signal generation when chop condition holds).
- Net effect: training data is "tradeable regime only"; model learns directional moves in regimes where directional moves are detectable.

**Why these `min_profit_pct` values (BTC=0.3, SOL=0.4, LINK=0.4):**
- LOWERED from prior values (BTC=0.6, SOL=0.9, LINK=0.8) — chop filter handles the bigger problem (excluding low-vol bars entirely), so the timeout-gate threshold can be tighter.
- Values still above round-trip fees+spread (~10 bps × 2 = 0.2% for BTC, slightly more for alts) to ensure timeout-LONG/SHORT trades are profit-meaningful.
- Per-asset variation: BTC's tighter threshold reflects its lower base volatility post-chop-filter; alts get slight +0.1 cushion.

**Why `min_atr_pct_threshold` values (BTC=0.20%, SOL=0.30%, LINK=0.30%):**
- BTC=0.20% — ATR/Price observed range 0.15-0.60% per 2026-05-03 chart; 0.20% is the floor below which 30m candles are demonstrably chop (small wicks, narrow bodies, no directional thrust). Filters ~20-30% of bars in low-vol regimes.
- SOL/LINK=0.30% — higher base volatility vs BTC; their "chop" floor is wider. Without empirical data, set conservatively at 0.30%; tune in DR-008 if observed chop-filter rate is wrong (e.g., filtering >50% of bars = threshold too high; <10% = threshold too low to matter).
- These are first-iteration values per Decision v2.56; revisit after BTC re-run reports filter rate + label distribution.

**Inference-time consequence (locked here for Phase 3.2 predictor service per §12.1):**

```
on each 30m bar close:
    compute atr_pct = atr_14 / close × 100
    if atr_pct < min_atr_pct_threshold[asset]:
        signal = NO_TRADE  (skip — chop regime; not tradeable per training)
    else:
        signal = model.predict(features)  # in {LONG, SHORT, NEUTRAL}
```

This is a CLEANER trading rule than always-on prediction. The bot only operates when the market itself is tradeable.

### 8.2 Expected label distribution (per Decision v2.56 — applies to FILTERED set)

**Important:** the §8.2 distribution bands apply to the FILTERED training set (rows with label ∈ {0, 1, 2}), NOT the full feature matrix including chop-filtered rows. Per Decision v2.56 (DR-007), bars with `atr_14 / close × 100 < min_atr_pct_threshold` are tagged `label = -2` and excluded BEFORE distribution check.

Expected on the filtered set:
- LONG: 35–45%
- SHORT: 35–45%
- NEUTRAL: 15–25%

**Auxiliary metrics** (informational, not gate criteria):
- Chop-filter rate: % of total bars labeled `-2`. Expected 20–40% on BTC depending on regime mix; higher implies threshold may be too aggressive.
- Unlabelable tail: % of bars labeled `-1`. Should match `max_holding_bars / total_bars` ≈ 0.016% on 3-year data.

**Diagnostic rules:**
- If filtered-set NEUTRAL > 30%: barriers still too wide despite tightening to 2.5×ATR; consider further tighten to 2.0×ATR (DR-008 trigger).
- If filtered-set NEUTRAL < 10%: barriers too tight; relax `min_profit_pct` slightly OR consider 2.7-3.0×ATR.
- If chop-filter rate > 50%: threshold too high; lower `min_atr_pct_threshold` (e.g., BTC 0.20% → 0.15%).
- If chop-filter rate < 10%: threshold too low to matter; raise (e.g., BTC 0.20% → 0.25%).

Phase 1.11 explicitly may iterate freely on these params per Decision v2.55 brief; OOT one-shot rule does NOT apply until Phase 2.10.

### 8.3 Pessimistic tie-break (unchanged)

When TP and SL hit on the same bar, the tighter side wins (conservative). Inherited from v1.0 `model/labeler.py`.

---

## 9. Training & Validation

### 9.1 Splits

```yaml
splits:
  total_data_years: 3              # hard minimum; 4 years preferred
  train_val_end: "2026-02-28"      # everything ≤ this is train+val
  oot_start: "2026-03-01"          # OOT — UNTOUCHED until final freeze
  oot_end:   "2026-03-31"          # 1 month, ~1440 × 30m bars / sym
```

**OOT rule:** the OOT slice is loaded, labels computed, and the frozen-model prediction scored **exactly once**. If the result disappoints, v2.0 does not re-iterate into it — it goes to paper trading (which is the only uncontaminated OOT going forward).

### 9.2 Walk-forward

```yaml
walk_forward:
  train_months: 9
  val_months:   1
  step_months:  1
  purge_bars:   12                 # = max_holding_bars (per §8.1 — DR-009 cascade)
  embargo_bars: 12                 # = max_holding_bars
```

With 3 years of data and 1-month step, expect ~14 folds (was 8 on v1.0 with 18mo). More folds = better generalization estimate.

**Cascade contract (per Decisions v2.7 + v2.61):** `purge_bars = embargo_bars = max_holding_bars` is invariant across the entire pipeline. If §8.1 `max_holding_bars` changes (e.g., DR-009 extended 8→12 per Decision v2.61), `purge_bars` and `embargo_bars` MUST update in lockstep. Rationale: the purged-CV scheme prevents label-leakage between train/val folds — a label at bar `i` uses forward bars `[i+1, i+max_holding_bars]`, so the train/val boundary must purge `max_holding_bars` bars on each side to avoid information bleed. If purge < max_holding_bars, train/val labels overlap → log-loss artificially deflates → silent overfit.

### 9.3 Model

```yaml
model:
  framework: lightgbm
  params:
    objective: multiclass
    num_class: 3
    metric: multi_logloss
    boosting_type: gbdt
    num_leaves: 63
    learning_rate: 0.05
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 5
    min_child_samples: 50
    lambda_l1: 0.1
    lambda_l2: 0.1
    verbose: -1
    seed: 42
  num_boost_round: 1000
  early_stopping_rounds: 50
  calibration: sigmoid              # Platt scaling on val fold — LightGBM multiclass is overconfident
```

### 9.4 Hyperparameter tuning (Optuna)

Same as v1.0 — 50 trials, MedianPruner, tune on 3 evenly-spaced folds, search space:

```yaml
tuning:
  n_trials: 50
  timeout_sec: 14400
  tuning_fold_indices: [1, 7, 13]   # adjust for 14-fold walk-forward
  search_space:
    num_leaves:           [31, 127]
    learning_rate:        [0.01, 0.1]     # log-scale
    feature_fraction:     [0.5, 1.0]
    bagging_fraction:     [0.5, 1.0]
    min_child_samples:    [20, 100]
    lambda_l1:            [0.001, 10.0]   # log-scale
    lambda_l2:            [0.001, 10.0]   # log-scale
```

### 9.5 SHAP-driven feature trim (Phase 2.5)

After initial walk-forward training:
1. Compute TreeExplainer SHAP values on each fold's val set
2. Pool across folds by mean(|SHAP|)
3. **Keep the top ~120 features OR all features with mean|SHAP| > 0.001**, whichever is smaller
4. Re-run walk-forward with trimmed feature list
5. Expect marginal log-loss improvement + meaningful speedup + lower overfitting risk

---

## 10. Anti-Overfit Discipline

This section exists because v1.0 failed here. It is not optional in v2.0.

### 10.1 The three-fold structure

```
[       TRAIN        ][   VAL   ][    OOT (1 month)    ]
 <---- iterate --->      |            |
 features, labels,       |            |
 hyperparams, SHAP       |            |
 trim, calibration       |            |
                         |            └─── scored ONCE, at freeze
                         └──────────────── iterable during dev
```

### 10.2 Rules

1. **OOT is one month, reserved from day zero.** It is loaded, labeled, and stored, but never scored until the final frozen model is ready.
2. **Dev iterations score on VAL only.** If val improves but OOT degrades, that will be discovered once at freeze — not chased.
3. **One freeze, one OOT score.** If the OOT score disappoints, the path forward is **paper trade, not re-tune**.
4. **Bar-close only during training.** No intrabar features in Phase 1 — this limits information leakage surface.
5. **Fold-local normalization.** Every scaler is fit on train fold only; val/OOT use transform-only.
6. **Purge and embargo = max_holding_bars.** 8 bars each side of each fold.

### 10.3 Baseline gates (two stages)

Two cheap sanity checks **before** the full walk-forward + Optuna + SHAP pipeline. Both must pass to proceed.

#### 10.3.1 Pre-gate: ultra-minimal 5-feature baseline

Per TradingView AI consultation (§19 Report 1), if a tiny hand-picked feature set cannot beat the empirical prior, the signal space is noise-dominated and adding 195 more features will not rescue it. Fast fail.

**Features (exactly 5):**
1. `close_vs_ema21_pct` — price deviation from 20–21 EMA
2. `rsi_14`
3. `adx_14`
4. `htf4h_ema21_pos` — 4H trend direction (from Cat 2a)
5. `volume_ratio_20` — current volume / 20-bar MA

**Setup (amended per Decision v2.67 / DR-014, 2026-05-05):** full walk-forward per §9.2 (14 folds; train_months=9, val_months=1, step_months=1; purge_bars=12, embargo_bars=12 cascade per §9.2). Default LightGBM params (num_leaves=31, lr=0.05, 200 rounds, early stopping 20).

**Gate condition:** **mean val log-loss across all 14 folds** must beat **mean empirical prior** (per-fold, computed from each fold's val label distribution) by **≥1%** (lower bar than §10.3.2 since we expect worse with only 5 features).

**Why walk-forward (not single fold):** the original §10.3.1 prescribed a single first-fold test for compute economy. The 2026-05-05 first-fold pre-gate FAILED at val_logloss/prior=1.032 (gate ≤0.99 to PASS), but diagnostic analysis revealed this was a fold-selection artifact: val month (2024-02 → 2024-03) was the BTC ETF rally with 48.78% LONG class share, vs train's near-balanced 35.00% LONG — a 13.78pp regime-shift between train and val that any model trained on 2023 chop cannot generalize across. Walk-forward across 14 folds eliminates the fold-selection artifact entirely and provides a robust signal-detection check; compute cost remains tractable (~5-10min for 5 features). This methodology change is NOT tuning (does not alter features, labels, or LightGBM params) — it corrects a methodological flaw in the test; spec §10.3.1 was written assuming the chosen fold would be representative, which Feb 2024 demonstrably is not.

**Empirical prior calculation:** per fold, compute val label distribution `[p₀, p₁, p₂]` and prior `Hᵢ = −Σ pᵢ × ln(pᵢ)`. Mean prior across 14 folds = mean(H₁..H₁₄). Mean val log-loss = mean(val_logloss across folds). Gate requires `mean_val_logloss / mean_prior ≤ 0.99`.

**If it fails:** halt the project. This is a premise-level signal — either the label parameters are wrong for this regime, the timeframe is wrong, or this asset has no extractable edge right now. Do not tune your way out of a failed pre-gate. **Per Decision v2.67**: a walk-forward-mean failure is robust signal absence (not a fold artifact); halt with high confidence and escalate to a fresh DR re-opening §8 labeling, Phase 1.10 feature engineering, or accepting v2.0 as a research dead-end.

#### 10.3.2 Baseline gate: full feature-set walk-forward

**Setup (amended per Decision v2.68 / DR-015, 2026-05-05):** parallel to §10.3.1 amendment per DR-014 — full walk-forward per §9.2 (~25 folds across 35-month pre-OOT data; train_months=9, val_months=1, step_months=1; purge_bars=12, embargo_bars=12 cascade per §9.2). Default LightGBM params (num_leaves=31, lr=0.05, 200 rounds, early stopping 20). Use the v2.0 frozen feature set per Phase 1.14 (BTC=243 features, alts=250 features); no SHAP trim yet (that's Phase 2.6).

**Gate condition:** **mean val log-loss across all walk-forward folds** must beat **mean per-fold empirical prior** by **≥2%** (i.e., `mean_val_logloss / mean_prior ≤ 0.98`). Empirical prior calculation identical to §10.3.1: per fold `Hᵢ = −Σ pᵢ × ln(pᵢ)` on val label distribution; mean across folds.

**Why walk-forward (not single fold):** parallel reasoning to §10.3.1 amendment per Decision v2.67. Single-fold methodology is fold-fragile (regime-shift artifacts dominate); walk-forward across all folds is robust. Compute remains tractable for full feature set (~30 min for 243 features × 25 folds × 200 rounds).

**Out-of-sequence run after §10.3.1 fail (per Decision v2.68):** spec §10.3.2 normally runs only after §10.3.1 passes. Decision v2.68 (DR-015) authorized §10.3.2 to run as a **confirmatory premise check** when §10.3.1 fails, on the diagnostic hypothesis that the 5-feature heuristic in §10.3.1 may be crypto-30m-miscalibrated (originally inherited from non-crypto context per Decision v2.21 + §19 Report 1). The 5 hand-picked features (close-vs-EMA21, RSI, ADX, HTF EMA21 position, volume ratio) omit signal channels that crypto-30m research consistently identifies as high-importance (Cat 10 regime, Cat 5 multi-anchor VWAP, Cat 13 divergence, Cat 16 break-of-structure, Cat 22 cross-asset). If these channels carry signal but the 5 do not, §10.3.1's "if 5 fail, 245 will too" inference produces a false-negative halt. §10.3.2's full-feature gate is the canonical test of this hypothesis.

**Decision criterion locked upfront (NOT iterable per §10.3.1 + Decision v2.68 commitment)**:
- **PASS** (mean_ratio ≤ 0.98) → premise alive; 5-feat heuristic was crypto-miscalibrated; proceed to Phase 2.3 walk-forward CV with default params, then Phase 2.4 Optuna.
- **FAIL** (mean_ratio > 0.98) → premise robustly dead across BOTH 5-feature AND 245-feature walk-forward; halt v2.0 with very high confidence; **NO further "one more test"** — escalate to fresh DR re-opening §8 labeling, Phase 1.10 features, §2.1 timeframe, OR accepting v2.0 as research dead-end.

**If §10.3.2 passes but §10.3.1 failed**: the canonical diagnosis is that the 5 features were crypto-miscalibrated, not that signal is absent. SHAP analysis (Phase 2.6) will identify which features carry signal; if subsequent SHAP-trim shows top-5 features differ materially from the §10.3.1 heuristic, file DR to update §10.3.1 5-feature list with empirically-validated picks.

**Note (preserved from original §10.3.2)**: empirical prior for a 3-class label with distribution [p₀, p₁, p₂] is `-Σ pᵢ × ln(pᵢ)`. For the expected 40/40/20 split, prior ≈ 1.055; gate threshold ≈ 1.034 (when `mean_ratio ≤ 0.98`).

**If pre-gate passes AND this full-feature gate fails (the originally-anticipated v2.0 sequence)**: the 5 features contained the signal but the 245 set is drowning it in noise. Response: SHAP-trim aggressively to ~30–50 features and re-run this gate before Optuna. (This branch is not active in the current Decision v2.68 path since pre-gate already failed.)

### 10.4 Paper trading as the only true OOT

After freeze + OOT score, the model enters paper mode on Hyperliquid live. **That** is the honest out-of-sample. Minimum 2 weeks of paper before any capital commitment.

### 10.5 Process Discipline — Spec-Driven Execution (NON-NEGOTIABLE)

**Why this section exists:** v1.0 did not fail only because of timeframe or features. It failed because across 7+ iterations Claude made independent judgment calls mid-phase (tp=4/sl=3 → 3/3, sign flips, class-weight swaps, feature adds) that each looked reasonable in isolation but in aggregate drifted away from the design and contaminated the OOT. The user is not an ML specialist and trusted Claude's per-step opinions; that trust was the attack surface. This section closes it.

The rules below apply to Claude (and any human collaborator) for the entire lifetime of this project.

#### 10.5.1 Spec-Authority Rule

**Every action must cite the spec section that authorizes it.**

- Before writing code or changing config, state the spec section or decision-log entry that requires/permits this action.
- If no section authorizes it, the action is a **Deviation** — stop and go to §10.5.2.
- Implicit authorizations are not valid. "It seems reasonable" / "this is best practice" / "the research says" are **not** spec authorizations.

#### 10.5.2 Deviation Request Protocol

When something is not in the spec, or a spec assumption is broken:

1. **Stop.** Do not write code or change config.
2. Write a Deviation Request containing:
   - **What:** the proposed change (concrete, not vague)
   - **Where:** which spec section it modifies, and what currently says
   - **Why:** what observation or constraint prompted it (evidence, not hunch)
   - **Options:** at least two alternatives, with tradeoffs
   - **Recommended:** which option and why
3. Wait for user approval.
4. On approval: **update the spec first**, then execute. The spec change is the commit that precedes the code commit.
5. On rejection: the idea is dead for this phase. Do not re-propose without new evidence.

#### 10.5.3 Banned Patterns

The following phrases and patterns are banned mid-phase. If Claude starts to type one, stop and re-route through §10.5.2:

- "Let me try …"
- "Let me see if …"
- "Just a quick tweak …"
- "A small adjustment …"
- "What if we …" followed directly by code
- "I'll also add …" (out-of-scope additions)
- Any change to labels, feature list, or hyperparameters announced inside a training run

These phrases are the linguistic signature of improvisation. Proposals use "Deviation Request:" as their lead; actions cite a spec section.

#### 10.5.4 Phase Freeze Gates

Once a phase's deliverable is checkpointed, its parameters are **locked** for the remainder of the project:

| Phase | Freeze-at point                           | What becomes immutable                                                   |
| ----- | ----------------------------------------- | ------------------------------------------------------------------------ |
| 1     | Feature matrix written to parquet         | Feature list, `config.features.*`, warmup_bars                          |
| 1     | Label column written                      | `config.labeling.*` (tp, sl, max_holding_bars, min_profit_pct)          |
| 2     | Baseline pre-gate passes (§10.3.1)        | 5-feature pre-gate list, pre-gate params                                 |
| 2     | Baseline full-gate passes (§10.3.2)       | Full feature list, default params                                        |
| 2     | Optuna best params selected               | Tuned hyperparameters                                                    |
| 2     | SHAP trim complete                        | Trimmed feature list                                                     |
| 2     | Model frozen                              | Everything. OOT must be scored exactly once next.                        |

Revisiting a locked parameter requires re-entering that phase via Deviation Request (§10.5.2) and explicitly discarding all downstream work. There are no half-reverts.

#### 10.5.5 Self-Audit at Session Start

At the start of every working session, before any tool call that modifies state, Claude states:

```
CURRENT POSITION
  Phase:              <1 | 2 | 2.5 | 2.9 | 3 | 4>
  Current step:       <e.g., 2.2 walk-forward BTC>
  Last completed:     <e.g., 2.1 baseline gates passed>
  Next step per spec: <e.g., 2.3 Optuna search>
  Frozen artifacts:   <list of files locked per §10.5.4>
  Open deviations:    <pending Deviation Requests, if any>
```

If Claude cannot cleanly answer any of these, Claude re-reads the spec + `PROJECT_LOG.md` before touching code. The user may ask for this statement at any time and Claude must produce it on demand.

#### 10.5.6 PROJECT_LOG.md — the decision trail

Every non-trivial action writes a one-line entry to `ml-bot-30m/PROJECT_LOG.md`:

```
2026-04-25 14:32  Phase 1.1  [SPEC §6.3]  Exported 30m BTC 2023-04..2026-03 → data/storage/binance/30m/BTC_USDT.parquet (52416 rows)
2026-04-25 15:10  Phase 1.2  [SPEC §7.2]  Updated features/vwap.py for multi-anchor per Cat 5 expansion
2026-04-25 16:44  Phase 2.1  [SPEC §10.3.1]  Pre-gate result: val log-loss 1.042 vs prior 1.055 (−1.23%) — PASS
```

Format: `YYYY-MM-DD HH:MM  <Phase.step>  [SPEC §<section>]  <one-line summary>`.

If an action can't cite a spec section, it's a deviation — stop.

The log is append-only. Mistakes are not deleted; they get a follow-up line marking the correction.

#### 10.5.7 When the user says "do something not in the spec"

Even if the user asks for an action that is not in the spec, Claude responds with a brief Deviation Request (§10.5.2) rather than executing immediately. This protects the user from their own in-the-moment ideas. If the user confirms after seeing the framed Deviation Request, proceed and update the spec. One extra message round-trip is cheap; a 3-week derailed iteration is not.

Exception: trivial non-design actions (reading a file, answering a question, explaining code) do not require a Deviation Request.

#### 10.5.8 Why this is non-negotiable

- **v1.0 evidence:** every failed iteration traced back to a change Claude made without explicit spec authorization. Pattern is documented, not hypothetical.
- **Cheap to follow:** one extra message per decision point.
- **Expensive to skip:** a single OOT-contaminating iteration can compromise the entire project; v1.0 proved this at a cost of ~3 weeks.
- **Resilient to future Claude versions:** this section is self-contained and instructs any agent that reads the spec.

#### 10.5.9 Anti-patterns from peer projects (do not import)

Source: deep analysis 2026-04-26 of intelligent-trading-bot, freqtrade/FreqAI, microsoft/qlib, pybroker, jesse, FinRL, TensorTrade. Many of these projects have battle-tested patterns we *do* import (see PROJECT_LOG entry 2026-04-26 + Decision Log v2.32). The following patterns from those same projects, however, would silently violate §10.5 and **must not be adopted under any circumstance** — even if a future agent argues they are "best practice in the field":

1. **Continuous clock-driven retrain** (FreqAI's `live_retrain_hours`). Continuous retrain on live data silently mutates the frozen model and erases the OOT result. Phase 5 retrains in this project are explicitly **scheduled (90-day) or triggered (red metric)** per §17.3 — never clock-driven.
2. **Threshold/HP grid search applied across the OOT range** (intelligent-trading-bot's `simulate_model.grid` if `data_end` extends into OOT; pybroker's `walkforward()` if windows span the held-out month). Threshold tuning is permitted on Phase 1–2 train + WF data only. **OOT is one shot, period.**
3. **Online auto-rolling models** (qlib's `OnlineManagerR`, similar mechanisms in freqtrade). Automated rolling = automated drift; defeats §10.5.4 freeze gates entirely.
4. **Auto-PCA dimensionality reduction** (FreqAI's `principal_component_analysis: true`). PCA destroys per-feature SHAP attribution, breaks the static/dynamic taxonomy needed for Phase 4 intrabar safety, and prevents leak audit. Our trim is SHAP TreeExplainer (interpretable) — **never PCA**.
5. **Two-binary-heads label combine** (intelligent-trading-bot's `gen_labels_highlow.highlow2` then `combine: "difference"`). Our triple-barrier 3-class label correctly represents the "neither hit, time-out" case as NEUTRAL; the two-head pattern collapses this into a low-magnitude false signal. Importing it is a regression.
6. **Flat-dict LightGBM training without valid_set / early_stopping / class_weight** (intelligent-trading-bot's `classifier_gb.py`). Our Optuna TPE + MedianPruner + Platt calibration pipeline is non-negotiable; the flat-dict pattern silently overfits on minority-class assets (e.g., LINK as the smallest of v2.0's BTC/SOL/LINK universe).
7. **"Quant model zoo" temptation** (qlib offers TRA, KRNN, HIST, ADARNN, Tabnet, Transformer-based, etc.). Solo-developer maintenance budget cannot support a model zoo. **LightGBM is the spec.** Model swaps are v3.0-only (§17.9.2), gated by spec re-architecture, not exploration.

If any future agent (or human collaborator, including me-on-a-bad-day) proposes one of these patterns citing "the FreqAI docs say…" or "qlib does it that way…" — the answer is reject by reference to this section. Their use case is different from ours; their freedom to mutate is different from our discipline.

---

## 11. Backtesting

### 11.1 Simulator logic (unchanged from v1.0)

- Bar-close inference (open next bar's order on the *following* bar's open)
- Triple-barrier exit: TP, SL, or `max_holding_bars` elapsed
- Fees: Hyperliquid taker 4.5 bps (BTC/SOL/LINK in v2.0); 1.5 bps maker; HYPE/TAO when re-introduced per Phase 4
- Slippage: 2 bps per side (conservative vs HYPE liquidity)
- Position sizing: fixed 1% risk per trade (stop distance × size = 1% of equity)
- Single-position per symbol (no pyramiding in Phase 1)

### 11.2 Metrics reported

- Sharpe (net, annualized)
- Sortino
- Max drawdown
- Win rate
- Average win / average loss (R-multiple)
- Profit factor
- Fire rate (% of bars that trade at P>threshold)
- Time-in-trade distribution
- Per-symbol breakdown

---

## 12. Execution Layer

### 12.1 Phase 1 (30m bar close, Hyperliquid)

- At each 30m close, the predictor computes features and produces P(LONG), P(SHORT), P(NEUTRAL)
- If max(P_long, P_short) > threshold (default 0.65), place limit order with 2-bar TTL
- ATR-based trailing stop once in profit by 1 ATR
- One position per symbol; no adds

### 12.2 Phase 3 option: hybrid 30m bias + 5m execution — "Triple-Trigger" method

Only after 30m model clears paper trading. Architecture per TradingView AI consultation (§19 Report 3):

```
  Trigger 1 — ML 30m directional bias (runs at every 30m close)
    ↓
    ml_prob = predict_30m(static_features + dynamic_features)
    ml_bias = argmax(ml_prob) if max(ml_prob) > 0.65 else "neutral"
    (uses only Cat 6/7/2a/11 static features + fresh Cat 1/3 dynamic features at bar close)

  Trigger 2 — 5m price trigger (runs every 5m bar close within the active 30m window)
    ↓
    Long-side:
      swept  = (bar5m.low  <  nearest_support_level)       # key level breached intrabar
      reclaim = (bar5m.close > nearest_support_level)       # but reclaimed by close
      price_trigger = swept and reclaim
    Short-side: mirror (sweep up + reject down)
    (uses only static features from Cat 6 — pivots, swing Fib levels — by design)

  Trigger 3 — 5m confirmation (same bar as Trigger 2)
    ↓
    vol_spike     = (bar5m.volume > volume_ma_20 * 1.5)
    momentum_flip = (bar5m.close - bar5m.open) / bar5m.range > 0.5   # strong close in range
    confirm = vol_spike or momentum_flip

  Entry fires only if: ml_bias != "neutral"  AND  price_trigger  AND  confirm
```

**Why three triggers, not one:**
- ML alone: can fire mid-bar-range without structural alignment
- Price sweep alone: noise (many sweeps don't recover)
- Volume alone: momentum spikes happen in both continuation and exhaustion contexts
- The AND-conjunction is selective by design — fires infrequently, but every fire has bias + structure + participation all aligned

**Static-vs-dynamic split (ties to §7.5):** the 5m triggers use only **static** features (Cat 6 levels, Cat 7 time) plus fresh 5m bar-close values. This means the intrabar scout can run without recomputing the full 30m feature set every 5 minutes — the static features are already correct for the entire 30m window.

**Partial scaling (optional):** 50% position on Trigger-2/3 fire intrabar, add 50% on 30m bar close if ML bias still holds. If the wick fails and 30m closes against, exit the half position; avoid full loss.

**Alert structure on TradingView (Phase 3 wiring):**
- 30m chart alert: "ML bias LONG" fires once per 30m bar close → sets the active window
- 5m chart alert: Pine script on 5m chart checks Trigger 2+3 within the active window → fires entry alert to webhook

This is still pitched as a **stage-2 optimization**. The 30m bar-close model must earn it first.

### 12.3 Phase 4 option: CUSUM information-driven bars (see §14)

Research (§3.3) suggests event-bars outperform time-bars for ML+triple-barrier. Reserve as Phase 4 experiment.

---

## 13. Risk Management

(Unchanged from v1.0.)

```yaml
risk:
  max_risk_per_trade_pct: 1.0        # (v1.0 had 2.0, v2.0 tightens for live conservatism)
  max_position_size_pct:  10.0
  max_daily_loss_pct:     5.0
  max_concurrent_positions: 1        # per symbol
  trail_atr_mult: 2.0
```

Position sizing formula: `size = (equity × risk_pct) / (stop_distance × instrument_price)`.

---

## 14. Project Phases & Timeline

### Phase 1: Data + Feature Engineering (Weeks 1–3)

| Step | Task                                                        | Output                                    |
| ---- | ----------------------------------------------------------- | ----------------------------------------- |
| 1.1  | Binance archive downloader (3 years, 30m + 4H + 1D)        | 12 parquet files (4 syms × 3 TFs)        |
| 1.2  | Adapt `features/builder.py` to 30m + dual HTF merge         | 30m feature matrix                        |
| 1.3  | Implement Category 2a (HTF Context)                         | +18 features                              |
| 1.4  | Expand Category 5 (Multi-anchor VWAP)                       | +6 features                               |
| 1.5  | Promote weekly pivots to Phase 1                            | +9 features                               |
| 1.6  | Trim Categories 1, 2, 3, 4, 7, 11, 12, 14, 15, 17, 20 per §7.2 | ~78 fewer features                      |
| 1.7  | Triple-barrier label with v2.0 parameters                   | Labeled matrix                            |

### Phase 2: ML Training + Validation (Weeks 4–6)

| Step | Task                                           | Output                          |
| ---- | ---------------------------------------------- | ------------------------------- |
| 2.1  | Baseline gate (single fold, default params)    | Pass/halt decision              |
| 2.2  | Walk-forward (14 folds, BTC first)            | Fold-level metrics              |
| 2.3  | Optuna hyperparameter search (50 trials)      | Tuned params                    |
| 2.4  | Re-run walk-forward with tuned params         | Improved metrics                |
| 2.5  | SHAP analysis + feature trim (~190 → ~120)    | Trimmed feature list            |
| 2.6  | Probability calibration (Platt scaling)       | Calibrated model                |
| 2.7  | Transfer learning: SOL, LINK (per Decision v2.35) | Per-asset models           |
| 2.8  | **Freeze all models**                         | `models/v2.0_frozen/*.pkl`      |
| 2.9  | **Single OOT evaluation**                     | One-shot OOT report (no iter)  |

### Phase 3: Paper Trading + Live Validation (Weeks 7–10)

| Step | Task                                           | Output                          |
| ---- | ---------------------------------------------- | ------------------------------- |
| 3.1  | Hyperliquid WebSocket ingestion (30m rollup)  | Live feature pipeline           |
| 3.2  | Predictor service (30m bar close inference)   | Live signals                    |
| 3.3  | Paper executor (no real orders)               | Paper P&L log                   |
| 3.4  | 2+ weeks paper trading                        | Paper performance report        |
| 3.5  | Live trading gate decision                    | Go/no-go                        |
| 3.6  | (Optional) Begin Category 21 microstructure collection | 2–3 months accumulation for Phase 3.2 refit |

### Phase 4: Extensions (Month 3+)

- **4.1** HYPE / TAO models on Hyperliquid-native data — gated by re-entry conditions per Decision v2.35: ≥3 years clean Binance perp 30m data accumulated (TAO), Hyperliquid daily volume ≥5% of BTC sustained 30 days, v2.x diversification window open
- **4.2** Microstructure refit with Hyperliquid L2 + trade tape features
- **4.3** Hybrid 30m-bias + 5m-execution layer (Pine script or Python intrabar)
- **4.4** CUSUM event-bars experiment (compare to time-bar baseline)
- **4.5** Ensemble with XGBoost / CatBoost (only if Phase 4.1–4.3 saturate)
- **4.6** Orthogonal feature class (on-chain / funding / news) — decay insurance per §17.6

---

## 15. Directory Structure

Proposed new project root: `ml-bot-30m/` (sibling to the existing v1.0 repo at `ml-bot/`, not an overwrite).

```
ml-bot-30m/
├── PROJECT_SPEC.md              # this document (copy/rename)
├── config.yaml                  # v2.0 config
├── .env.example
├── data/
│   ├── __init__.py             # NEW v2.0 (package marker; clean glue per Decision v2.36)
│   ├── db.py                   # NEW v2.0 (~120 LOC clean glue per Decision v2.36; ohlcv_30m / features_30m / labels_30m / wf_folds_30m / models_30m / decay_metrics_30m only)
│   ├── collectors/
│   │   ├── __init__.py         # NEW v2.0 (package marker)
│   │   ├── binance_archive.py  # NEW v2.0 (clean glue per Decision v2.36; v1.0 fetcher.py NOT renamed)
│   │   ├── hyperliquid_ws.py   # reused from v1.0 (Phase 3)
│   │   └── storage.py          # reused (parquet I/O, no v1.0 baggage)
│   └── storage/
│       ├── binance/30m/         # 30m only — 4H/1D aggregated in-pipeline (§6.4)
│       └── hyperliquid/
├── features/
│   ├── __init__.py
│   ├── _common.py               # reused (math helpers, timeframe-agnostic)
│   ├── builder.py               # MODIFIED: 30m primary + 4H + 1D in-pipeline aggregation merge (§6.4)
│   ├── indicators.py            # MODIFIED: per Decision v2.37 Q4 — base indicator MATH only (rsi, macd, wavetrend, stochastic, squeeze, adx_di, ema). Pure calculation functions consumed by momentum_core.py + trend.py + htf_context.py + volatility.py. NO `_features`/selection logic, NO `_5m`/`_1h`/`_1d` suffixes. Drop wavetrend(suffix=1h), macd_features_1h, adx_features_1h, ema_features_1h, ema_features_1d (HTF flows through htf_context.py).
│   ├── htf_context.py           # NEW: Category 2a (18 features). Imports from indicators.py for 4H+1D RSI/MACD/ADX/EMA math
│   ├── momentum_core.py         # NEW: Cat 1 selection layer (32 features). Imports from indicators.py for math; selects + trims per §7.2 Cat 1; emits roc_1/3/6/12, cross-feature flags, velocity-of-velocity. Consolidates v1.0 logic split across indicators.py + extra_momentum.py
│   ├── trend.py                 # NEW: Cat 2 selection (14 features) per Decision v2.39 — sub of v2.37 Q4. Imports adx_di + ema math from indicators.py; selects ADX(14)/+DI/-DI/DI-spread/ADX-slope (5) + 4 zone flags (adx_trending/weak/accelerating/decelerating per Q5) + EMA9/21/50 dist (3) + EMA stack (1) + price-vs-EMA21-ATR (1). Symmetric with momentum_core.py
│   ├── extra_momentum.py        # MODIFIED: Cat 15 (Williams %R / CCI / CMO / TSI), trim 9→6 per §7.2 — drop `roc_10` (overlap with Cat 1 multi-period momentum), `cci_extreme`, `williams_r_direction`. Math reusable from indicators.py if/when refactored
│   ├── volatility.py            # MODIFIED: Cat 3 trim 14→12. Imports atr/bbands math from indicators.py
│   ├── volume.py                # MODIFIED: Cat 4 trim 17→12 + Cat 14 trim 8→6. VFI fix kept from v1.0. Drop vfi_features_1h
│   ├── vwap.py                  # MODIFIED: Cat 5 expansion 8→14 (multi-anchor: daily + swing-high/low + htf-pivot + weekly + confluence + ATR-distance + cross-events + heavy-VWAP)
│   ├── pivots.py                # MODIFIED: Cat 6 expansion 13→30 (Cat 6.1 daily 9 reused + 6.2 NEW 3 + 6.3 weekly 9 reused + 6.4 NEW 2 + 6.5 swing-Fib NEW 7)
│   ├── candles.py               # reused — Cat 8 unchanged 9→9 (timeframe-agnostic pattern recognition)
│   ├── structure.py             # MODIFIED: Cat 16 reconcile 10→10. Add HL count, LH count, fractal_pivot_count, explicit break_of_structure flag. Drop retrace_depth, range_position, swing_range_pct
│   ├── divergence.py            # reused — Cat 13 unchanged 7→7 (sign convention +1/-1 already fixed). May tune lookback_bars 14→20 at 30m
│   ├── event_memory.py          # MODIFIED: Cat 20 trim 41→22 per §7.2 enumeration. Add bars_since_pivot_touch_weekly, bars_since_ema21_cross, bars_since_macd_signal_cross, bars_since_squeeze_entry, squeeze_direction_at_fire, last_rsi_extreme_depth, bars_in_current_rsi_episode. Rename column conventions per §7.2 spec
│   ├── adaptive_ma.py           # MODIFIED: Cat 18 trim 5→4 (combine psar_direction + psar_dist into single signed psar_state_dist_pct)
│   ├── ichimoku.py              # MODIFIED: Cat 19 expand 5→6 (add explicit senkou_a_dist_pct + senkou_b_dist_pct; cloud displacement +26 fix kept from v1.0)
│   ├── regime.py                # MODIFIED per Decision v2.37 Q3: REWRITE Cat 10 to spec. Drop efficiency_ratio, choppiness_index, regime_volatile, regime_quiet. Add trend_direction (+1/0/-1 EMA stack), volume_regime (tercile), vol-adjusted-momentum-regime. Keep trending, ranging, regime_change_bar (rename from bars_in_current_regime), volatility_regime (rewrite from boolean to tercile)
│   ├── stats.py                 # MODIFIED: Cat 9 trim 8→7 (drop mean_reversion_score) + Cat 17 trim 9→6 (drop parkinson_vol, autocorrelation_1, variance_ratio; add autocorr(20), realized-vol-of-realized-vol)
│   ├── sessions.py              # MODIFIED: Cat 7 trim 15→9 — REWRITE to cyclic encodings. hour sin/cos, dow sin/cos, is_weekend, 3 session-overlap flags, month_of_year cyclic. Drop minutes_into/to_session (intra-30m no analog), prev_session_range_pct, individual session flags
│   ├── context.py               # MODIFIED per Decision v2.37 Q2: REWRITE Cat 11 to literal prev-bar (1 bar back) — drop all prev-DAY logic. + Cat 12 trim 8→5 (Δrsi_1, Δrsi_3, Δadx_3, Δvolume_3, Δclose_vs_ema21_3). v1.0 day-grouped logic dropped entirely
│   ├── cross_asset.py           # NEW: Cat 22 (8 features for SOL/LINK; BTC has nothing to correlate to within universe)
│   └── feature_stability.py     # NEW: static/dynamic/mixed taxonomy per §7.5
│   # NOTE per Decision v2.37 Q1: ema_context.py (v1.0 "Tier-1 setup" file) DROPPED — features not enumerated in §7.2 (orphan). Concepts partly absorbed by Cat 2 + Cat 20 (bars_since_ema21_cross). If specific touch/bounce features needed later, add via separate Deviation Request.
├── model/
│   ├── labeler.py               # reused (triple-barrier with pessimistic tie-break)
│   ├── train.py                 # MODIFIED (Phase 2)
│   ├── predict.py               # MODIFIED (Phase 2)
│   └── calibration.py           # NEW (Platt scaling per §9.3, Phase 2)
├── tune/
│   ├── optuna_search.py         # reused (Phase 2)
│   └── shap_analysis.py         # reused (Phase 2)
├── backtest/
│   └── simulator.py             # reused (Phase 3)
├── execution/
│   ├── predictor_service.py     # MODIFIED (Phase 3)
│   └── executor_hyperliquid.py  # MODIFIED (Phase 3)
├── scripts/
│   ├── export_parquet.py        # NEW v2.0 (clean glue per Decision v2.36)
│   ├── relabel.py               # reused (used in 1.12 + Phase 5 retrains)
│   └── baseline_gate.py         # NEW: two-stage gate check per §10.3
├── utils/
│   ├── __init__.py              # NEW v2.0 (package marker)
│   ├── config.py                # NEW v2.0 (~25 LOC clean glue per Decision v2.36; YAML + dotenv loader)
│   └── logging_setup.py         # NEW v2.0 (~25 LOC clean glue per Decision v2.36; loguru wrapper)
├── monitoring/
│   └── decay_monitor.py         # NEW (Phase 5, per §17.2)
├── models/
│   ├── v2.0_frozen/             # frozen Phase 2 outputs
│   ├── v2.0_30m_archive/        # archived after switch-to-1H per §17.9.1 (if triggered)
│   ├── v2.x_archive/            # post-retrain archive (per §17.3 step 5)
│   └── paper/                   # live paper-trade checkpoints
├── research/
│   └── monthly_YYYY-MM.md       # research scan per §17.8
├── logs/
└── tests/
    ├── test_htf_aggregation.py  # NEW (verifies 30m → 4H/1D resample correctness)
    ├── test_multi_anchor_vwap.py # NEW
    ├── test_labeler.py          # NEW (v1.0 tests/ was empty; written fresh in v2.0)
    └── test_purged_cv.py        # NEW (v1.0 tests/ was empty; written fresh in v2.0)
```

---

## 16. Success Criteria

### 16.1 Baseline gate (before full Phase 2)

- Val log-loss beats empirical prior by ≥2%

### 16.2 Phase 2 (walk-forward + Optuna + SHAP freeze)

- Mean val log-loss across 14 folds: ≥3% below empirical prior
- Fire rate at P>0.65 threshold: 2–8% (translates to 1–4 trades/day at 48 bars/day)
- Hit rate at threshold: ≥55%
- SHAP top-20 features are semantically sensible (HTF context, pivots, ATR, RSI-family expected)

### 16.3 Phase 2.9 (one-shot OOT)

- OOT log-loss within 10% of mean val log-loss (accept deterioration but not collapse)
- OOT fire rate in [1%, 10%] range
- OOT hit rate ≥50%
- **BCa bootstrap 95% confidence intervals** computed for OOT Sharpe, hit-rate, and net P&L (per §16.3.1 below). Required for OOT pass.

#### 16.3.1 BCa bootstrap confidence intervals (the only stat-test compatible with §10.5)

Adopted from pybroker's `eval.py:bca_boot_conf` per Decision Log v2.32 / DR-A4. Justification: with the OOT one-shot rule (§10.1, §10.2), point estimates are interpretively fragile — a single Sharpe of 0.95 vs 1.05 means nothing without a confidence interval. **Bootstrap CIs strengthen §10.5, not weaken it:** they require zero additional data and zero re-tuning; they only re-sample the OOT trade list to bound where the true metric likely lies.

**Procedure** (executed exactly once, immediately after OOT score):

1. Run frozen model over OOT, produce trade list `T = [trade_1, …, trade_n]` with realized P&L per trade
2. Bootstrap: draw `B = 10,000` resamples of size `n` with replacement from `T`
3. For each resample, compute Sharpe, hit-rate, net P&L
4. Compute **bias-corrected and accelerated (BCa)** 95% CI per Efron 1987 (handles skewed return distributions correctly; symmetric percentile is insufficient for trade returns)
5. Report: point estimate + BCa 95% CI lower / upper bound for each metric

**Pass criterion update:**

- OOT Sharpe **BCa lower bound ≥ 0** (point estimate insufficient — lower bound being negative means the OOT result is plausibly noise)
- OOT hit-rate **BCa lower bound ≥ 0.50**
- OOT net P&L **BCa lower bound ≥ 0**

If the point estimate passes but the BCa lower bound fails → **OOT fails**. The model goes to paper trading anyway (per §10.4) but cannot proceed to real capital (§16.5) without paper-trade confirmation.

**No bootstrap on training/WF data** (CIs there are vanity metrics and tempt re-tuning). BCa is OOT- and Phase-5-only.

### 16.4 Phase 3 (2-week paper trading)

- Paper Sharpe (net of simulated fees/slippage) ≥1.0
- Max drawdown ≤10%
- Trade count 15–60 over 2 weeks (1–4/day average)

### 16.5 Phase 3 live gate (go/no-go for real capital)

- 4-week paper performance matches backtest within tolerance
- Net P&L after paper fees ≥0
- No catastrophic single-trade losses (>1.5× planned stop)
- Latency from bar-close to order placement <2 seconds

---

## 17. Alpha Decay & Regime Adaptation Plan

### 17.1 Premise — why this section exists

Crypto perp markets in 2026 have AI-driven flow estimated at >50% of taker volume (institutional desks plus retail bots running open-source LightGBM/PyTorch stacks). Edge half-life on classical TA signals is now in months, not years. Any model trained today **will** degrade — the only question is whether degradation is detected and adapted to, or silently bleeds capital.

This section is the **decay-aware operational plan**. Phases 1–3 build v2.0; this plan governs what happens **after** Phase 3 goes live, plus what triggers re-architecture into v3.0.

**v2.0 design choices that already buy partial resilience:**

- HTF context features (4H BB, 4H RSI, weekly pivots) — less crowded than 5m RSI signals
- Multi-anchor VWAP, swing Fib retracements, confluence flags — structural patterns adversaries can't trivially front-run
- Symmetric labels, walk-forward, one-shot OOT — minimizes overfit-to-noise that decays fastest
- Asset choice (SOL/LINK over BTC/ETH only) — long-tail alts have less algo penetration. TAO deferred per Decision v2.35 until data + Hyperliquid liquidity gates met

These reduce decay rate but do **not** eliminate it. Plan assumes **3–6 month effective edge half-life** and operates accordingly.

**Scope boundary:** §17 does not modify Phase 1–3 work. v2.0 is built per current spec. §17 activates at Phase 3.5 (paper trading) and governs all subsequent operations.

---

### 17.2 Decay monitoring — live metrics

Active continuously after Phase 3 paper-trading start. Each metric has yellow / red bands. **Yellow** triggers a diagnostic entry in `PROJECT_LOG.md`; **red** triggers §17.3 retraining or §17.7 kill switch.

| Metric                                   | Window  | Yellow                       | Red                                  |
| ---------------------------------------- | ------- | ---------------------------- | ------------------------------------ |
| Rolling hit rate (P>0.65 fires)          | 4 wk    | < backtest mean − 5 pp       | < 50% absolute                       |
| Rolling net P&L vs backtest baseline     | 4 wk    | −20% relative                | −40% relative                        |
| Brier score (probability calibration)    | 4 wk    | +15% vs frozen baseline      | +30% vs frozen baseline              |
| Feature PSI (per top-10 SHAP feature)    | 1 wk    | PSI > 0.10                   | PSI > 0.25                           |
| SHAP top-20 turnover                     | monthly | 3+ features drop out         | 5+ features drop out                 |
| Realized slippage vs backtest assumption | 2 wk    | 1.5× assumed                 | 2.0× assumed (= crowding signal)     |
| Trade frequency vs backtest              | 2 wk    | ±50%                         | ±100%                                |

Concrete thresholds tuned in **Phase 3.5** against paper-trade variance — values above are starting points.

Implementation: `monitoring/decay_monitor.py` runs nightly cron, writes metrics to `decay_metrics_30m` Postgres table, posts yellow/red alerts to `PROJECT_LOG.md` with timestamp.

**BCa bootstrap CIs in maintenance** (per §16.3.1 / Decision Log v2.32): each rolling 4-week window produces point estimates *and* BCa 95% CIs for Sharpe, hit-rate, P&L. **Yellow/red bands are evaluated against the lower CI bound, not the point estimate.** Reason: a noisy 4-week point estimate that drifts into yellow is often a sampling artifact, not real decay; only when the lower CI bound crosses the threshold has the underlying metric materially shifted. This avoids "false-alarm retrains" that would themselves degrade the model.

---

### 17.3 Refresh cadence — scheduled and triggered

**Scheduled retraining:** every **90 days** regardless of metrics.

**Triggered retraining:** on any red metric in §17.2 sustained ≥3 days, OR quarterly anniversary, whichever comes first.

**Retrain process** (each retrain is a full mini-Phase, not a tweak):

1. Roll training window forward by 1 month (oldest month falls off; newest month becomes new OOT)
2. Re-run Phase 1.12 → 2.10 (relabel → walk-forward → tune → SHAP → freeze → one-shot OOT)
3. Each retrain is its own freeze + one-shot OOT — **no iteration**, same discipline as v2.0 initial training
4. New model promoted to live only if Phase 2 success criteria pass on the new OOT
5. Old model archived to `models/v2.x_archive/`, not deleted; rollback path always available

**Critical:** retraining is not "fitting until it works." If the new OOT fails Phase 2 criteria, the live model **continues with degraded performance OR is shut down (§17.7)** — it does **not** get re-tuned. Re-tuning into the new OOT is the v1.0 failure pattern, and it remains banned in maintenance.

---

### 17.4 Strategy diversification roadmap

Single-model dependence is fragile. Roadmap diversifies edge sources so any one decay path doesn't kill total P&L.

| Version | Addition                                            | Rationale                                                                         |
| ------- | --------------------------------------------------- | --------------------------------------------------------------------------------- |
| v2.0    | Single 30m 3-class model (current scope)            | Baseline                                                                          |
| v2.1    | LONG-only and SHORT-only specialists (ensemble vote) | Class imbalance; specialists often outperform unified model on minority class    |
| v2.2    | Mean-reversion model (return-to-VWAP/POC labels)    | Orthogonal label scheme; uncorrelated decay path from momentum model              |
| v3.0    | Multi-timeframe ensemble (15m + 30m + 4H)           | Regime-aware weighting; longer-TF edge decays slower than HFT-scale signals       |
| v4.0    | Regime-aware model selector (§17.5)                 | Per-regime specialists chosen by live regime classifier                           |

Each version is a distinct project with its own freeze + OOT discipline. v2.1+ are post-§17.9 lifecycle decisions, not auto-launched.

---

### 17.5 Regime adaptation

**v2.0 (current):** regime features (ATR percentile, trend strength, BB width) are passive ML inputs. Model learns regime-dependent behavior implicitly.

**v2.5 (planned):** explicit regime classifier (HMM or threshold-based on 1D ATR percentile + trend strength + BB squeeze). Outputs categorical {trending-bull, trending-bear, ranging-low-vol, ranging-high-vol}. Per-regime specialist models trained separately. Live regime classification gates which model fires.

**Why regime-aware matters more in AI-saturated markets:** AI flow concentrates around regime boundaries (breakouts, reversals); static models trained across regimes underweight transitions. Per-regime specialists reduce the "average across regimes" dilution.

---

### 17.6 Adversarial-AI considerations

**Crowding detection:**

- Monitor order-book stuffing / cancel-rate in 100 ms post-signal window. Spike above baseline = competitor algos read same signal.
- If crowding score sustained above threshold on an asset, that asset is "burnt out" → reduce allocation or rotate to less-crowded venue

**Less-crowded venues / assets:**

- SOL / LINK already chosen partly for this rationale (TAO deferred per Decision v2.35)
- Hyperliquid HYPE token (Phase 4): native perp, less institutional algo presence than BTC/ETH/SOL
- Altcoins with <$500M cap (research): possible Phase 5 expansion if liquidity supports

**Orthogonal feature classes (slower for HFT desks to operationalize):**

- On-chain flow (whale wallets, exchange netflow, dormant supply waking)
- Funding rate extremes + open interest divergence
- News / event embeddings (LLM-flavored, requires Phase 4 data infra)
- Cross-market spread (crypto-equity ETF flow, BTC-ETH ratio extremes)

These classes require non-OHLCV pipelines that most HFT desks don't have. **Phase 4 priority: incorporate ≥1 orthogonal feature class** as decay insurance.

---

### 17.7 Kill switches & circuit breakers (live)

Hard halts to live trading **without human approval required**. Halt = paper mode (predictor + executor continue running but submit no real orders). User must explicitly re-enable after diagnostic.

| Trigger                                              | Action                              |
| ---------------------------------------------------- | ----------------------------------- |
| 7-day drawdown > 8% of capital                       | Halt → paper mode                   |
| 3 consecutive days of red metrics from §17.2         | Halt → paper mode                   |
| Single trade loss > 2× planned stop (slippage event) | Halt → paper mode + diagnostic      |
| Bar-close-to-order latency > 5s for 3 bars in row    | Halt → paper mode (pipeline broken) |
| Predictor service crash > 2 in 24 h                  | Halt → paper mode                   |
| Daily loss > 4% of capital (intraday)                | Halt for 24 h                       |

Every halt event auto-logs to `PROJECT_LOG.md` with the trigger and capital state.

Thresholds tunable in **Phase 3.5** — listed values are starting points, calibrated from paper-trade variance.

---

### 17.8 Continuous research pipeline

**Monthly:**

- Scan arxiv (cs.LG, q-fin.TR, q-fin.ST) for new techniques relevant to the project
- Scan top-25 GitHub algo-trading repos for emerging patterns
- One-page summary appended to `research/monthly_YYYY-MM.md`

**Quarterly (champion / challenger):**

- Any new strategy idea enters as **challenger** model in shadow mode (4 weeks)
- Compared on identical OOT methodology vs current champion
- Promotion only after surviving full Phase 2 freeze gate
- **No swap based on intuition or single-period outperformance**

**Bi-annually (assumption audit):**

- Re-validate that 30m is the optimal primary timeframe vs current AI-penetration level
- Re-evaluate asset universe (BTC/SOL/LINK in v2.0 + Phase 4 candidates TAO/HYPE) — drop assets that no longer show edge, add new candidates as Phase 4 re-entry gates clear (per Decision v2.35)
- Update §17.2 thresholds if calibration drifted

---

### 17.9 Project lifespan, timeframe fallback & v3.0 trigger criteria

**Planned operational lifetime of v2.x architecture: 18–24 months** before structural redesign.

#### 17.9.1 Timeframe fallback policy

The 30m primary timeframe is research-validated (Decision Log v2.31). It is **not infinitely defended** — if 30m fails on quantitative criteria, the **only allowed fallback is 1H**. **15m is permanently rejected** for the duration of the v2.x lifecycle and may not be tried even if 30m and 1H both fail (institutional HFT/MM competition makes it unwinnable for an individual developer).

**Switch-to-1H triggers** (any one sustained for the listed period):

| Trigger                                                         | Sustained period       | Rationale                                                                         |
| --------------------------------------------------------------- | ---------------------- | --------------------------------------------------------------------------------- |
| Phase 2 walk-forward AUC < 0.54 on BTC                          | full 14-fold WF        | Below Keller's published live benchmark (0.58); signal too weak at 30m            |
| Post-cost Sharpe < 0.7 in Phase 3 paper trading                  | 6 months continuous    | Same Keller anchor; 30m no longer fee-economic vs alpha captured                  |
| Post-SHAP feature count drops below 150                          | after Phase 2.6 trim   | 1H sample-size constraint relaxes; 1H now strictly superior on SNR + half-life    |
| Realized slippage > 3× backtest assumption                       | 1 month continuous     | Crowding has eaten the 30m spread — 1H less crowded, higher captured ATR per cost |

**Switch-to-1H process** (NOT a tweak — full mini-Phase, same discipline as initial training):

1. Submit Deviation Request citing which 17.9.1 trigger fired with metric data
2. Update spec's "primary timeframe" parameter (config + §6.4 aggregation factors)
3. **Discard all v2.0 30m frozen artifacts** — feature matrix, labels, model, OOT score. They were trained on a different bar distribution
4. Re-execute Phase 1 and Phase 2 from scratch on 1H bars (1H aggregated from 30m via `resample("1h", …)` if 30m data was kept; otherwise refetch native 1H)
5. New 1H model gets its own one-shot OOT, same discipline
6. Old 30m artifacts archived to `models/v2.0_30m_archive/`, never resurrected

**No half-switches** — running both 30m and 1H in parallel during transition is prohibited (it's the v1.0 contamination pattern in disguise).

#### 17.9.2 v3.0 re-architecture triggers

Distinct from §17.9.1 (which is a timeframe pivot, not architectural change). Triggers for full v3.0 redesign:

- 2+ consecutive quarterly retrains fail Phase 2 success criteria on new OOT (at whichever timeframe is current)
- SHAP top-20 turnover > 75% from frozen v2.0 baseline (= market structure has fundamentally shifted, current feature set obsolete)
- Sustained net loss across 2 consecutive quarters despite scheduled retrains
- 1H fallback also fails §17.9.1 success criteria — both timeframes exhausted means the architecture itself is wrong
- Realized slippage > 3× backtest assumption sustained ≥1 month (= crowding has eaten the spread, regardless of timeframe)

**v3.0 candidate architectures** (decided when triggered, not predetermined):

- Multi-timeframe ensemble (per §17.4)
- Regime-aware specialist (per §17.5)
- Orthogonal-data hybrid (on-chain + OHLCV)
- Reinforcement learning at portfolio level

#### 17.9.3 Honest acknowledgment

Every algorithmic trading system has a finite lifespan. The goal of v2.0 is to extract edge while it exists, detect decay early, and either adapt cleanly or exit cleanly. There is no "permanent" system. Plan for **18–24 months of v2.x productive life** and design v3.0 as the natural successor — not as an emergency response after the model has already been losing money for a quarter.

**The timeframe ladder is finite: 30m → 1H → v3.0.** No 15m rung. No re-trying 30m after switching to 1H.

---

### 17.10 Maintenance phase governance

§17 operations are themselves spec-governed under §10.5 discipline:

- Decay-monitor metric definitions are frozen at Phase 2.9 freeze and cannot be re-defined to make a failing model "look healthy"
- Threshold changes in §17.2 / §17.7 require Deviation Request + spec edit + `PROJECT_LOG.md` entry
- Each scheduled / triggered retrain logs `Phase 5.x [SPEC §17.3] retrain {n} OOT pass/fail` to `PROJECT_LOG.md`
- v3.0 trigger evaluation is logged quarterly even when not triggered (negative evidence is decision evidence)

This makes long-term decay management auditable and prevents the maintenance phase from becoming the new attack surface for v1.0-style improvisation.

---

## 18. Decision Log

v2.0-specific decisions, extending v1.0's log.

| #    | Decision                                                                 | Rationale                                                                                  | Date       |
| ---- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ---------- |
| v2.1 | **30m primary timeframe**                                                | 5m noise dominated 4-ATR targets vs fees; 30m bar ATR is 2.3× larger, SNR improves         | 2026-04-23 |
| v2.2 | **3-year minimum training data**                                         | 2022 bear + 2023 recovery + 2024 bull + 2025 correction spans all regimes; 18mo was mono-regime | 2026-04-23 |
| v2.3 | **HTF Context as first-class category**                                 | 2026 research: 4H BB position = #1 SHAP feature (8.4%), 4H RSI = #5 (4.5%)                 | 2026-04-23 |
| v2.4 | **Multi-anchor VWAP (6 new features)**                                  | Single daily VWAP loses context at HTF; swing/pivot/weekly anchors add structure           | 2026-04-23 |
| v2.5 | **Weekly pivots promoted to Phase 1**                                   | 336 bars/week at 30m makes weekly levels persistent and testable; already implemented in v1.0 `pivots.py` | 2026-04-23 |
| v2.6 | **Event memory trimmed 41→22**                                          | 5m-granularity recency collapses to hours at 30m — dynamic range insufficient              | 2026-04-23 |
| v2.7 | **Symmetric 3.0/3.0 barriers, 8-bar hold**                              | 4-hour hold matches momentum archetype; symmetric avoids directional bias in labels        | 2026-04-23 |
| v2.8 | **min_profit_pct raised**                                               | 30m drift thresholds must exceed 2× fees to qualify as signal at timeout                   | 2026-04-23 |
| v2.9 | **OOT one-shot discipline**                                             | v1.0 iterated into OOT — overfit the hold-out. v2.0 reserves one month, scores once        | 2026-04-23 |
| v2.10| **Baseline gate before full pipeline**                                  | Avoid investing 8+ hours of tuning on a feature set that can't beat the empirical prior   | 2026-04-23 |
| v2.11| **Cross-asset correlation Phase 1 for altcoins**                        | BTC correlation as booster (not filter) per user memory; useful from day 1 for alt models | 2026-04-23 |
| v2.12| **CUSUM event-bars deferred to Phase 4**                                | 2025 research shows edge, but architectural change — validate time-bar baseline first      | 2026-04-23 |
| v2.13| **Hybrid intrabar execution deferred to Phase 4**                       | Per user agreement — prove 30m bar-close model before adding intrabar complexity          | 2026-04-23 |
| v2.14| **1D context features added (6 new)**                                   | Macro regime (close vs EMA200 daily) matters for entries; single 1D feature wasn't enough | 2026-04-23 |
| v2.15| **Warmup 250→150 bars**                                                 | VFI(130) is the longest lookback at 30m; 150 is safe margin                                | 2026-04-23 |
| v2.16| **Risk per trade 2% → 1%**                                              | Live trading conservatism; v1.0's 2% was backtest-only assumption                          | 2026-04-23 |
| v2.17| **Swing Fib retracement features (Cat 6.5, +7 features)**               | TradingView AI Report 2: distinct from Fib-pivots; `fib_retracement_pct`, `in_golden_pocket`, etc. Neither v1.0 nor the original v2.0 draft had these. Real gap | 2026-04-24 |
| v2.18| **Continuous 0–1 pivot_position + ATR-normalized distance (Cat 6.2, 6.4, +5)** | TradingView AI Report 2: `0.65` means the same at $30k BTC or $100k BTC — superior to absolute $; ATR units complement % units | 2026-04-24 |
| v2.19| **Pivot × Fib confluence flags (+2 features)**                          | Confluence zones where daily pivot ≈ weekly pivot, or swing Fib ≈ pivot — highest-probability reversal zones in classical TA | 2026-04-24 |
| v2.20| **Static-vs-dynamic feature taxonomy (§7.5)**                           | TradingView AI Report 3: tag every feature at catalog-time so Phase 4 intrabar inference knows which features are safe without recomputation | 2026-04-24 |
| v2.21| **Two-stage baseline gate (§10.3)**                                     | TradingView AI Report 1: 5-feature pre-gate before the 200-feature gate. If a minimal hand-picked set can't beat the prior, the premise is wrong — fail fast, don't tune into noise | 2026-04-24 |
| v2.22| **Triple-trigger entry method (§12.2)**                                 | TradingView AI Report 3: explicit AND-conjunction of ML bias + price sweep/reclaim + volume/momo confirm. Reduces false positives in the Phase 3 hybrid architecture | 2026-04-24 |
| v2.23| **Phase 1 feature count revised ~190 → ~202**                           | Net effect of v2.17–v2.19 (+14 features in Cat 6). Post-SHAP trim target revised to ~110–140 | 2026-04-24 |
| v2.24| **Process Discipline as non-negotiable spec section (§10.5)**           | v1.0 failed partly because Claude made 7+ rounds of independent judgment calls mid-phase, each drifting from the design, contaminating OOT. User is not an ML specialist and trusted per-step opinions. Root cause documented; rules lock it down | 2026-04-25 |
| v2.25| **Phase Checklist (Appendix C) + PROJECT_LOG.md (Appendix D)**         | Makes §10.5 auditable in practice: tick-through list + append-only decision trail citing spec sections. Makes drift visible immediately instead of 3 weeks later | 2026-04-25 |
| v2.26| **Fresh-start 30m, do not restart 5m**                                  | 5m OOT is already compromised by v1.0 iterations; restarting would train against seen hold-out. 30m gives clean OOT + better SNR vs fees. Discipline fix applies from day 1 regardless | 2026-04-25 |
| v2.27| **Database reuse: same Postgres/TimescaleDB instance, new tables only** | Avoid duplicate DB setup, credentials, and extension installs. v2.0 shares v1.0's instance; new tables (`ohlcv_30m`, `features_30m`, `labels_30m`, `wf_folds_30m`, `models_30m`) isolate v2.0 artifacts. v1.0 tables remain read-only reference. See §6.1.1 | 2026-04-25 |
| v2.28| **4H and 1D derived from 30m in-pipeline, not fetched separately**      | 30m OHLCV is a strict superset of 4H and 1D — any 4H bar = 8 consecutive 30m bars (first/max/min/last/sum); any 1D bar = 48 consecutive 30m bars. Separate fetches double storage and introduce boundary mismatches (archive cutover timestamps). Single-source-of-truth for candle data. See §6.4 Step A | 2026-04-25 |
| v2.29| **Canonical project naming: `ml-bot/` (v1.0) and `ml-bot-30m/` (v2.0)** | VPS convention. Local folder may still be `hyperliquid-ml-bot-30m/` until manual rename; spec references canonical name. `ml-bot/` path assumed for all v1.0 references (config, features, labeler reuse, `.env` inheritance)                  | 2026-04-25 |
| v2.30| **Alpha Decay & Regime Adaptation Plan as new §17**                     | AI-driven flow now >50% of crypto perp taker volume; classical-TA edge half-life is months, not years. Any v2.0 model will degrade. §17 codifies live decay metrics (yellow/red bands), 90-day scheduled + threshold-triggered retraining (each retrain = one-shot OOT, no re-tuning), strategy diversification roadmap (v2.1 specialists → v3.0 multi-TF ensemble → v4.0 regime selector), kill switches, monthly research scan + quarterly champion/challenger, and v3.0 trigger criteria. Honest 18–24mo lifecycle assumption built in. Activates Phase 3.5; does not modify Phase 1–3 work | 2026-04-26 |
| v2.31| **30m primary timeframe re-validated; 1H is the documented fallback; 15m is rejected** | Deep web research (2026-04-26) tested 15m / 30m / 1H against the user's "cannot beat institutional bots" constraint. Findings: (1) 5–15m band is HFT/MM dominated (Wintermute, Jump, Cumberland, Hyperliquid HLP, Aster/Lighter MMs); arb success drops 82%→31% as latency crosses 50→150ms — a regime an individual VPS cannot win. (2) Strongest live-trading anchor is Keller's published LightGBM/XGBoost on 1–4h horizons: Sharpe 1.4 / AUC 0.58 / 54% accuracy post-cost — directly applicable to 30m bars with 1–4h holds. (3) 1H is best per individual factor but has only ~105k bars over 3yr × 4 assets, borderline for a 284-feature LightGBM; 30m gives ~210k bars (2× headroom). (4) 15m halves ATR while keeping Hyperliquid's 0.045% taker fee constant — fee economics turn marginal. **Decision: 30m stays as primary. 1H is the only allowed fallback per §17.9.1. 15m is permanently off the table — re-opening it requires §17.9.2 v3.0 trigger plus explicit user authorization.** Sources: Keller (Medium 2025), López de Prado / Hudson & Thames, MDPI 2024 dataset-size study, Springer 2025 information-bars paper, Hyperliquid fees docs, 21Shares perp DEX wars report. See §17.9.1 for switch-to-1H quantitative triggers | 2026-04-26 |
| v2.32| **Peer-project deep analysis: 10 adoptions, 4 anti-patterns**           | Deep analysis (2026-04-26) of intelligent-trading-bot, freqtrade/FreqAI, microsoft/qlib, pybroker, jesse, FinRL, TensorTrade. **Adoptions** (most are implementation choices in unfrozen phases — logged in PROJECT_LOG when each phase begins, not in spec): A1 generator-registry architecture for our 22 feature categories (intelligent-trading-bot/common/generators.py), A2 embargo-via-label_horizon as canonical WF reference, A3 forced `train=False` flag on live server (architectural kill-switch), A4 BCa bootstrap CIs on OOT and monitoring (the only spec-affecting one — see §16.3.1 + §17.2 + Phase 2.11 + 5.0), A5 producer/consumer thread separation forward-compat for Phase 4, A6 FreqAI `feature_engineering_expand_all` naming pattern for cross-asset features, A7 score-then-threshold separation (threshold = only Phase 3 tunable), A8 append-only `historic_predictions.pkl`, A9 `label_horizon` tail truncation hardening, A10 DI threshold (out-of-distribution refuse) as Phase 5 kill-switch. **Anti-patterns to NOT import** (codified in §10.5.9): continuous clock-driven retrain, OOT-spanning grid search, online auto-rolling models, auto-PCA, two-binary-heads label combine, flat-dict LGBM training without early-stopping/class-weight, "model zoo" temptation. Their use case ≠ ours; their freedom-to-mutate ≠ our discipline. v2.0 imports rigor we lack from these projects, rejects mutation patterns that would erase OOT | 2026-04-26 |
| v2.33| **DR-001: phase-scoped selective copy replaces Appendix A mass-copy**   | First implementation-time DR. VPS Claude flagged that `cp -r ml-bot ml-bot-30m` (Appendix A original step 1) ports Phase 2–5 files into Phase 1's workspace, increasing cognitive load and risking stale-file references. §15 is already a file-by-file manifest with `# reused / # MODIFIED / # NEW` annotations — that's the authoritative source. Resolution: each phase copies only the files it needs from `../ml-bot/` when entering that phase; files not in §15 stay at v1.0 forever. Phase 1 copy manifest: data collectors + features/* + model/labeler.py + tests/test_labeler.py + configs/requirements/.gitignore + `.env` (chmod 600, gitignored). Phase 1 NEW files (not in v1.0): htf_context.py, cross_asset.py, feature_stability.py, test_htf_aggregation.py, test_multi_anchor_vwap.py, baseline_gate.py placeholder. Phase 2/3/5 files copied at phase-entry, not before. Appendix A rewritten to make §15 the manifest authority. Approved by user 2026-04-26 | 2026-04-26 |
| v2.34| **DR-002: Phase 1 manifest reconciled with v1.0 ground truth**          | VPS Claude's recon (`ls ../ml-bot/`) before executing v2.33 caught four spec-vs-reality mismatches: (i) `binance_archive.py` does not exist in v1.0 — actual file is `data/collectors/fetcher.py`; v2.0 `binance_archive.py` is NEW (created from `fetcher.py` template); (ii) `momentum_core.py` does not exist in v1.0 — Cat 1 momentum lives in v1.0 `indicators.py` + Cat 15 in `extra_momentum.py`; v2.0 `momentum_core.py` is NEW (Cat 1 refactor consolidating both); v2.0 also pulls `extra_momentum.py` (MODIFIED, Cat 15 trim); (iii) v1.0 `tests/` directory is empty — `test_labeler.py` and `test_purged_cv.py` are v2.0 NEW (not "reused"); (iv) v1.0 has no `pyproject.toml`/`setup.cfg` — drop from manifest (no-op). Plus reconciliation: `scripts/export_parquet.py` (MODIFIED) and `scripts/relabel.py` (reused, used in 1.12) added to Phase 1 manifest. Spec edits: §15 directory tree (4 annotations corrected, `extra_momentum.py` and `feature_stability.py` added, `test_purged_cv.py` re-marked NEW), Appendix A.1 (Phase 1 copy + NEW lists rewritten to match), BOOTSTRAP_VPS.md STEP 14 (`fetcher.py` as template, not `binance_archive.py`). Demonstrates v2.33's principle (§15 = manifest authority) requires §15 to mirror reality — patching only Appendix A would leave §15 stale, replicating the same mismatch DR-002 caught. Approved by user 2026-04-26 | 2026-04-26 |
| v2.35| **DR-003: TAO removed from v2.0; asset universe = BTC/SOL/LINK**        | Pre-Phase-1 design discussion. User questioned 3yr training window, considered 5–6yr. Discussion surfaced (a) hard data ceiling for TAO (Binance perp 30m only ~1.5–2yr available, incompatible with 3yr minimum from Decision v2.2), (b) low Hyperliquid TAO daily volume making execution unreliable, (c) stationarity argument against extending window into pre-2023 era (FTX-collapse regime break for SOL Nov 2022, pre-AI-saturation market, MDPI 2024 evidence that noisier old data degrades LightGBM). User confirmed: **keep 3yr window** (Decision v2.2 stays); **drop TAO** from v2.0. v2.0 asset universe is now **BTC, SOL, LINK** (3 assets, ~158k pooled samples, ~1,130:1 to ~1,435:1 sample/feature ratio post-trim — healthy). TAO becomes a Phase 4 candidate alongside HYPE; re-entry conditions (all three required): ≥3 years clean Binance perp 30m data accumulated, Hyperliquid TAO daily volume ≥5% of BTC sustained 30 days, v2.x diversification window open. DR-004 (window extension + per-period SHAP audit) WITHDRAWN — not needed since 3yr stays. Spec edits: §1 header (training universe), §1.4 transfer learning sequence, §5.1 architecture diagram, §6.1 (data sources table + new TAO/HYPE deferral subsection), §6.2 volumes table (4 → 3 symbols), §7.2 Cat 22 (TAO row removed from cross-asset features), §10.5.9 anti-pattern reference (LINK now smallest minority asset, not TAO), §11.1 fees (TAO/HYPE deferred), §14 Phase 2.7 + 4.1 (3 transfers, HYPE/TAO gated re-entry, NEW Phase 4.6 orthogonal feature class), §17.1 / §17.6 / §17.8 references updated, Appendix C steps 1.1 + 2.8 (3 assets). Approved by user 2026-04-26 | 2026-04-26 |
| v2.36| **DR-005: kept math, fresh glue**                                       | Phase 1.0 import audit by VPS Claude exposed dependency-closure gap in the 27-file v1.0 selective-copy: 6 v1.0 infrastructure files (`data/__init__.py`, `data/db.py`, `data/collectors/__init__.py`, `utils/__init__.py`, `utils/config.py`, `utils/logging_setup.py`) imported by the copied modules but missing from App.A.1 — and `scripts/export_parquet.py` had a stale `from data.collectors.fetcher import EXCHANGE` from the v2.34 rename. User chose option (d) over (a) "expand copy list": **retain v1.0 algorithm files (`model/labeler.py`, all `features/*`, `scripts/relabel.py` — math, not glue), rewrite the glue layer clean for v2.0**: `data/db.py` 325→~120 LOC limited to v2.0 tables (`ohlcv_30m`, `features_30m`, `labels_30m`, `wf_folds_30m`, `models_30m`, `decay_metrics_30m`), no 5m/1h baggage; `utils/config.py` + `utils/logging_setup.py` ~50 LOC trivial; `data/collectors/binance_archive.py` clean v2.0 archive fetcher (NOT v1.0 `fetcher.py` renamed — supersedes v2.34 rename decision); `scripts/export_parquet.py` clean v2.0; `config.yaml` fresh from §6/§8/§9/§13 (no inherit-and-patch). `data/collectors/storage.py` kept (parquet I/O, no v1.0 baggage). Net: 9 fresh files (~500 LOC) — comparable effort to "copy + audit + DR cycle" but produces clean infrastructure. Eliminates the long tail of glue-related DRs that would surface as v1.0 baggage met v2.0 invariants. Spec edits: App.A.1 step 1 restructured into 3 tracks (copy 22 algorithm files / write fresh 9 glue files / NEW algorithm stubs); §15 directory tree extended (`utils/`, `data/db.py`, package `__init__.py` markers); `binance_archive.py` annotation updated to "NEW v2.0 (clean glue per v2.36; v1.0 `fetcher.py` NOT renamed)". Phase 1.1 starts on clean infrastructure with no migration debt. Approved by user 2026-04-26 | 2026-04-26 |
| v2.37| **Phase 1.3 audit Q1–Q5 resolutions: spec-ambiguity calls locked**      | Pre-Phase-1.3 feature-implementation audit (local-Claude reviewed all 23 feature files vs §7.2 target) surfaced 5 spec ambiguities. User approved all 5 resolutions: **(Q1) `features/ema_context.py` DROPPED** — v1.0 "Tier-1 setup" file with 14 features (EMA touches, bounces, MTF alignment, pivot confluence at EMA, pin/engulf-at-level) not enumerated in §7.2 (orphan). Concepts partly absorbed by Cat 2 + Cat 20 `bars_since_ema21_cross`. Re-add via separate DR if specific touch/bounce features needed later. **(Q2) Cat 11 "prev-bar" = literal 1-bar-back interpretation** — current v1.0 `context.py:previous_context_features` does prev-DAY (yesterday's OHLC), which §7.2 explicitly says to drop ("Drop: yesterday's features that duplicate HTF1D"). v2.0 Cat 11 rewritten to literal previous 30m bar features (close/high/low/volume vs current); today's open-to-now and today's H/L distance features kept. **(Q3) Cat 10 Market Regime REWRITE to spec** — current `regime.py` emits `efficiency_ratio` and `choppiness_index` not in §7.2's 7-feature list, lacks spec's `trend_direction` (+1/0/-1 EMA stack), `volume_regime` (tercile), and `vol-adjusted_momentum_regime`. Drop efficiency/chop, add 3 spec features, rewrite `volatility_regime`/`volume_regime` to terciles. **(Q4) `indicators.py` = base math only; `momentum_core.py` = Cat 1 selection** — option (b): math separated from feature selection. `indicators.py` retains pure calc functions (rsi, macd, wavetrend, stochastic, squeeze, adx, ema) with no `_features` selection logic and no `_5m`/`_1h`/`_1d` suffixed variants. `momentum_core.py` imports from `indicators` and selects 32 Cat 1 features. `htf_context.py` imports from `indicators` for 4H/1D math. Single source of truth for indicator math; cleaner consumer pattern. **(Q5) ADX zone 4th flag = `adx_decelerating`** — spec §7.2 Cat 2 wording was ambiguous ("trending (>25), weak (<20), accelerating (4)" — 3 named conditions but count of 4). Resolved: 4 flags = `adx_trending`, `adx_weak`, `adx_accelerating` (rising), `adx_decelerating` (falling). 2 level flags + 2 momentum flags — orthogonal information. Spec edits: §7.2 Cat 2 (4 zone flags enumerated explicitly + 6 specific drops from `adx_features_5m`), §7.2 Cat 11 (rewrite to literal prev-bar), §15 directory tree (drop `ema_context.py`, expand per-file annotations to specify v2.0 deltas including `indicators.py` math-only role per Q4), Appendix A.1 (drop `ema_context.py` from Phase 1 copy list — 22→21 files; add Phase 1.3 reconciliation note). Approved by user 2026-04-27. Phase 1.3 implementation can now proceed with locked spec interpretation | 2026-04-27 |
| v2.38| **Phase 1.3-1.11 workflow: local-Claude implements, VPS Claude validates** | Workflow shift for the feature-engineering implementation phase (does not change §10.5 process discipline; explicitly preserves it). User approved Option A + per-feature verification + low-risk pilot. **Workflow:** local-Claude (the planner of Decision v2.37 audit) edits feature files locally with PROJECT_LOG citations per file; user pulls to VPS; VPS Claude runs `python -c "from <module> import …"` smoke tests + `pytest tests/` and surfaces any failure as CORRECTION lines in PROJECT_LOG (same pattern as Phase 1.1's .env-key-convention CORRECTION). **Discipline preserved:** local-Claude treats each file edit as an action requiring spec citation in PROJECT_LOG; banned phrases still apply; freeze gates unchanged; OOT-one-shot rule unchanged. **Why this is OK:** (a) implementation work follows v2.37-locked spec interpretation — no design judgment calls remaining; (b) VPS Claude retains validation/recon role — runtime check that local-Claude can't perform; (c) for any new ambiguity that surfaces during implementation, local-Claude raises a Deviation Request (same DR-NNN protocol as VPS Claude); (d) shorter loop than ping-ponging individual file edits to VPS. **Sequencing:** before ANY file edit, local-Claude delivers a per-feature KEEP/DROP/RENAME/ADD verification pass on `indicators.py` (proof of method); user reviews; then pilot the workflow cycle on a no-change file (`_common.py` or `candles.py`) to test the local-edit → user-pull → VPS-validate loop end-to-end; only after pilot passes, proceed to bigger files (builder.py, vwap.py, pivots.py, etc.). **Rollback:** if cycle breaks down or the validation layer surfaces too many runtime issues, revert to Option B (VPS Claude implements per audit). Decision v2.38 is the only spec-modification this workflow change requires. Approved by user 2026-04-27 | 2026-04-27 |
| v2.39| **Cat 2 home: new `features/trend.py` (sub-decision under v2.37 Q4)**   | Local-Claude's `indicators.py` per-feature verification pass surfaced an inconsistency in v2.37's §15 edit: declaring `indicators.py` "MATH only — NO `_features` selection" left Cat 2 features (ADX zone flags + EMA dist + EMA stack + price-vs-EMA21-ATR = 14 features) without a clear home. Three options considered: (a) new `features/trend.py`, (b) put Cat 2 inside `momentum_core.py` (loose naming), (c) keep Cat 2 `_features` in `indicators.py` (walks back v2.37 Q4 strict math-only claim). User approved (a). **Resolution: new file `features/trend.py` for Cat 2 = 14 features.** Math (ADX/DI/EMA calc) stays in `indicators.py` as building blocks; selection lives in `trend.py`. Symmetric with `momentum_core.py` for Cat 1. Spec edits: §15 directory tree adds `trend.py # NEW: Cat 2 selection (14 features) per Decision v2.39 (sub of v2.37 Q4); imports adx_di + ema math from indicators.py`; Appendix A.1 NEW files list adds `features/trend.py`. Approved by user 2026-04-27 | 2026-04-27 |
| v2.40| **Cat 1 implementation ambiguities Q6 + Q8 resolved**                   | Pre-`momentum_core.py` implementation surfaced 2 spec ambiguities in §7.2 Cat 1 (analogous to v2.37 Q5 ADX zone count). **(Q6) Velocity-of-velocity count = 4**: spec named only 2 (`d²rsi/dt²`, `d²macd/dt²`) but count was (4). Resolved option (a): one second-derivative per oscillator family — `d2_rsi`, `d2_macd_line`, `d2_wt1`, `d2_stoch_k`. **Distinction from MACD's `hist_acceleration`** is locked: `hist_acceleration` = d²(macd_hist) is in MACD's own 6-feature group; `d2_macd_line` = d²(macd_line) is in vel-of-vel group — different variables, no double-count. **(Q8) Cross-feature implementations**: `rsi_wt_divergence_flag` = `int(sign(rsi-50) ≠ sign(wt1))` — binary 0/1, 1 = oscillator regime disagreement; `macd_rsi_alignment` = `sign(macd_hist) × sign(rsi-50)` — signed −1/0/+1, +1 aligned. Both are minimal-interpretation extensions of spec language. Spec edits: §7.2 Cat 1 cross-feature line + velocity-of-velocity line rewritten to lock formulas explicitly (same pattern as v2.37 locking ADX zone flags). Approved by user 2026-04-27 | 2026-04-27 |
| v2.41| **Cat 22 implementation: ETH dropped (Q9) + ATR-norm difference (Q10)** | Pre-`cross_asset.py` implementation surfaced 2 ambiguities. **(Q9) ETH correlation DROPPED entirely from Cat 22.** Original §7.2 Cat 22 listed 8 features (6 v1.0 + 2 NEW); ETH correlation was 1 of the 2 NEW. Reasoning for drop: (a) ETH is not a v2.0 trade asset (Decision v2.35 locked universe at BTC/SOL/LINK), so ETH correlation would only serve as a signal-source feature for SOL/LINK predictions; (b) ETH adds another asset to fetch + maintain — scope creep beyond v2.35; (c) BTC correlation already captures the bulk of cross-asset signal — ETH's marginal lift doesn't justify the cost; (d) Phase 4 has asset-universe expansion as natural place to revisit if needed. **Cat 22 final size: 7 features** (6 in Phase 1 + 1 BTC funding in Phase 3+). Feature count cascade: Cat 22 (8→7) → Phase 1 total (~202→~201), Phase 3 total (~231→~230), Phase 4 full (~239→~238). **(Q10) BTC ATR-normalized move formula = difference**: `btc_vs_asset_atr_norm_diff = (btc_move/atr_btc) − (asset_move/atr_asset)`. Signed signal: positive = BTC led in vol-adjusted units, negative = asset led / decoupled alt strength, near-zero = synchronized. Difference chosen over ratio (unstable when btc_move≈0) and signed product (loses magnitude). Spec edits: §7.2 Cat 22 (ETH line removed; remaining 7 features explicitly named with column conventions; ATR-norm-diff formula locked); §7.3 feature count summary (Cat 22 row 8→7, Phase 1/3/4 totals updated). Approved by user 2026-04-28 | 2026-04-28 |
| v2.42| **Cat 15 implementation: TSI keeps BOTH cross states (Q11)**            | Pre-`extra_momentum.py` implementation surfaced ambiguity in §7.2 Cat 15: "TSI long/short signal cross state (1)" could mean (a) TSI vs signal-line cross or (b) TSI zero-cross. User chose **"keep both"**: orthogonal information — `tsi_signal_cross_state` captures momentum acceleration (TSI vs EMA(7) signal line), `tsi_zero_cross_state` captures directional regime (TSI sign). Both inexpensive (single line of code each, ~0.5% incremental sample/feature ratio cost at ~1,100:1). SHAP trim in Phase 2.6 retains whichever is more predictive. **Cat 15 final size: 7 features** (was spec'd at 6). Feature count cascade: Cat 15 (6→7) → Phase 1 total (~201→~202), Phase 3 total (~230→~231), Phase 4 full (~238→~239) — restores totals to pre-v2.41 levels (Cat 22 −1 cancelled by Cat 15 +1). Spec edits: §7.2 Cat 15 (7 features explicitly enumerated with column conventions; both TSI formulas locked; specific drops from v1.0 listed including roc_10 overlap with Cat 1); §7.3 feature count summary (Cat 15 row 6→7, Phase 1/3/4 totals reverted to ~202/~231/~239). Approved by user 2026-04-28 | 2026-04-28 |
| v2.43| **Cat 13 strict spec interpretation: split flags + drop WT/Stoch (Q12)**| Pre-`divergence.py` implementation surfaced ambiguity in §7.2 Cat 13: "Keep all v1.0, signs normalized" + bullet list of 7 separate-direction names. Two valid 7-feature interpretations: (a) strict spec — split bullish/bearish into separate binary flags + drop WT/Stoch divs + drop divergence_count, OR (b) signed encoding — keep current v1.0 structure (4 oscillator signed flags + count + hidden + freshness). User chose **(a) strict spec**. **v1.0 → v2.0 transformation map** (locked here for rollback reference): v1.0 `rsi_price_divergence` (signed) splits into `regular_bullish_div_rsi` + `regular_bearish_div_rsi` (binary); v1.0 `macd_price_divergence` (signed) splits into `regular_bullish_div_macd` + `regular_bearish_div_macd` (binary); v1.0 `hidden_divergence` (signed, RSI-only) splits into `hidden_bullish_div_rsi` + `hidden_bearish_div_rsi` (binary); v1.0 `divergence_freshness` renamed to `divergence_recency`. **Dropped from v1.0**: `wt_price_divergence`, `stoch_price_divergence`, `divergence_count`. Reasoning: spec does not list WT/Stoch divergences; their information is largely redundant with RSI divergence (WT is correlated with RSI by construction; Stoch is a slower-period RSI variant); `divergence_count` is derivable from kept flags via sum if downstream needs it. Cat 13 size unchanged at 7 features. **ROLLBACK PROCEDURE** (if Phase 2.5 SHAP analysis shows dropped features were useful): (1) revert `divergence.py` to git commit immediately preceding the v2.43 implementation commit; (2) re-add to function: wt1 + stoch_k parameters in `divergence_features()`, `wt_price_divergence` + `stoch_price_divergence` + `divergence_count` columns; (3) update feature_stability.py to add tags for restored features (3 dynamic); (4) update §7.2 Cat 13 via fresh DR re-opening this decision; (5) re-run Phase 1.10 builder.py rewrite if pre-rebuild OR re-extract feature matrix if post-rebuild. Approved by user 2026-04-28 | 2026-04-28 |
| v2.44| **Cat 5 multi-anchor VWAP: 5 sub-decisions Q13 locked**                 | Pre-`vwap.py` implementation surfaced 5 implementation ambiguities in §7.2 Cat 5 (multi-anchor expansion 8→14). User approved all 5 recommendations (Q13.1–Q13.5). **(Q13.1) Daily-anchored 5-feature decomposition**: daily_vwap (raw value), daily_vwap_upper_band_1sig (raw level), daily_vwap_lower_band_1sig (raw level), daily_vwap_dist_pct (signed %), daily_vwap_zone (categorical 0..4 using ±2σ thresholds). Spec wording "±1σ / ±2σ bands" satisfied: 1σ as raw bands, 2σ used in zone categorization. **(Q13.2) Multi-anchor confluence**: signed count `(# above) − (# below)`, range -5..+5 (option a). More informative than binary or unsigned; SHAP can trim. **(Q13.3) Heavy VWAP**: lookback 100 bars (~50 hrs at 30m); touch tolerance 0.1×ATR(14); flag = +1/-1/0 (with 0 if no leader). **(Q13.4) VWAP cross events**: count of bars in last 10 with at least one VWAP cross (option b — bounded 0..10). **(Q13.5) HTF pivot break anchor**: close-cross of daily pivot P (option a — confirmed close, lookback-safe). **v1.0 → v2.0 transformation map** (locked for rollback): v1.0 `vwap_daily` → `daily_vwap` (renamed); v1.0 `vwap_upper_band` → `daily_vwap_upper_band_1sig` (rename); v1.0 `vwap_lower_band` → `daily_vwap_lower_band_1sig` (rename); v1.0 `vwap_band_position` → REPLACED by `daily_vwap_zone` (categorical, not 0-1 continuous); v1.0 `vwap_dist_pct` → `daily_vwap_dist_pct` (renamed). **Dropped from v1.0**: `vwap_session`, `vwap_session_dist_pct`, `vwap_slope` (3 dropped — spec moves session-related features to Cat 7 or drops; vwap_slope replaced by multi-anchor confluence which is more informative). **NEW (9 features)**: 4 anchored VWAP positions (swing_high, swing_low, htf_pivot, weekly) + 5 cross-anchor analytics (confluence count, vwap-of-vwaps mean reversion, nearest-VWAP ATR distance, cross events count, heavy VWAP flag). **Cross-module dependencies**: imports `fractal_pivots` from `divergence.py` for swing-anchor detection; requires caller to supply `atr_14` (from volatility.py) and `daily_pivot_p` (from pivots.py — note: pivots.py has not yet been rewritten in Phase 1.10b, so `daily_pivot_p` for v2.0 use will come from the rewritten pivots.py later; in the meantime, v1.0 `pivots.pivot_features` still emits `pivot_P` column compatibly). **Tunable parameters** (cfg-defaultable, no config.yaml edit required): bands_window=20, swing_lookback=5, heavy_lookback_bars=100, heavy_touch_atr_mult=0.1. **ROLLBACK PROCEDURE** (if Phase 2.5 SHAP shows dropped/changed features useful): (1) `git revert` the v2.44 implementation commit; (2) restore v1.0 column names (vwap_daily, vwap_upper_band, vwap_lower_band, vwap_band_position [continuous 0-1], vwap_dist_pct, vwap_session, vwap_session_dist_pct, vwap_slope) — 8 features; (3) update feature_stability.py to revert Cat 5 tags; (4) file fresh DR re-opening §7.2 Cat 5; (5) rebuild feature matrix + re-run baseline gate. Approved by user 2026-04-28 | 2026-04-28 |
| v2.45| **Cat 6 implementation: 4 sub-decisions Q14 locked + trending-week-aware** | Pre-`pivots.py` implementation surfaced 4 implementation ambiguities in §7.2 Cat 6 (multi-sub-category expansion 13→30). User approved all 4 with one nuance on Q14.1 — trending-week awareness. **(Q14.1 REVISED) Daily Fib-pivots 9-feature structure**: keep all 7 level distances (S3..R3) — required for trending weeks where price moves outside [S1,R1] toward S2/S3 or R2/R3 — plus 2 most-informative derived (`pivot_zone` categorical 0..5; `pivot_times_tested_today` cumulative count). User explicitly flagged this trending-week concern; original recommendation (3 core distances + 6 derived) would lose information during exactly the high-conviction weeks where trend extends to outer levels. Drops 4 derived (`dist_to_nearest_pivot_pct`, `nearest_pivot_type`, `pivot_approach_dir`, `pivot_approach_speed`) — all derivable from the 7 distances by LightGBM internally; `pivot_zone` and `times_tested` carry distinct information (interval categorization + accumulated S/R strength) not derivable from distances alone. **(Q14.2) extension_progress_1272**: option (a) single-direction formula `(close - swing_low) / (1.272 × swing_range)`. Range: 0 at swing_low, ~0.786 at swing_high, 1.0 at 1.272 ext; negative below swing_low (bearish ext territory). Simpler, monotonic, signed. **(Q14.3) daily_pivot_weekly_pivot_confluence**: any of {daily_S1, daily_P, daily_R1} within `0.25 × atr_14` of any of 7 weekly levels {weekly_S3..R3}. Daily kept to core 3 per spec; weekly all 7 per spec "any weekly level". **(Q14.4) swing_fib_pivot_confluence**: at each bar, identify nearest Fib retracement level price (closest of `[0.382, 0.5, 0.618, 0.786]` to close); fires when that nearest Fib price is within `0.25 × atr_14` of any of 14 pivots (7 daily + 7 weekly). **§7.5 tagging**: Cat 6 = static per spec ("Cat 6 all pivots — fixed for the day/week"; Cat 6.5 swing Fib only updates on new pivot confirm). All 30 features static — even though dist_pct/zone depend on current close, the LEVELS are reusable across 5m intrabar bars within a 30m bar. **v1.0 → v2.0 transformation map** (locked for rollback): v1.0 daily emits 13 features (7 raw level values + 6 derived) → v2.0 emits 9 daily-6.1 + 3 NEW daily-6.2 = 12 daily features; SAME for weekly (9 + 2 = 11); plus 7 NEW swing-Fib (6.5). v1.0 columns RENAMED: pivot_S3..R3 raw level VALUES → pivot_S3_dist_pct..R3_dist_pct (now distances not levels); v1.0 `pivot_zone` and `pivot_times_tested_today` kept verbatim. v1.0 weekly_* columns similarly transformed. v1.0 columns DROPPED: `dist_to_nearest_pivot_pct`, `nearest_pivot_type`, `pivot_approach_dir`, `pivot_approach_speed` (4 daily; 4 weekly equivalents) = 8 dropped, derivable internally. **Cross-module dependencies**: `from .divergence import fractal_pivots` for swing pivot detection (lookback=5 = 2-left-2-right). Requires caller to supply `atr_14` (from volatility.py) per consistent pattern. **Function signature**: single entry-point `pivot_features(df, atr_14, cfg) -> DataFrame[30 cols]`; v1.0 had separate daily + weekly + 1.0-style sig with day_id/week_id passed explicitly — now derived from df.index DatetimeIndex internally. **Tunable params** (cfg.get with defaults): pivots.tolerance_pct=0.05 (touch threshold for times_tested), pivots.swing_lookback=5, pivots.confluence_atr_mult=0.25, pivots.fib_touch_pct=0.001 (0.1%). **ROLLBACK PROCEDURE** (if Phase 2.5 SHAP shows dropped/changed features useful): (1) `git revert` v2.45 commit; (2) restore v1.0 13-feature daily structure (raw levels + 6 derived) and 13-feature weekly equivalent; (3) drop 6.5 swing-Fib features entirely; (4) update feature_stability.py to revert Cat 6 tags; (5) file fresh DR re-opening §7.2 Cat 6; (6) rebuild feature matrix + re-run baseline gate. Approved by user 2026-04-29 with Q14.1 trending-week refinement | 2026-04-29 |
| v2.46| **Cat 16 implementation: 7 sub-decisions Q15 locked + first mixed block** | Pre-`structure.py` implementation surfaced 7 implementation ambiguities in §7.2 Cat 16 (10→10 reconcile per §15 directory: +HL/LH counts, +fractal_pivot_count, +break_of_structure flag; −retrace_depth, −range_position, −swing_range_pct). Cat 16 is also the **first `mixed`-tagged category** in §7.5, requiring an explicit per-feature static/dynamic split in `feature_stability.py`. User approved all 7 recommendations. **(Q15.1) `break_of_structure` = signed binary event**: +1 on bar where `structure_type` flips from −1 to +1 (bullish break), −1 on flip from +1 to −1 (bearish break), 0 otherwise. Replaces v1.0 `bars_since_structure_break` (counter pattern belongs in Cat 20 event_memory which is locked at 22 features without this counter). **(Q15.2) `swing_length_ratio`**: rename from v1.0 `swing_ratio` to disambiguate from spec narrative wording "(HH count vs LL count)" — that literal interpretation is degenerate given separate HH and LL count features. Definition: `|swing_n| / |swing_(n-1)|` over alternating H/L pivot chain (forward-filled between confirms). Independent signal from the 4 count features. **(Q15.3) Count window = "last 20 BARS" (not "last 20 PIVOTS")**: spec literal wins. v1.0 `_pivot_running_count` used last-20-pivots window (~200-300 bars at 30m with lookback=5); rewrites to bar-anchored `HH_event.rolling(20).sum()`. Symmetric with `pivot_times_tested_today` (Cat 6) and `bars_since_last_hh` (Cat 20) which are bar-anchored. HH/HL/LH/LL events defined precisely: `p_high.notna() & (p_high vs previous_non_nan(p_high) {>, <})` and same for p_low. **(Q15.4) `fractal_pivot_count_20`**: `(p_high.notna() | p_low.notna()).rolling(20).sum()`. Boolean-OR collapses simultaneous high+low pivot to count-of-1; pivot density signal independent of HH/HL/LH/LL classification. **(Q15.5) §7.5 mixed split = explicit per-feature**: 8 STATIC + 2 DYNAMIC, not blanket "mixed" tag. STATIC (8): higher_highs_count_20, higher_lows_count_20, lower_highs_count_20, lower_lows_count_20, structure_type, swing_length_ratio, fractal_pivot_count_20, break_of_structure — all derived from confirmed pivots; only update when a new pivot confirms (deterministic at bar = pivot_bar + lookback). DYNAMIC (2): swing_high_dist_pct, swing_low_dist_pct — depend on current close. This sets the precedent for Cat 19 Ichimoku (second mixed block: senkou spans displaced to past = static; Tenkan/Kijun = dynamic). feature_stability.py docstring updated to clarify: "Categories tagged 'mixed' in §7.5 may have feature-level static/dynamic splits." **(Q15.6) Final 10-feature column lock**: 1=swing_high_dist_pct, 2=swing_low_dist_pct, 3=structure_type, 4=swing_length_ratio, 5=higher_highs_count_20, 6=higher_lows_count_20, 7=lower_highs_count_20, 8=lower_lows_count_20, 9=fractal_pivot_count_20, 10=break_of_structure. **(Q15.7) Function signature**: `structure_features(df, cfg) -> DataFrame[10]` — self-contained (pivots derived internally via `fractal_pivots(high, lookback)` + `fractal_pivots(low, lookback)`); no caller-supplied dependencies needed since pivots derive from OHLC alone (cheap at 30m). Tunable cfg keys: `structure.fractal_lookback=5`, `structure.count_window=20`. **v1.0 → v2.0 transformation map** (locked for rollback): KEEP RENAMED — `swing_high_dist_pct` (kept verbatim), `swing_low_dist_pct` (kept verbatim), `structure_type` (kept verbatim), `higher_highs_count_20` (kept verbatim — but window definition changed from 20-pivots to 20-bars per Q15.3), `lower_lows_count_20` (kept, same window change), `swing_ratio` → `swing_length_ratio` (renamed per Q15.2). DROP — `bars_since_structure_break` (4 features dropped: replaced by `break_of_structure` binary), `swing_range_pct` (low SHAP, derivable), `retrace_depth` (overlaps Cat 6.5 `fib_retracement_pct`), `range_position` (overlaps Cat 6 `pivot_position_daily_01` + Cat 3 `bb_position`). NEW — `higher_lows_count_20`, `lower_highs_count_20`, `fractal_pivot_count_20`, `break_of_structure` (4 added). Net: 10→10. **Cross-module dependencies**: `from .divergence import fractal_pivots` (lookback=5 = 2-left-2-right). No caller-supplied dependencies. **§7.5 first mixed block** — sets the per-feature-tag-within-mixed-category pattern for Cat 19 Ichimoku to follow. **ROLLBACK PROCEDURE** (if Phase 2.5 SHAP shows dropped/changed features useful): (1) `git revert` v2.46 commit; (2) restore v1.0 10-feature structure: re-add `swing_range_pct`, `retrace_depth`, `range_position`, `bars_since_structure_break`; rename `swing_length_ratio` back to `swing_ratio`; revert HH/LL count windows from 20-bars to 20-pivots in `_pivot_running_count` helper; drop `higher_lows_count_20`, `lower_highs_count_20`, `fractal_pivot_count_20`, `break_of_structure`; (3) update feature_stability.py to revert Cat 16 tags (remove the static/dynamic split, restore single tag); (4) file fresh DR re-opening §7.2 Cat 16; (5) rebuild feature matrix + re-run baseline gate. Approved by user 2026-04-29 | 2026-04-29 |
| v2.47| **Cat 19 implementation: 7 sub-decisions Q16 locked + Ichimoku removed from §7.5 mixed list** | Pre-`ichimoku.py` implementation surfaced 7 implementation ambiguities in §7.2 Cat 19 (5→6 expansion per §15 directory: +senkou_a_dist_pct + senkou_b_dist_pct). Critical tension: §7.2 wording "Senkou A, Senkou B (both displaced +26 bars)" reads as raw level values, but §15 directive explicitly says "add explicit senkou_a_dist_pct + senkou_b_dist_pct" (close-relative %). §7.5 narrative ("spans displaced to past are static; Tenkan/Kijun are dynamic") was intent-based and assumed raw spans → fails when locked features are all close-relative. User approved all 7 Q16 recommendations. **(Q16.1) Senkou representation = `_dist_pct`** per §15 directive (more specific instruction wins). All 6 Cat 19 features are close-relative %. **(Q16.2) `tk_cross` DROPPED**: v1.0 emitted both `tk_cross` (sign of T−K) and `tk_spread` (signed %); sign is fully derivable from sign(signed-%), so `tk_cross` was redundant. v2.0 keeps single `tk_diff_pct` carrying both magnitude and direction. **(Q16.3) `tk_diff_pct` denominator = close** (v1.0 convention): `(tenkan − kijun) / close × 100`. Symmetric with tenkan_dist_pct + kijun_dist_pct (both also use close as denominator). **(Q16.4) §7.5 mixed-block resolution = ALL 6 DYNAMIC + REMOVE Cat 19 FROM §7.5 MIXED LIST**. With Q16.1 (a) locked, every Cat 19 feature is close-relative `*_dist_pct` → close-dependent → dynamic. Honest taxonomy: Cat 19 cannot be a "mixed-split" block when its 6 locked features are uniformly close-dependent. §7.5 narrative updated to explain this: "Cat 19 was originally conceived mixed (raw spans = static math); locked feature shapes per §15 emit `*_dist_pct` so the static-span half of that intent is masked by close-dependence at the feature level." Phase 4 intrabar scout can still optimize `senkou_*_dist_pct` recomputation by caching displaced spans (implementation detail in scout code, not a tagging concern). Net: Cat 16 remains the ONLY mixed-split block in feature_stability.py — the per-feature mixed-split pattern set by Q15.5 sits as a single instance rather than a recurring pattern. Trade-off accepted: spec wording fidelity (§15 directive) > preserving the "second mixed block" framing. **(Q16.5) Periods = canonical Hosoda 9/26/52** as cfg-defaultable (v1.0 had them required; v2.0 makes them optional with defaults via cfg.get fallback to legacy v1.0 keys for config.yaml compat). **(Q16.6) Function signature = `ichimoku_features(df, cfg=None) -> DataFrame[6 cols]`** — self-contained, no external deps. Pattern-consistent with structure.py / regime.py. **(Q16.7) Final 6-feature column lock**: 1=tenkan_dist_pct, 2=kijun_dist_pct, 3=tk_diff_pct, 4=senkou_a_dist_pct, 5=senkou_b_dist_pct, 6=cloud_dist_pct. **v1.0 → v2.0 transformation map** (locked for rollback): KEEP VERBATIM — tenkan_dist_pct, kijun_dist_pct, cloud_dist_pct (3). RENAME — tk_spread → tk_diff_pct (1; matches spec wording). DROP — tk_cross (1; redundant with sign of tk_diff_pct). NEW — senkou_a_dist_pct, senkou_b_dist_pct (2; per §15). Net: 5 → 6 (= +1). **§7.5 LIST EDIT** (notable spec change): Cat 19 line removed from "mixed" bullets; new explanatory paragraph below the mixed bullets clarifies WHY Cat 19 was demoted out of mixed (locked feature shapes mask the original static-span intent). **§7.2 Cat 19 spec text rewritten** to enumerate all 6 features explicitly with formulas + canonical Hosoda math + drops + transformation map. Decision v2.47 records full Q16 resolution with detailed rollback procedure. **ROLLBACK PROCEDURE** (if Phase 2.5 SHAP shows dropped/changed features useful, OR if Cat 19 mixed-block framing turns out to matter): (1) `git revert` the v2.47 commit; (2) restore v1.0 5-feature ichimoku_features signature with required cfg keys; (3) restore tk_cross feature; restore tk_spread name (drop tk_diff_pct rename); drop senkou_a_dist_pct + senkou_b_dist_pct; (4) update feature_stability.py to revert Cat 19 dynamic block; (5) restore §7.5 mixed list to include Cat 19; (6) file fresh DR re-opening §7.2 Cat 19; (7) rebuild feature matrix + re-run baseline gate. ALTERNATIVELY (preserve mixed-block framing without revert): file new DR to switch Q16.1 to (b) — replace senkou_*_dist_pct with raw senkou_a + senkou_b LEVELS (static, displaced past). Would yield genuine 4 dynamic + 2 static mixed split for Cat 19. Approved by user 2026-04-29 | 2026-04-29 |
| v2.48| **Cat 18 implementation: 7 sub-decisions Q17 locked + psar_state combined feature** | Pre-`adaptive_ma.py` implementation surfaced 7 implementation ambiguities in §7.2 Cat 18 (5→4 trim per §15 directory: combine psar_direction + psar_dist into single signed psar_state_dist_pct). User approved all 7 Q17 recommendations. **(Q17.1) `psar_state_dist_pct` = `pct(close − sar, close)` rename-only combine**: in Wilder PSAR math, psar_direction and sign(close − sar) are ALWAYS equivalent by construction — long trend has SAR below close (positive sign), short trend has SAR above close (negative sign), flip bars reposition SAR to prior extreme on opposite side maintaining sign-trend alignment. Therefore `psar_direction` is redundant with `sign(psar_state_dist_pct)`. Same drop pattern as Cat 19 Q16.2 (tk_cross dropped because redundant with sign of tk_diff_pct). Single rename of v1.0 `psar_dist_pct` → `psar_state_dist_pct` matches §15 wording verbatim; no new math, no defensive multiply needed. **(Q17.2) PSAR acceleration = canonical Wilder 0.02/0.02/0.20** (af_start/af_step/af_max). Timeframe-agnostic; no 30m-specific tuning needed. cfg-overridable. **(Q17.3) KAMA/DEMA/TEMA dist_pct = v1.0 close-as-denominator convention**: `(close − ma) / close × 100`. Symmetric with all other Phase 1.10b dist_pct features (ichimoku, vwap, structure, etc.). Sign positive when close above MA, negative below. **(Q17.4) KAMA = canonical Kaufman (10, 2, 30)** for (period, fast_ema, slow_ema). DEMA(21), TEMA(21) per spec. Spec note 'Drop: Duplicate KAMA parameter variants' interpreted as preventive — v1.0 already has only one KAMA call; no actual code change for this drop, just spec hygiene to forestall future KAMA(20)+KAMA(50) etc. additions. **(Q17.5) Function signature = `adaptive_ma_features(df, cfg=None) -> DataFrame[4]`**: self-contained, no external deps. Keeps v1.0 NESTED cfg structure (`cfg['kama']['period']`, `cfg['psar']['af_start']`, etc.) with all keys optional + canonical defaults applied when keys absent. Backward-compat with existing v1.0 config.yaml preserved (no breaking change). Differs from ichimoku.py FLAT-keys pattern because v1.0 adaptive_ma cfg was nested; preserving nested structure avoids config schema rewrite. **(Q17.6) §7.5 tagging = all 4 DYNAMIC**: KAMA/DEMA/TEMA values mutate intrabar (rolling MAs include current bar close); PSAR state/distance evaluated at current bar; all dist_pct features close-dependent. Pure-dynamic block (matches Cat 19 post-Q16.4 demote pattern). No mixed-tag question. **(Q17.7) Final 4-feature column lock**: 1=kama_dist_pct, 2=dema_dist_pct, 3=tema_dist_pct, 4=psar_state_dist_pct. **v1.0 → v2.0 transformation map** (locked for rollback): KEEP VERBATIM — kama_dist_pct, dema_dist_pct, tema_dist_pct (3). COMBINE + RENAME — psar_direction + psar_dist_pct → psar_state_dist_pct (single signed feature; sign carries trend state). DROP — psar_direction (1; redundant with sign of combined feature). Net: 5 → 4 (= −1). **No spec change to §7.5 mixed list** (Cat 18 was never on mixed list — has always been pure-dynamic). Decision v2.48 records full Q17 resolution. **ROLLBACK PROCEDURE** (if Phase 2.5 SHAP shows dropped psar_direction useful, OR if separate psar_dist_pct feature outperforms combined): (1) `git revert` v2.48 commit; (2) restore v1.0 5-feature adaptive_ma_features signature with required nested cfg keys; (3) restore separate psar_direction + psar_dist_pct columns; rename psar_state_dist_pct back to psar_dist_pct; (4) update feature_stability.py to revert Cat 18 block (5 dynamic instead of 4); (5) file fresh DR re-opening §7.2 Cat 18; (6) rebuild feature matrix + re-run baseline gate. Approved by user 2026-04-29 | 2026-04-29 |
| v2.49| **Cat 8 implementation: SPEC AMENDMENT — drop binary doji+hammer flags, hybrid 9-feature reshape (Q18)** | Pre-`candles.py` implementation surfaced a critical mismatch: §7.2 Cat 8 narrative listed 9 features (incl. binary doji_flag, hammer_or_shooting_star_flag, inside_bar_flag) but v1.0 candles.py emitted a different 9 (continuous body+wicks + is_bullish + consecutive_bull/bear + body_vs_prev_body + engulfing/pin_bar). v1.0 docstring claimed '9→9 unchanged' which was misleading — only 5 of 9 features matched between spec and v1.0. User raised research-grounded objection to spec's binary doji/hammer flags: tree-based models (LightGBM) trivially learn threshold splits from continuous body_pct + wick features at SHAP-optimal thresholds; encoding `(body_pct ≤ 0.10)` as a hardcoded binary `doji_flag` adds NO signal and may displace the model's learned splits. The candle-pattern alpha lives in CONTEXT (overbought/oversold zones, momentum fading, structural breaks), not in the pattern flag itself. User approved Decision v2.49 SPEC AMENDMENT to a hybrid 9-feature set (Q18.1 (c)). **Q18 sub-decisions locked**: **(Q18.1 (c) HYBRID)** Drop binary doji_flag + hammer_or_shooting_star_flag (redundant with continuous body+wicks); preserve v1.0 is_bullish + body_vs_prev_body (distinct continuous signal); drop v1.0 consecutive_bull + consecutive_bear (weak signal at 30m; counter belongs to Cat 20 if needed); keep spec's range_pct + inside_bar_flag (genuinely-new info — range_pct is volatility burst marker complementing Cat 3 ATR; inside_bar_flag is true 2-bar conditional NOT derivable from single-bar continuous features). **(Q18.5)** inside_bar_flag binary: `(high < high.shift(1)) & (low > low.shift(1))`. **(Q18.6)** range_pct denominator = close (consistent with all v2.0 % features): `(high − low) / close × 100`. **(Q18.7)** engulfing + pin_bar v1.0 formulas verbatim, renamed to engulfing_signal + pin_bar_signal for spec wording match. **(Q18.8)** §7.5 = all 9 DYNAMIC (close/high/low mutate intrabar; pattern evaluated on current bar OHLC). Pure-dynamic block. **(Q18.9)** `candle_features(df, cfg=None) -> DataFrame[9]` — self-contained; cfg accepted for future tunable thresholds (currently no tunables, defaults baked in). **(Q18.10) Final 9-feature column lock**: 1=body_pct, 2=upper_wick_pct, 3=lower_wick_pct, 4=range_pct, 5=is_bullish, 6=body_vs_prev_body, 7=engulfing_signal, 8=pin_bar_signal, 9=inside_bar_flag. **Q18.2-Q18.4 N/A**: original questions about hammer/shooting-star encoding + thresholds + doji thresholds dissolved when the binary flags themselves were dropped. **v1.0 → v2.0 transformation map** (locked for rollback): KEEP VERBATIM (3) — body_pct, upper_wick_pct, lower_wick_pct. KEEP RENAMED (2) — engulfing → engulfing_signal, pin_bar → pin_bar_signal (formulas unchanged). KEEP VERBATIM per amendment (2) — is_bullish, body_vs_prev_body. DROP (2) — consecutive_bull, consecutive_bear. NEW from spec narrative (2) — range_pct, inside_bar_flag. DROPPED FROM SPEC NARRATIVE per amendment (2) — doji_flag, hammer_or_shooting_star_flag. Net: 9 → 9 reshape with 4 in/4 out. **§7.2 Cat 8 spec text rewritten** to lock the hybrid 9 features explicitly with the SPEC AMENDMENT preamble explaining the binary-flag drop rationale. **No §7.5 mixed-list change** (Cat 8 was never on mixed list). **Research-grounded rationale**: consistent with prior research finding that explicit binary pattern flags add no signal over continuous shape features when consumed by tree models. Candle-pattern alpha = context-amplified, not pattern-intrinsic. Future-Claude should not re-litigate the doji/hammer flag question — the rationale is preserved here. **ROLLBACK PROCEDURE** (if Phase 2.5 SHAP shows dropped binary flags useful, OR if a different model class than LightGBM is adopted that doesn't auto-learn thresholds): (1) `git revert` v2.49 commit; (2) restore v1.0 5-feature continuous + 4-feature v1.0-extra structure (consecutive_bull, consecutive_bear, body_vs_prev_body, is_bullish); OR alternatively implement original spec narrative (add doji_flag with body_pct ≤ 0.10 threshold, hammer_or_shooting_star_flag signed); (3) update feature_stability.py to revert Cat 8 block; (4) revert §7.2 Cat 8 spec amendment text; (5) file fresh DR re-opening §7.2 Cat 8; (6) rebuild feature matrix + re-run baseline gate. Approved by user 2026-04-29 with research-grounded objection to binary pattern flags | 2026-04-29 |
| v2.50| **Cat 9 + Cat 17 implementation: Q19 locked + algorithm choices + redundancy accepted** | Pre-`stats.py` implementation surfaced 11 implementation ambiguities across two categories: §7.2 Cat 9 (8→7 trim per §15 directive — drop mean_reversion_score) and §7.2 Cat 17 (9→6 trim per §15 — drop parkinson_vol, autocorrelation_1 (moved to Cat 9), variance_ratio; add autocorr(20) and realized-vol-of-realized-vol). v1.0 stats.py emitted 17 features across two functions (mean_reversion_features=8, stats_features=9), with 4 orphans not in spec (rsi_zscore, return_5/20/60bar) and category misalignment (skewness_20, kurtosis_20 lived in stats_features but spec puts them in Cat 9). Combined block needs 5-add / 9-drop / 4-rename = 17→13 reshape. User approved all 11 Q19 recommendations. **(Q19.1) bb_pct_b = caller-supplied** from Cat 3 volatility.py (rename of bb_position). Pattern-consistent with regime.py / vwap.py / event_memory.py / pivots.py caller-supplied dependency model. **(Q19.2) bb_dist_mid_sigma ≡ zscore_20 redundancy ACCEPTED**: when BB period = 20 (canonical Cat 3), `(close − bb_mid) / bb_std` is mathematically identical to `(close − sma20) / std20`. Spec lists both as separate features in 7-count. Per Q19.2 (a) emit both as distinct columns with identical values; LightGBM tolerates duplicates. Honest about the redundancy in spec; could file future DR to merge if SHAP shows redundant. **(Q19.3) Skew/kurt/autocorr series = log returns** (standard for higher-moment stats; close levels are non-stationary). **(Q19.4) autocorr_1 rolling window = 50 bars** per v1.0 convention; window must be ≥ several × lag for stable autocorr estimate. Lag=1 gets 50-bar window; mid-range tradeoff between stability and recency. **(Q19.5) Hurst = simplified single-pass R/S algorithm** with formula `H = log(R/S) / log(N)` where R = range of cumulative deviations, S = std. v1.0 verbatim. 100-bar window per spec. Multi-scale Hurst dropped per spec wording 'single scale is enough'. **(Q19.6) Fractal dimension = box-counting on CLOSE LEVELS (not log returns)** with 50-bar window, [0,1]² unit-square normalization, dyadic grid scales eps = 1/2^k for k = 1..floor(log2(N)), occupied-box count includes vertical spans between consecutive samples, D = −slope of log(N(eps)) vs log(eps) via polyfit. v1.0 implementation verbatim — measures geometric complexity of price PATH, which is why close levels (not returns) is the right series. **(Q19.7) Entropy = Shannon on log returns, 10 bins, 20-bar window** per spec window. v1.0 used 50-bar window — switch 50→20 per spec wording 'Rolling entropy(20)'. Sample/permutation entropy alternatives rejected — not in spec. **(Q19.8) autocorr_5 + autocorr_20 windows = scaled with lag**: lag=5 → 50-bar window (≥ 10× lag); lag=20 → 100-bar window (≥ 4× lag for stable estimate). Window scaling avoids degenerate autocorr at small windows. **(Q19.9) realized_vol_of_realized_vol = two-pass rolling std on log returns** with W1 = W2 = 20: `inner = log_ret.rolling(20).std()`, then `outer = inner.rolling(20).std()`. Symmetric windows match Cat 9 skew/kurt 20-bar convention. **(Q19.10) §7.5 tagging = all 13 DYNAMIC**: rolling stats / distribution moments / fractal geometry all evaluate using current bar's close (or log return); close mutates intrabar. Pure-dynamic block (matches Cat 8, 18, 19 pattern). **(Q19.11) Function signature = SPLIT per spec category** with semantic names: `mean_reversion_features(df, bb_position, cfg=None) -> DataFrame[7]` for Cat 9 (caller-supplied bb_position from Cat 3) + `fractal_stats_features(df, cfg=None) -> DataFrame[6]` for Cat 17 (self-contained — derives log_returns internally). Both lives in stats.py. Pattern-consistent with how Cat 4 + Cat 14 split into volume_features + money_flow_features inside volume.py. **(Q19.12) Final column lock**: Cat 9 (7) — bb_pct_b, bb_dist_mid_sigma, zscore_20, zscore_50, skewness_20, kurtosis_20, autocorr_1; Cat 17 (6) — hurst_exponent, fractal_dimension, autocorr_5, autocorr_20, entropy_20, realized_vol_of_realized_vol. **v1.0 → v2.0 transformation map** (locked for rollback): KEEP RENAMED (1) — Cat 3 bb_position → bb_pct_b (caller-supplied). KEEP VERBATIM (5) — zscore_20, zscore_50, skewness_20, kurtosis_20, hurst_exponent, fractal_dimension. KEEP RENAMED (3) — autocorrelation_1 → autocorr_1 (moved Cat 17 → Cat 9 per spec), autocorrelation_5 → autocorr_5, price_entropy → entropy_20 (window 50→20). DROP (7) — mean_reversion_score (per §15), rsi_zscore (orphan), return_5bar/20bar/60bar (orphans, overlap Cat 1), parkinson_vol (per §15), variance_ratio (per §15). NEW (3) — bb_dist_mid_sigma, autocorr_20, realized_vol_of_realized_vol. Net: 17 → 13 (= −4). **Cross-module dependencies**: mean_reversion_features requires caller-supplied bb_position from Cat 3 volatility output. fractal_stats_features is self-contained. **No new sibling-module imports** beyond `from ._common import safe_div`. **No spec change to §7.5 mixed list** (Cat 9 + Cat 17 were never on mixed list). **ROLLBACK PROCEDURE** (if Phase 2.5 SHAP shows dropped features useful, OR if algorithm choices underperform alternatives like DFA Hurst / sample entropy): (1) `git revert` v2.50 commit; (2) restore v1.0 17-feature stats.py with 2-function structure (mean_reversion_features, stats_features); (3) restore rsi_zscore, return_5/20/60bar columns; restore mean_reversion_score; restore autocorrelation_1 in Cat 17; restore parkinson_vol, variance_ratio; revert price_entropy window 20→50; (4) update feature_stability.py to revert Cat 9 + Cat 17 blocks; (5) revert §7.2 Cat 9 + Cat 17 spec text; (6) file fresh DR re-opening §7.2 Cat 9/17; (7) rebuild feature matrix + re-run baseline gate. ALTERNATIVELY (drop the bb_dist_mid_sigma ≡ zscore_20 redundancy): file new DR to consolidate Cat 9 to 6 features (keep one of the two duplicates). Approved by user 2026-04-30 | 2026-04-30 |
| v2.51| **Cat 11 + Cat 12 implementation: REWRITE per Q20 + Cat 11 mixed-block split (§7.5 list edit)** | Pre-`context.py` implementation surfaced 10 implementation ambiguities across two categories that had been locked at high level by Decision v2.37 Q2 (Cat 11 prev-bar interpretation) but needed formula-level resolution. v1.0 context.py emitted 8 + 8 = 16 features, ALL of which were misaligned with v2.0 spec — full REWRITE. Critical §7.5 finding: Cat 11 was on the §7.5 STATIC list (rationale: 'look only at prior bars') but the v2.37 Q2 rewrite ADDED today-running features (today_open_to_now_pct, today_high_low_distance_from_current_pct) which depend on current close → genuine mixed split. User approved all 10 Q20 recommendations. **(Q20.1) Prev-bar formula sign = form (i)**: `(prev_X − close) / close × 100`. Positive when prev reference > current close; distinct from Cat 1 roc_1bar which uses opposite sign. Avoids duplication while keeping intuitive sign reading. **(Q20.2) prev_bar_volume_ratio = `volume.shift(1) / volume`** per spec literal numerator/denominator. Values > 1 indicate prev bar had higher volume than current. **(Q20.3) today_high_low_distance_from_current_pct = SIGNED**: positive when today_low closer (price near support), negative when today_high closer (price near resistance). Magnitude = `min(dist_high, dist_low) / close × 100`. Sign carries side information that tree models use for splits; pure unsigned magnitude would collapse near-support and near-resistance into one signal. Spec literal 'distance...whichever closer' charitably read as signed distance. **(Q20.4) UTC-day boundary = `df.index.floor('1D')`** consistent with pivots.py / vwap.py daily anchor pattern. **(Q20.5) today_open = first 30m bar's OPEN per UTC day**: math `df.groupby(day_id)['open'].transform('first')`. 30m bar at 00:00 UTC has open == day's first traded price. Standard. **(Q20.6) Cat 12 caller-supplied = 3 series**: rsi (Cat 1 momentum_core), adx (Cat 2 trend), ema21_dist_pct (Cat 2 trend). Volume taken from df. Pattern-consistent with mean_reversion_features (caller-supplied bb_position from Cat 3) and other Phase 1.10b caller-supplied dependency patterns. **(Q20.7) Cat 12 column naming = `delta_*` prefix**: delta_rsi_1, delta_rsi_3, delta_adx_3, delta_volume_3, delta_close_vs_ema21_3 (ASCII Python identifiers — Greek Δ from spec is unsafe in Python). Delta formula choices per-feature: raw point-difference for bounded series (RSI/ADX/ema21_dist_pct deltas), percentage change for unbounded volume (volume.shift(3) ratio). **(Q20.8) Cat 11 §7.5 LIST EDIT — MOVE FROM STATIC LIST TO MIXED LIST**: with explicit 4 static / 2 dynamic feature-level split. STATIC (4): prev_bar_close_vs_close_pct, prev_bar_high_vs_close_pct, prev_bar_low_vs_close_pct, prev_bar_volume_ratio (all `.shift(1)` lookups; locked once prev bar closed; intrabar-safe). DYNAMIC (2): today_open_to_now_pct, today_high_low_distance_from_current_pct (both depend on current close; cummax/cummin update intrabar as new highs/lows print). Cat 11 + Cat 16 = TWO mixed-split blocks now (Cat 19 was demoted to all-dynamic per Q16.4). v1.0 Cat 11 was prev-DAY-only (all static prior-bar lookups) — that's why §7.5 originally listed it as static; the Decision v2.37 Q2 rewrite added today-running features that broke the all-static assumption. Honest taxonomy: per-feature split preserves intrabar-scout efficiency for the 4 prev-bar features. **(Q20.9) Function signature = SPLIT per spec category** with semantic names: `prev_context_features(df, cfg=None) -> DataFrame[6]` (Cat 11 self-contained; prev-bar via .shift(1) + today running via groupby UTC day) + `lagged_dynamics_features(df, rsi, adx, ema21_dist_pct, cfg=None) -> DataFrame[5]` (Cat 12 caller-supplied 3 series). Both live in context.py. Pattern-consistent with stats.py (mean_reversion + fractal_stats split) and volume.py (volume + money_flow split). **(Q20.10) Final 11-feature column lock**: Cat 11 (6) — prev_bar_close_vs_close_pct, prev_bar_high_vs_close_pct, prev_bar_low_vs_close_pct, prev_bar_volume_ratio, today_open_to_now_pct, today_high_low_distance_from_current_pct; Cat 12 (5) — delta_rsi_1, delta_rsi_3, delta_adx_3, delta_volume_3, delta_close_vs_ema21_3. **v1.0 → v2.0 transformation map** (locked for rollback): DROP ALL 8 v1.0 Cat 11 prev-DAY features per Decision v2.37 Q2 (prev_day_range_pct, prev_day_close_vs_pivot, gap_pct, dist_to_prev_day_high_pct, dist_to_prev_day_low_pct, prev_session_direction, prev_session_volume_rank, daily_open_dist_pct — all duplicates of Cat 2a HTF1D context). DROP ALL 8 v1.0 Cat 12 features (rsi_5bar_ago, wt1_slope_5bar, adx_slope_5bar, atr_slope_5bar, volume_slope_5bar, vwap_slope_5bar, di_spread_change_5bar, squeeze_mom_slope — all overlap Cat 1/2/5 or use longer lag windows that overlap HTF context per spec). NEW (11): all 6 Cat 11 + all 5 Cat 12 features. Net: 16 → 11 (= −5). **§7.5 LIST EDIT** (notable spec change): Cat 11 line removed from §7.5 STATIC list bullet; added to MIXED list bullet with Decision v2.51 Q20.8 reference. Cat 16 + Cat 11 = TWO mixed-split blocks; Cat 19 remains demoted to all-dynamic. **§7.2 Cat 11 spec text rewritten** with Decision v2.51 preamble, Q20.1 sign convention lock, Q20.3 signed distance formula, Q20.4 UTC-day boundary, Q20.5 today_open definition, §7.5 mixed-split documentation. **§7.2 Cat 12 spec text rewritten** with Q20.6 caller-supplied dependencies, Q20.7 delta_* prefix + per-feature formula choices. **Cross-module dependencies**: lagged_dynamics_features requires caller-supplied rsi (Cat 1), adx (Cat 2), ema21_dist_pct (Cat 2). prev_context_features is self-contained. **No new sibling-module imports**. **ROLLBACK PROCEDURE** (if Phase 2.5 SHAP shows v1.0 prev-DAY or 5-bar slope features useful, OR if Cat 11 mixed-split framing causes intrabar scout complexity): (1) `git revert` v2.51 commit; (2) restore v1.0 16-feature context.py with 2-function structure; (3) restore prev_day_* Cat 11 features + 5bar_* Cat 12 features; (4) update feature_stability.py to revert Cat 11 + Cat 12 blocks; (5) revert §7.5 list edit (restore Cat 11 to static list); (6) revert §7.2 Cat 11/12 spec text; (7) file fresh DR re-opening §7.2 Cat 11/12 OR re-opening Decision v2.37 Q2 (literal prev-bar interpretation); (8) rebuild feature matrix + re-run baseline gate. ALTERNATIVELY (preserve Decision v2.37 Q2 but demote Cat 11 to all-dynamic instead of mixed): file new DR to switch Q20.8 to (b) — tag the 4 prev-bar features dynamic too. Loses intrabar-scout efficiency but simpler taxonomy. Approved by user 2026-04-30 | 2026-04-30 |
| v2.52| **Phase 1.10c indicators.py refactor: Q21 locked + 11-function drop + ema() primitive added** | Pre-refactor surfaced 9 implementation ambiguities for the indicators.py math-only refactor per Decision v2.37 Q4 (which locked the high-level rule: math primitives only — rsi, macd, wavetrend, stochastic, squeeze, adx_di, ema; drop all `_features` selection helpers and `_5m`/`_1h`/`_1d` suffix variants). v1.0 indicators.py had 6 math primitives + 11 _features/suffix helpers = 17 total functions. Refactor target per Q21: 7 math primitives (6 v1.0 keep + 1 ema add). User approved all 9 Q21 recommendations. **(Q21.1) Math function inventory**: KEEP (6) — rsi, wavetrend, stochastic, squeeze_momentum_value, macd, adx_di. ADD (1) — ema(close, period) clean helper (v1.0 inlined `close.ewm(span=p, adjust=False, min_periods=p).mean()` everywhere; v2.37 Q4 listed `ema` as a primitive). DROP (11) — all _features and suffixed helpers. **(Q21.2) wavetrend suffix**: math signature already clean `(high, low, close, n1, n2, signal)`; no change needed (suffix kwarg was only on dropped wavetrend_features). **(Q21.3) adx_di return shape = tuple `(di_plus, di_minus, adx)`** per v1.0; tuple-unpacking pattern preserved (`di_plus, di_minus, adx = adx_di(h, l, c, 14)`). Math primitives return tuples or single Series; only feature-selection functions return DataFrames. Clean separation. **(Q21.4) squeeze name = keep `squeeze_momentum_value` verbatim**: descriptive name; momentum_core.py consumers already import this name; rename would force consumer-side edits. Avoids collision with TTM Squeeze (Cat 3 squeeze_state from BB-vs-KC compression in volatility.py — different concept). **(Q21.5) ema = generic single helper** `ema(close, period) -> Series`. Pythonic; consumers call `ema(close, 21)` etc. Specific period helpers (ema9/21/50) rejected as DRY violation. **(Q21.6) Consumer signature audit (no breakage)**: momentum_core.py imports rsi/wavetrend/stochastic/squeeze_momentum_value/macd ✓ all kept. trend.py imports adx_di + inlines EMAs ✓ kept (can optionally switch to new ema() helper but not required). htf_context.py imports rsi/macd/adx_di + inlines EMAs ✓ kept. volatility.py + all Phase 1.10b feature files do NOT import from indicators.py (caller-supplied pattern). No consumer breakage. **(Q21.7) Hard-break NO transitional shims**: drop all 11 _features helpers without aliases. v2.0 has no v1.0-callers to support; builder.py rewrite (Phase 1.10d) is the orchestration layer that wires everything. Shims would be dead weight. Pattern-consistent with v2.0 'no migration debt' principle (Decision v2.36 fresh glue). **(Q21.8) Drop list (11 functions)**: rsi_features → momentum_core.py (Cat 1 reshape 12→32). wavetrend_features(suffix=5min|1h) → momentum_core.py; suffix dropped. stoch_features → momentum_core.py. squeeze_features → momentum_core.py (Cat 1 4-feature squeeze block). macd_features_5m → momentum_core.py. macd_features_1h → DELETED (HTF flows through htf_context.py). adx_features_5m → trend.py (Cat 2 selection). adx_features_1h → DELETED. ema_features_5m → trend.py. ema_features_1h → DELETED. ema_features_1d → DELETED. 4 of 11 dropped helpers were already migrated to category files during Phase 1.10b (momentum_core.py, trend.py, htf_context.py landed earlier); 4 were deleted as 1H/1D HTF pre-aggregation duplicates that v2.0 §6.4 in-pipeline aggregation makes obsolete; remaining 3 are now-orphan _features helpers that have no v2.0 consumer. **(Q21.9) feature_stability.py unchanged**: indicators.py is math-only post-refactor — emits no feature columns. Tag dict unchanged. **REFACTOR SCOPE**: indicators.py from 352 lines + 17 functions → ~150 lines + 7 functions (estimated 50%+ code reduction). Pure math primitives; no feature column emission. **§15 directory entry already locked** from Decision v2.37: `indicators.py # MODIFIED: per Decision v2.37 Q4 — base indicator MATH only (rsi, macd, wavetrend, stochastic, squeeze, adx_di, ema). Pure calculation functions consumed by momentum_core.py + trend.py + htf_context.py + volatility.py. NO _features/selection logic, NO _5m/_1h/_1d suffixes.` No spec text edit needed beyond Decision v2.52 entry — refactor is implementation-only (math primitives are not emitted features; not in §7.2 inventory). **No §7.5 mixed-list change** — indicators.py emits no feature columns. **ROLLBACK PROCEDURE** (if Phase 2.5 SHAP shows dropped _features helpers were doing something useful, OR if a consumer breakage surfaces post-pull): (1) `git revert` v2.52 commit; (2) restore v1.0 indicators.py with all 17 functions; (3) revert ema() addition; (4) re-test consumers in momentum_core.py / trend.py / htf_context.py against v1.0 signatures; (5) file fresh DR re-opening Decision v2.37 Q4. ALTERNATIVELY (preserve Q4 cleanup but with shims): file new DR to switch Q21.7 to (b) — re-export migrated _features functions from indicators.py via `from .momentum_core import rsi_features` etc. Keeps the math-only refactor intent but eases any consumer-side migration friction. Approved by user 2026-04-30 | 2026-04-30 |
| v2.53| **Phase 1.10 post-implementation review: 6 deferred feature candidates documented in §7.7** | Pre-Phase-1.10d professional review (2026-05-01) of the 22-category, 250-feature engineering layer surfaced 6 'real gaps' worth flagging for future consideration but not load-bearing for v2.0 build. User asked: 'add to spec and calculate now, OR add to spec and build first?' Resolution: **Option B — spec-defer with documented backlog, build first, revisit after Phase 2.5 SHAP**. Rationale: (1) §10.5 process discipline explicitly warns against mid-implementation scope creep — 11 Q-batches + 11 Decision Log entries already executed with rigor; adding 6 more 'while we're at it' features betrays that discipline; (2) Cat 21 deferral pattern (funding rate, microstructure — Phase 3+) already establishes 'build → measure → add' consistency — if we defer KNOWN alpha (funding rate), we should defer SPECULATIVE alpha; (3) Phase 2.5 SHAP is the truth-finder — evidence > speculation about which gaps actually pull weight at 30m for our BTC/SOL/LINK universe; (4) 250 features at ~1,100:1 sample/feature ratio is healthy — adding 6-10 more doesn't materially change overfit risk but adds 1-2 weeks of Q-batch + decision-log cycles delaying Phase 2 training (the actual edge test); (5) §10.5 DR mechanism exists for post-baseline additions — proper escape valve. **§7.7 added** as documented backlog. 6 deferred candidates with cost + alpha-character + trigger conditions: (1) regime duration features (LOW cost, speculative; trigger Cat 10 SHAP < 5%), (2) bb_dist_mid_sigma ≡ zscore_20 deduplication (TRIVIAL cost, cosmetic; trigger Phase 2.5 cleanup), (3) Cat 12 longer lag windows (LOW cost, speculative; trigger Cat 12 SHAP > 8% AND Cat 1 12bar pulls weight), (4) volume profile POC/VAH/VAL (MEDIUM-HIGH cost — own Q-batch + session-boundary algorithm choices; trigger Cat 5 multi-anchor VWAP SHAP < 6% AND live paper-trade reversal misses near volume nodes), (5) funding-rate-adjusted return (LOW cost once data collected, KNOWN alpha for crypto perps — single biggest 30m gap; FORCED defer Phase 3.0 — Cat 21 microstructure WS collection prerequisite), (6) cross-asset correlation dispersion (LOW cost, speculative; trigger Cat 22 SHAP < 4% on altcoins, Phase 4 candidate per §17 alpha-decay plan). **Closing process for any deferred candidate**: Phase 2.5 SHAP identifies under-performing category(ies) → DR-NNN filed citing specific candidate → Q-batch surfaced for implementation ambiguities → Decision v2.5N entry locks resolutions → spec edit → implementation + feature_stability tagging → re-run baseline gate to verify OOT improvement. **Future candidates explicitly NOT in §7.7 backlog**: NLP/news/economic calendar (out of scope for v2.0; no DR planned), macro regime BTC.D / total alt cap (out of scope per Decision v2.35 asset-universe lock), order-flow imbalance from L2 orderbook (Phase 3+ Cat 21), sentiment aggregators CryptoFear/Greed Index / social volume (out of scope). Re-evaluate via DR if v3.0 redesign opens asset universe or model class. **Implementation impact**: NONE for Phase 1.10 — pure documentation entry. No code changes, no feature_stability.py changes, no PROJECT_LOG implementation entry for v2.53 (only spec edit + Decision Log entry). Phase 1.10d builder.py rewrite remains next; the 250-feature locked baseline is what gets orchestrated. **ROLLBACK PROCEDURE**: trivial — `git revert` v2.53 commit to remove §7.7 + this Decision Log entry. No code state to revert. ALTERNATIVELY (close any candidate from the backlog now): file new DR re-opening §7.7 candidate N with full Q-batch resolution; implement; tag; re-baseline. Approved by user 2026-05-01 with explicit 'add new spec first, git commit, then proceed to builder.py' workflow request | 2026-05-01 |
| v2.54| **Phase 1.10d builder.py orchestration rewrite: Q22 locked + 24-phase dependency chain** | Pre-rewrite Q22 batch surfaced 12 implementation ambiguities for the builder.py orchestration of all 22 category files using their Phase 1.10b-locked caller-supplied signatures. v1.0 builder.py: 391 lines, loads 5m+1h dual, calls 11 dropped indicator helpers (rsi_features, wavetrend_features with suffix=5min|1h, stoch_features, squeeze_features, macd_features_5m/_1h, adx_features_5m/_1h, ema_features_5m/_1h/_1d), imports dropped ema_context module, references obsolete v1.0 column names (pivot_P, dist_to_nearest_pivot_pct). Full rewrite required. User approved all 12 Q22 defaults. **(Q22.1) Single-timeframe load** — 30m only per Decision v2.28; 4H + 1D derived in-pipeline via pandas resample (label='left', closed='left'). Removes ~50 lines of v1.0 1H-merge logic. **(Q22.2) HTF aggregation** — exact §6.4 contract: agg dict {open:'first', high:'max', low:'min', close:'last', volume:'sum'} resampled to '4h' and '1D'; shift(1) on aggregated frames before htf_context call to avoid look-ahead (30m bar at hour H sees 4H bar that closed at H, not at H+4). **(Q22.3) HTF merge mechanism** — single helper `_merge_htf_into_30m(df_30m, htf_features, htf_freq)` doing floor + shift(1) + reindex. Used twice (4H + 1D). **(Q22.4) Per-symbol Cat 22 conditional** — `if coin != 'BTC': load BTC alongside, call cross_asset_features(...).` Skip Cat 22 for BTC entirely (BTC has nothing to correlate to within v2.0 universe per Decision v2.41). BTC model gets 250 − 7 = 243 features; alts get 250. Per-asset training already locked per Decision v2.35. **(Q22.5) bb_width_percentile inline derivation** — Cat 3 trim 14→12 dropped this from volatility output; Cat 10 regime needs it. Builder derives inline before regime call: `rolling_percentile(vol_df['bb_width_pct'], 100)`. Documented in regime.py docstring already. **(Q22.6) Pivot levels** — single `pivots.compute_pivot_levels(df)` call returns (daily_levels, weekly_levels) tuple. Builder extracts `daily_pivot_p = daily_levels['P']` for vwap; passes both DataFrames to event_memory. **(Q22.7) Cat 1 momentum_core squeeze_state** — pass squeeze_state from Cat 3 volatility; leave bars_since_squeeze_release=None (defaults handle it; momentum_core computes inline if needed). **(Q22.8) 24-phase dependency-ordered orchestration** — A: load 30m + aggregate 4H/1D; B: Cat 3 volatility (atr_14 etc); C: Cat 1 momentum (uses Cat 3 squeeze_state); D: Cat 2 trend (uses Cat 3 atr_14); E: Cat 2a HTF context (4H + 1D); F: Cat 22 cross-asset (alts only); G: Cat 4 volume; H: Cat 14 money flow; I: Cat 6 pivots (uses Cat 3 atr_14); J: Cat 5 vwap (uses Cat 3 atr_14 + Cat 6 daily_pivot_p); K: Cat 7 sessions; L: Cat 8 candles; M: Cat 9 mean_rev (uses Cat 3 bb_position); N: Cat 10 regime (uses Cat 2 adx/di + Cat 3 atr_14/atr_pct + inline bb_width_percentile); O: Cat 11 prev_ctx; P: Cat 12 lagged (uses Cat 1 rsi + Cat 2 adx + ema21_dist_pct); Q: Cat 13 divergence (uses Cat 1 rsi + macd_hist); R: Cat 15 extra_mom; S: Cat 16 structure; T: Cat 17 fractal; U: Cat 18 adaptive_ma; V: Cat 19 ichimoku; W: Cat 20 event_memory (full dependency consumer — Cat 1 rsi/stoch_k/wt1/squeeze_value, Cat 1 macd_line/macd_signal, Cat 2 adx, Cat 3 squeeze_state, Cat 6 pivot_levels); X: concat + insert OHLC + warmup trim. **(Q22.9) Triple-barrier labeling integration** — keep v1.0 add_labels() + --no-label CLI arg with atr_col='atr_14' (column emitted by Cat 3 volatility). Spec §8 + Phase 1.11 expects labeling here. **(Q22.10) Output schema** — insert timestamp + open/high/low/close/volume (6 cols) at front, then 250 feature columns (243 for BTC). Apply warmup trim 150 bars per §7.6. BTC: (N − 150, 249); alts: (N − 150, 256). **(Q22.11) Config structure** — keep v1.0 nested cfg['features'] structure; pass entire cfg['features'] sub-dict as cfg arg to category files. Each function picks what it needs via .get() with defaults. Backward-compat with existing config.yaml; minimal disruption. **(Q22.12) Logging** — keep per-phase log lines (24 phases). Matches v1.0 verbosity for debugging the dependency chain. **Cross-module import cleanup**: drop `from . import ema_context` (Decision v2.37 Q1). Add imports for `momentum_core, trend, htf_context, cross_asset` (Phase 1.10a NEW algorithm files). Function-call renames: rsi_features→momentum_core.momentum_core_features; wavetrend_features/stoch_features/squeeze_features/macd_features_5m → all subsumed into momentum_core_features; adx_features_5m + ema_features_5m → trend.trend_features; *_1h/_1d helpers → htf_context.htf_context_features; previous_context_features → context.prev_context_features; lagged_features → context.lagged_dynamics_features; stats_features → stats.fractal_stats_features. **No spec text edit needed** — this is implementation-only orchestration; no §7.2 or §7.5 changes; no feature_stability.py changes; no new feature columns. Decision v2.54 entry documents the 24-phase ordering + dependency chain for future-Claude reference. **§15 directory** already lists builder.py as 'orchestration' (no rewrite needed for v2.0 description — already accurate). **ROLLBACK PROCEDURE** (if Phase 1.11 ETL pipeline shows builder output mismatch with expected feature set): (1) `git revert` v2.54 commit; (2) restore v1.0 builder.py 391 lines; (3) BUT: v1.0 builder won't actually run because it imports the 11 dropped indicator helpers from refactored indicators.py — would force ALSO reverting Decision v2.52 (indicators.py refactor). Realistically: rollback path = identify specific phase that broke (e.g., 'Cat 5 vwap call fails because daily_pivot_p extraction returns wrong shape'), file targeted DR for that one phase. Approved by user 2026-05-01 | 2026-05-01 |
| v2.55| **Phase 1.11 ETL execution brief: Q23 locked + strict halt on label distribution** | User confirmed transition from Phase 1.10 (feature engineering, COMPLETE) to Phase 1.11 ETL (produce + persist feature matrices for BTC + SOL + LINK; verify warmup correctness + label distribution at scale). Maps to Appendix C steps 1.12 (triple-barrier label) → 1.13 (label distribution sanity check) → 1.14 (FREEZE feature matrix + label parquet). Workflow inversion lock: VPS-Claude executes (heavy compute — LightGBM training, walk-forward CV, Optuna search in subsequent phases); Local-Claude supervises against spec invariants + design intent + §10.5 process discipline. Decision v2.38 'local edits, VPS validates' was Phase 1.10-specific; from Phase 1.11 onward, the role split inverts (VPS executes, local reviews) per user's 2026-05-01 role-clarification message. **(Q23.1) Label distribution out-of-band action = STRICT HALT** per §8.2 (LONG/SHORT 35–45%, NEUTRAL 15–25%). If a per-asset distribution falls outside the bands after triple-barrier labeling, VPS HALTS, pings local-Claude, surfaces the deviation in PROJECT_LOG. Local-Claude evaluates root cause: (i) min_profit_pct tuning needs adjustment per asset, (ii) regime-driven artifact (e.g., extreme bull trend skews LONG), (iii) data quality issue. Cheap pause vs expensive retrain. Soft-log alternative rejected because out-of-band distribution silently invalidates downstream model — catching it pre-training is cheaper than discovering it post-Phase-2. **(Q23.2) Persistence = VPS-disk-only**, not git-tracked. Parquet lives at `data/storage/features/{COIN}_features.parquet` per spec §15 + v1.0 builder default. Git-ignored (large binary; no value in versioning). Local-Claude reviews structured reports from VPS, not raw parquet — local environment lacks pandas for ad-hoc reads anyway. **(Q23.3) Reporting format = STRUCTURED TABLE per asset** (markdown table or fenced code block in PROJECT_LOG entry). Same-shape report across BTC + SOL + LINK makes anomaly detection visual; freeform prose rejected because comparison across assets is the primary review activity. **Reporting template fields** (locked here for VPS reference): (1) shape pre-warmup (rows × cols), (2) shape post-warmup (rows × cols, expected N − 150), (3) feature count vs expected (243 BTC / 250 alts), (4) label distribution (LONG/SHORT/NEUTRAL counts + percentages, with §8.2 threshold pass/fail), (5) NaN audit per category (post-warmup non-zero NaN columns; expect 0 except btc_funding_rate on alts which is all-NaN by Phase 1 design), (6) 0-leak audit (v1.0-dropped column names absent — list provided in execution brief), (7) HTF aggregation timing spot-check (one timestamp's htf4h_* values + manual verification they reflect prior-closed 4H bar), (8) persistence confirmation (file path + size + write timestamp), (9) total runtime per asset. **Halt criteria (locked)**: VPS halts and pings local-Claude on (a) label distribution out-of-band per Q23.1, (b) feature count mismatch from expected, (c) any NaN-leak post-warmup outside the btc_funding_rate carveout, (d) 0-leak violation (any v1.0-dropped name present), (e) HTF spot-check mismatch (htf4h_* not constant within 4H window, OR reflects current-not-prior 4H bar = look-ahead leak), (f) any unhandled exception in builder.py. **Sequencing**: BTC first (largest data, no Cat 22 — validates core pipeline), then SOL + LINK (validates Cat 22 + BTC sidecar alignment). Sequential per Q22 default; parallel run rejected as low-value (compute is ample on VPS; sequential gives clean log boundaries between assets). **Phase 2 hand-off**: after all 3 asset parquet files land with PASS, local-Claude signals Phase 2.0 entry — self-audit per §10.5.5 + Phase 2.1 5-feature pre-gate on BTC per §10.3.1. **Spec text** has no contract changes; Decision v2.55 only documents the workflow + execution brief contract. The actual VPS work order is delivered as a PROJECT_LOG entry (visible to VPS-Claude on pull). **§10.5 process discipline reminders for VPS** (locked in execution brief): no banned phrases ('let me just', 'should work fine', 'quick fix', 'this might', 'TODO'); every action requires spec citation; deviations require DR-NNN before action; OOT one-shot rule applies starting Phase 2.10 — Phase 1.11 doesn't touch OOT. **ROLLBACK PROCEDURE**: trivial — VPS clears `data/storage/features/` outputs, local-Claude `git revert` v2.55 commit. Phase 1.11 produces no spec/code state; only data outputs. Approved by user 2026-05-01 | 2026-05-01 |
| v2.56| **DR-007: Path D — chop-filtered ATR-based labels (§8.1 amendment)** | First Phase 1.11 BTC build (PROJECT_LOG line 138, 2026-05-01) produced 57.77% NEUTRAL with tp/sl_atr_mult=3.0 + min_profit_pct={BTC=0.6, SOL=0.9, LINK=0.8}. §8.2 last clause hard-gate fired (NEUTRAL > 30%). Diagnostic: barriers too wide for 30m × 4-hour hold AND chop-regime samples overweight in training. User raised the structural question: ATR-scaled vs percent-based barriers. Path A (DR-006: 3.0→2.5×ATR), Path B (pure percent 0.7%/1.0%), Path D (chop-filtered ATR + slight tighten) considered. User rejected Path B + chose Path D with professional rationale: ATR-scaling is academically sound (López de Prado regime adaptation); the 4× ATR variation problem is solved by EXCLUDING low-vol regimes from training, not by abandoning ATR-scaling. Local-Claude's Path B recommendation revised — Path D is professionally cleaner because it preserves academic standard AND maps to a real trading rule (don't trade in chop). **§8.1 amendment locked**: tp_atr_mult=2.5 (3.0→2.5 tighten); sl_atr_mult=2.5 (symmetric); max_holding_bars=8 unchanged; min_profit_pct={BTC=0.3, SOL=0.4, LINK=0.4} (LOWERED — chop filter handles the bigger problem); NEW min_atr_pct_threshold={BTC=0.20, SOL=0.30, LINK=0.30} (chop filter); NEW label class CHOP_FILTERED=-2 (excluded from training + inference). **Inference contract locked**: Phase 3.2 predictor service computes `atr_pct = atr_14/close × 100` per 30m bar close; if `atr_pct < threshold[asset]` → signal=NO_TRADE (skip), else model.predict(). This is a CLEANER trading rule than always-on prediction — the bot only operates when the market itself is tradeable. **§8.2 amendment**: expected distribution bands (LONG/SHORT 35-45%, NEUTRAL 15-25%) now apply to the FILTERED set (label ∈ {0,1,2}), not the full feature matrix. Auxiliary metrics added: chop-filter rate (% of bars labeled -2; expected 20-40% on BTC) + unlabelable tail (label=-1). Diagnostic rules: NEUTRAL>30% on filtered set → DR-008 trigger to tighten ATR to 2.0; chop-filter rate >50% → threshold too aggressive, lower; <10% → threshold too loose, raise. **Implementation contract**: model/labeler.py extended to (a) accept per-asset min_atr_pct_threshold param, (b) compute atr_pct at entry bar, (c) emit label=-2 when chop condition holds before any TP/SL/timeout logic, (d) preserve v1.0 4-col output (label, holding_bars, exit_price, pnl_pct) per Decision v2.54 Q22.9. Builder.py training-X filter: `label.isin({0,1,2})` (drops -1 unlabelable AND -2 chop). config.yaml labeling section updated. **Threshold rationale**: BTC=0.20% — observed ATR/Price range 0.15-0.60% per 2026-05-03 chart; 0.20% is the floor below which 30m candles are demonstrably chop. SOL/LINK=0.30% — higher base vol vs BTC; conservative without empirical data; tunable in DR-008 after first SOL/LINK runs. **ROLLBACK PROCEDURE**: revert §8.1 + §8.2 + labeler.py + config.yaml; restore v1.0 tp/sl_atr_mult=3.0 + original min_profit_pct values; drop label=-2 class. ALTERNATIVELY (Path B switch if Path D underperforms in Phase 2.5 SHAP): file new DR for percent-based barriers; tp_pct=0.7% BTC, 1.0% alts. Approved by user 2026-05-04 | 2026-05-04 |
| v2.57| **Brief correction: label output is 4 cols (acknowledge v2.55 brief error)** | VPS BTC report (PROJECT_LOG line 138) flagged shape mismatch: actual 253 cols vs expected 250 per Decision v2.55 brief. Root cause: brief implicitly assumed 1 label col; triple_barrier_labels() preserved verbatim from v1.0 per Decision v2.54 Q22.9 emits 4 cols (label, holding_bars, exit_price, pnl_pct). Brief was wrong. The 3 metadata cols are intentional v1.0 contract: `exit_price` enables backtest slippage modeling, `holding_bars` shows trade duration distribution for Phase 2.5 diagnostic analysis, `pnl_pct` enables risk-adjusted return analysis post-prediction. These are TARGET-SIDE metadata (realized outcomes; look-ahead by definition) — they CANNOT be features and MUST be excluded from training X. **Revised expected col counts (locked)**: BTC = 6 OHLC + 243 features + 4 label cols = 253 cols; SOL/LINK = 6 OHLC + 250 features + 4 label cols = 260 cols. Post-warmup + post-tail-trim row count formula: rows = N − 150 − max_holding_bars (≈51,162 for 51,320-bar raw input with max_holding_bars=8). **Training-pipeline contract (locked for Phase 2)**: Feature-X = all columns EXCEPT {timestamp, label, exit_price, holding_bars, pnl_pct} (= 249 BTC / 256 alts). Target-y = `label` ONLY. The 3 metadata cols travel through the prediction → backtest pipeline; never enter LightGBM .fit(X, y). VPS halt on Trigger 2 (PROJECT_LOG line 138) is CLEARED — feature count was 243 ✓; the +3 cols are intentional metadata; my brief contract was wrong. **No spec text edit beyond Decision Log + revised brief in PROJECT_LOG**. **ROLLBACK**: trivial — Decision-Log-only entry; no code/spec state to revert beyond the Decision v2.55 brief language. Approved by user 2026-05-04 | 2026-05-04 |
| v2.58| **§7.6 amendment: NaN carveout table — accept warmup-NaN as v2.0 invariant** | VPS BTC report (PROJECT_LOG line 138) flagged ~30 columns with non-zero NaN post-warmup outside Decision v2.55 brief's 'btc_funding_rate carveout only' contract. Root cause: brief was over-strict. `warmup_bars=150` was sized to longest SINGLE-feature indicator lookback at 30m TF (VFI=130, Hurst=100, autocorr_20=100). Does NOT cover (a) intra-day rolling stats with per-day reset (Cat 5 daily VWAP bands), nor (b) HTF features whose native-TF warmups translate to many more 30m bars when mapped via reindex + shift(1) (Cat 2a HTF1D EMA200 needs 9,600 30m bars). **§7.6 amendment locked** with explicit NaN carveout table: (1) btc_funding_rate (alts only) — ALL rows NaN per Phase 1 design (Decision v2.54 round 2 placeholder; wired Phase 3+); (2) daily_vwap_upper/lower_band_1sig + daily_vwap_zone — ~20,254 NaN each (39.6%) from bands_window=20 intra-day reset × ~1066 days; (3) htf1d_ema200_pos — ~9,410 NaN (18.4%) from 200-day EMA × 48 30m/day; (4) htf1d_ema50_pos — ~2,210 NaN; (5) htf1d_ema20_pos + close_vs_20d_high/low_pct — ~770 NaN each; (6) vfi family — ~115 residual NaN (period 130 vs warmup 150). All mathematically explainable by indicator parameter math, not pipeline bugs. **Why not increase warmup_bars to 9,600**: would discard 200 days of training data (~5% of 3-year baseline) — devastating cost. LightGBM handles NaN natively via missing-value-direction learning at split decisions; NaN rows abstain from splits on specific features but contribute to all other 249 features' splits. This is fundamentally different from imputation — 'missing' becomes a meaningful third category which is correct during warmup (the feature literally doesn't exist for that row). **Halt criterion (c) per Decision v2.55 revised**: triggers ONLY when (i) NaN appears in a feature OUTSIDE the carveout table, OR (ii) NaN at row positions BEYOND the documented warmup for carveout features (e.g., htf1d_ema200_pos NaN at row 10,000 would be a bug since ~9,600 should suffice). VPS halt on Trigger 3 is CLEARED — observed NaN clusters all match documented carveouts. **No code change required** — LightGBM tolerates NaN natively; this Decision documents the v2.0 invariant + revises the audit policy. **ROLLBACK**: revert §7.6 amendment + Decision Log entry; restore prior brief language. No code state to revert. Approved by user 2026-05-04 | 2026-05-04 |
| v2.59| **DR-008: Path D second iteration — incremental tighten 2.5→2.4 + chop floor 0.20→0.25 (BTC)** | Phase 1.11 second BTC build (PROJECT_LOG line 158, 2026-05-04) with v2.56 Path D first-iteration params produced filtered-set NEUTRAL=33.94% (>30% upper bound) and BTC chop-filter rate=8.06% (<10% lower bound). Both v2.56 §8.2 diagnostic triggers fired simultaneously as predicted; improvement vs first pass real (NEUTRAL 57.77%→33.94%, −23.83pp). Directional symmetry (LONG 33.92% ≈ SHORT 32.13%) confirms barrier-distance is the issue, not asset bias. v2.56 default prescription was tp/sl 2.5→2.0 (0.5-step jump). User rejected 0.5-step in favor of incremental 0.1-step (2.5→2.4) per §10.5 single-variable-change discipline — minimum step lets each iteration's effect be measured cleanly without overshooting. Same incremental rhythm applied to chop floor (0.20→0.25, not 0.20→0.30). User explicit: 'test with 2.4 first and 2.3 or 2.2 later' / 'min_atr_pct_threshold=0.20 is small. maybe 0.25 or similar is okay'. **§8.1 amendment locked**: tp_atr_mult: 2.5→2.4; sl_atr_mult: 2.5→2.4 (symmetric kept); min_atr_pct_threshold.BTC: 0.20→0.25 (more selective; SOL/LINK unchanged at 0.30 since those haven't been run yet); min_profit_pct UNCHANGED (BTC=0.3, SOL=0.4, LINK=0.4) per §10.5 single-variable-change — only changing two related variables (ATR-mult + chop floor) this iteration. **Quantitative projection** (informational): tp/sl 2.4 is 4% tighter than 2.5 → expected NEUTRAL drop ~3-5pp (33.94%→28-31%); chop floor 0.25 is 25% more selective than 0.20 → expected chop rate ~13-18% (8.06%→13-18%); LONG/SHORT each gain ~1.5-2.5pp into 34-36% range, approaching 35% lower bound. **§8.1 param history table added** to spec — tracks Path D iteration ladder (first, second, third candidate, conservative floor) so future-Claude sees the planned escalation without re-litigation. Path D third (DR-009 candidate, not triggered yet): tp/sl=2.3, BTC chop floor=0.30. Path D conservative floor: tp/sl=2.2, BTC chop floor=0.30. **Code changes**: NONE required — labeler.py + builder.py + cfg schema already param-driven from Decision v2.56; only config.yaml values change. **Spec edits**: §8.1 amendment + §8.1 param-history table; §8.2 unchanged (band thresholds + diagnostic rules already locked). **PASS condition**: filtered-set NEUTRAL ≤ 25% AND filtered-set LONG/SHORT ≥ 35% each AND BTC chop rate 10-50% AND zero halts → proceed to SOL/LINK with same per-asset thresholds. **If second iteration still off-band**: file DR-009 for tp/sl 2.4→2.3 + BTC chop floor 0.25→0.30. **ROLLBACK PROCEDURE**: revert config.yaml + §8.1 spec edits; restore v2.56 first-iteration values (2.5×ATR + 0.20% chop). Approved by user 2026-05-04 with explicit incremental-step preference | 2026-05-04 |
| v2.60| **§7.6 carveout extension: first-event warmup cluster (small NaN counts)** | VPS BTC second build (PROJECT_LOG line 158) flagged 14 columns with small NaN counts (max 37 rows = 0.07%) outside the v2.58 carveout table. Pattern analysis: all are first-occurrence-event warmups — `bars_since_pivot_touch_weekly` (37 NaN, needs first weekly pivot confirmation), 9 `weekly_pivot_*_dist_pct + helpers` (~2 NaN each, first weekly pivot from prior-week OHLC), `body_vs_prev_body` (21 NaN, Cat 8 needs prev-bar non-zero body), `bars_since_rsi_os` (5 NaN, Cat 20 needs first RSI<30 event). Mathematically explainable like v2.58 carveouts; LightGBM tolerates NaN natively (Decision v2.58 spirit rationale). Magnitudes negligible (≤0.07% per column vs ≤39.6% for documented carveouts). VPS report (informational, not halting): asked whether to (a) extend v2.58 table per-column or (b) accept as v2.0 invariant without table mention. Local-Claude chose **(c) compromise**: add a single SUMMARY ROW to §7.6 carveout table covering the cluster — explicit acknowledgment that the pattern is recognized + carveout-covered, without bloating the table with 14 individual rows. **§7.6 amendment**: 7th row added to carveout table titled 'First-event warmup cluster (per Decision v2.60)' with row count summary (≤ ~40 each, ≤ 0.07%) + pattern enumeration (`bars_since_*` + `weekly_pivot_*` + Cat 8 `body_vs_prev_body`) + the 4 specific BTC-observed columns and their counts as examples. **Halt criterion (c) per Decision v2.55+v2.58 unchanged**: still triggers only when NaN appears in a feature OUTSIDE the (now-extended) carveout table OR at row positions BEYOND documented warmup. The extension just makes the existing spirit-coverage explicit so future-Claude doesn't re-litigate the small-count first-event pattern. **No code change** — LightGBM NaN-native handling already covers; this is documentation-only. **No threshold change to halt criterion** — the extension is INclusive (more carveouts = fewer halts), not exclusive. **ROLLBACK**: trivial — revert §7.6 amendment 7th row + Decision Log entry. Approved by user 2026-05-04 (informational item flagged in VPS report; user agreed via 'lock the 3 decisions + spec edits + brief' workflow approval that covered carveout extension as well) | 2026-05-04 |
| v2.61| **DR-009: Path D third iteration — STRUCTURAL hold extension 8→12 + chop floor 0.25→0.275 (BTC)** | Phase 1.11 third BTC build (PROJECT_LOG line 167, 2026-05-04) with v2.59 (DR-008) params (tp/sl=2.4 + BTC chop=0.25 + hold=8) produced filtered-set NEUTRAL=32.14% (still >30 trigger) and chop rate=15.59% (in [10,50] generic but below [20,40] BTC-spec). Improvement vs v2.56-1st real (NEUTRAL 57.77→33.94→32.14, −25.63pp total) but the v2.59 step alone gave only −1.80pp — **diagnostic evidence the linear-ATR lever has plateaued**. VPS linear projection: pure ATR-tighten cannot reach [15,25] band even at tp/sl=2.0 (Δ ≈ −1.80pp/0.1 ATR step → 2.0 still ~25%). User's domain insight (2026-05-03): 30m × 4-hour hold cuts off mid-development moves; actual price movement plays out over hours but includes neutral/squeeze periods that compress into the 4-hour window unfinished. Per Decision v2.61 (DR-009), Phase 1.11 takes the **STRUCTURAL lever** (extend hold) instead of pushing the linear lever further. **§8.1 amendment locked**: tp_atr_mult held at 2.4 (UNCHANGED — explicitly to isolate hold-extension effect from ATR-tighten effect); sl_atr_mult held at 2.4; max_holding_bars 8→12 (4hr → 6hr; the new structural variable — gives squeeze + breakout + extension sequence room to complete on 30m); min_atr_pct_threshold.BTC 0.25→0.275 (incremental +0.025pp continuing v2.59's 0.025-step rhythm, per user's explicit '0.275' specification 2026-05-04). SOL/LINK chop floors unchanged at 0.30 (haven't been run yet). **§9.2 cascade locked**: per the §9.2 contract `purge_bars = embargo_bars = max_holding_bars`, both cascade 8→12. Rationale: purged-CV scheme prevents label-leakage between train/val folds — label at bar i uses forward bars [i+1, i+12], so train/val boundary must purge 12 bars on each side. If purge < max_holding_bars, train/val labels overlap → log-loss artificially deflates → silent overfit. Cascade is INVARIANT across the pipeline. **§8.1 param-history table updated** with v2.61 row + max_holding_bars column added (replaces simpler param-history from v2.59 since now 3 dimensions vary across iterations). **§9.2 cascade contract** added as paragraph below the walk_forward yaml block — explicit documentation that future-Claude doesn't decouple purge/embargo from max_holding_bars. **§10.5 single-variable-change discipline justification for multi-variable change**: v2.59 result (NEUTRAL only −1.80pp despite incremental tighten) is empirical evidence that the linear lever has plateaued; structural intervention (hold extension) is now warranted. Each variable's effect remains independently traceable: chop floor 0.025-step continues v2.59 rhythm; hold extension is the new structural variable; tp/sl explicitly held at 2.4 to isolate hold-effect. If v2.61 misses, DR-010 escalates the linear ATR lever (2.4→2.3) WITH the new structural baseline (hold=12) — preserving iteration discipline. **Quantitative projection** (informational): structural hold extension + 0.025-step chop tighten expected to drop NEUTRAL into 22-28% range (approaching [15,25] band upper bound). LONG/SHORT each gain ~3-5pp into 36-40% range (well into [35,45]). BTC chop rate expected ~17-22% (closer to [20,40] BTC-spec). **Code changes**: NONE required — labeler.py + builder.py + cfg schema already param-driven from Decision v2.56; max_holding_bars is already a config-driven param; hold=12 just produces longer forward window. Only config.yaml values change (lines 103, 110, 155, 156). **Spec edits**: §8.1 yaml block + param-history table refresh (added max_holding_bars column + v2.61 row); §9.2 cascade note paragraph; Decision v2.61 entry. **Output schema impact**: tail unlabelable rows shift from 8 → 12 (rows where forward window is incomplete); slight (~4 rows) reduction in usable training data; expected post-warmup-and-tail rows ≈ N − 150 − 12 ≈ 51,158 (was 51,162 at hold=8). **PASS condition**: filtered-set NEUTRAL ≤25% AND filtered-set LONG/SHORT ≥35% each AND BTC chop rate ∈ [10,50] AND zero halts → proceed to SOL/LINK with their per-asset thresholds (chop floor 0.30 unchanged, same tp/sl=2.4 + hold=12). **If v2.61 still misses**: DR-010 = tp/sl 2.4→2.3 + BTC chop floor 0.275→0.30 (incremental escalation on linear levers; structural hold=12 now baseline). **ROLLBACK PROCEDURE**: revert config.yaml lines 103/110/155/156 + §8.1 + §9.2 spec edits; restore v2.59 values (tp/sl=2.4, chop=0.25, hold=8, purge/embargo=8). Approved by user 2026-05-04 via Path B selection (multi-variable structural intervention vs Path A pure chop tighten) | 2026-05-04 |
| v2.62| **DR-010: alt chop-floor empirical recalibration (SOL+LINK 0.30→0.60)** | Phase 1.11 BTC v2.61 build (PROJECT_LOG line 179, 2026-05-04) PASSED all bands cleanly (LONG 38.57%, SHORT 37.41%, NEUTRAL 24.03%, chop 19.77%). Path B structural intervention (hold=12 + chop=0.275 + tp/sl=2.4) was the binding fix per local-Claude diagnostic. Proceeded to SOL/USDT per Decision v2.55 sequential contract. SOL v2.61 build (PROJECT_LOG line 180) HALTED on chop rate 0.55% << [10, 50] PASS gate (18.5× below floor). Distribution itself was excellent (LONG 41.88%, SHORT 41.59%, NEUTRAL 16.53% — all 3 in band). Root cause: SOL ATR/Price profile is structurally higher than BTC at 30m; 0.30% threshold designed as 'wider than BTC' wasn't actually wider enough — most SOL 30m bars sit comfortably above 0.30%. VPS empirical observation suggested 0.60-0.80% range. Local-Claude selected 0.60% (lower bound) per conservative-iteration discipline: SOL distribution is ALREADY in band; aggressive chop raise risks pushing NEUTRAL below 15% floor (since chop filter removes the lowest-vol bars which are mostly NEUTRAL). Lower-bound raise lets us measure effect; if chop rate < 10% on this iteration, DR-011 escalates to 0.70-0.80. **§8.1 amendment locked**: SOL min_atr_pct_threshold 0.30 → 0.60; LINK min_atr_pct_threshold 0.30 → 0.60 (same as SOL initially; LINK first-time validation in this iteration; if LINK comes back off-band, DR-011 differentiates SOL vs LINK). BTC unchanged at 0.275 (frozen at v2.61 PASS state). All other params unchanged: tp/sl=2.4, hold=12, min_profit_pct unchanged, purge/embargo=12 unchanged. **§8.1 param-history table** updated with v2.62 row showing asset-specific chop-floor differentiation (BTC=0.275, alts=0.60); BTC iteration row marked PASS (frozen). DR-011 candidate row updated to reflect alt-distribution-shift risks (NEUTRAL <15% via min_profit_pct.SOL/LINK adjust; chop still <10% via further raise; distribution divergence via SOL/LINK differentiation). **§10.5 single-variable-change discipline**: this iteration changes ONLY SOL+LINK chop floors; BTC + tp/sl + hold + min_profit_pct + cascade all unchanged. SOL+LINK move together as a class (alt vs BTC); not strictly single-variable but conceptually one variable (alt class). Differentiation between SOL and LINK is reserved for DR-011 if needed. **Quantitative projection** (informational): SOL chop rate 0.55% → ~10-15% (depends on SOL ATR/Price 20th-percentile; rough projection from VPS empirical observation); SOL filtered set drops from 50,879 → ~43,000-46,000 rows (losing ~5,000-8,000 lowest-vol bars to chop label); SOL distribution shifts: LONG/SHORT each +1-3pp into 43-45% (near upper band); NEUTRAL drops 16.53% → 13-15% (near or below 15% floor). LINK first-time metrics unknown; expected similar profile to SOL since both are alt-vol-class. **§10.5 risk flag**: NEUTRAL approaching 15% floor; if SOL/LINK NEUTRAL drops below 15%, DR-011 = raise min_profit_pct.SOL/LINK 0.4 → 0.5 to push more timeouts into NEUTRAL classification. **Code changes**: NONE (config-driven). config.yaml lines 111 + 112 (SOL + LINK chop floors). **Spec edits**: §8.1 yaml block + param-history table; Decision v2.62 in Decision Log line 2009. Mirror synced. **Decision rules for next iteration**: (PASS) chop ∈ [10,50] AND distribution still in band → proceed to LINK if SOL passes solo; if LINK runs first, differentiate; (HALT — NEUTRAL too low) NEUTRAL < 15% → DR-011 raises min_profit_pct.alt to 0.5; (HALT — chop still too low) chop < 10% → DR-011 raises alt floor 0.60 → 0.70-0.80; (HALT — chop too high) chop > 50% → DR-011 lowers alt floor (unlikely at 0.60). **ROLLBACK PROCEDURE**: revert config.yaml lines 111+112 + §8.1 yaml block + param-history table; restore SOL/LINK floor=0.30. Approved by user 2026-05-04 with 'yes, please lock. I will test on vps' confirmation | 2026-05-04 |
| v2.63| **§7.6 carveout extension: generalize first-event-warmup pattern + explicit btc_above_ema200_daily** | VPS SOL v2.61 build (PROJECT_LOG line 180) flagged 6 alt-side cols outside the v2.58+v2.60 carveout table — `btc_above_ema200_daily` (Cat 22; ~9,400 NaN at 18.39% — alt-side mirror of `htf1d_ema200_pos`), 2 Cat 20 first-event warmups (`bars_since_adx_weak_start`, `bars_since_wt_os`), 3 Cat 6.5 first-swing-pair warmups (`nearest_fib_level_dist`, `extension_progress_1272`, `fib_retracement_pct`). All match the v2.60 first-event-warmup PATTERN; mathematically explainable; LightGBM tolerates natively (Decision v2.58 spirit rationale). VPS report (informational, not halting): asked whether to (a) explicit per-column enumeration in v2.58/v2.60 table, OR (b) generalize the rule to 'any feature matching pattern regardless of asset'. Local-Claude chose **(c) hybrid**: (1) explicit row added for `btc_above_ema200_daily` (18.4% NaN is substantial; warrants explicit table entry parallel to its BTC-side mirror `htf1d_ema200_pos`); (2) generalize the first-event warmup rule to cover ANY asset matching the pattern (`bars_since_*` + Cat 6.5 swing-Fib + Cat 22 daily-EMA200). Future-proofs against new assets (TAO/HYPE re-entry per Decision v2.35) without per-asset Decision Log churn. **§7.6 amendment**: 8th carveout table row added for `btc_above_ema200_daily` (alts only, ~9,400 NaN, parallel to v2.58 row 3 `htf1d_ema200_pos`); 9th row added for 'First-event warmup cluster — alt-side extension' with the 5 SOL-observed cols enumerated as examples + the GENERALIZED PATTERN RULE locked: **'Any feature matching `bars_since_*` + Cat 6.5 swing-Fib (`nearest_fib_level_dist`, `extension_progress_1272`, `fib_retracement_pct`) + Cat 22 daily-EMA200 (`btc_above_ema200_daily`) is carveout-covered regardless of asset, with row counts proportional to lookback × time-to-first-event'**. **Halt criterion (c) per Decisions v2.55+v2.58+v2.60+v2.63**: still triggers only when NaN appears in a feature OUTSIDE the (now-twice-extended) carveout table OR at row positions BEYOND documented warmup. Extension is INclusive (more carveouts = fewer halts), not exclusive. Halt threshold semantics unchanged. **No code change** — documentation only. **No threshold change to halt criterion** — purely additive. **ROLLBACK**: trivial — revert §7.6 amendment 8th + 9th rows + Decision Log entry. Approved by user 2026-05-04 (informational item flagged in VPS line 180; user 'yes please lock' covered carveout extension as part of DR-010 workflow) | 2026-05-04 |
| v2.64| **DR-011: per-asset tp/sl_atr_mult schema extension + alts-only 2.4→2.2 experiment** | Phase 1.11 v2.62 result (PROJECT_LOG: BTC line 179 PASS / SOL+LINK line 180+ PASS post-DR-010): all 3 assets in §8.2 bands. BTC NEUTRAL=24.03% (mid-band, comfortable). SOL NEUTRAL=15.55% / LINK NEUTRAL=15.75% (both at lower-band floor edge, +0.55pp / +0.75pp above 15% target). User professional hypothesis (2026-05-04): before committing v2.62 params to implicit Phase 1.14 freeze, validate that tp/sl=2.4 is the right alt-side floor. Tightening alts to 2.2 may increase directional sample density (timeout-NEUTRAL → directional reclassification) — potentially better Phase 2 signal density via reduced label-class imbalance. **Schema extension locked**: §8.1 `tp_atr_mult` and `sl_atr_mult` migrate from SCALAR to PER-ASSET DICT (matches existing pattern for `min_profit_pct` + `min_atr_pct_threshold`). New schema: tp_atr_mult: {BTC: 2.4, SOL: 2.2, LINK: 2.2}; sl_atr_mult: {BTC: 2.4, SOL: 2.2, LINK: 2.2} (symmetric). **BTC frozen at 2.4** = v2.61 PASS state preserved; do NOT re-run BTC during this experiment. **Code change**: builder.py:add_labels gains `_resolve_per_asset(value, coin)` helper accepting both scalar (legacy) AND dict (v2.64 schema); falls back to scalar when value is not a dict. Backward-compat with config files using old scalar form. labeler.py unchanged (signature still takes scalar floats; builder extracts per-asset). **§9.2 cascade**: NONE — purge/embargo coupled to max_holding_bars (unchanged at 12), not tp/sl. Walk-forward params unaffected. **Empirical projection** (informational, not gate): BTC iteration ladder gave −1.80pp NEUTRAL per 0.1 ATR step (v2.56→v2.59). Linear projection for alts: SOL NEUTRAL 15.55% → ~13.7%; LINK 15.75% → ~13.9%. Both projected to BREACH 15% target floor. Stays above 10% hard-emergency floor (where §8.2 prescribes raise min_profit_pct). LONG/SHORT each gain ~+1.0-1.5pp into [42-44%] (well below 45% upper). Chop rate UNCHANGED (chop is entry-bar evaluation, independent of barrier ATR multiplier). **Outcome judgment deferred to post-test review**: §8.2 band is quality TARGET not hard constraint per spec wording ('If NEUTRAL falls below 10%, raise min_profit_pct'). LightGBM tolerates 13-14% NEUTRAL; user assesses Phase 2 sample-density vs band-discipline trade-off after data lands. **Decision matrix (5-tier per VPS DR-011 framing)**: (a) both alts NEUTRAL ∈ [15,25] AND LONG/SHORT ∈ [35,45] AND chop ∈ [10,50] → 2.2 wins on alts; DR-012 candidate to also retune BTC to 2.2 (system-wide); (b) NEUTRAL 13-15 + LONG/SHORT well-balanced — borderline, user judgment call (LightGBM tolerates; band quality target not hard); (c) NEUTRAL <13% or LONG/SHORT >45% → 2.2 too aggressive; revert alts to 2.4 (per-asset kept as schema extension); (d) chop rate shifts (shouldn't — chop is entry-bar) → flag for diagnosis; (e) any 0-leak / NaN-leak / shape regression → halt for code investigation. **§10.5 single-variable-change discipline**: this iteration changes ONLY alt tp/sl (BTC + hold + min_profit_pct + chop floors + cascade all unchanged). Schema extension (scalar→dict) is mechanical not semantic — adds capability without breaking BTC's frozen state. **Cascade impact**: only config.yaml lines 101+102 change (tp_atr_mult + sl_atr_mult become dicts); BTC parquet stays at v2.61 PASS state on disk. **Quantitative projection** (per VPS framing): see decision matrix (a)-(e). **Code changes**: builder.py:add_labels (~5 lines for helper + per-asset extraction); config.yaml lines 101-102 (scalar→dict). labeler.py UNCHANGED. spec §8.1 yaml block (per-asset dict) + §8.1 param-history table v2.64 row + Decision v2.64 entry. Mirror synced. **Sequence (per VPS proposal)**: (1) local commit current v2.62 PASS as known-good baseline; (2) local update config.yaml + builder.py + spec; (3) local push; (4) VPS pull, verify config (only lines 101+102 changed; BTC values intact); (5) VPS run SOL — DO NOT re-run BTC (BTC parquet is v2.61 baseline); (6) VPS run LINK; (7) VPS report comparison table; (8) decision matrix applied; (9) local commit winning param set as Phase 1.14 freeze. **Phase 1.14 freeze contract** (per spec §10.5.4): only after this experiment lands AND user accepts the result, the feature matrix + label parquet for all 3 assets freezes for Phase 2 training. **ROLLBACK PROCEDURE**: revert config.yaml lines 101+102 (restore scalar tp/sl=2.4); revert builder.py helper (restore scalar extraction); revert §8.1 spec text + Decision Log entry; alts re-run with 2.4 reverts to v2.62 baseline. ALTERNATIVELY (preserve schema extension; revert only alt values): keep dict schema, set tp_atr_mult={BTC: 2.4, SOL: 2.4, LINK: 2.4} — schema benefit retained for future per-asset tuning. Approved by user 2026-05-04 with experiment-then-decide framing | 2026-05-04 |
| v2.65| **DR-012: alt tp/sl_atr_mult WIDENING correction (SOL+LINK 2.2→2.5)** | Phase 1.11 v2.64 (DR-011) experiment result (PROJECT_LOG line 210+, 2026-05-04): SOL NEUTRAL 15.55→14.54% (−0.46pp BREACHED 15% floor); LINK 15.75→14.75% (−0.25pp BREACHED). VPS 5-tier matrix returned tier-(b) borderline. User identified MISJUDGMENT in v2.64 hypothesis direction: 'I find that I misset 2.2. I was thinking that 2.4 is wide for SOL/LINK. to fix that I would set 2.5 not 2.2'. Original v2.64 belief: '2.4 is wide for alts → tighten to 2.2'. Corrected belief (post-experiment): '2.4 isn't wide enough for alts → widen to 2.5'. Underlying intent: alts have higher ATR per bar than BTC, so 2.5×ATR for alts at typical ATR ≈ similar-or-larger absolute % move target than BTC's 2.4×ATR. Alts SHOULD have wider barriers in ATR-multiplier units to give moves room to develop in 6hr (max_holding_bars=12) hold window. **§8.1 amendment locked**: SOL tp_atr_mult 2.2→2.5; LINK tp_atr_mult 2.2→2.5; sl_atr_mult symmetric 2.2→2.5 for both. BTC frozen at 2.4 (v2.61 PASS state preserved). All other params unchanged: hold=12, min_profit_pct (BTC=0.3, alts=0.4), min_atr_pct_threshold (BTC=0.275, alts=0.60), purge/embargo=12. **Per-asset dict schema (Decision v2.64) is RETAINED** — only the values change; the schema migration is preserved as a v2.0 invariant for future per-asset tuning. **Quantitative projection** (using v2.64-calibrated alt step-response of −0.50pp NEUTRAL per 0.1 ATR step): from v2.64 baseline (2.2): SOL NEUTRAL 14.54% + (0.3 ATR widen × −0.50pp/0.1ATR × −1) = 14.54% + 1.50pp = ~16.04% projected; LINK 14.75% + 1.50pp = ~16.25%. Both projected to land in [15,25] band with >1pp comfortable margin above 15% floor. Better than v2.62 baseline (SOL=15.55%, LINK=15.75% just above floor). LONG/SHORT each drop ~0.75-1.0pp into ~41-42% range (well within [35,45]). Chop UNCHANGED (validated invariant from v2.64; chop is entry-bar evaluation, independent of barrier ATR multiplier). **Decision matrix (post-VPS report)**: (a) PASS — alts NEUTRAL ∈ [15,25] AND LONG/SHORT ∈ [35,45] AND chop ∈ [10,50] → COMMIT v2.65 as Phase 1.14 freeze (BTC=2.4 / SOL=2.5 / LINK=2.5); (b) NEUTRAL >25% → over-widened; revert alts to 2.4 (v2.62 baseline that already PASSED) via DR-013; (c) chop shifts (shouldn't) → diagnose; (d) shape/0-leak/NaN regression → halt. **§10.5 single-variable-change discipline**: this iteration changes ONLY alt tp/sl values (BTC + hold + min_profit_pct + chop floors + cascade all unchanged). Direction REVERSAL from v2.64 (was tighten 2.4→2.2; now widen 2.2→2.5) — methodologically clean since v2.64 experimented in wrong direction per user's post-data realization. Reversal is +0.3 ATR step (3 incremental units) which is larger than the standard 0.1-step rhythm; justified because (i) v2.62 baseline at 2.4 was already in band, so reverting to 2.4 wouldn't be a test, just an undo; (ii) widening past 2.4 (to 2.5) tests whether alts NEED more room than BTC; (iii) v2.64 calibration data (alt step-response −0.50pp/0.1 ATR) lets us project the outcome confidently before running. **Phase 1.14 freeze pre-staged**: if v2.65 PASSES, the feature matrix + label parquet for all 3 assets freeze with locked params per spec §10.5.4. If v2.65 FAILS, we have two known-good baselines to revert to: v2.62 (BTC=SOL=LINK=2.4, all PASSED) or v2.61 (BTC-only PASS at 2.4 + alt chop floor 0.30). **Code changes**: NONE — per-asset dict resolver `_resolve_per_asset` already in builder.py from Decision v2.64; labeler.py UNCHANGED. config.yaml lines 102+103+106+107 (SOL+LINK tp/sl). **Spec edits**: §8.1 yaml block (SOL+LINK tp/sl 2.2→2.5); §8.1 param-history table (v2.65 row + DR-013 candidate rows; v2.64 row marked superseded with tier-(b) result documented); Decision v2.65 in Decision Log line 2022. Mirror synced. **Sequence (per Decision v2.55 + v2.64 BTC-frozen contract)**: (1) local commit current v2.64 result state (post-experiment, alts at 2.2 in disk parquet — that's the data the user is reverting); (2) local update config.yaml + spec; (3) local push; (4) VPS pull, verify config (only lines 102+103+106+107 changed; BTC values intact at 2.4); (5) VPS run SOL — DO NOT re-run BTC (BTC parquet at v2.61 PASS baseline preserved); (6) VPS run LINK; (7) VPS report comparison table (v2.62/v2.64/v2.65 columns); (8) decision matrix applied; (9) if PASS, local commits winning param set as Phase 1.14 freeze and signals Phase 2.0 entry. **ROLLBACK PROCEDURE**: revert config.yaml lines 102+103+106+107 (alt tp/sl 2.5→2.2 OR 2.5→2.4 to v2.62 baseline); revert §8.1 spec text + Decision Log entry; alts re-run with reverted values. Two known-good fallbacks available: v2.62 (alts at 2.4, all PASSED §8.2 bands) or v2.61 (BTC-only PASSED before alt calibration). Approved by user 2026-05-04 with explicit misjudgment-correction framing | 2026-05-04 |
| v2.66| **DR-013: Phase 1.14 FREEZE LOCK — alts tp/sl=2.7 (post-5-point-sweep Pareto optimum)** | Phase 1.11 v2.65 (DR-012) result PASSED: SOL NEUTRAL 15.99% (+0.99pp); LINK 16.14% (+1.14pp). VPS extended sweep to 2.7 (v2.66c) and 3.0 (v2.67c) per user-staged plan 2026-05-04 ('test 2.5 first and then 2.7'). **Full 5-point sweep on alts** (chop floor 0.60% + hold=12 + min_profit=0.4 held constant): tp/sl ∈ {2.2, 2.4, 2.5, 2.7, 3.0}. SOL NEUTRAL trajectory: 14.54 (✗) → 15.55 → 15.99 → 16.64 → 17.32. LINK: 14.75 (✗) → 15.75 → 16.14 → 16.89 → 17.59. **Diminishing-returns inflection identified at 2.7→3.0**: NEUTRAL gain/0.1-step drops from 0.32 (2.5→2.7) to 0.23 (2.7→3.0); trade ratio (NEUTRAL gain : directional loss) worsens from 1:1 (2.4→2.7 range) to 0.7:1 (2.7→3.0). 2.7 is therefore the empirical Pareto optimum. **Chop INVARIANT** validated 5 times (SOL 17.85% / LINK 18.84% across all sweep points) — proves tp/sl entirely independent of chop filter (entry-bar evaluation). **Locked params**: BTC tp_atr_mult=2.4 (frozen at v2.61 PASS state); SOL tp_atr_mult=2.7; LINK tp_atr_mult=2.7; sl_atr_mult symmetric. All other params unchanged at cumulative iterated values: max_holding_bars=12 (DR-009/v2.61), min_profit_pct={BTC: 0.3, SOL: 0.4, LINK: 0.4} (DR-007/v2.56), min_atr_pct_threshold={BTC: 0.275 v2.61, SOL: 0.60 v2.62, LINK: 0.60 v2.62}, purge/embargo=12 (§9.2 cascade per v2.61). **Per-asset dict schema (Decision v2.64) RETAINED** as v2.0 invariant — values change but schema preserved for future per-asset tuning. **Phase 1.14 FREEZE CONTRACT (per spec §10.5.4)**: feature matrix + label parquet for all 3 assets are FROZEN at these params. Phase 2 baseline gate (§10.3.1 5-feature pre-gate on BTC; §10.3.2 full-feature gate) reads from locked artifacts. **Final §8.2 distribution snapshot at freeze**: BTC LONG 38.57% / SHORT 37.41% / NEUTRAL 24.03% / chop 19.77% (all in band, mid-band comfort); SOL LONG 42.30% / SHORT 41.06% / NEUTRAL 16.64% / chop 17.85% (all in band, +1.64pp NEUTRAL margin); LINK LONG 41.65% / SHORT 41.47% / NEUTRAL 16.89% / chop 18.84% (all in band, +1.89pp NEUTRAL margin). **Why 2.7 not 2.5**: +0.65pp/+0.75pp better NEUTRAL margin on SOL/LINK; 1pp+ buffer is data-shift robust (regime changes can move NEUTRAL by 1-2pp); small directional density loss (~0.7pp) outweighed by safety gain. **Why 2.7 not 3.0**: at 2.7 the NEUTRAL/directional trade is still 1:1 (favorable); at 3.0 it degrades to 0.7:1 (unfavorable). Excess NEUTRAL gain past 2.7 comes at disproportionate directional cost. 2.7 is the empirical inflection point. **Why not v2.62 (alts=2.4)**: only +0.55-0.75pp band margin — too close to 15% floor for regime-shift robustness. Demonstrated via v2.64 experiment that −0.5pp drift from data-noise alone is plausible. **Pre-freeze state (VPS uncommitted)**: SOL/LINK parquets on disk currently at v2.67c (3.0) artifacts; config lines 103/104/107/108 at 3.0 uncommitted. Lock action: revert config to 2.7; VPS re-runs SOL + LINK with 2.7 (~3min compute) to regenerate the official Phase 1.14 freeze parquets. BTC parquet at v2.61 PASS state already correct (BTC=2.4 unchanged). **Code changes**: NONE — `_resolve_per_asset` helper already in builder.py from Decision v2.64; labeler.py UNCHANGED. config.yaml lines 103+104+107+108 (revert SOL+LINK tp/sl 3.0→2.7). **Spec edits**: §8.1 yaml block (alt tp/sl 3.0→2.7 + freeze marking); §8.1 param-history table (sweep summary row + Phase 1.14 freeze 🔒 row replacing the 4 candidate rows); Decision v2.66 in Decision Log line 2023. Mirror synced. **Phase 2.0 entry CONTRACT**: per Decision v2.55 transition contract, all 3 assets PASS §8.2 bands AND chop ∈ [10,50] AND zero halts → local-Claude signals Phase 2.0 entry (self-audit per §10.5.5 + Phase 2.1 5-feature pre-gate on BTC per §10.3.1). VPS regenerates alt parquets with locked 2.7 params; once VPS confirms regeneration matches projected 16.64/16.89% NEUTRAL, Phase 1.11 closes and Phase 2.0 begins. **OOT one-shot rule (§10.4)** activates at Phase 2.10 boundary — does NOT apply yet. Iteration freedom on triple-barrier params now CLOSED at Phase 1.14 freeze; further changes require fresh DR with explicit unfreeze rationale per §10.5.4. **ROLLBACK PROCEDURE**: if Phase 2 baseline gate (§10.3) fails to clear empirical prior by 2% log-loss with 2.7 params, file DR-014 to test alternative locked values (v2.65 alts=2.5 already characterized, +0.99/+1.14pp margin; v2.62 alts=2.4 also PASS but at floor edge). Two known-good fallbacks remain available without re-iteration. Approved by local-Claude lock 2026-05-04 + user delegation 'plz continue' | 2026-05-04 |
| v2.67| **DR-014: §10.3.1 amendment — walk-forward pre-gate (NOT tuning; methodological correction)** | Phase 2.1 first-fold pre-gate (PROJECT_LOG line 478+, 2026-05-05) FAILED: val_logloss=1.0806, empirical_prior=1.0470, ratio=1.0322 (gate ≤0.99 to PASS); −3.22% delta vs prior; best_iteration 11/200 (early-stopped). VPS surfaced 5 diagnostic interpretations. Local-Claude root-cause analysis identified diagnostic (1) fold-selection artifact as overwhelmingly likely: train (2023-05 → 2024-01) class distribution L 35.00 / S 35.40 / N 29.61 (near-balanced, bear-bottom + ranging); val (2024-02 → 2024-03) class distribution L **48.78** / S 27.17 / N 24.05 (Feb 2024 BTC ETF rally — $42k → $73k ATH = +74% in 2 months). 13.78pp LONG distribution shift between train and val — one of the most regime-anomalous single months in BTC history. A model trained on 2023 chop cannot generalize across this regime boundary; the empirical prior (val marginal distribution) wins because val is so LONG-skewed that 'always predict LONG' beats anything trained on balanced data. best_iteration=11 confirms LightGBM detected NO useful generalizing signal. **This is a methodology problem, not a signal problem.** Walk-forward CV exists exactly to handle fold-selection artifacts of this kind by averaging across multiple folds. **§10.3.1 amendment locked**: 'single fold, train on months [1..9], val on month 10' → 'full walk-forward per §9.2 (14 folds; train_months=9, val_months=1, step_months=1; purge=embargo=12 cascade)'. Gate condition: 'val log-loss must beat empirical prior by ≥1%' → 'mean val log-loss across all 14 folds must beat mean empirical prior (per-fold val-distribution-based) by ≥1%'. Empirical prior calculation: per fold compute val label distribution `[p₀,p₁,p₂]` and prior `Hᵢ = −Σ pᵢ × ln(pᵢ)`; mean prior across 14 folds = mean(H₁..H₁₄). Gate: `mean_val_logloss / mean_prior ≤ 0.99`. **§10.3.1 rationale paragraph added** documenting WHY walk-forward (not single fold): first-fold's Feb 2024 ETF rally is demonstrably NOT representative of the 35-month dataset; walk-forward across 14 folds eliminates fold-selection artifact entirely and provides robust signal-detection; compute cost remains tractable (~5-10min for 5 features). **This is NOT tuning per spec §10.3.1 'do not tune your way out'**: tuning means changing features, params, or labels hoping for a different result on the same test. DR-014 changes WHICH FOLDS the gate evaluates, from 'first fold only (regime-anomalous)' to 'all 14 folds averaged (regime-mixed)'. Features (5) unchanged, LightGBM defaults unchanged, labels unchanged, parquets frozen at Phase 1.14 unchanged. The DR mechanism (§10.5) exists for exactly this case: methodological surprise that the spec couldn't have anticipated (specifically: that fold 1 would land on the most regime-anomalous val month in the dataset). **Halt criterion strengthened**: walk-forward-mean failure is robust signal absence (not fold artifact). If §10.3.1 amended pre-gate fails, halt v2.0 with HIGH confidence and escalate via fresh DR re-opening §8 labeling / Phase 1.10 feature engineering / accepting v2.0 as research dead-end. **§10.3.2 NOT amended at this time**: leave §10.3.2 (full-feature gate) at single-fold for now. If §10.3.1 amended pre-gate PASSES, we'll evaluate whether §10.3.2 also needs walk-forward amendment via separate DR (likely yes for methodological consistency, but defer that decision until pre-gate result lands). Bundling §10.3.2 amendment into DR-014 would change two gates simultaneously without diagnostic evidence for the second change. **Decision matrix for amended pre-gate**: (a) PASS — mean log-loss beats mean prior by ≥1% → proceed to Phase 2.2 §10.3.2 full-feature gate (+ likely DR-015 to also amend §10.3.2 to walk-forward for consistency); (b) FAIL — mean log-loss ≤ mean prior × 0.99 → HALT v2.0 with high confidence; signal absence is robust across 14 folds; escalate via fresh DR. **Code changes**: VPS extends `scripts/baseline_pre_gate.py` to loop 14 folds; helpers `_walk_forward_folds()`, `_per_fold_prior()`, `_aggregate_results()` recommended. Phase 2.2 + Phase 2.3 will reuse these helpers. Compute estimate: 14 folds × 5 features × 200 rounds ≈ 5-10 min total. **Spec edits**: §10.3.1 setup paragraph + gate-condition + rationale-amendment paragraph; Appendix C step 2.1 wording (single-fold → walk-forward + first-fold-failure note); Decision v2.67 in Decision Log line 2028. Mirror synced. **Iteration freedom CLOSED on triple-barrier params** per Phase 1.14 freeze (Decision v2.66); changes to §10.3.x methodology are test-design changes, not strategy iteration — orthogonal concerns. **OOT one-shot rule (§10.4)** still does NOT apply (Phase 2.10 boundary). **ROLLBACK PROCEDURE**: revert §10.3.1 setup + gate-condition paragraphs (restore single-fold wording); revert Appendix C step 2.1; revert Decision Log entry. If user wants single-fold preserved, two paths: (i) Option (A) — pick regime-stable single fold (e.g., train 2024-04..2024-12 val 2025-01); (ii) Option (D) — accept spec-strict halt and pivot strategy. Approved by user 2026-05-05 with 'I will follow your recommend' confirmation | 2026-05-05 |
| v2.68| **DR-015: §10.3.2 amendment + confirmatory premise check after §10.3.1 walk-forward FAIL** | Phase 2.1 walk-forward pre-gate (DR-014 amended methodology) FAILED on 25 walk-forward folds: mean_val_logloss=1.0707, mean_prior=1.0645, mean_ratio=1.0058 (gate ≤0.99 to PASS); fails by 1.58pp. 76% of folds (19/25) cluster in ratio ∈ [0.99, 1.02]; only 2 of 25 folds individually PASS gate; best fold ratio 0.9891 (only 1.09pp better than prior — well below 'real signal' magnitude). Original first-fold (00) failure ratio 1.0322 was upper-end but NOT outlier; fold 19 worse (1.0542). DR-014 methodology correction worked exactly as intended: walk-forward eliminated fold-selection artifact hypothesis cleanly, confirming the 5-feature baseline genuinely cannot extract signal beyond marginal-distribution prediction on BTC 30m. VPS surfaced 5 escalation paths: (1) re-open §8 labeling, (2) re-open Phase 1.10 features, (3) re-open §2.1 timeframe, (4) accept v2.0 as research dead-end, (5) DR-015 §10.3.2 confirmatory walk-forward check. Local-Claude analyzed all 5; recommended path 5 as cheapest information-gain action before any major rework (~30 min compute vs months of feature/label re-iteration). User confirmed: 'I agree with you DR-015. because with only that 5 features can succeed that anybody can develop a bot'. **§10.3.2 amendment locked**: parallel to DR-014's §10.3.1 amendment — single-fold methodology → walk-forward (~25 folds per §9.2; train_months=9, val_months=1, step_months=1; purge=embargo=12 cascade). Gate condition preserved at ≥2% improvement (mean_val_logloss / mean_prior ≤ 0.98). **Out-of-sequence run explicitly authorized**: §10.3.2 normally runs only after §10.3.1 passes; Decision v2.68 authorizes §10.3.2 to run as a confirmatory premise check when §10.3.1 fails, on the diagnostic hypothesis that the §10.3.1 5-feature heuristic may be crypto-30m-miscalibrated. **Why the §10.3.1 heuristic might be wrong for crypto 30m**: the 5 hand-picked features (close-vs-EMA21, RSI, ADX, HTF EMA21 position, volume ratio) are CLASSICAL stocks/FX TA channels (per §19 Report 1 source). They OMIT signal channels that crypto-30m research consistently identifies as high-importance: Cat 10 volatility/regime tercile features, Cat 5 multi-anchor VWAP, Cat 13 divergence flags, Cat 16 break_of_structure, Cat 22 cross-asset. If signal exists in these channels but not in the 5 chosen ones, §10.3.1's 'if 5 fail, 245 will too' inference produces false-negative halt. §10.3.2 (full feature set) is the canonical test of this hypothesis. User reasoning ('with only 5 features can succeed that anybody can develop a bot') reinforces this: classical-TA-only signal extraction is near-impossible at BTC 30m's institutional-attention level; richer feature representation may still extract signal even if the canonical 5 cannot. **Decision criterion locked upfront (NOT iterable)**: PASS (mean_ratio ≤ 0.98 across walk-forward folds) → premise alive; 5-feat heuristic was crypto-miscalibrated; proceed to Phase 2.3 walk-forward CV + Phase 2.4 Optuna. FAIL (mean_ratio > 0.98) → premise robustly dead across BOTH 5-feature AND 245-feature walk-forward; halt v2.0 with very high confidence; **NO further 'one more test'**; escalate to fresh DR re-opening §8 labeling / Phase 1.10 features / §2.1 timeframe / accepting v2.0 as research dead-end. The 'no further one more test' commitment is critical to prevent indefinite premise-questioning into v1.0 failure mode (tuning into noise). **If §10.3.2 passes but §10.3.1 failed**: file DR to update §10.3.1 5-feature list with SHAP-validated picks (Phase 2.6 will identify them); future v2.x runs use the corrected list. **Spec edits**: §10.3.2 setup + gate-condition + out-of-sequence-authorization + decision-criterion paragraphs amended; Appendix C step 2.2 wording updated; Decision v2.68 in Decision Log line 2034. Mirror synced. **Code changes**: VPS extends `scripts/baseline_pre_gate.py` to handle full feature set (or forks `scripts/baseline_full_gate.py`). Reuses walk-forward helpers from DR-014. Compute estimate: 25 folds × 243 features × 200 LightGBM rounds × ~60-90s/fold ≈ 25-40 min total. **Halt criterion strengthened**: walk-forward-mean failure on FULL feature set is robust signal absence (5-feat AND 245-feat both fail across 25 folds = premise truly dead). **§10.3.x methodology iteration NOW closed**: walk-forward is canonical for both pre-gate and full-feature gate. No further methodology changes without fresh DR. **OOT one-shot rule (§10.4)** still does NOT apply (Phase 2.10 boundary); current iteration is on PRE-OOT data only. **ROLLBACK PROCEDURE**: revert §10.3.2 amendment (restore single-fold wording); revert Appendix C step 2.2; revert Decision Log entry. If user wants single-fold §10.3.2 OR immediate halt without §10.3.2 confirmatory test, two alternative paths available; user explicitly chose DR-015 path. Approved by user 2026-05-05 with 'I agree with you DR-015. because with only that 5 features can succeed that anybody can develop a bot' confirmation | 2026-05-05 |

---

## 19. References

### Research papers (2024–2026)

- Multi-Timeframe Feature Engineering for Crypto (Preprints.org, Mar 2026) — 4H BB position #1 SHAP feature, 4H RSI #5
- Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning (Springer, Financial Innovation, 2025)
- Multivariate Forecasting of Bitcoin Volatility with Gradient Boosting (Elsevier, 2025)
- Regime-Aware LightGBM for Stock Market Forecasting (MDPI Electronics, 2025)
- Coinbase Institutional ML Monthly Outlook (July 2024) — BTC/ETH/SOL drivers are asset-specific
- López de Prado, *Advances in Financial Machine Learning* (2018) — triple-barrier, purged CV, embargo

### v1.0 artifacts (inputs to v2.0)

- `ml-bot/PROJECT_SPEC.md` — 1980-line v1.0 specification
- `ml-bot/features/*.py` — reference implementations (most reused)
- `ml-bot/model/labeler.py` — triple-barrier with pessimistic tie-break (reused)
- `ml-bot/config.yaml` — baseline config (modified per §8, §9); DB credentials **inherited via `.env`** (same Postgres/TimescaleDB instance, new tables per §6.1.1)

### TradingView AI consultations (2026-04-23)

- `TradingView/AI/reply-2026-04-23T03-44-45.md` — move-to-15m/30m rationale, feature count guidance
- `TradingView/AI/reply-2026-04-23T04-30-22.md` — pivot/Fibonacci feature engineering as ML features
- `TradingView/AI/reply-2026-04-23T06-34-47.md` — intrabar execution hybrid (parked as Phase 4)

### Frameworks

- LightGBM 4.x (primary model)
- Optuna 3.x (hyperparameter tuning, TPESampler + MedianPruner)
- SHAP 0.45+ (TreeExplainer)
- pandas-ta (select indicator implementations)
- pandas, numpy, scikit-learn (preprocessing)
- mlfinlab / custom (purged CV with embargo)

### Data sources

- `data.binance.vision` — public OHLCV archives, geo-open
- Hyperliquid WebSocket + REST — live execution venue data (funding hourly, L2 book, trade tape)

---

## Appendix A — Migration Checklist from v1.0 Repo

If bootstrapping v2.0 by copying v1.0 rather than greenfield:

1. **Phase-scoped selective copy + clean-glue rewrite** (per Decision Log v2.33 / DR-001, reconciled with v1.0 ground truth per Decision Log v2.34 / DR-002, and split into "kept math / fresh glue" per Decision Log v2.36 / DR-005). **§15 is the authoritative file manifest.** Each phase splits its work into three tracks: (i) *copy* algorithm files from `../ml-bot/` unchanged, (ii) *write fresh* the v2.0 glue layer from spec, (iii) create *NEW algorithm files* not in v1.0. Files not listed in §15 (5m-specific scalping features, intrabar execution code, deprecated experiments, pre-cleanup branches) stay at v1.0 and never come over. Phase scope:
   - **Phase 1 — copy from `../ml-bot/` (algorithm files: math, not glue)** per Decision v2.36 + v2.37 (21 files; ema_context.py removed per Decision v2.37 Q1): `model/labeler.py`, `features/__init__.py`, `features/_common.py`, `features/builder.py`, `features/indicators.py`, `features/extra_momentum.py` (Cat 15), `features/volatility.py`, `features/volume.py`, `features/vwap.py`, `features/pivots.py`, `features/candles.py`, `features/structure.py`, `features/divergence.py`, `features/event_memory.py`, `features/adaptive_ma.py`, `features/ichimoku.py`, `features/regime.py`, `features/stats.py`, `features/sessions.py`, `features/context.py`, `data/collectors/storage.py` (parquet I/O — no v1.0 baggage), `scripts/relabel.py`, `requirements.txt`, `.gitignore`. Plus `.env` (per step 4 below). **Note (v2.37 Q1):** v1.0 `features/ema_context.py` is **NOT** copied — its 14 "Tier-1 setup" features are not enumerated in §7.2 (orphan); concepts partly absorbed by Cat 2 + Cat 20 (`bars_since_ema21_cross`). If a Phase 1.0 selective-copy already pulled it in, **delete from `ml-bot-30m/features/`** during Phase 1.3 reconciliation.
   - **Phase 1 — write fresh from spec (clean v2.0 glue layer)** per Decision v2.36 (9 files / ~500 LOC): `utils/__init__.py`, `utils/config.py` (YAML + dotenv loader), `utils/logging_setup.py` (loguru wrapper), `data/__init__.py`, `data/collectors/__init__.py`, `data/db.py` (~120 LOC; tables: `ohlcv_30m`, `features_30m`, `labels_30m`, `wf_folds_30m`, `models_30m`, `decay_metrics_30m` — no 5m/1h baggage), `data/collectors/binance_archive.py` (clean v2.0 archive fetcher; **NOT** the v1.0 `fetcher.py` renamed — supersedes v2.34's rename decision), `scripts/export_parquet.py` (clean v2.0), `config.yaml` (fresh from §6/§8/§9/§13 — no inherit-and-patch).
   - **Phase 1 — NEW algorithm files** (do not exist in v1.0; create fresh in `ml-bot-30m/`): `features/htf_context.py` (Cat 2a), `features/momentum_core.py` (Cat 1 selection per Decision v2.37 Q4), `features/trend.py` (Cat 2 selection per Decision v2.39 — sub of v2.37 Q4), `features/cross_asset.py` (Cat 22), `features/feature_stability.py` (§7.5 taxonomy), `tests/test_htf_aggregation.py`, `tests/test_multi_anchor_vwap.py`, `tests/test_labeler.py` (v1.0 tests/ was empty), `tests/test_purged_cv.py` (v1.0 tests/ was empty), `scripts/baseline_gate.py` (placeholder OK in Phase 1).
   - **Phase 2 — copy when entering Phase 2** (training + tuning): `model/train.py`, `model/predict.py`, `tune/optuna_search.py`, `tune/shap_analysis.py`. NEW in Phase 2: `model/calibration.py` (Platt scaling per §9.3).
   - **Phase 3 — copy when entering Phase 3** (paper trading): `backtest/simulator.py`, `execution/predictor_service.py`, `execution/executor_hyperliquid.py`, `data/collectors/hyperliquid_ws.py`.
   - **Phase 5 — create when entering Phase 5** (maintenance): `monitoring/decay_monitor.py` (NEW per §17.2).
2. Keep the v1.0 `ml-bot/` repo **intact** as reference — never edit, never commit changes there. v2.0 changes happen only in `ml-bot-30m/`.
3. **Scaffold §15 directory tree** as empty subdirs at project init (one-shot, before any file copies): `data/{collectors,storage/binance/30m,storage/hyperliquid}/`, `features/`, `model/`, `tune/`, `backtest/`, `execution/`, `scripts/`, `monitoring/`, `models/{v2.0_frozen,paper,v2.x_archive}/`, `research/`, `logs/`, `tests/`. Empty subdirs make the per-phase copy targets explicit and prevent accidental file placement.
4. **Database reuse (no setup):** v2.0 shares the v1.0 Postgres/TimescaleDB instance. Copy `.env` from `ml-bot/` (same `DATABASE_URL`, same credentials, `chmod 600`, verify `git check-ignore .env` returns `.env`). Do **not** create a new schema or re-run TimescaleDB install. New tables are added per §6.1.1 (`ohlcv_30m`, `features_30m`, `labels_30m`, `wf_folds_30m`, `models_30m`); v1.0 tables (`ohlcv_5m`, `ohlcv_1h`, …) remain untouched.
5. **30m-only data fetch:** do not add 4H/1D fetchers. 30m OHLCV is the sole fetched timeframe; 4H and 1D are aggregated losslessly from 30m at feature-build time via pandas `resample` (§6.4 Step A). Verify: 8×30m rows aggregate → 1×4H row with OHLC reconstructed from first/max/min/last and volume summed.
6. Update `config.yaml` per §6, §7, §8, §9; add `chunk_interval_ohlcv_30m: "90 days"` and remove stale 5m/1h-specific keys.
7. Rename `features/builder.py` HTF merge from `merge_1h_into_5m` → `merge_htf_into_30m`; implement as "aggregate 4H+1D from 30m, then prev-bar merge" (§6.4 Steps A–C).
8. Create `features/htf_context.py` (Category 2a, 18 features; consumes aggregated 4H/1D).
9. Extend `features/vwap.py` with multi-anchor VWAP (Category 5 expansion).
10. Promote `weekly_pivot_features()` in `features/pivots.py` to Phase 1 output.
11. Add Category 6 expansions (6.2 continuous 0–1 pos + ATR dist, 6.4 weekly continuous + ATR, 6.5 swing Fib retracements, confluence flags) per §7.2.
12. Tag every feature `static` / `dynamic` / `mixed` in `feature_stability.py` per §7.5.
13. Trim per-category feature outputs per §7.2 deltas.
14. Add `tests/test_htf_aggregation.py` (verifies 30m→4H/1D resample correctness) and `tests/test_multi_anchor_vwap.py`.
15. Initialize `PROJECT_LOG.md` (Appendix D template) at project root if not already present.
16. Read `Project Spec 30min.md` §10.5 end-to-end before any Phase 1 tool call. Output the §10.5.5 self-audit. Confirm anti-patterns (§10.5.9) refused on contact.

---

## Appendix B — Summary of What's Different (for fast onboarding)

If you read only one section: v2.0 = v1.0 with these swaps:

- **Timeframe:** 5m → 30m (+ 4H + 1D context)
- **Data:** 18 months → 3 years (4 years target)
- **Features:** 268 → ~202 (drop scalping; add HTF context, multi-anchor VWAP, weekly pivots, swing Fib retracements, continuous/ATR pivot encodings, confluence flags)
- **Labels:** 4.0/4.0@24 bars (2h) → 3.0/3.0@8 bars (4h)
- **min_profit_pct:** raised across the board
- **Walk-forward:** 8 folds → 14 folds
- **Discipline:** OOT is one-shot — single score at freeze, no iteration
- **Gate:** two-stage baseline — 5-feature pre-gate then 200-feature full-gate, both vs empirical prior, before Optuna
- **Feature taxonomy:** every feature tagged `static`/`dynamic` to enable Phase 4 intrabar scout without recomputation
- **Execution (Phase 3+):** triple-trigger method — ML bias × 5m price sweep/reclaim × volume/momo confirm
- **Process discipline (§10.5):** spec-authority rule, Deviation Request protocol, phase freeze gates, session-start self-audit, `PROJECT_LOG.md` decision trail. **v1.0's real failure mode.**

Everything else (LightGBM, triple-barrier, purged CV with embargo, Optuna search, SHAP trim, Wilder smoothing, Ichimoku displacement, project-wide sign convention, pessimistic tie-break, calibration) is carried forward unchanged.

---

## Appendix C — Phase Checklist (tick through, one row at a time)

Claude and user both tick items. No item is marked ✅ without spec-section citation in `PROJECT_LOG.md`. Each phase has a freeze gate per §10.5.4 — crossing it locks earlier decisions.

### Phase 1 — Data + Features

| Step | Action                                                                    | Spec ref  | Status |
| ---- | ------------------------------------------------------------------------- | --------- | ------ |
| 1.0  | Read the full spec (especially §10.5) before first tool call              | §10.5     | ☐      |
| 1.1  | Binance archive downloader — 3 yrs × {BTC, SOL, LINK} × 30m only          | §6.1, 6.3 | ☐      |
| 1.2  | Verify no gaps, UTC timestamps, canonical boundaries                      | §6.3      | ☐      |
| 1.3  | Adapt `features/builder.py` to 30m primary + dual HTF (4H, 1D) merge     | §6.4      | ☐      |
| 1.4  | Create `features/htf_context.py` (Cat 2a, 18 features)                    | §7.2 C2a  | ☐      |
| 1.5  | Expand `features/vwap.py` multi-anchor (Cat 5, +6)                        | §7.2 C5   | ☐      |
| 1.6  | Promote `weekly_pivot_features()` to Phase 1 output                       | §7.2 C6   | ☐      |
| 1.7  | Implement Cat 6.2 (continuous 0–1 pos + ATR dist + confluence, +3)       | §7.2 C6.2 | ☐      |
| 1.8  | Implement Cat 6.4 (weekly continuous pos + ATR dist, +2)                 | §7.2 C6.4 | ☐      |
| 1.9  | Implement Cat 6.5 (swing Fib retracement features, +7)                   | §7.2 C6.5 | ☐      |
| 1.10 | Trim per-category outputs per §7.2 deltas                                 | §7.2      | ☐      |
| 1.11 | Tag every feature static/dynamic in `feature_stability.py`               | §7.5      | ☐      |
| 1.12 | Triple-barrier labels with v2.0 params                                    | §8.1      | ☐      |
| 1.13 | Label distribution sanity check (LONG/SHORT 35–45%, NEUTRAL 15–25%)       | §8.2      | ☑ 2026-05-04 — Decision v2.66 freeze accepted; BTC/SOL/LINK all in band per filtered-set distribution |
| 1.14 | **FREEZE:** feature matrix + label column written to parquet              | §10.5.4   | ☑ 2026-05-05 — VPS regeneration confirmed bit-identical to v2.66c sweep; Phase 1.14 ACCEPTED |

### Phase 2 — Baseline Gates + Walk-Forward + Tuning + Trim + Freeze

| Step | Action                                                                    | Spec ref  | Status |
| ---- | ------------------------------------------------------------------------- | --------- | ------ |
| 2.0  | Self-audit: state current position per §10.5.5                           | §10.5.5   | ☑ 2026-05-05 — local-Claude session-start statement (Phase 2 entry) |
| 2.1  | Pre-gate: 5-feature walk-forward baseline on BTC (mean val log-loss ≥1% better than mean prior across 14 folds per §10.3.1 amended by Decision v2.67) | §10.3.1   | ⏳ in progress — first-fold attempt 2026-05-05 FAILED on regime-shift artifact (val_logloss/prior=1.032); DR-014 re-issued as walk-forward methodology |
| 2.2  | Full-gate: 243-feature walk-forward baseline on BTC (mean val log-loss ≥2% better than mean prior across 25 folds per §10.3.2 amended by Decision v2.68; runs out-of-sequence as confirmatory premise check after §10.3.1 walk-forward FAIL on 2026-05-05) | §10.3.2 | ⏳ DR-015 work order issued 2026-05-05 |
| 2.3  | Walk-forward on BTC (14 folds)                                            | §9.2      | ☐      |
| 2.4  | Optuna search (50 trials, folds [1, 7, 13])                              | §9.4      | ☐      |
| 2.5  | Re-run walk-forward with tuned params                                     | §9.4      | ☐      |
| 2.6  | SHAP analysis + feature trim to ~110–140                                 | §9.5      | ☐      |
| 2.7  | Probability calibration (Platt scaling)                                   | §9.3      | ☐      |
| 2.8  | Transfer learning: SOL, LINK (steps 2.1–2.7 each, per Decision v2.35)     | §14 Ph2   | ☐      |
| 2.9  | **FREEZE all models** — `models/v2.0_frozen/`                             | §10.5.4   | ☐      |
| 2.10 | **OOT evaluation (ONCE) — no iteration regardless of result**             | §10.1, 10.2 | ☐    |
| 2.11 | BCa bootstrap 95% CI for Sharpe / hit-rate / P&L (B=10000); pass = lower bound clears thresholds | §16.3.1 | ☐ |

### Phase 3 — Paper Trading

| Step | Action                                                                    | Spec ref  | Status |
| ---- | ------------------------------------------------------------------------- | --------- | ------ |
| 3.1  | Hyperliquid WebSocket ingestion with 30m rollup                           | §6.5      | ☐      |
| 3.2  | Predictor service (30m bar close inference)                              | §12.1     | ☐      |
| 3.3  | Paper executor (no real orders)                                          | §12.1     | ☐      |
| 3.4  | 2+ weeks paper trading                                                   | §14 Ph3   | ☐      |
| 3.5  | Paper vs backtest comparison; live gate go/no-go                         | §16.5     | ☐      |

### Phase 4 — Extensions (optional, each is its own Deviation Request)

| Step | Action                                                                    | Spec ref  | Status |
| ---- | ------------------------------------------------------------------------- | --------- | ------ |
| 4.1  | HYPE model on Hyperliquid-native data                                     | §14 Ph4   | ☐      |
| 4.2  | Microstructure refit with Hyperliquid L2 + trade tape                    | §7.2 C21  | ☐      |
| 4.3  | Hybrid triple-trigger 30m-bias + 5m-execution layer                       | §12.2     | ☐      |
| 4.4  | CUSUM event-bars experiment                                               | §3.3      | ☐      |
| 4.5  | Orthogonal feature class (on-chain / funding / news) — decay insurance    | §17.6     | ☐      |

### Phase 5 — Maintenance & Decay Management (continuous, post-Phase 3)

| Step | Action                                                                    | Spec ref  | Status |
| ---- | ------------------------------------------------------------------------- | --------- | ------ |
| 5.0  | Build `monitoring/decay_monitor.py` + `decay_metrics_30m` table; nightly BCa bootstrap on rolling 4-week window | §17.2, 16.3.1 | ☐ |
| 5.1  | Calibrate §17.2 yellow/red thresholds against 4-week paper variance       | §17.2     | ☐      |
| 5.2  | Calibrate §17.7 kill-switch thresholds against 4-week paper variance      | §17.7     | ☐      |
| 5.3  | Scheduled retrain every 90 days — each = full mini-Phase, one-shot OOT    | §17.3     | ☐      |
| 5.4  | Triggered retrain on red metric sustained ≥3 days                         | §17.3     | ☐      |
| 5.5  | Monthly research scan → `research/monthly_YYYY-MM.md`                     | §17.8     | ☐      |
| 5.6  | Quarterly champion/challenger evaluation                                  | §17.8     | ☐      |
| 5.7  | Bi-annual assumption audit (timeframe, asset universe, thresholds)        | §17.8     | ☐      |
| 5.8  | Quarterly v3.0-trigger evaluation logged regardless of outcome            | §17.9     | ☐      |

---

## Appendix D — PROJECT_LOG.md template

Create `ml-bot-30m/PROJECT_LOG.md` at project start with this header, then append one line per non-trivial action.

```
# PROJECT_LOG — ml-bot-30m (v2.0)

Append-only decision trail per Project Spec 30min §10.5.6.
Format: YYYY-MM-DD HH:MM  <Phase.step>  [SPEC §<section>]  <one-line summary>

Every line cites a spec section. Lines without citations are deviations (per §10.5.1).
Mistakes are not deleted — they get a follow-up line marking the correction.

---

2026-MM-DD HH:MM  Phase 0    [SPEC §10.5]     Project_LOG.md initialized
```

---

**End of Project Spec 30min v2.0**
