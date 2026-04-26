# PROJECT_LOG — ml-bot-30m (v2.0)

Append-only decision trail per Project Spec 30min §10.5.6.
Format: `YYYY-MM-DD HH:MM  <Phase.step>  [SPEC §<section>]  <one-line summary>`

Every line cites a spec section. Lines without citations are deviations (per §10.5.1).
Mistakes are not deleted — they get a follow-up line marking the correction.

Canonical project name: `ml-bot-30m/` (VPS). Local working folder may still be `hyperliquid-ml-bot-30m/` pending rename — same project.

---

2026-04-25        Phase 0     [SPEC §10.5]      Project_LOG.md initialized; spec v2.0 with process-discipline section locked in
2026-04-25        Phase 0     [SPEC §10.5.4]    Freeze gates defined; spec itself is frozen until Phase 1.1 begins
2026-04-25        Phase 0     [SPEC §6.1.1]     DB reuse policy: v2.0 shares v1.0 Postgres/TimescaleDB instance; new tables ohlcv_30m/features_30m/labels_30m/wf_folds_30m/models_30m; no schema setup (Decision v2.27)
2026-04-25        Phase 0     [SPEC §6.4]       4H/1D aggregated in-pipeline from 30m via pandas resample (label=left, closed=left); 30m is sole fetched timeframe (Decision v2.28)
2026-04-25        Phase 0     [SPEC §17 v2.29]  Canonical naming adopted: ml-bot/ (v1.0 reference) and ml-bot-30m/ (v2.0 active); spec + Appendix A + Appendix D references updated
2026-04-26        Phase 0     [SPEC §17]        Alpha Decay & Regime Adaptation Plan added as new §17; Decision Log renumbered to §18, References to §19; Decision Log v2.30 records rationale; Appendix C extended with Phase 5 Maintenance steps 5.0–5.8 and Phase 4.5 (orthogonal feature class)
2026-04-26        Phase 0     [SPEC §17.9.1]    Timeframe ladder fixed: 30m primary → 1H fallback only → v3.0 redesign. 15m permanently rejected. Decision v2.31 records deep-research validation; §17.9.1 codifies four quantitative switch-to-1H triggers
2026-04-26        Phase 0     [SPEC §10.5]      Local project folder hyperliquid-ml-bot-30m/ was deleted between sessions; recreated from TradingView mirror (canonical source for spec). PROJECT_LOG.md re-seeded with full prior history. No spec content lost — all v2.30 + §17 + v2.31 edits preserved via mirror
2026-04-26        Phase 0     [SPEC §10.5.9]    Anti-patterns from peer projects codified: continuous retrain, OOT-spanning grid, auto-rolling models, auto-PCA, two-binary-heads label, flat-dict LGBM train, model-zoo temptation. Permanently rejected for v2.x lifecycle (Decision v2.32)
2026-04-26        Phase 0     [SPEC §16.3.1]    BCa bootstrap 95% CIs added as Phase 2.11 step and Phase 5 monitoring requirement; OOT pass criterion now requires BCa lower bound, not just point estimate (DR-A4 from Decision v2.32)
2026-04-26        Phase 0     [SPEC §17 v2.32]  Peer-project deep analysis recorded: 10 adoptions (A1-A10) with phase + integration mapping, 4 anti-patterns codified in §10.5.9. Only A4 (BCa) modified spec; A1-A3, A5-A10 are implementation choices logged when each phase begins
