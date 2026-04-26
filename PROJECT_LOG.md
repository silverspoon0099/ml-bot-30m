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
