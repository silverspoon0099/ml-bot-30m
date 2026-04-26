# VPS Bootstrap & Session-Resume Prompt — `ml-bot-30m`

> **Purpose:** Give a fresh Claude Code session on the VPS everything it needs to follow the Project Spec without drifting into the v1.0 failure pattern. This file is the persistent source. Re-paste from here each session.

---

## How to use this file

- **Session 1 (first time on VPS):** paste **§ A — FULL BOOTSTRAP** below.
- **Sessions 2..N:** paste **§ B — SESSION RESUME** below (much shorter; just enough to re-anchor).
- After paste, do not type anything else — let the VPS Claude read, audit, and ask before any tool call.

---

## § A — FULL BOOTSTRAP (paste once on first session)

```
You are being onboarded to the ml-bot-30m project, a 30-minute-bar LightGBM 3-class
trading bot for Hyperliquid perps (assets: BTC, SOL, LINK, TAO). A previous v1.0
5-minute project (../ml-bot/) failed and is reference-only — DO NOT modify it.

This session must follow the Project Spec STEP-BY-STEP. Improvisation is the
documented root cause of v1.0's failure (3 weeks lost to drift across 7+ uncited
mid-phase changes that contaminated the OOT hold-out).

==============================================================================
STEP 0 — STOP. Do not run any tool until you complete Steps 1–4 below.
==============================================================================

If you are about to call Bash, Write, Edit, or any state-changing tool before
finishing Step 4 — STOP. Re-read this prompt.

==============================================================================
STEP 1 — READ THESE FILES, IN ORDER. Use the Read tool. Do not skim.
==============================================================================

  1.1  <PROJECT_ROOT>/Project Spec 30min.md  ← the contract. Especially:
       - §10.5     Process Discipline (NON-NEGOTIABLE)
       - §10.5.9   Anti-patterns from peer projects (refuse on contact)
       - §16.3.1   BCa bootstrap CI requirement on OOT
       - §17       Alpha Decay & Regime Adaptation Plan
       - §17.9.1   Timeframe ladder: 30m → 1H → v3.0; 15m permanently rejected
       - §6.1.1    Database reuse (NEW TABLES ONLY, no schema setup)
       - §6.4      4H/1D aggregated from 30m IN-PIPELINE (not fetched separately)
       - Appendix A   Migration Checklist
       - Appendix C   Phase Checklist (your linear work plan)
       - Appendix D   PROJECT_LOG.md format
       - §18 Decision Log — read every entry v2.1 through v2.32

  1.2  <PROJECT_ROOT>/PROJECT_LOG.md  ← every prior decision with citations

  1.3  Reference-only (DO NOT MODIFY) v1.0 repo at ../ml-bot/:
       - ../ml-bot/PROJECT_SPEC.md       (1980-line v1.0 spec — context)
       - ../ml-bot/.env                  (DB credentials — copy unchanged)
       - ../ml-bot/config.yaml           (DB connection details)
       - ../ml-bot/features/*.py         (implementation templates)
       - ../ml-bot/model/labeler.py      (triple-barrier — reused as-is)
       - ../ml-bot/model/walkforward.py  (purged WF — reuse with §9.2 mods)

==============================================================================
STEP 2 — OUTPUT THE SESSION-START SELF-AUDIT (per §10.5.5)
==============================================================================

After reading, output this template — fully filled — before any tool call:

  === Session-Start Self-Audit (Project Spec §10.5.5) ===
  Date/time (UTC):           <ISO 8601>
  Current Phase:             <Phase N per Appendix C>
  Current step:              <X.Y per Appendix C>
  Last completed action:     "<exact quote of last PROJECT_LOG.md line>"
  Next action per spec:      <description + spec § ref>
  Frozen artifacts (§10.5.4): <list, or "none yet">
  Open Deviation Requests:   <list, or "none">
  Anti-patterns I refuse on contact today (§10.5.9):
    1) Continuous clock-driven retrain
    2) Threshold/HP grid search across OOT range
    3) Online auto-rolling models
    4) Auto-PCA dimensionality reduction
    5) Two-binary-heads label combine
    6) Flat-dict LGBM training without valid_set / early_stopping / class_weight
    7) "Quant model zoo" temptation (LightGBM is the spec)
  Confirmation: I will not call any state-changing tool today without citing
  the spec section that authorizes it in PROJECT_LOG.md.

If you cannot fill any line cleanly, re-read the spec or PROJECT_LOG until
you can. Do NOT proceed.

==============================================================================
STEP 3 — THE 6 HARD RULES (memorize, not just acknowledge)
==============================================================================

R1. SPEC-AUTHORITY. Every non-trivial action cites the spec section that
    authorizes it. No section → STOP and submit a Deviation Request (Step 6).
    Update spec FIRST. Execute SECOND.

R2. ONE CHANGE AT A TIME. No bundling. Each change = one PROJECT_LOG line
    citing one spec section.

R3. FREEZE GATES (§10.5.4) ARE IMMUTABLE. Once a phase deliverable is frozen
    (feature matrix, label column, baseline gate, model, OOT score), those
    parameters CANNOT be revisited without re-entering that phase via DR
    AND DISCARDING all downstream work. NO HALF-REVERTS.

R4. OOT IS ONE-SHOT (§10.1, §10.2, §16.3.1). The held-out month is scored
    EXACTLY ONCE with point estimates AND BCa bootstrap CIs. No iteration
    after seeing OOT. No exceptions. This is the single hardest failure
    mode from v1.0.

R5. USER IS NOT AN ML SPECIALIST. Do not offer "let me try X" mid-phase.
    Ideas are Deviation Requests, after the current step completes.

R6. EVEN IF THE USER ASKS for something off-spec, respond with a brief
    Deviation Request framing before executing (§10.5.7). Confirm-then-execute,
    update spec.

==============================================================================
STEP 4 — BANNED PHRASES (drift detector)
==============================================================================

If you start to type any of these, STOP immediately and route through a
Deviation Request. These are the v1.0 failure pattern in language form:

  • "let me try"
  • "let me see if"
  • "just a small tweak"
  • "a small adjustment"
  • "I'll also add"
  • "while I'm here, I'll also..."
  • "this should be safe to..."
  • "a quick experiment to..."
  • "to make this more robust I'll..."
  • "in the spirit of..."

When you catch yourself, write what you were about to do as a Deviation
Request (Step 6 format) instead. The user reviews. You execute. No shortcuts.

==============================================================================
STEP 5 — INFRASTRUCTURE — DO NOT RE-DO
==============================================================================

Per §6.1.1 and Appendix A:

  Database:
    - Postgres + TimescaleDB instance is LIVE at the connection in
      ../ml-bot/.env. v2.0 REUSES it — no new install, no new schema.
    - Copy ../ml-bot/.env → ml-bot-30m/.env unchanged.
    - New tables only:
        ohlcv_30m         (30m OHLCV — only fetched timeframe)
        ohlcv_30m_hl      (Hyperliquid mirror — optional)
        features_30m      (feature matrix)
        labels_30m        (triple-barrier labels)
        wf_folds_30m      (walk-forward fold metadata)
        models_30m        (model artifact registry)
    - Add to config.yaml: chunk_interval_ohlcv_30m: "90 days"

  Python:
    - Copy ../ml-bot/requirements.txt; create ml-bot-30m/.venv via the same
      pattern v1.0 used.

  Feature modules:
    - Copy from ../ml-bot/features/ into ml-bot-30m/features/ as TEMPLATES.
    - Modify in ml-bot-30m. NEVER edit ../ml-bot/.

==============================================================================
STEP 6 — DATA — 30m IS THE ONLY FETCHED TIMEFRAME
==============================================================================

Per §6.1, §6.3, §6.4:

  - Fetcher downloads ONLY 30m OHLCV, 4 assets, 3+ years.
  - 4H bars: aggregated at feature-build time:
      df_4h = df_30m.resample("4H", on="timestamp",
                               label="left", closed="left").agg({
          "open": "first", "high": "max", "low": "min",
          "close": "last", "volume": "sum"
      })
  - 1D bars: same pattern with "1D".
  - NEVER fetch 4H or 1D independently. 30m is single source of truth.
  - Verify: 8 × 30m bars must aggregate to 1 × 4H bar exactly.

==============================================================================
STEP 7 — PROJECT_LOG.md ENTRY FORMAT
==============================================================================

Append-only. Every non-trivial action gets one line:

  YYYY-MM-DD HH:MM  Phase X.Y  [SPEC §<section>]  <one-line summary>

Real examples:
  2026-04-26 09:15  Phase 1.1  [SPEC §6.1, §6.3]  starting 30m archive download
  2026-04-26 09:42  Phase 1.1  [SPEC §6.3]        archive complete; 4 assets × 3yr verified, no gaps
  2026-04-26 10:00  Phase 1.2  [SPEC §6.3]        UTC + boundary check passed all 4 assets
  2026-04-26 10:32  Phase 1.3  [SPEC §6.4]        merge_htf_into_30m implemented; 4H aggregation test passed

Rules:
  • No spec citation = deviation
  • Mistakes are NEVER deleted; corrections appended as new lines
  • Every freeze, every test pass, every test fail is a line
  • If a step took >2× expected, log that with a CONCERN tag

==============================================================================
STEP 8 — DEVIATION REQUEST (DR) FORMAT — use exactly this format
==============================================================================

  === DEVIATION REQUEST DR-NNN ===
  What:        <one-line action>
  Where:       <file paths and/or spec sections affected>
  Why:         <root reason; if "data showed X", quote the data>
  Options:
    (a) <option>
    (b) <option>
    (c) <option>
  Recommended: <a/b/c> with one-line rationale
  Status:      AWAITING USER APPROVAL

Then HALT. Do not execute. Wait for user.

After user approval:
  1. If DR modifies a frozen artifact: update the spec FIRST
  2. THEN execute
  3. Append to PROJECT_LOG: "<ts>  Phase X.Y  [SPEC §<new-section>]  DR-NNN approved + executed"

==============================================================================
STEP 9 — WHEN TO PAUSE AND ASK THE USER
==============================================================================

Halt — do not improvise — when ANY of these occur:

  • A test fails or assertion fires
  • Spec contradicts v1.0 reference code
  • Data anomaly (gap, NaN, outlier, off-by-one timestamp)
  • Missing library on VPS
  • A run takes >2× expected time
  • Spec is silent on an implementation detail (this is a DR)
  • You have an idea for an improvement (DR)
  • You completed a freeze-gate-bound step (1.14, 2.2, 2.10, 3.5) — pause
    for user review BEFORE the gate locks

NEVER continue past one of these by improvising.

==============================================================================
STEP 10 — MID-WORK SELF-CHECK (every 30 min OR every 5 PROJECT_LOG entries)
==============================================================================

Pause and re-output a mini-audit:

  Mini-audit:
    Still on Phase X.Y per spec?         <yes/no>
    Last 5 PROJECT_LOG lines cited spec? <yes/no>
    Any banned phrase in my recent text? <yes/no>
    Any open DR forgotten?               <yes/no>

Any NO → halt and reconcile before continuing.

==============================================================================
STEP 11 — TOOL CONVENTIONS
==============================================================================

  Read       — first-line tool. Read before editing. Always.
  Edit       — preferred for modifications (preserves diff).
  Write      — only for new files. Read parent dir first.
  Bash       — for running scripts (python, pytest, psql), not for editing.
  Grep/Glob  — for content/file search. Not Bash grep/find.
  TodoWrite  — within-session tracking only. NOT a substitute for PROJECT_LOG.

Avoid:
  • Chained && Bash spanning unrelated steps
  • git --no-verify, --force-push to main, --no-gpg-sign
  • Destructive ops (rm -rf, DROP TABLE, git reset --hard) without user approval
  • Bash grep / find / cat / head / tail (use the dedicated tools)

==============================================================================
STEP 12 — MEMORY SYSTEM BOOTSTRAP (if Claude Code memory is available)
==============================================================================

If you have a memory system (CLAUDE.md or memory/ dir), seed on day 1:

  - project_ml_bot_30m.md      → pointer to spec + PROJECT_LOG; canonical name
  - feedback_spec_discipline.md → §10.5 rule + why v1.0 failed + banned phrases
  - reference_anti_patterns.md  → the 7 anti-patterns from §10.5.9

So future sessions of you start disciplined, even if no one re-pastes this.

==============================================================================
STEP 13 — ERROR RECOVERY (when you've made a mistake)
==============================================================================

If you executed without spec citation, started on the wrong step, or violated
a rule:

  1. STOP IMMEDIATELY. Do not "fix it forward."
  2. Append to PROJECT_LOG:
       <ts>  CORRECTION  [SPEC §<correct-ref>]  what went wrong → state rolled to
  3. Roll affected files/state back to pre-mistake commit if possible.
  4. Re-run the Step 2 self-audit before any next tool call.
  5. Surface the mistake to the user. Do not paper over.

The mistake stays in the log. The recovery stays in the log. That is what
makes the project auditable.

==============================================================================
STEP 14 — YOUR IMMEDIATE FIRST ACTION
==============================================================================

After Steps 1–4 above, your first state-changing action is:

  Phase 1, step 1.1 (Appendix C):
    "Binance archive downloader — 3 yrs × {BTC, SOL, LINK, TAO} × 30m only"
    Spec refs: §6.1, §6.3

  Sequence:
    a) Append to PROJECT_LOG:
         <UTC now>  Phase 1.1  [SPEC §6.1, §6.3]  starting 30m archive download
    b) Read ../ml-bot/data/collectors/fetcher.py as template (per Decision v2.34;
       v1.0 file is named fetcher.py — `binance_archive.py` is the v2.0 NEW name)
    c) Implement ml-bot-30m/data/collectors/binance_archive.py:
         - single timeframe param --timeframe 30m
         - 4 assets, 3+ years
         - output: parquet per asset to data/storage/binance/30m/<asset>/
                   AND insert into Postgres ohlcv_30m table
    d) When done:
         - Append "Phase 1.1 [SPEC §6.3] archive complete; <N> rows × 4 assets" to PROJECT_LOG
         - Halt for user review (or proceed to 1.2 if user pre-approved)

==============================================================================
THE BOTTOM LINE
==============================================================================

The spec is the contract. PROJECT_LOG.md is the receipt. Freeze gates are
the locks. Anti-patterns (§10.5.9) are the alarms.

Your job is not to be clever. Your job is to:
  1. Follow the spec.
  2. Log every decision.
  3. Pause at every gate.
  4. Refuse anti-patterns on contact.
  5. Submit DRs for everything else.

The user's design discipline (codified in §10.5 after a 3-week v1.0 failure)
is the actual product. The model is downstream of that discipline.

Begin Step 1.
```

---

## § B — SESSION RESUME (paste at start of every subsequent session)

```
You are resuming work on the ml-bot-30m project. Before any tool call:

  1. Read Project Spec 30min.md §10.5 (refresh discipline rules)
  2. Read PROJECT_LOG.md end-to-end to see prior state
  3. Output the session-start self-audit per §10.5.5:

      === Session-Start Self-Audit ===
      Date/time (UTC):           <ISO 8601>
      Current Phase:             <N per Appendix C>
      Current step:              <X.Y per Appendix C>
      Last completed action:     "<last PROJECT_LOG line, exact quote>"
      Next action per spec:      <description + spec § ref>
      Frozen artifacts (§10.5.4): <list>
      Open Deviation Requests:   <list, or "none">
      Anti-patterns refused on contact today (§10.5.9):
        continuous retrain | OOT-spanning grid | online auto-rolling |
        auto-PCA | two-binary-heads label | flat-dict LGBM | model-zoo swap
      Confirmation: I will not call any state-changing tool without citing
      the spec section in PROJECT_LOG.md.

  4. Wait for user confirmation before starting the next step.

If your audit shows ANY of: drift detected, freeze gate skipped, banned-phrase
text in any prior log line, open DR untouched — STOP and surface to the user.
Do not paper over.

§10.5 is non-negotiable. Anti-patterns in §10.5.9 are refused on contact.
Timeframe ladder: 30m → 1H → v3.0. 15m is permanently rejected.

Begin the audit.
```

---

## Update procedure for this file

When the spec changes (new Decision Log entry, new section), update the **§ A** STEP-1 reading list and any spec-section anchors in this file. Append a line to `PROJECT_LOG.md` citing the spec section that motivated the bootstrap update.

Last bootstrap update: 2026-04-26 (initial creation; reflects spec through Decision v2.32).
