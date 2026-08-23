# Prompts

One file per phase. Each is sized for a single Claude Code session.

## How to use these

1. Open a **fresh** Claude Code session in the repo root. Fresh matters — a
   session that has been running for hours has a context full of earlier phases
   and will start conflating them.
2. Paste the whole prompt file.
3. Work through it. Expect back-and-forth; these are briefs, not incantations.
4. **Verify the acceptance criteria yourself.** Do not take "done" on trust —
   run the command, read the output, check the number is plausible.
5. Commit, then start the next phase in a new session.

## Why phases instead of one big prompt

This project has a crawler, a rating metric, a leakage-sensitive feature
pipeline, a model, a live client integration, a dashboard, and an LLM layer.
Handed all of that at once, a coding agent either goes deep on one part and
stubs the rest, or spreads thin across all of it — and runs out of context
before any of it is real.

Phases with hard acceptance criteria also mean each stage is *verified* before
the next depends on it. Phase 4's leakage audit is worthless if Phase 2's
temporal store was never actually tested.

## If a phase goes badly

Do not push forward with failing acceptance criteria. Every later phase assumes
the earlier ones hold — a broken temporal store in Phase 2 produces a model in
Phase 5 whose results are meaningless, and you will not find out until you
notice the AUC is suspiciously high.

Start a fresh session, describe what failed specifically, and fix it before
moving on.

## Order

| File | Phase |
|---|---|
| `00-scaffold.md` | environment, deps, reference data |
| `01-collector.md` | rate-limited snowball crawler |
| `02-normalize.md` | schema + the temporal store |
| `03-rating.md` | the player rating metric |
| `04-features.md` | time-gated feature builder |
| `05-model.md` | baselines, training, calibration |
| `06-live-client.md` | lockfile auth, match detection |
| `07-dashboard.md` | FastAPI + browser front end |
| `08-coach.md` | Claude API coaching layer |
