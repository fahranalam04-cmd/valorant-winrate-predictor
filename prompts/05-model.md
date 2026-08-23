# Phase 5 — Model

Read `CLAUDE.md` and **all of `docs/MODELING.md`** before starting. Phase 4's
leakage audit must pass — if it does not, stop and fix that first, because
nothing measured here means anything otherwise.

## Goal

Climb the baseline ladder, calibrate the output, and produce honest measured
results for the README. **At the end of this phase the project is a complete
portfolio piece** even if nothing after it gets built.

## Context

I have ML coursework but no end-to-end projects. Explain what you are doing and
why at each rung, particularly around calibration, which I have not worked with
in practice.

Expected performance is **AUC 0.60–0.68, accuracy 58–64%**. Matchmaking is
actively working against the model — it exists to make matches even, and any
signal is the residual after it has done its job. Above AUC 0.75, assume
leakage and go looking.

## Build

**1. Splits — `valwr/model/split.py`**

Time-ordered by match start: 70% train, 15% validation, 15% test.

**Never random k-fold.** Match data is a time series; shuffling trains on the
future and the inflation is large.

Deduplicate by `match_id` before splitting. The crawler collects the same match
via multiple players, so duplicates across splits are a live risk here, not a
theoretical one.

The test slice is touched **once**, at the end. Tuning against it repeatedly is
a slower form of overfitting.

**2. Baselines — `valwr/model/baselines.py`**

Permanent fixtures, kept in the repo, reported in every evaluation:

- Coin flip (0.5 for everything) — the floor
- **Higher average rank wins** — the honest bar, and stronger than it sounds,
  because rank is already most of what there is to know about a player

**3. Models — `valwr/model/train.py`**

In order, each beating the previous on the test set before moving up:

- Logistic regression on 5–10 features. Interpretable, and the sanity check on
  the whole pipeline. **If this cannot beat the rank baseline, the features are
  wrong — do not reach for a bigger model to paper over it.** Stop and
  investigate.
- Gradient boosting (LightGBM or XGBoost) on the full feature set. This is where
  map × agent and composition features earn their place, since trees find
  interactions a linear model cannot express.
- Optional calibrated ensemble, only if there is time and the above is solid.

**4. Calibration — `valwr/model/calibrate.py`**

The product displays "58%". That has to *mean* 58% — of matches predicted at
58%, roughly 58% should be won. A model can have decent AUC and useless
calibration.

Fit isotonic (or Platt) calibration on the **validation** slice, never on test.
Produce a **reliability diagram** — predicted probability against observed
frequency, binned. Save it to `reports/`. This is the single most convincing
artefact the project produces; it goes in the README.

Report expected calibration error.

**5. Evaluation — `valwr/model/evaluate.py`**

Log loss (primary), Brier score, AUC, accuracy, ECE. Every model, every
baseline, one table.

**6. The shuffled-target check — into `test/test_leakage.py`**

Shuffle the labels, retrain, assert AUC collapses to ~0.5 (below 0.55). If a
model can predict shuffled labels, a feature encodes the outcome. Cheap, and it
catches leakage the traceability audit misses.

**Run this before believing any result.**

**7. Attribution — `valwr/model/explain.py`**

SHAP on the final model. Global feature importance plot for the README, and
per-match attributions — Phase 8's coach depends on these being real.

## Then, the analysis that makes it interesting

- **Equal-rank subset.** Filter to matches where team average ranks are within a
  tier. This is where matchmaking did its job and the residual signal is all
  there is — performance here is the honest measure of whether the features add
  anything beyond rank.
- **Which features actually matter?** If map × agent contributes nothing, say
  so. A clearly-reported negative result is a credibility signal.
- **Temporal decay.** Train early, test on progressively later slices. Meta
  shifts should show as degradation, and quantifying it tells you the retraining
  cadence.
- **Where is it most wrong?** Errors clustered on a map, rank band, or comp
  point at a missing feature.

## Finally

Fill in the README results table with **measured** numbers. Never an
aspirational one. Add the reliability diagram and the feature importance plot.
Write the limitations honestly.

## Acceptance criteria

1. Gradient boosting beats the rank baseline on log loss on the held-out test set
2. Reliability diagram close to the diagonal
3. Shuffled-target check collapses to ~0.5
4. README results table filled in with real numbers

If #1 fails, that is a genuine result worth reporting, not a failure to hide —
but investigate the features first.

If AUC comes out above 0.75, **do not celebrate**. Go find the leak.
