# Modelling

This is the document that decides whether the project is credible. The model
itself is the easy part — a gradient booster is twenty lines. Everything that
makes the result *mean* something is here.

---

## What is actually being predicted

`P(team_a wins)` for a competitive match, using only information available at
the loading screen.

**Why this is hard, stated honestly:** matchmaking exists to make matches even.
It is actively working against you. Any signal you find is the residual after
rank-based matchmaking has already equalised the obvious differences. This is
not a problem where more feature engineering eventually yields 90% accuracy —
the ceiling is set by how much matchmaking leaves on the table.

### Expected performance

| Metric | Realistic range |
|---|---|
| AUC-ROC | **0.60 – 0.68** |
| Accuracy | **58 – 64%** |
| Log loss | ~0.66 – 0.68 |

**Above AUC 0.75, assume leakage.** Not "be cautious" — assume it, and go
looking. Every implausibly good result in this problem space has turned out to
be a feature carrying outcome information.

Beating the rank baseline by a few points of log loss, with good calibration, is
a genuinely good result here. Write that up confidently.

---

## The baseline ladder

Climb in order. Each rung must beat the one below on the **held-out time-split
test set** before moving up. `valwr/model/baselines.py` stays in the repo
permanently and every evaluation reports against it.

**1. Coin flip — 50%.** The floor. If anything scores below this, something is
inverted.

**2. Higher average rank wins.** The honest bar, and the one people skip. It is
much stronger than it sounds, because rank is the single best available proxy
for skill and it is already most of what there is to know. A complicated model
that fails to beat this has demonstrated nothing.

**3. Logistic regression, 5–10 features.** Interpretable, fast, and the sanity
check on the whole feature pipeline. If a linear model on shrunk win rates and
rank deltas cannot beat #2, the features are wrong — do not reach for a bigger
model to paper over it.

**4. Gradient boosting on the full feature set.** LightGBM or XGBoost. This is
where map × agent interactions and composition features earn their place, since
trees find interactions that the linear model cannot express.

**5. Calibrated ensemble.** Optional. Only if #4 is solid and there is time.

Report every rung in the README table. Showing the progression is more
persuasive than showing one number, because it demonstrates you know what your
model is being compared against.

---

## Features

All per-player features are computed **only** from matches with
`started_at < this_match.started_at`, via `valwr/store/temporal.py`. See
[DATA.md](DATA.md).

### Per player

**Skill**
- Rating from `valwr/rating/` (see below)
- Rank tier, RR
- Account level, total games played — experience proxies

**History, all shrunk**
- Overall win rate
- Win rate on **this map**, plus games played on it
- Win rate on **this agent**, plus games played on it
- Win rate on **this map × this agent** — the interaction from the original
  brief ("better on Ascent specifically as Jett"). Very sparse; shrink hard.

**Performance detail**
- ACS, ADR (average damage per round)
- Headshot percentage
- First-blood rate, first-death rate — entry impact, and its inverse
- Clutch rate, trade participation, multi-kill rate

**Form**
- Last-N win rate (N ≈ 10, 20)
- Rating trend — is this player improving or sliding
- Session length so far today, as a tilt proxy
- Days since last played — rust

### Per team (aggregations)

- Mean, **max, min, and standard deviation** of player rating.
  The standard deviation matters and is easy to overlook: a team with one smurf
  and four weak players behaves very differently from five average players, and
  the mean erases that distinction entirely.
- Role composition from the agent→role map: duelist count, has-controller,
  has-sentinel, has-initiator, role balance score
- **Off-role count** — players on an agent outside their usual role
- Composition strength **on this specific map** — some comps are map-dependent
- **Party structure** from `party_id`: largest party size, number of distinct
  parties. A five-stack is a large, cleanly-observable signal.
- Rank spread within the team

### Match context

- Map
- Attack or defence first
- Episode/act — absorbs meta shifts rather than learning them as noise
- Region (fixed to `na` for now, but keep the column)

### Team-pair structure

The model sees the **difference** between team vectors, plus symmetric
aggregates. Critically:

**The target must be defined from one fixed perspective and the representation
must be antisymmetric** — otherwise the model can learn "team A wins more
often", which is an artefact of how rows were written, not football. Either
construct features as `team_a - team_b` differences (naturally antisymmetric),
or mirror-augment every training row with its swapped counterpart. Verify by
asserting that swapping the two teams produces exactly `1 - p`.

---

## Leakage: the trap list

Leakage makes results look **better**, which is why it survives. Nobody
investigates a good number. Treat each of these as a specific thing to check,
not a general principle to keep in mind.

**1. No statistic from the match being predicted.** Obvious in the abstract,
easy in practice: a "last 20 games" window computed from a match list that
includes the current match is the classic version. Strict `<` on timestamps,
enforced at the query layer.

**2. No post-match data.** Final score, rounds won, whether the match went to
overtime — all resolved after the prediction point. Excluded.

**3. Time-based splits only. Never random k-fold.** Match data is a time series.
Shuffling trains on the future and tests on the past, and the inflation it
produces is large. Train on the earliest slice, validate on the middle, test on
the latest.

**4. Shrink every rate feature.** A player with 3 games at 100% is not a 100%
win rate player. Use empirical-Bayes shrinkage toward the population mean:

```
shrunk = (wins + prior_weight * population_rate) / (games + prior_weight)
```

with `prior_weight` tuned on validation (start around 20–50 for overall win
rate, higher for sparse cells like map × agent). Without this, sparse cells
dominate the model and generalise to nothing. This is the single most common
mistake in projects of this shape.

**5. Match overlap between splits is fatal; player overlap is fine.** The same
player appearing in train and test is realistic — that is what happens live.
The same *match* in both is leakage. Deduplicate by `match_id` before splitting.
Note that the crawler naturally collects the same match via multiple players,
so this is a live risk, not a theoretical one.

**6. The mirror trap.** Covered above: verify `swap(features) → 1 - p`.

**7. Rank at time of match, not current rank.** `match_players.tier` is the rank
during that match. A player's *current* rank reflects everything since,
including the match being predicted. Use the stored per-match tier.

### Enforcement

Two mechanical checks in `test/test_leakage.py`, both required to pass:

- **Traceability audit.** Sample matches, rebuild each feature, assert every
  contributing source row has `started_at < target.started_at`.
- **Shuffled-target check.** Shuffle labels, retrain, assert AUC collapses to
  ~0.5 (say, below 0.55). If a model can predict shuffled labels, a feature
  encodes the outcome. This catches leakage that the traceability audit misses,
  and it is cheap to run.

Run both before believing any result.

---

## Validation protocol

**Splits.** Time-ordered, by match start:
```
|--------- train (70%) ---------|--- val (15%) ---|--- test (15%) ---|
                                                   ^ touched once
```
The test slice is for the final number. Tuning against it repeatedly is a slower
form of the same overfitting.

**Metrics.** Report all four:

- **Log loss** — the primary. Punishes confident wrong answers, which is what
  matters for a probability product.
- **Brier score** — mean squared error on probabilities.
- **AUC-ROC** — ranking quality, threshold-independent.
- **Accuracy** — least informative, but it is what people ask about.

**Calibration is the priority here.** The product displays "58%". That number
has to *mean* something: of matches predicted at 58%, roughly 58% should be won.
A model can have good AUC and useless calibration.

- Fit isotonic (or Platt) calibration on the validation slice, never on test
- Produce a **reliability diagram** — predicted probability vs observed
  frequency, binned. It goes in the README; it is the most convincing single
  artefact the project can produce.
- Report expected calibration error (ECE)

**Attribution.** SHAP values on the final model, for two purposes: a global
feature-importance plot for the README, and per-match attributions that feed the
coach in Phase 8. The coach's grounding depends on these being real.

---

## Things worth checking, beyond the headline number

- **Does it beat the rank baseline where ranks are equal?** Filter to matches
  where average ranks are within a tier of each other. This is the interesting
  subset — where matchmaking did its job and the residual signal is all there
  is. Performance here is the honest measure of whether the features add
  anything.
- **Which features actually matter?** If map × agent contributes nothing, say
  so in the README. A negative result reported clearly is a credibility signal.
- **Does performance decay over time?** Train on early data, test on progressively
  later slices. Meta shifts should show up as degradation, and quantifying it
  tells you how often the model needs retraining.
- **Where is it most wrong?** Errors clustered on a map, a rank band, or a comp
  point at a missing feature.
