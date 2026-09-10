# Which model, and what else was tried

Three questions get asked about this project often enough to answer once,
properly: *what does each model actually do*, *why the linear model rather than
the gradient booster*, and *what else could raise accuracy*. All three are
settled by measurement below.

Reproduce everything here with:

```bash
python tools/experiments.py --coverage --json       # what was tried, on validation
python tools/validate_potential.py --json           # the player score
python tools/validate_potential.py --flag --json    # the above-rank flag
python tools/model_metrics.py                       # every model, every metric, below
```

The experiments read the feature matrix and the database and never load or
overwrite `models/model.joblib`. **The test slice is scored once, by
`train.py`, and never used to choose.** `model_metrics.py` recomputes the
candidates' test predictions from the saved bundle to add precision and recall,
and first proves they are the same predictions by reproducing every recorded
log loss — so nothing in the tables below was selected on.

---

## The models, in plain terms

Every win model answers the same question — *what is the chance Blue wins?* —
from the same pre-match information, and differs only in how it combines it.

**Coin flip.** Always 50%. The floor: anything scoring below it is broken.

**Rank baselines** — *average rank* and *best player's rank*. One number each
(how far apart the teams are), turned into a probability by a fitted curve.
They answer: did matchmaking leave a rank gap worth betting on? Barely — rank
alone is almost a coin flip, because matchmaking exists to remove that gap.

**Single-stat baselines** — *average player rating* and *average ACS*. The same
idea with one performance statistic. ACS is included as a diagnostic, to test
whether the project's own rating measures anything ACS does not. It does not.

**Logistic regression — shipped.** All 52 features at once. Each is a
difference between the teams — Blue's average ACS minus Red's, best rating,
worst KAST, rank spread, win rates, map and agent history, roles, parties, form
— and each gets one learned weight. The weighted sum goes through a logistic
curve to become a probability. Three details matter: strong regularisation keeps
weights small on noisy data; there is no intercept and no centring, so swapping
the teams gives *exactly* `1 − p`; and every prediction can be explained exactly,
feature by feature, which is what the dashboard shows.

**Gradient boosting (LightGBM).** Hundreds of small decision trees built one
after another, each correcting the errors of those before it. It can learn
things the linear model cannot — interactions such as "good on this map *as
this agent*", and effects that level off — at the cost of needing more signal
to learn them from, and with no guarantee that swapping the teams mirrors the
answer.

**Margin regression.** Instead of the binary win/loss, it predicts the round
margin — 13-3 and 13-11 are the same win but very different evidence — and a
second small model, fitted on validation, turns that margin into a probability.

**Logistic + margin blend.** The average of the two probabilities, on the idea
that two models making different mistakes beat either alone.

**Calibrated variants** — *+ isotonic*, *+ Platt*. The logistic model and the
booster, with their probabilities reshaped by a curve fitted on validation, so
that "58%" wins 58% of the time. `train.py` tries both and keeps whichever is
better on validation; for both models that was isotonic.

**Tried and rejected on validation** — each described in its section below:
exact symmetry against the intercept fit it replaced; training on every match
both ways round; a learned strength per player (Bradley-Terry); admitting
matches with less known history; the above-rank flag as a feature; and a model
restricted to what the live client can observe.

**The two player-level models** answer a different question — not who wins, but
who plays well. The **potential score** blends ACS, the rating, K/D and a map
term into a 0–100 percentile. The **above-rank flag** marks players rated well
above their own rank who also top their lobbies far more often than chance.

## Why logistic regression is the one that ships

1. **It is tied for best.** The top four finish within one standard error of
   each other on 10,304 held-out matches — log loss 0.6872 to 0.6882, against
   an error of 0.0012. Choosing the lowest would be choosing noise: the order
   among them has changed from one retrain to the next.
2. **Among tied models, the simplest ships.** That is the one-standard-error
   rule, fixed in `train.py` before results were seen. Margin regression and
   the blend each add a second model and a validation-fitted link for a gain
   smaller than the noise.
3. **It is symmetric by construction.** Swap the teams and it mirrors to
   machine precision. The booster gives two identical teams different odds.
4. **It is already calibrated.** Expected calibration error 0.008. Adding an
   isotonic layer made held-out log loss *worse* (0.6893 against 0.6874) and
   had it call Blue in three matches out of four.
5. **It explains itself exactly.** Each feature's contribution is its weight
   times its standardised value, and they sum to the prediction. The dashboard
   shows those, not an approximation.

---

## Every model, every metric

<!-- metrics:start -->
Held-out test set: **10,304 matches**, of which Blue won 50.7%. Precision and recall treat *Blue wins* as the positive class at a 0.5 threshold; PR-AUC's chance level is Blue's win rate (0.507), not 0.5. Generated by `tools/model_metrics.py`.

| Model | What it does | Log loss | AUC | PR-AUC | Precision | Recall | F1 | Accuracy | Calls Blue |
|---|---|---|---|---|---|---|---|---|---|
| margin regression | Ridge regression on the round margin (13-3 vs 13-11), mapped to a probability on validation. | 0.6872 | 0.560 | 0.563 | 0.549 | 0.557 | 0.553 | 54.3% | 51.5% |
| logistic + margin blend | The average of the logistic and margin-regression probabilities. | 0.6873 | 0.560 | 0.563 | 0.551 | 0.551 | 0.551 | 54.4% | 50.8% |
| **logistic regression (shipped)** | A weighted sum of all 52 team-difference features, through a logistic curve. | **0.6874** | 0.560 | 0.562 | 0.550 | 0.542 | 0.546 | 54.3% | 50.0% |
| gradient boosting | LightGBM: many small decision trees over the same 52 features. Can learn interactions. | 0.6882 | 0.555 | 0.558 | 0.543 | 0.537 | 0.540 | 53.6% | 50.2% |
| logistic + isotonic | The logistic model, recalibrated by a stepwise fit on validation. | 0.6893 | 0.557 | 0.552 | 0.531 | 0.784 | 0.634 | 53.9% | 74.9% |
| gbm + isotonic | The booster, recalibrated by a stepwise fit on validation. | 0.6902 | 0.555 | 0.552 | 0.531 | 0.698 | 0.603 | 53.4% | 66.7% |
| acs alone | One feature, the gap in average combat score. A diagnostic, not a training candidate. | 0.6908 | 0.539 | 0.549 | 0.538 | 0.529 | 0.533 | 53.0% | 49.9% |
| avg rating (fitted) | One feature, the gap in average player rating. | 0.6914 | 0.532 | 0.542 | 0.529 | 0.524 | 0.527 | 52.2% | 50.3% |
| avg rank (fitted) | One feature, the gap in average rank, through a logistic curve. | 0.6928 | 0.507 | 0.517 | 0.509 | 0.544 | 0.526 | 50.2% | 54.3% |
| best player rank | One feature, the gap between each team's highest rank. | 0.6928 | 0.516 | 0.520 | 0.513 | 0.685 | 0.586 | 51.0% | 67.8% |
| coin flip | Always 50%. The floor. | 0.6931 | 0.500 | 0.507 | 0.507 | 1.000 | 0.673 | 50.7% | 100.0% |

**Tried on the validation period** (9,079 matches; the test set is never used to choose). A candidate had to beat the incumbent's log loss by more than one standard error (0.0012) to count. Generated by `tools/experiments.py --json`.

| Candidate | Log loss | vs incumbent | AUC | PR-AUC | Precision | Recall | F1 | Accuracy | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| logistic, exact symmetry (shipped) | 0.6863 | +0.0000 | 0.563 | 0.556 | 0.544 | 0.546 | 0.545 | 54.1% | null |
| logistic, intercept and centring | 0.6863 | -0.0000 | 0.563 | 0.556 | 0.544 | 0.546 | 0.545 | 54.0% | null |
| logistic, augmented | 0.6863 | -0.0000 | 0.563 | 0.556 | 0.544 | 0.545 | 0.545 | 54.1% | null |
| gradient booster | 0.6873 | +0.0009 | 0.558 | 0.552 | 0.537 | 0.535 | 0.536 | 53.3% | null |
| gradient booster, augmented | 0.6863 | -0.0001 | 0.563 | 0.555 | 0.544 | 0.547 | 0.545 | 54.0% | null |
| logistic, reduced train (C=0.02) (n=25,422) | 0.6870 | +0.0006 | 0.560 | 0.553 | 0.543 | 0.545 | 0.544 | 54.0% | null |
| logistic + Bradley-Terry (C=0.02) (26,839 fitted strengths) | 0.6870 | +0.0007 | 0.560 | 0.552 | 0.544 | 0.544 | 0.544 | 54.1% | null |
| Bradley-Terry strength alone (C=0.02) (diagnostic) | 0.6942 | +0.0079 | 0.495 | 0.504 | 0.498 | 0.493 | 0.495 | 49.5% | null |
| logistic, reduced train (C=0.1) (n=25,422) | 0.6870 | +0.0006 | 0.560 | 0.553 | 0.543 | 0.545 | 0.544 | 54.0% | null |
| logistic + Bradley-Terry (C=0.1) (26,839 fitted strengths) | 0.6869 | +0.0006 | 0.560 | 0.553 | 0.543 | 0.544 | 0.543 | 54.0% | null |
| Bradley-Terry strength alone (C=0.1) (diagnostic) | 0.7023 | +0.0159 | 0.495 | 0.504 | 0.500 | 0.497 | 0.499 | 49.7% | null |
| logistic, reduced train (C=0.5) (n=25,422) | 0.6870 | +0.0006 | 0.560 | 0.553 | 0.543 | 0.545 | 0.544 | 54.0% | null |
| logistic + Bradley-Terry (C=0.5) (26,839 fitted strengths) | 0.6869 | +0.0006 | 0.560 | 0.553 | 0.544 | 0.546 | 0.545 | 54.1% | null |
| Bradley-Terry strength alone (C=0.5) (diagnostic) | 0.7497 | +0.0634 | 0.495 | 0.503 | 0.504 | 0.498 | 0.501 | 50.0% | null |

Coverage sweep, on its own fixed validation rows (10,224, standard error 0.0011):

| Training threshold | Log loss | vs shipped | AUC | PR-AUC | Precision | Recall | F1 | Accuracy | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| train coverage >= 5 (shipped) (train 40,001) | 0.6875 | +0.0000 | 0.557 | 0.549 | 0.539 | 0.542 | 0.541 | 53.7% | null |
| train coverage >= 0 (train 52,292) | 0.6874 | -0.0001 | 0.557 | 0.550 | 0.540 | 0.540 | 0.540 | 53.8% | null |
| train coverage >= 3 (train 47,062) | 0.6873 | -0.0002 | 0.558 | 0.550 | 0.540 | 0.540 | 0.540 | 53.8% | null |
| train coverage >= 4 (train 43,655) | 0.6874 | -0.0001 | 0.558 | 0.549 | 0.541 | 0.541 | 0.541 | 53.9% | null |
| train coverage >= 6 (train 35,307) | 0.6877 | +0.0002 | 0.556 | 0.547 | 0.537 | 0.539 | 0.538 | 53.5% | null |

**The player score** ranks the five players on a team; the question is who had the best game. Measured on 1,500 complete test-period teams. Picking one player per team, precision and recall are the same number -- the top-1 rate, chance 20%. AUC and PR-AUC score every player against *had their team's best game* (base rate 0.200). Generated by `tools/validate_potential.py --json`.

| Ranked by | Top pick right (precision = recall) | AUC | PR-AUC |
|---|---|---|---|
| potential score | 30.5% | 0.607 | 0.292 |
| existing rating alone | 29.9% | 0.602 | 0.289 |
| career ACS alone | 30.3% | 0.607 | 0.292 |
| shuffled (control) | 20.3% | 0.506 | 0.207 |

**The above-rank flag** is a yes/no call, checked against *finished in the top third of their lobby* across 8,184 players in 952 test-period lobbies (base rate 0.299). AUC scores the rank-relative rating the flag thresholds. Generated by `tools/validate_potential.py --flag --json`.

| Flags | Precision | Recall | F1 | AUC (rating z) | PR-AUC |
|---|---|---|---|---|---|
| 4.9% of players | 0.440 | 0.072 | 0.124 | 0.588 | 0.383 |
<!-- metrics:end -->

### Reading precision and recall here

This is a two-sided problem, so which side counts as "positive" is arbitrary,
and precision and recall inherit that arbitrariness where AUC and log loss do
not. Three things follow:

- **A model that calls Blue about half the time has precision close to recall.**
  The shipped model calls Blue in 50.0% of matches, and its precision (0.550)
  and recall (0.542) sit together.
- **A model that leans to one side buys recall with precision.** The calibrated
  logistic model calls Blue in 74.9% of matches, which lifts its recall to 0.784
  while its precision falls to 0.531 and its log loss gets worse. The coin flip,
  calling Blue every time, has perfect recall and the base-rate precision. High
  recall here is a symptom, not an achievement.
- **PR-AUC's chance level is Blue's win rate, 0.507** — not 0.5 — so the shipped
  model's 0.562 is about as far above chance as its AUC of 0.560 is.

Log loss decides, because the product is a probability. AUC, precision and
recall are here for completeness, and because a reader will ask.

---

## Why the linear model, in detail

Three independent arguments. The first rules out the obvious objection;
the other two are what actually decide.

**1. The held-out test set does not favour the booster.** Measured once, on
10,304 matches:

| | Log loss | AUC | Accuracy (95% CI) |
|---|---|---|---|
| **Logistic regression** | **0.6874** | **0.560** | **54.3% ± 1.0%** |
| Gradient boosting | 0.6882 | 0.555 | 53.6% ± 1.0% |

Read that honestly. Logistic regression is not even the lowest log loss of the
ten candidates — margin regression is, at 0.6872 — and the booster trails
logistic by 0.0008 against a standard error of 0.0012, two thirds of one.
Logistic now leads on accuracy by 0.7 points, also inside the noise; on the
previous bundle the booster led by 0.4, which is exactly what noise looks like
when you retrain. The test set says these are indistinguishable; it does not
crown the linear model. Arguments 2 and 3 are what decide.

Every figure in this table moves with each retrain. `python tools/audit.py`
re-derives them from `reports/results.json` and names any that have drifted.

**2. The one-standard-error rule selected it.** Four models finish
statistically tied. Consecutive runs crowned different winners on the same
data, because the gaps are smaller than the noise — so the rule ships the
*simplest* model within one standard error of the best, not whichever happened
to come first. See `train.py`.

**3. The booster is not symmetric, and that is a correctness failure.**
Swapping the two teams must mirror the prediction: `P(A) + P(B) == 1`. The
feature vector negates exactly (measured `0.00e+00`), so any asymmetry is
purely the model. On 9,079 validation rows:

| Model | Mean mirror error |
|---|---|
| **Logistic as now shipped** | **8.6e-18** — machine zero |
| Logistic with intercept and centring | 5.2e-04 |
| Gradient booster, augmented | 9.2e-03 |
| Gradient booster | 2.4e-02 |

Trees carry no symmetry constraint, and this one learned a side preference from
noise. It gets identical teams wrong — 49.1% instead of 50.0%. Earlier bundles
had it at 48.3% and at 51.4%, favouring first one side and then the other,
which is what a preference learned from noise looks like when you retrain it.

---

## The one change worth making: exact symmetry — now shipped

The logistic used to miss the mirror by 1.8e-03. Not because of the model —
because of two implementation details:

- `StandardScaler` subtracts a non-zero training mean, so the *standardised*
  vector does not negate even though the raw one does.
- The intercept adds a constant that does not negate either.

Remove both — `StandardScaler(with_mean=False)` and `fit_intercept=False` —
and `P(A) + P(B) == 1` holds **by construction** rather than to a documented
tolerance:

| | Log loss | delta | Mirror error |
|---|---|---|---|
| Logistic, with intercept and centring | 0.6863 | — | 5.15e-04 |
| Logistic, exact symmetry (shipped) | 0.6863 | +0.0000 | **8.61e-18** |

Measured on the current validation slice; when the change was made, on less
data, the old fit missed the mirror by 1.76e-03 at the same zero cost.

Free. It costs nothing measurable in log loss and turns an approximate
invariant into an exact one. This is a **correctness** win, not an accuracy
one, and it is the only change here that earned its place.

**Shipped.** `fit_logistic` now fits this way, and the retrained bundle was
verified against the full sandbox catalog:

| | Before | After |
|---|---|---|
| Identical teams | 0.500130 | **0.5000000000** |
| Worst mirror error, 159 scenarios | ~1.8e-03 | **1.1e-16** |

`test_mirrored_probabilities_sum_to_one` now asserts `< 1e-12` instead of
`< 1e-3`, so a change that reintroduced an intercept would fail the suite
rather than pass quietly.

One caveat on reading the numbers below against the current README: this
change shipped alongside a retrain that picked up roughly 10,000 matches
collected since the previous run, so the test-set figures moved for reasons
that have nothing to do with symmetry. The validation measurement above
(+0.0000, same data, same split) is the one that isolates it.

---

## What was tried and did not work

All deltas are validation log loss against the incumbent — the shipped fit —
on 9,079 validation matches, standard error **0.0012**. Nothing below clears
it, so everything below is **null**. Recorded anyway — a null result that says
*why* is worth more than an unreported one. The full table, with AUC,
precision and recall for each, is in *Every model, every metric* above.

### Bradley-Terry player strength — the idea with the best odds, and it failed

Give every player one strength learned from *who actually won*, rather than
from ADR/KAST/ACS, which are only a proxy for skill. Team strength is the sum
of five, and `P(Blue) = sigmoid(S_blue - S_red)`. It is the family behind Elo,
and it is the textbook model for team-vs-team outcomes. Implemented in
`valwr/model/strength.py`; fitting is ordinary logistic regression on a sparse
±1 design matrix, one column per player.

It does not work here, and the diagnostic says exactly why:

```
C=0.02  strength alone: auc 0.495  | weight the model gave it: sum -0.0020
C=0.1   strength alone: auc 0.495  | weight the model gave it: sum +0.0072
C=0.5   strength alone: auc 0.495  | weight the model gave it: sum +0.0111
                                     (mean |weight| on the other 52: 0.032)
```

**Strength on its own is AUC 0.495 — no better than a coin flip**, across all
26,839 fitted strengths. So this is not redundancy with the existing features;
there is no signal in it at all. The cause is sparsity, and it is structural rather than a snapshot: **53% of the players in this dataset appear in exactly one
match**, and only 17% appear five or more times. A
player seen once has no learnable strength. The downstream model correctly
assigns the two columns a near-zero weight, and the model with them scores the
same as the same model without them, trained on the same rows (0.6869–0.6870
against 0.6870).

Two things kept this honest rather than accidentally impressive:

- **Strengths are fitted only on matches before the boundary.** A strength that
  absorbed a test match would hand the model the answer.
- **Training features use an expanding window.** A training match's own outcome
  contributes to its players' strengths, so a single fit would make those
  features partly self-fulfilling — the model would over-trust them and then
  fail on validation, which reads as "Bradley-Terry does not work" when the
  real fault was the construction. Rows before the first cut are dropped, so
  the incumbent is refitted on the same reduced set for the comparison.

`valwr/model/strength.py` is **not used by the shipped model.** It is kept
because the negative result is reproducible and worth being able to re-run.

This partially revisits the decision in `valwr/rating/rating.py` to prefer a
hand-weighted composite over a learned rating. That decision stands, and now
has evidence behind it rather than only a preference.

### Symmetric augmentation — works as theory says, changes nothing

Train on every match twice, as `(X, y)` and `(-X, 1-y)`.

| | Log loss | delta | Mirror error |
|---|---|---|---|
| Logistic, augmented | 0.6863 | −0.0000 | 8.0e-18 |
| Booster, augmented | 0.6863 | −0.0001 | 9.2e-03 |
| Booster, plain | 0.6873 | +0.0009 | 2.4e-02 |

For the logistic it is redundant — the no-intercept fit already gives exact
symmetry more cheaply. For the booster it cuts mirror error by more than half
and brings its log loss level with the logistic's — which says the asymmetry
was costing the booster something real. Level is all it reaches, though, and
even augmented it remains ~15 orders of magnitude worse on the mirror than the
symmetric logistic.

### Coverage threshold — the current value is already right

A match is only used if at least 5 of its 10 players have prior history, which
drops 12,291 of 52,292 training-period matches. Does admitting the rest help?
The evaluation set is held **fixed** at coverage ≥ 5 while only the training
threshold moves — sweeping both would change the validation rows at every step
and make the log losses incomparable.

| Training threshold | Train rows | Log loss | delta |
|---|---|---|---|
| ≥ 0 | 52,292 | 0.6874 | −0.0001 |
| ≥ 3 | 47,062 | 0.6873 | −0.0002 |
| ≥ 4 | 43,655 | 0.6874 | −0.0001 |
| **≥ 5 (shipped)** | 40,001 | 0.6875 | — |
| ≥ 6 | 35,307 | 0.6877 | +0.0002 |

Standard error 0.0011; nothing clears it. Admitting low-coverage matches is
very slightly positive and well inside noise; tightening to 6 is mildly
negative. **5 is fine, and now that is measured rather than assumed.**

### Not attempted, with reasons

XGBoost, CatBoost, random forests and small neural networks were considered and
skipped deliberately. Four models already tie within one standard error, and
the *entire* spread from coin flip (0.6931) to best (0.6872) is 0.0059 log
loss. LightGBM already represents gradient boosting here and does not beat the
logistic; XGBoost and CatBoost are the same family with different defaults. A
neural network needs far more signal than AUC 0.560 offers. Adding them would
lengthen the results table without changing any conclusion.

---

## The per-player potential score

Separate question, separate model. The win model asks which team wins; this
asks **who on your team is likely to play best**, and prints a 0-100 score per
teammate during a live match (`valwr/rating/potential.py`).

Four components, z-scored against the training population and blended:
ACS, the existing composite rating, K/D, and `map_edge` -- how much better this
player is *on this map* than they are in general. `map_edge` is deliberately a
delta rather than a level: an absolute map rating would mostly restate overall
skill, which `rating` already carries, and a player with no history on the map
scores exactly 0 on it rather than being guessed at.

The 0-100 number is a **percentile against the training period**, so 70 means
"likely to outperform 70% of players".

### Measured, because "likely to play best" is a prediction

The claim the live table makes is precisely: *the player at the top of this
list will have the best game*. So that is what gets measured -- inside real
five-player teams from the **test** period, with chance at exactly 20%.

| Ranked by | Picks the best of five | vs chance | AUC |
|---|---|---|---|
| **Potential score (shipped)** | **30.5%** | +10.5 | 0.607 |
| Career ACS alone | 30.3% | +10.3 | 0.607 |
| Existing rating alone | 29.9% | +9.9 | 0.602 |
| Shuffled control | 20.3% | +0.3 | 0.506 |

1,500 teams, standard error 1.0 points. The score beats chance by **10.1
standard errors**, and the shuffled control lands on 20% as it must. AUC scores
every player against *had their team's best game*.

**The honest finding: ACS alone is as good.** The composite does not beat it --
30.5% against 30.3%, well inside noise; the AUCs are identical, and Spearman
agrees (+0.194 against +0.191). A first draft that led with `rating` instead scored 29.0% on
validation, worse than the single feature it was built on top of.

Weights were chosen on the **validation** period
(`tools/validate_potential.py --sweep`, which refuses to run on test), and test
was scored once afterwards. Every candidate weighting from "ACS only" down to
"ACS + rating" landed within one standard error of the best, so the components
are close to interchangeable for ranking.

They are kept anyway, for a reason that is not accuracy: they are what turns a
bare number into "wins duels" or "strong on this map". A single ACS figure
cannot say why. That is a presentation argument, and it is labelled as one
rather than dressed up as a modelling gain.

### It is biased by role, and that is measured

The score is ACS-led, and ACS depends heavily on which role a player mains.
Measured across 484,520 player-rows:

| Role | Mean ACS | Mean K/D | Rows |
|---|---|---|---|
| Duelist | **225.3** | 1.12 | 182,648 |
| Controller | 215.8 | 1.03 | 101,758 |
| Sentinel | 200.3 | 1.10 | 106,805 |
| Initiator | **194.2** | 1.03 | 93,309 |

A 31-point ACS spread against a population standard deviation of 20.8 -- one
and a half standard deviations of pure role.

What that does to the score, two ways:

- **In the wild**, grouping 4,000 test-period players by the role they main:
  duelist mains average **54.1**, initiator mains **41.3**. A 12.8-point gap.
- **In isolation**, holding ability exactly fixed and varying only the
  measured per-role ACS and K/D (`potential_role_bias` in the sandbox):
  **52 points**, duelist 72 against initiator 20.

Both numbers are real and they measure different things. 52 is the pure effect
at identical skill; 12.8 is what survives once individual variation is mixed
back in. Either way a strong Sova main will often rank below a mediocre Reyna
main, and the score should not be read as "who is better at the game".

The sandbox scenario asserts the gap stays between 30 and 70 points -- two-
sided on purpose. The bias must not be quietly papered over, and it must not
silently grow either.

### What it is not

Ranking the top player correctly 30.5% of the time is a real edge over 20% and
a long way from reliable. Individual performance is noisy and strongly
mean-reverting. The live output says so on screen rather than only here.

---

## The "playing above their rank" flag

Alongside the 0-100 score, the live view marks players who are **both**
performing above their own rank band **and** topping their lobbies far more
often than chance. Two conditions, because either alone is weaker.

The first is free: `rate_performance` already z-scores within `band_of(tier)`,
so a high `rating` literally means "better than others at this rank".

### The second condition was wrong the first time

The first version gated on account level, on the intuition that smurfs play on
fresh accounts. Raced on held-out data against a 29.7% base top-third rate:

| Signal | Fires on | Top-third rate | Lift |
|---|---|---|---|
| Account level < 100 | 1,103 | 29.6% | **−0.2** |
| Band-relative z ≥ 0.90 | 1,077 | 39.0% | +9.3 |
| z ≥ 0.90 AND level < 100 *(first version)* | 234 | 39.7% | +10.0 |
| Lobby dominance ≥ 0.50 | 444 | 44.1% | +14.4 |
| **z ≥ 0.90 AND dominance ≥ 0.50** *(shipped)* | 337 | 46.3% | **+16.5** |
| Headshot % ≥ 0.30 | 1,999 | 33.3% | +3.5 |
| Rank climb ≥ 3 tiers | 135 | 31.1% | +1.4 |
| Performance consistency | 700 | 31.1% | +1.4 |

**Account level does nothing on its own — −0.2 points.** The intuition it was
built on is simply wrong for this data. What an irregular account actually
looks like is in the *match history*: finishing top-2 of a ten-player lobby far
more often than one game in five. Rank climb and headshot percentage, both
plausible-sounding, are noise.

`temporal.lobby_dominance` computes it, time-gated like everything else in that
module. Population mean is 0.20 exactly — the 2-in-10 chance rate — which is
the sanity check that it is computed correctly.

### One guard that matters

A match only counts toward dominance if we hold at least `MIN_LOBBY` players
from it. Without that, a partially-scraped match counts as a win by default,
and a *synthetic* solo match scores a perfect 1.00 — you cannot top a lobby of
one. It costs nothing on real data, where 100% of collected matches have all
ten players, and it is the difference between `elite` correctly not flagging
and `elite` being reported as a smurf.

A consequence worth stating: **the sandbox cannot exercise this half of the
flag.** Synthetic players have no lobby-mates, so dominance is undefined for
them and no synthetic player flags. `test_the_sandbox_cannot_measure_dominance_and_says_so`
asserts exactly that rather than pretending otherwise; the dominance half is
verified on held-out real lobbies.

### Calibrated, not guessed

`tools/build_perf_index.py` picks the cut firing on 5% of the training period
— about one player per two lobbies — and stores it in the index. It currently
lands at **z ≥ 0.70**. Controlling the *rate* is what matters; a hand-picked
z-score drifts as the population shifts.

### Does it mean anything? Yes

No smurf label exists, so the flag cannot be checked against ground truth. The
nearest honest proxy: inside a complete ten-player lobby, how often does a
flagged player finish in the **top third** by actual performance? Base rate is
33.3% by construction.

| | n | Top-third rate |
|---|---|---|
| **Flagged players** | 400 | **44.0%** |
| Everyone else | 7,784 | 29.2% |

**+14.8 points, 6.3 standard errors** on the test period, across 952 lobbies.
Read as a classifier of "finished in the top third", its **precision is 0.44
and its recall 0.07** — deliberately: it fires on about one player in twenty,
so it can only ever catch a small share of strong games, and it is meant to
be right when it does. Reproduce with
`python tools/validate_potential.py --flag`.

### It does not help the win model

Two team-level features — count of flagged players, and the largest
band-relative rating per side — added to the 52 and scored on validation:

| | Log loss | AUC | delta |
|---|---|---|---|
| Incumbent | 0.6889 | 0.553 | — |
| + flag features | 0.6888 | 0.553 | −0.0001 |

Standard error 0.0016, so **null**, as expected in advance: the model already
carries `d_rating_max`. The flag is information for the reader, not a model
input — it tells you a carry is in the lobby, it does not make the win
prediction better.

### What it is not

It cannot distinguish a smurf from a returning player or someone mid-climb, and
the output says so on screen rather than only here.

---

## Replaying the live path (Phase 9)

Everything else here exercises the *training* path. That left the code that
actually runs in a match verified by whatever games got played -- five, at the
time this was written. `tools/backtest_live.py` replays real matches exactly as
the client would deliver them: ten puuids, teams and agents, nothing else.

### It found a leak before it produced a single number

`live/predict._roster_rows` read each player's most recent rank and account
level with **no `as_of` filter**. Live that is harmless, because nothing later
than now exists -- which is why it survived review. In a replay it reads the
player's *future* rank, so the backtest would have quietly flattered itself.
Every other read in this project goes through the temporal layer for exactly
this reason; this one did not. Now time-gated, with a regression test.

### The two paths disagree on inputs, and here is why

Replaying 2,500 held-out matches, **2,487 produced a different probability**
from the training path, the largest gap being 28 percentage points. The cause
is precise: seven of the fifty-two features cannot be observed correctly at
inference.

| Feature | Why the live path cannot have it |
|---|---|
| `d_tier_mean/max/min/std` | the client does not expose rank; we substitute the last rank seen, or nothing |
| `d_account_level_mean` | same |
| `d_max_party`, `d_n_grouped` | party is not exposed pre-match **at all** -- every player is sent as unpartied, always |

### But it costs almost nothing, which was the surprise

| Path | Log loss | AUC | Accuracy |
|---|---|---|---|
| Training | 0.6893 | 0.549 | 53.6% |
| Live | 0.6922 | 0.548 | **53.8%** |

On 2,500 matches: **+0.0029 log loss (about one standard error), no AUC gap,
and marginally better accuracy.** A smaller 800-match run suggested a large
AUC penalty (0.573 against 0.584); that was sampling noise and does not
survive the bigger sample.

The reason the inputs can diverge so much while the outputs barely move is
that those seven features carry almost no weight -- dropping them from
training costs +0.0007 log loss, itself null against a 0.0016 standard error.

### A train/serve-parity model is not better either

Retraining on the 45 live-available features and scoring **through the live
path** gives 0.6899 against the shipped model's 0.6923 -- directionally right,
but −0.0024 against a 0.0029 standard error, so null. Worth revisiting if the
feature set ever changes; not worth a retrain today.

---

## Does more data help? Measured, and no

Every null result here concluded the same thing: the ceiling is the data and
the domain, not the estimator. That claim went untested until the crawler ran
long enough to grow the dataset **24%** — 45,854 resolved matches to 56,825 —
and the model was refitted on it.

| | Before | After |
|---|---|---|
| Training matches | 22,020 | **29,040** |
| Test matches | 6,132 | **7,677** |
| Log loss | 0.6883 | **0.6875** |
| AUC | 0.558 | **0.557** |
| Accuracy | 54.3% ± 1.2% | **53.4% ± 1.1%** |

Log loss improved by 0.0008 against a standard error of **0.0013**. AUC moved
−0.001. **Null.** A quarter more data did not move the model.

That is the strongest evidence yet that the limit is the problem, not the
pipeline: matchmaking exists to make these games close, and it succeeds. More
of the same data buys precision on the estimate, not a better estimate.

**It held a second time.** A week later the crawler had grown the training set
another 38%, to 40,001 matches, with a 10,304-match test set. Log loss went
from 0.6875 to 0.6874 — a change of 0.1 standard errors. Accuracy rose to 54.3%
and AUC to 0.560, but on a newer and larger test set those are not like for
like, and log loss, the metric the model is selected on, did not move. Two
independent growth runs, both null.

Two things worth noting from the first run. The one-standard-error rule still ships
logistic regression, now with **5 models tied** rather than 4. And the coverage
gradient came back **non-monotonic again** (5-6: 0.551, 7-8: 0.543, 9-10:
0.572) after being monotonic on the previous bundle — the flip-flop across
retrains is itself the evidence that the earlier retraction of that finding was
right.

### The live-path gap is not stable either

Re-running the Phase 9 replay on the new bundle:

| Path | Log loss | AUC | Accuracy |
|---|---|---|---|
| Training | 0.6885 | 0.547 | 52.3% |
| Live | 0.6941 | 0.535 | 52.2% |

That is +0.0056 log loss, about **two standard errors**, where the previous
bundle showed +0.0029 and no AUC gap. The gap moves between retrains, which is
consistent with it being small and noisy rather than a fixed penalty — but it
is not zero, and it is not shrinking. Recorded in both directions rather than
settling on whichever run flattered the conclusion.

---

## The map barely moves the score, and that is correct

Noticed in use: the per-player score "feels consistent rather than adapting to
the specific map". It is, and measurement says it should be.

**Map history does not predict map performance.** For 12,000 held-out
player-matches, comparing a player's map-specific history against how much
better or worse they then did than their overall history predicts:

| Map games in history | n | Spearman |
|---|---|---|
| 0 | 4,931 | +0.000 |
| 1 | 2,971 | −0.036 |
| 2 | 1,677 | +0.031 |
| 3 | 1,000 | −0.057 |
| 5 | 349 | +0.065 |
| 6+ | 488 | +0.017 |
| **all** | **12,000** | **−0.010** |

Signs flip at random, magnitudes are noise, and it stays flat even at six or
more games — so this is not merely thin history at the low end.

Two further facts explain what a user sees. **39.6% of players have zero games
on the current map** and the median is **one**, so shrinkage toward no-opinion
(`PRIOR_N_MAP = 6`) correctly erases most of what little is there: the median
contribution to the composite is 0.049 against a span of about 2.0, roughly two
percentile points.

### Turning it up makes the score worse

| `map_edge` weight | Picks the best of five |
|---|---|
| 0.00 | 29.4% |
| **0.15 (shipped)** | **29.9%** |
| 0.30 | 29.2% |
| 0.50 | 27.3% |
| 0.75 | 23.4% |

The shipped weight is already at its optimum, and at 0.75 the score is barely
above the 20% chance line. Dropping the component entirely is null (−0.3
against a 1.0 standard error), so it stays — but it earns its place by costing
nothing, not by contributing.

### What did change: the score no longer claims a map effect

`explain()` used to be able to print **"strong on this map"**. On a component
with no measurable predictive content that is a confident sentence the data
does not support, so `map_edge` was removed from the phrasing entirely. It
still contributes to the number; it is simply never given as the reason.

Removing it from the phrase table alone was not enough — `explain()` still
ranged over every weight when picking the largest deviation, so a map-dominated
player hit a `KeyError` in the live view. The candidate set is now the
narratable components, and a test asserts the map is never named even when
`map_edge` is the single largest deviation.

---

## Why the live score was frozen

Reported as "the rating is stagnant, the same 67-68 every map and every game".
It was not stagnant so much as **frozen**, and the cause was two bugs stacked
on each other, neither of which raised anything.

### The live path never stored what it fetched

`HenrikClient.matches()` fetches a matchlist and caches the raw body. It does
**not** normalise. `resolve` called it and then asked whether the player now
had history, on the strength of a comment reading *"the crawler normalises
inline, so a fetched player is queryable immediately"* -- true of
`collect/crawl.py`, false of the client.

So every live fetch spent an API call, wrote a blob nothing ever read, and left
the player exactly as unknown as before. Nothing caught it: the request
returned 200, the response was stored, no exception was raised, and lobby
coverage still looked plausible because the crawler had independently collected
most of those players by other means.

`normalize.ingest(conn, payload)` now parses one response in place, and
`resolve` calls it. Verified against the real cached response: 10 matches, 100
players, 0 errors, from a body that had been sitting unparsed.

### "Known" meant known *ever*, not known *recently*

`has_history` returns true for any stored row, so an account last seen in
August counted as known and was never refetched. The local account is the worst
case: crawled players sit at a median staleness of 0.6 days because the crawl
follows them, but nothing follows the person running the tool.

Measured on the reported account, score over time **before** the fix:

| | today | 7 days ago | 14 days | 21 days |
|---|---|---|---|---|
| Score | 77 | 77 | 27 | 40 |

It moved whenever data arrived and had not moved since 24 August. **After**
both fixes, with 28 matches instead of 18: 45, 67, 30, 40 -- and the current
value dropped from 77 to 45, because the frozen number had been flattering.

`resolve` now refreshes the local account unconditionally before any deadline
accounting, then stale others within the budget.

### The map gate

The same investigation found the score swinging 20 points across maps off 0-5
games -- 57 on Ascent from a single 9/17 game, 77 on Lotus from three. Since
map history has no measured predictive power, that movement was noise
presented as insight.

`MIN_MAP_GAMES = 6`: below it `map_edge` is exactly zero, no opinion rather
than a shrunk guess. Top-1 on the current index is 30.5% ± 1.0 — inside one
standard error of the 29.9% measured at a gate of 4 on less data — so the raise
costs no measurable signal. What it
buys is exposure: the map term now moves 1.4% of samples rather than 14.1%, at
the same strength when it does fire. Since map history has no measured
predictive power on its own (Spearman −0.010 over 12,000 player-matches),
touching fewer players is the right side to err on.

**Raising the gate exposed a worse bug, and briefly made it worse.** A gated
component is set to exactly 0.0, and the index was fitting `map_edge`'s
standard deviation across those zeros — 98.7% of the sample. That measures the
width of a spike at zero: 0.0042 against 0.039 over the players who actually
clear the gate, so the divisor was **9.4x too small**. Every player who cleared
the gate got a z-score of ±12 to ±18, and at 15% weight that one component
outweighed the other three combined. One real account read 74 on one map and 2
on another off nothing else; after the fix the same account reads 48 and 37.

The gate sweep could not see this. Top-1 ranks players *within a team*, where
nearly everyone is gated to zero, so a mis-scaled tail moves almost no
comparisons — the sweep reported 29.5% against 29.7% and called the gates
equivalent. They are equivalent on that metric and were not equivalent in the
scores people read.

`potential.fit_scales` now fits that scale on the gate-clearing subset and pins
the mean to exactly 0.0, so *gated implies no contribution* is an exact
property rather than an accident of where the sample mean landed. It needs 200
such samples to trust the subset; the index build had been drawing 15,000 rows,
which yielded 128, so it silently fell back to the contaminated scale. The
default draw is now 60,000, which yields about 546 on the current data.

### The card

A single repeated phrase was the whole explanation -- asked for a score on all
eight maps, the reason came back `"consistently strong"` eight times.
`potential.detail()` now returns the components with how far each sits from
average, the record and agents played on this map, recent form, and **how old
the data is**. That last line is the one whose absence hid all of the above: a
score computed from two-week-old history looked identical to a live one.

---

## The conclusion worth stating plainly

**Model class is not the bottleneck.** Every candidate above is null, and the
gap between a coin flip and the best model is 0.0059 log loss. The limit is the
data and the domain: Valorant's matchmaker is *designed* to produce even games,
so it actively suppresses the signal this project is trying to detect. An AUC
of 0.560 from pre-match public statistics may be close to the practical
ceiling.

The remaining levers are more data and better features — not a better
estimator.
