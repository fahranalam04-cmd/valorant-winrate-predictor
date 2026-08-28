# Which model, and what else was tried

Two questions get asked about this project often enough to answer once,
properly: *why the linear model rather than the gradient booster*, and
*what else could raise accuracy*. Both are settled by measurement below.

Reproduce everything here with:

```bash
python tools/experiments.py --coverage
```

It reads the feature matrix and the database, writes nothing, and never loads
`models/model.joblib`. The **test slice is deliberately not scored** — it was
touched once, by `train.py`. A candidate chosen by repeatedly consulting the
test set is just overfitting more slowly.

---

## Why the linear model

Three independent arguments, and they agree.

**1. It beats the booster on the held-out test set.** Measured once, on 6,132
matches:

| | Log loss | AUC | Accuracy |
|---|---|---|---|
| **Logistic regression** | **0.6883** | **0.558** | **54.3% ± 1.2%** |
| Gradient boosting | 0.6885 | 0.557 | 54.1% |

It is not the lowest log loss overall — margin regression reaches 0.6878 —
but the gap is a third of a standard error, and argument 2 is what settles
that.

**2. The one-standard-error rule selected it.** Seven models finish
statistically tied. Consecutive runs crowned different winners on the same
data, because the gaps are smaller than the noise — so the rule ships the
*simplest* model within one standard error of the best, not whichever happened
to come first. See `train.py`.

**3. The booster is not symmetric, and that is a correctness failure.**
Swapping the two teams must mirror the prediction: `P(A) + P(B) == 1`. The
feature vector negates exactly (measured `0.00e+00`), so any asymmetry is
purely the model. On validation rows:

| Model | Mean mirror error |
|---|---|
| **Logistic as now shipped** | **7.2e-18** — machine zero |
| Logistic as previously shipped | 1.8e-03 |
| Gradient booster, augmented | 1.1e-02 |
| Gradient booster | 3.1e-02 |

Trees carry no symmetry constraint, and this one learned a side preference from
noise. It gets identical teams wrong — 0.4831 instead of 0.5000. An earlier
bundle had it at 0.5138, favouring the *other* side by a similar margin, which
is what a preference learned from noise looks like when you retrain it.

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
| Logistic, with intercept and centering | 0.6893 | — | 1.76e-03 |
| Logistic, exact symmetry | 0.6893 | +0.0000 | **7.18e-18** |

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

All deltas are validation log loss against the incumbent, standard error
**0.0022**. Nothing below clears it, so everything below is **null**. Recorded
anyway — a null result that says *why* is worth more than an unreported one.

### Bradley-Terry player strength — the idea with the best odds, and it failed

Give every player one strength learned from *who actually won*, rather than
from ADR/KAST/ACS, which are only a proxy for skill. Team strength is the sum
of five, and `P(Blue) = sigmoid(S_blue - S_red)`. It is the family behind Elo,
and it is the textbook model for team-vs-team outcomes. Implemented in
`valwr/model/strength.py`; fitting is ordinary logistic regression on a sparse
±1 design matrix, one column per player.

It does not work here, and the diagnostic says exactly why:

```
C=0.02  strength alone: auc 0.506  | weight the model gave it: sum +0.0074
C=0.1   strength alone: auc 0.511  | weight the model gave it: sum -0.0179
C=0.5   strength alone: auc 0.515  | weight the model gave it: sum -0.0203
                                     (mean |weight| on the other 52: 0.039)
```

**Strength on its own is AUC 0.506–0.515 — barely above a coin flip.** So this
is not redundancy with the existing features; there is close to no signal in it
at all. The cause is sparsity: **56.8% of the 162,002 players in this dataset
appear in exactly one match**, and only 13.4% appear five or more times. A
player seen once has no learnable strength. The downstream model correctly
assigns the two columns a near-zero weight, and adding them still costs a
little (+0.0004) because they are two more noisy inputs.

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
| Logistic, augmented | 0.6895 | +0.0002 | 1.05e-17 |
| Booster, augmented | 0.6871 | −0.0021 | 1.05e-02 |
| Booster, plain | 0.6887 | −0.0006 | 3.08e-02 |

For the logistic it is redundant — the no-intercept fit already gives exact
symmetry more cheaply. For the booster it cuts mirror error threefold and
improves log loss by 0.0021, which lands *just* under the 0.0022 threshold and
so is still null. It confirms the asymmetry costs the booster something real,
but even augmented it remains ~15 orders of magnitude worse on the mirror than
the symmetric logistic.

### Coverage threshold — the current value is already right

A match is only used if at least 5 of its 10 players have prior history, which
discards roughly 25,000 of 44,838 resolved matches. Does admitting the rest
help? The evaluation set is held **fixed** at coverage ≥ 5 while only the
training threshold moves — sweeping both would change the validation rows at
every step and make the log losses incomparable.

| Training threshold | Train rows | Log loss | delta |
|---|---|---|---|
| ≥ 0 | 31,591 | 0.6897 | −0.0002 |
| ≥ 3 | 26,984 | 0.6897 | −0.0002 |
| ≥ 4 | 24,251 | 0.6899 | +0.0000 |
| **≥ 5 (shipped)** | 21,570 | 0.6899 | — |
| ≥ 6 | 18,291 | 0.6904 | +0.0005 |

Standard error 0.0015; nothing clears it. Admitting low-coverage matches is
very slightly positive and well inside noise; tightening to 6 is mildly
negative. **5 is fine, and now that is measured rather than assumed.**

### Not attempted, with reasons

XGBoost, CatBoost, random forests and small neural networks were considered and
skipped deliberately. Seven models already tie within one standard error, and
the *entire* spread from coin flip (0.6931) to best (0.6878) is 0.0053 log
loss. LightGBM already represents gradient boosting here and loses to the
logistic; XGBoost and CatBoost are the same family with different defaults. A
neural network needs far more signal than AUC 0.558 offers. Adding them would
lengthen the results table without changing any conclusion.

---

## The conclusion worth stating plainly

**Model class is not the bottleneck.** Every candidate above is null, and the
gap between a coin flip and the best model is 0.0053 log loss. The limit is the
data and the domain: Valorant's matchmaker is *designed* to produce even games,
so it actively suppresses the signal this project is trying to detect. An AUC
of 0.558 from pre-match public statistics may be close to the practical
ceiling.

The remaining levers are more data and better features — not a better
estimator.
