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

| Ranked by | Picks the best of five | vs chance |
|---|---|---|
| Career ACS alone | **31.3%** | +11.3 |
| **Potential score (shipped)** | **30.5%** | +10.5 |
| Existing rating alone | 28.8% | +8.8 |
| Shuffled control | 19.1% | -0.9 |

1,500 teams, standard error 1.0 points. The score beats chance by **10.2
standard errors**, and the shuffled control lands on 20% as it must.

**The honest finding: ACS alone is as good.** The composite does not beat it --
31.3% against 30.5%, well inside noise, and Spearman agrees (+0.180 against
+0.179). A first draft that led with `rating` instead scored 29.0% on
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

Alongside the 0-100 score, the live view marks players whose band-relative
performance is high **and** whose account is young. Two conditions, because
either alone is noise.

The first is free: `rate_performance` already z-scores every component within
`band_of(tier)`, so a high `rating` literally means "better than others at this
rank". The second is what separates a smurf pattern from a good player.

### Why account level, and why not games played

Measured over 484,520 player-rows, mean ACS barely moves with account level --
209.6 under level 40 against 213.5 at level 300+ -- while mean tier moves a
lot: **6.7 against 19.9**. New accounts frag like veterans while ranked far
below them. That gap is the signal.

Games played looked like it belonged in the same test and does not.
`n_games` counts matches *this crawler has collected*, not matches the player
has played: on the training period **99.8% of players fall under 40 games and
the median is 3**. Including it made the second condition inert -- 9,033 of
9,067 accounts qualified -- which would have quietly reduced the flag to "top
5% of band-relative performance" and lost the distinction it exists to draw.
Only account level gates now.

### Calibrated, not guessed

`tools/build_perf_index.py` picks the cut that fires on 5% of the training
period -- about one player per two lobbies -- and stores it in
`models/perf_index.json`. It currently lands at **z >= 0.90**. Controlling the
*rate* is what matters; a hand-picked z-score would drift every time the
population shifted.

### Does it mean anything? Yes

No smurf label exists, so the flag cannot be checked against ground truth. The
nearest honest proxy: inside a complete ten-player lobby, how often does a
flagged player finish in the **top third** by actual performance? The base rate
is 33.3% by construction.

| | n | Top-third rate |
|---|---|---|
| **Flagged players** | 400 | **40.2%** |
| Everyone else | 8,245 | 29.4% |

**+10.9 points, 4.6 standard errors**, on the test period. The flag identifies
players who really do outperform their lobby.

Reproduce with `python tools/validate_potential.py --flag`.

### It does not help the win model

Two team-level features -- count of flagged players, and the largest
band-relative rating per side -- added to the 52 and scored on validation:

| | Log loss | AUC | delta |
|---|---|---|---|
| Incumbent | 0.6889 | 0.553 | — |
| + flag features | 0.6888 | 0.553 | −0.0001 |

Standard error 0.0016, so **null**, as expected in advance: the model already
carries `d_rating_max`, and a carry on either side is partly captured by it.
The flag is shipped as information for the reader, not as a model input.

### What it is not

It cannot distinguish a smurf from a returning player or someone mid-climb, and
the wording says so on screen: *"performing well above their rank -- possible
smurf"*, followed by a line making the ambiguity explicit. `rank_only_smurf`
-- high rank, ordinary output -- correctly does not fire, and neither does
`elite`, whose band-relative z of +2.56 is *higher* than the designed smurf's
+2.36 but sits on a mature account.

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
