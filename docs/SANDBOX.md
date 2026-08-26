# The prediction sandbox

A synthetic environment for asking whether the predictor **behaves sensibly**
across the space of plausible pre-match situations.

```
held-out test set  ->  does it generalise to real matches?
static sandbox     ->  does it behave sensibly on exact benchmark cases?
variance sandbox   ->  does that behaviour survive realistic noise?
```

## What it does not do

**The sandbox does not measure predictive accuracy, and no result from it is
evidence of real-world performance.** Every scenario is invented. A model can
be perfectly coherent — symmetric, monotonic, well-behaved at the boundaries —
and still be wrong about actual VALORANT matches.

Held-out evaluation on real historical matches remains the only source of truth
on predictive quality. That number is in the README, measured on matches the
model never saw.

What the sandbox *can* establish: that the system is internally consistent,
that a feature does what its name suggests, that noise does not flip a
conclusion, that unknown players are treated as unknown, and that a change to
the model moved what you thought it moved.

---

## Running it

```bash
python -m valwr.sandbox list                       # scenarios and coverage
python -m valwr.sandbox run --scenario fair_match  # one scenario, detailed
python -m valwr.sandbox run --scenario all         # the whole catalog
python -m valwr.sandbox run --scenario carry       # a whole category
python -m valwr.sandbox run --mode variance --scenario single_smurf \
                           --samples 1000 --seed 42
python -m valwr.sandbox sweep --feature rating --csv
python -m valwr.sandbox grid  --pair rating,wr_map
python -m valwr.sandbox benchmark
python -m valwr.sandbox compare
```

`--model` selects the estimator: `logistic` (shipped), `gbm`, `margin`,
`baselines`, `all`, or a comma-separated subset. Comparing `logistic` against
`gbm` is the most informative single thing the sandbox does — see *Which model
to probe* below.

`--generated` includes the ~184 machine-generated sweep, grid and context
scenarios alongside the 159 curated ones.

**Cost.** The static catalog runs in about 19 seconds. Variance rebuilds the
whole synthetic world for every sample, so 1,000 samples is roughly a minute
*per scenario* — `--scenario all --samples 1000` is hours, not minutes. Use a
category or a single scenario, or drop `--samples`.

---

## Architecture

```
profiles  ->  world  ->  features/build.build_match  ->  model  ->  probability
```

The important property is that only the first two boxes are sandbox code. The
rest is production.

| Module | Role |
|---|---|
| `schema.py` | `PlayerProfile`, `TeamProfile`, `MatchScenario`, `VarianceSpec`, results |
| `profiles.py` | ~28 canonical archetypes |
| `world.py` | profiles → synthetic history in an in-memory SQLite |
| `predictor.py` | adapter over `models/model.joblib` |
| `scenarios.py` | 159 curated scenarios in 21 categories |
| `sweeps.py` | generated single-factor, pairwise and context scenarios |
| `variance.py` | correlated stochastic realisations |
| `runner.py` | execution, auto-mirroring |
| `report.py` | human-readable and batch output |
| `benchmark.py` | save / compare |

### Profiles declare history, not features

A `PlayerProfile` says *"60 games, 50% win rate, 212 ACS, 12 of those games on
this map"*. It does **not** say what `d_wr_mean` should be.

That is the whole design. Declaring features directly would be easier and would
prove nothing: it would bypass shrinkage, recency weighting, the rating
composite and the map/band normalisation — precisely the machinery worth
testing. A profile declaring *"3 games, 100% win rate"* exists so that the
pipeline can be observed shrinking it toward the prior.

`world.py` materialises a profile into real `match_players` rows using the same
`store/normalize.upsert_*` functions the crawler uses, then
`features.build.build_match(..., require_outcome=False)` — the same call the
live client makes — reads them back. **No feature formula is duplicated
anywhere in the sandbox.**

### Why an in-memory database

It keeps the production path intact while guaranteeing isolation. The
connection is always `:memory:`, asserted by a test, so the sandbox cannot read
or write the real crawl, cannot leak a real PUUID, and needs no network.

**Sandbox data must never enter training.** It is not appended to
`features.parquet` or the real database, and the model is never fitted on it.
Training a model on synthetic expectations would teach it to agree with our
assumptions, which is the opposite of a test.

---

## Scenario taxonomy

159 curated scenarios across 21 categories:

| Category | n | Probes |
|---|---|---|
| `boundary` | 15 | domain edges — 0% and 100% rates, tier extremes, zero and huge histories |
| `contradiction` | 12 | signals that disagree; the highest-value group |
| `coverage` | 11 | 5/5 known down to 0/5, and asymmetric coverage |
| `shrinkage` | 10 | 3 games at 100% versus 500 at 55%, per rate family |
| `rank`, `form`, `composition`, `party` | 9 each | rank gaps and contradictions, streaks, comps, stack shapes |
| `map`, `agent`, `rust` | 8 each | map and agent comfort, inactivity |
| `carry` | 7 | smurfs, hidden smurfs, rank-only smurfs, double carries |
| `weak_link`, `distribution` | 6 each | liabilities; equal means with different spreads |
| `skill`, `map_agent`, `experience`, `off_role`, `dominance` | 5 each | graded gaps, the interaction in isolation, level-vs-skill |
| `cancellation` | 4 | deliberately offsetting advantages |
| `sanity` | 3 | identical teams, equal-but-different, mirrors |

Plus **184 generated** scenarios: 105 single-factor sweeps (21 features × 5
levels), 72 pairwise grid cells (8 curated pairs × 3×3), and 7 context sweeps
across maps.

Every scenario also gets a **mirrored counterpart generated automatically**.
Hand-writing mirrors would create two copies that eventually drift apart, and
the drift would look like a model finding.

---

## Feature completeness

There is exactly one authoritative feature list and it lives in production:
`features.player.FEATURE_NAMES` and `features.team.feature_names()`. The
sandbox imports it and never keeps a copy.

`sweeps.KNOBS` maps every player feature to a way of moving it.
`sweeps.ROSTER_LEVEL` lists the team-level features that have no single-player
knob — composition, party, coverage — each naming the category that exercises
it. That is an allow-list with reasons, not a silent pass.

`test_every_production_feature_is_covered` asserts nothing is missing.
**Adding a production feature without sandbox coverage fails the build.**

### Adding a feature

1. Add it to the production feature list as usual.
2. `pytest` fails, naming it.
3. Add a knob to `sweeps.KNOBS`, or an entry to `ROSTER_LEVEL` with the
   category that covers it.

### Adding a scenario

Add it to the relevant section of `scenarios.build()`. The mirror, the
validation and the benchmark entry are all automatic.

---

## Variance

Static scenarios are exact benchmarks. Variance asks whether the conclusion was
balanced on a knife edge.

**Correlated, not independent.** Perturbing each statistic on its own produces
players who cannot exist — elite damage with terrible KAST, a rising trend with
a collapsing win rate. Each player draws latent shocks instead:

| Shock | Moves |
|---|---|
| `skill` | ACS, ADR, KAST, first-blood rate, win rate |
| `form` | recent win rate, trend |
| `map` | map win rate |
| `agent` | agent win rate, map × agent win rate |
| `noise` | small independent per-statistic error |

Loadings are in `variance.SKILL_LOADING`. First-death rate loads *negatively* on
skill, because better players die first less often.

These are judgement calls, not fitted values. The goal is plausible variance,
not a generative model of VALORANT, and being crude on purpose is better than
pretending otherwise.

**Bounded.** Rates clip to [0, 1], counts stay integers, tier jitters by one
step, and the count hierarchy (map ⊆ games, map×agent ⊆ both) is maintained.

**Identity is preserved.** A realisation of `single_smurf` where the smurf is
the worst player in the lobby is not a noisy version of that scenario — it is a
different scenario. A `VarianceSpec` may declare a predicate, and realisations
are rejection-sampled against it. If no valid draw is found within
`max_attempts`, the exact scenario is returned rather than a broken one; that
shows up as unusually low variance, not as missing samples.

**Robustness labels** — `stable`, `moderately sensitive`, `highly sensitive` —
come from thresholds on standard deviation and favourite-flip rate. They exist
to sort a long report. They are a developer heuristic and not a scientific
claim.

---

## Three kinds of expectation

Treating these identically is how a test suite becomes theatre.

**Mechanical invariants — these fail `pytest`.**
Deterministic construction, valid domains, no NaN or infinity, probability in
[0, 1], antisymmetric features negating exactly, mirrored probabilities summing
to 1 within tolerance, seeded reproducibility, feature completeness, and the
sandbox never touching disk or network.

**Directional expectations — reported, never enforced.**
"A stronger team should be favoured." These print as `MODEL WARNING` in the
report and set a non-zero exit only under `--strict`. **If the trained model
violates a reasonable directional assumption, that is surfaced, not tuned
away.** Three currently fail; see below.

**Observational comparisons — reported only.**
Composition and party effects, where no universal expected direction is
defensible. Tagged `observational`; the report shows the number and declines to
judge it.

### The mirror tolerance is 1e-3, not zero

The *raw* difference vector negates exactly — measured error is `0.00e+00`,
asserted at `< 1e-9`.

The *probability* does not. `StandardScaler` subtracts non-zero training means
from all 52 columns, so the standardised vector does not simply negate even
though the raw one does. For the shipped linear model the resulting error is
about 2.6e-4, and identical teams score 0.500130 rather than 0.5.

---

## Which model to probe

The shipped model is a **linear** logistic regression on antisymmetric
difference features. That means:

- single-factor sweeps are monotonic **by construction**
- pairwise interactions are exactly **zero** by construction
- mirroring holds automatically to ~3e-4

So those checks passing says very little about the shipped model. Run
`--model gbm` for the interesting comparison: a tree ensemble has no symmetry
constraint, no monotonicity guarantee, and can express real interactions.

Measured on identical teams:

| Model | P(A) for two identical teams | typical mirror error |
|---|---|---|
| logistic | 0.5001 | 0.0003 |
| **gbm** | **0.5138** | **0.027** |
| margin | 0.5077 | 0.015 |

The gradient booster is biased 1.4 points toward Team A on a perfectly
symmetric match, and its mirror error is roughly a hundred times the linear
model's. Since the raw features are exactly antisymmetric, that asymmetry is
entirely the model. It is an independent argument for the linear model, beyond
the one-standard-error rule that selected it.

---

## Benchmarks

```bash
python -m valwr.sandbox benchmark     # freeze the catalog
python -m valwr.sandbox compare       # diff a later run against it
```

`reports/sandbox/static_benchmark.json` holds every scenario's probability and
full feature vector, and is committed so model evolution is visible in the
diff.

**These are not pytest golden values.** Retraining legitimately changes every
probability, and a benchmark that made retraining fail would simply be deleted
the first time it was inconvenient. `compare` reports what moved, which
features moved, and whether any favourite changed sides. Interpret a large
delta as *"the model changed, here is where"* — not as a regression.

---

## Compositional coupling

Rate features are **nested**: map games are a subset of all games, and
map x agent games are a subset of both. So a subset rate and its parent cannot
be chosen independently -- the complement absorbs the difference.

This is not a limitation to work around, it is a property of the feature space,
and ignoring it produces scenarios that look like model findings but are
arithmetic. The first version of these profiles held the overall win rate at
0.50 while moving the map rate, which forced the complement to the boundary:

```
map_specialist   off-map record   0/15 = 0.000
map_weak         off-map record  15/15 = 1.000
```

Those two are not mirror images. The "specialist" never won off-map and the
"weak" player always did, and the resulting `good_map` scenario was measuring
that, not map strength.

The sandbox therefore pins the **complement** at neutral and derives the parent
rate. A player better on one map genuinely has a slightly better overall
record, which is also true in reality. The same applies to sweeps: `wr` moves
every rate together, while `wr_map` moves one subset and lets the overall rate
follow.

**When adding a scenario that sets a subset rate, check what the complement
had to become.** `world._buckets` and `world._allocate_wins` will show you.

---

## Findings so far

The sandbox is meant to surface surprises. Three so far, all reported rather
than fixed, because a sandbox tuned until it agrees with expectations has
stopped being a test.

### Overall win rate is inverted

Sweeping `wr` from 0.30 to 0.70 -- with every subset rate tracking it, so the
player is simply better -- moves P(A) from **53.0% down to 46.0%**. Higher win
rate, lower predicted chance of winning.

The coefficients show why: `d_wr_mean` is **-0.068** while `d_wr_max` is
**+0.093** and `d_wr_min` is **-0.098**. This is the classic multicollinearity
sign flip. Win rate is highly correlated with rating, ACS and ADR, so the fit
loads the signal onto those and lets win rate absorb residual in the opposite
direction.

It is **not** evidence that winning is bad, and no individual coefficient in a
collinear fit is interpretable on its own. It does mean the model's internal
attribution should not be read as a causal story, and it is a reason to prefer
the single-feature rating baseline when explaining a prediction to a human.

### Map strength barely registers, and not monotonically

The `wr_map` sweep gives 49.0 / 48.8 / 50.0 / 46.6 / 51.1 across its range --
non-monotonic and inside a few points of noise, against +28 points for rating.
The `good_map` scenario, where every player has a 68% map record over 45 games,
predicts **45.0%**: against the team with the map advantage. That is largely
the inverted win-rate coefficient bleeding through, since a map specialist also
has a better overall record.

The map x agent interaction behaves better: `map_agent_specialist` predicts
75.8%, and the sparse-cell version is correctly shrunk.

### The gradient booster is not symmetric

Two identical teams should be a coin flip. The shipped linear model says
0.5001. The gradient booster says **0.5138**, with a mirror error of 0.027 --
roughly a hundred times the linear model's 0.0003, and it fails the mirror
check on every sanity scenario.

Since the raw feature vector is exactly antisymmetric (measured `0.00e+00`),
that asymmetry is entirely the model. Trees have no symmetry constraint, and
this one has learned a side preference from noise. It is an independent
argument for the linear model, beyond the one-standard-error rule that
selected it.
