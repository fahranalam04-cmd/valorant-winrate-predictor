# Phase 4 — Feature engineering

Read `CLAUDE.md` and **all of `docs/MODELING.md`** before starting. Phases 2 and
3 must pass.

## Goal

Turn a match into a feature vector, using only information that existed before
the match started. This phase is where leakage gets introduced if it is going to
be, so the audit matters as much as the features.

## Build

**1. Per-player features — `valwr/features/player.py`**

Everything in the "Per player" section of `docs/MODELING.md`: rating and its
components, rank, experience, shrunk history (overall, map, agent, **map ×
agent**), performance detail, and form.

**Shrinkage is mandatory on every rate feature:**

```
shrunk = (wins + prior_weight * population_rate) / (games + prior_weight)
```

A player with 3 games at 100% is not a 100% win rate player. Without this,
sparse cells dominate the model and generalise to nothing — this is the single
most common mistake in projects of this shape.

Tune `prior_weight` on the validation split, not by feel. Sparser cells need
stronger priors: map × agent needs far more shrinkage than overall win rate.
Report the tuned values and how sensitive results are to them.

**2. Team features — `valwr/features/team.py`**

Aggregate the ten players into two team vectors:

- Mean, max, min, and **standard deviation** of rating. The standard deviation
  is easy to skip and matters: one smurf plus four weak players is a very
  different team from five average players, and the mean erases that.
- Role composition from `ref_agents` — duelist count, has-controller,
  has-sentinel, has-initiator, role balance
- **Off-role count** — players on an agent outside their usual role
- **Party structure** from `party_id` — largest party size, number of distinct
  parties. A five-stack is a large, cleanly-observable signal that the naive
  version of this project would miss entirely.
- Rank spread within team

**3. Match context — `valwr/features/context.py`**

Map, attack/defence first, episode/act, region.

**4. Assembly — `valwr/features/build.py`**

Produce the final matrix. Two requirements from `docs/MODELING.md`:

- **Antisymmetric representation.** Either construct `team_a - team_b`
  differences, or mirror-augment every row with its swapped counterpart.
  Otherwise the model learns "team A wins more often", which is an artefact of
  row-writing order, not a real effect.
- Target is `P(team_a wins)` from one fixed perspective.

Write to parquet so Phase 5 does not rebuild every run. Include `match_id` and
`started_at` so the time-based split is possible downstream.

## The leakage audit — `test/test_leakage.py`

This is the deliverable that matters most in this phase.

**Traceability audit.** Sample matches. For each, rebuild every feature and
assert that every contributing source row has `started_at < target.started_at`,
strictly. Not a spot check on one feature — walk them all.

**Direct-query check.** Assert no module under `valwr/features/` queries
`match_players` directly; all reads go through `valwr/store/temporal.py`. A
grep-based test is fine and genuinely useful here.

**Symmetry check.** Swapping the two teams must produce exactly the negated
difference vector. Assert it.

## Constraints

- `as_of` is always the target match's `started_at`, and comparisons are always
  strict `<`.
- Rank comes from `match_players.tier` — the rank *at the time of that match*.
  A player's current rank reflects everything since, including the match being
  predicted. Using it is leakage, and it is a subtle one.
- Features must be computable at prediction time. If something is only knowable
  after the match, it cannot be a feature — this rules out anything derived from
  the target match's own stats.

## Acceptance criteria

`pytest test/test_leakage.py` passes in full. Feature matrix built and written
to parquet. Report: row count, feature count, and the fraction of rows with
missing values per feature — a feature that is 90% missing needs a decision, not
silent imputation.

## Do not build yet

No training, no evaluation. Features and the audit only. It is tempting to fit
something to see if it works; resist it — a model fitted before the audit passes
tells you nothing.
