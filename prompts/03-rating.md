# Phase 3 — Player rating

Read `CLAUDE.md` and `docs/MODELING.md` before starting. Phase 2 must pass.

## Goal

Build the opponent-adjusted player rating that replaces Tracker Score. This is
the headline component of the project — the thing that makes it "engineered a
player rating from raw match data" rather than "consumed someone else's number".

## Context

tracker.gg's developer program does not cover VALORANT, so Tracker Score is
unavailable to third-party developers. That is why this exists.

I have ML coursework but no end-to-end projects. **Explain the reasoning behind
each design decision as you go** — especially the normalisation and the
opponent adjustment. I want to be able to defend these choices in an interview,
which means understanding why they are there and what the alternatives were.

## Build

**1. Component metrics — `valwr/rating/components.py`**

Per player per match, from `match_players`:

- ACS (average combat score) — `score / rounds`
- ADR (average damage per round) — `damage_dealt / rounds`
- K/D, KDA
- Headshot percentage from `shots{head,body,leg}`
- First-blood rate and first-death rate, from kill event timings
- Trade participation, multi-kill rate, clutch rate where derivable

Some of these need kill-event data. If a metric is not reliably derivable from
what the API returns, **say so and leave it out** rather than approximating it
badly — a noisy component makes the composite worse, not more complete.

**2. Normalisation — `valwr/rating/normalize.py`**

Raw ACS is not comparable across contexts. 200 ACS in Iron is not 200 ACS in
Immortal; some maps produce systematically higher damage than others.

Normalise each component to a z-score **within rank band and within map**.
Explain how you are handling thin cells — a map/band combination with few
observations gives an unstable mean and standard deviation, and needs shrinkage
toward the global distribution just like the rate features do.

**3. Opponent adjustment — `valwr/rating/adjust.py`**

A 250 ACS game against Radiant opponents is worth more than 250 against Silver.
Adjust performance by the average rank of the *enemy* team.

Discuss the approach before implementing. A simple linear adjustment on rank
delta is probably right for a first version; note what a more principled
approach would look like (something Elo-like or a full random-effects model) and
why it is not worth it yet.

**4. Composite — `valwr/rating/rating.py`**

Combine into one number. **Keep it interpretable** — a weighted z-score
composite, with weights documented and justified in a comment, is more valuable
here than a learned rating that scores marginally better and cannot be
explained. Being able to say "this component is weighted this much because X" is
the point.

Also expose the components individually; the feature builder and the coach both
want them.

## Validation — `test/test_rating.py`

The rating is only worth anything if it measures something real and stable.
Three checks:

**1. Correlates with rank.** Sanity. Higher-ranked players should rate higher on
average. If not, something is inverted or the normalisation removed all signal —
note that normalising *within* rank band deliberately removes some of this, so
think about what correlation you actually expect and check against that, not
against "as high as possible".

**2. Split-half reliability.** Split a player's history into odd and even
matches, compute the rating on each half, correlate across players. This asks
whether the rating measures a stable property of the player or just noise. A
weak correlation means the composite is dominated by variance and needs either
more matches or fewer, better components.

**3. Beats raw ACS at predicting next-match performance.** For players with
enough history, does the rating from matches 1..n predict performance in match
n+1 better than raw ACS does? This is the one that shows the engineering added
value rather than just complexity.

Report all three numbers. **If #3 fails, say so plainly** — that is a real
finding, and it is better to know now than to build features on a rating that
does nothing.

## Constraints

- All history reads go through `valwr/store/temporal.py` with an explicit
  `as_of`. The rating is a feature; it is subject to the same time gating as
  everything else.
- No model training here — this is a hand-designed metric, deliberately.

## Acceptance criteria

`pytest test/test_rating.py` passes and the three validation numbers are
reported. A stable, documented, interpretable rating.

## Do not build yet

No feature vectors, no team aggregations, no model. Per-player rating only.
