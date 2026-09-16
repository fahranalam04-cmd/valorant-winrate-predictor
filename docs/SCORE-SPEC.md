# The 0-100 player score: specification

Status: **shipped.** The live view scores players this way; the previous single
formula in `valwr/rating/potential.py` still provides the rest of the card
(career, form, this map, freshness, the above-rank flag).

It ships having **failed the criteria in section 10**: 27.7% against the old
score's 29.7%, and worse on the three roles it was built for. That was a
deliberate call -- the score is meant to say what each role is trying to do,
and one match is 84% noise, so the ranking it gives up is small and the thing
it buys is a number that means the same for a Sage as for a Jett. The
measurement stays in this document rather than being quietly dropped.

The current score is one formula for everybody: four components
(`acs` 45%, `rating` 25%, `kd` 15%, `map_edge` 15%), z-scored against the whole
training population, combined, and mapped to a percentile. Two things are wrong
with it. It treats a Sage and a Jett as the same kind of player, and it is
effectively one number wearing three hats -- ACS, the composite rating and K/D
correlate 0.78 to 0.89 with each other, and ACS alone ranks a lobby as well as
all four together.

This replaces it with **four formulas, one per role**, each built from stats
chosen for what that role is actually trying to do.

---

## 1. What the score means

A player's score is their **percentile among players of the same role**.

> 82 means: played better than 82% of the Sentinels in the training
> population.

A Sentinel's 82 and a Duelist's 82 therefore mean the same thing, which the
current score cannot claim. Roles sit at genuinely different levels -- Duelists
average 226 ACS against an Initiator's 197 -- and pooling them means the
scoreboard rewards picking a Duelist.

Rank is handled one step earlier (section 3), so the percentile is taken across
all rank bands pooled. Every input has already had rank removed by then.

---

## 2. The inputs

All of these are already stored for every player in every match, at 100%
coverage, except ability casts (section 7).

| input | definition |
|---|---|
| **ACS** | combat score ÷ rounds played |
| **ADR** | damage dealt ÷ rounds played |
| **K/D** | total kills ÷ total deaths |
| **(K+A)/D** | (total kills + total assists) ÷ total deaths |
| **Assists** | assists ÷ rounds played |
| **KAST** | rounds with a kill, assist, survival or a traded death, ÷ rounds |
| **First bloods** | first kills of a round ÷ rounds played |
| **First deaths** | first deaths of a round ÷ rounds played |
| **HS%** | headshots ÷ (headshots + bodyshots + legshots) |
| **Abilities** | ability casts ÷ rounds played, all four slots summed |

Note that HS% is a share of *hits*, not of shots fired. Misses are not in the
data, so this is not an accuracy measure and never can be.

Each is computed over a player's history **strictly before the match starts**.
That rule is enforced by `store/temporal.py` and is not negotiable: a score
shown on the loading screen cannot use the match it is predicting.

---

## 3. What each input is compared against

Every input is turned into a **z-score against players in the same rank band**:
how many standard deviations above or below that band the player sits. So 250
ACS in Iron and 250 ACS in Immortal do not score the same.

Rank bands are the nine already used by `rating/normalize.py`, with the usual
hierarchical shrinkage (global -> band), so a thin band borrows from the
overall population rather than trusting four observations.

Two consequences worth being explicit about, because they look like bugs
otherwise:

- **The comparison group is everyone in the band, not everyone in the role.** A
  Sentinel will have a systematically negative ACS z-score, because Sentinels
  score less than Duelists. That is fine: the percentile in section 1 is taken
  within the role, so a level difference shared by every Sentinel cancels out.
- **Abilities are the exception.** They are z-scored against *the same agent*,
  never the band (section 7).

---

## 4. The weights

Each role's score is a weighted sum of z-scores. Weights are percentages of the
total absolute weight, so a negative weight still consumes its share.

### Duelist

| part | weight |
|---|---|
| ACS | 23% |
| K/D | 21% |
| ADR | 16% |
| KAST | 14% |
| Abilities | 11% |
| First bloods | 9% |
| HS% | 6% |
| *(map edge)* | *see section 6* |

Assists are **not** counted for Duelists.

### Controller

| part | weight |
|---|---|
| K/D | 21% |
| KAST | 20% |
| Abilities | 18% |
| Assists | 14% |
| ACS | 11% |
| ADR | 10% |
| HS% | 6% |

### Initiator

| part | weight |
|---|---|
| (K+A)/D | 24% |
| KAST | 23% |
| Abilities | 20% |
| ACS | 14% |
| ADR | 13% |
| HS% | 6% |

### Sentinel

| part | weight |
|---|---|
| (K+A)/D | 23% |
| KAST | 20% |
| Abilities | 19% |
| ACS | 15% |
| ADR | 11% |
| HS% | 6% |
| **First deaths** | **−6%** |

First deaths are negative: dying first repeatedly is the opposite of holding a
site.

### Two deliberate choices that look like mistakes

**ACS and ADR are kept together despite correlating 0.98.** They are close to
the same measurement, so the two of them together are really one dial: moving
one without the other has about half the effect the number suggests. How much
of the score sits on that one dial is a deliberate choice per role:

| role | ACS + ADR |
|---|---|
| Duelist | 39% -- damage is meant to lead for this role |
| Sentinel | 26% |
| Initiator | 27% |
| Controller | 21% -- lowest, because these roles are not judged on fragging |

The support roles were originally drafted with ACS leading, at 22-24%. That was
wrong for the same reason the whole spec exists: it scored a Controller on the
thing a Duelist is for. KAST, assists and abilities take that weight instead.

**Controllers count K/D and assists separately; Initiators and Sentinels fold
assists into (K+A)/D.** So two players with identical stats on different roles
are scored through different shapes, not just different weights. This is as
specified.

---

## 5. Which formula a player gets

The formula follows **the agent they are playing in this match**, since that is
what the live client tells us before the match starts.

**Per-agent exceptions.** Role weights are the default. An agent gets its own
weight set only where the kit genuinely differs from its role:

- **Chamber** — a 50/50 blend of the Duelist and Sentinel weights. His kit is
  gun-based, and he casts 1.67 abilities per round against Cypher's 3.55.

New exceptions need a stated reason, not a hunch. Everything else uses its
role's weights.

**Unknown agent.** In agent select, before players lock in, there is no agent.
The score then uses a role-neutral set: the mean of the four role weight sets,
with assists and first bloods at their average weight. This is a fallback, not
a fifth formula, and the page should mark it as provisional.

---

## 6. History, recency, and thin data

**Recency.** Matches are weighted with a 30-day half-life, matching the rest of
the project. Rates are recency-weighted means; ratios like K/D are
recency-weighted sums (Σw·kills ÷ Σw·deaths), not averages of per-match ratios,
so one 1-death match cannot produce a career K/D of 14.

**Off-role, and the hybrid.** 91% of player-role pairs in the database have
fewer than 5 games on that role; the median is one. A formula that needs role
history would have nothing to work with. So each input is a blend of the
player's history **on this role** and their history **overall**:

```
w_role = n_role / (n_role + 5)
value  = w_role · (value on this role) + (1 − w_role) · (value across all roles)
```

At 5 games on the role the role-specific figure carries half the weight; by 20
it carries 80%. Below that, their overall play fills the gap rather than the
score pretending to know something it doesn't.

**Thin history overall.** After the blend, the value is shrunk toward the
**role average** with a prior weight of 4 games, as the current score does:

```
shrunk = (value · n + role_average · 4) / (n + 4)
```

where `n` is their total prior matches. 53% of players have fewer than 5. Their
score should sit near the middle, not swing on one good game.

**No history at all.** No score. The page shows "—" and says why. Inventing a
number for an unknown player is the failure this project keeps guarding
against.

---

## 7. Abilities

**What the data is.** A count of ability casts per match, split into four slots
(grenade, two abilities, ultimate). It is available for 99.8% of the players in
the stored API responses, but it is **not currently saved as a column** --
ingestion discards it. Populating it means re-parsing the stored responses
(`python -m valwr.store.normalize`, which exists for exactly this). How many of
the 76,956 stored matches that recovers has not been measured yet; the count
needs re-running before this section can be relied on.

**Compared against the same agent, always.** Kits differ far too much for
anything else:

| role | fewest casts/round | most casts/round |
|---|---|---|
| Sentinel | Chamber 1.67 | Cypher 3.55 |
| Controller | Viper 1.73 | Clove 2.82 |
| Initiator | Gekko 1.56 | Fade 2.39 |
| Duelist | Reyna 1.41 | Yoru 2.19 |

Scored against the band, a Cypher would beat a Chamber for picking Cypher.
Agents with fewer than 300 training samples fall back to their role's
distribution, and the fallback is recorded so it can be audited.

**What this measures, stated honestly.** It is a count of presses, not of
effect: a flash that blinds nobody counts the same as one that wins the round.
Within the same agent, casting more does go with winning (+0.13 to +0.19). But
once kills and deaths are accounted for, that drops to **+0.02 to +0.04**. So
most of what abilities measure is *staying alive long enough to use them*.

This is the weakest-evidenced input in the spec, and it carries 17-19% on three
of the four roles. That is a deliberate choice about what the score should
reward, not a claim the data supports. It should be revisited once the live
scorecard has enough matches to measure it.

---

## 8. Map

The map component survives, at a lower threshold: **3 games on the map**, down
from 6. It is the difference between a player's level on this map and their
level overall, shrunk toward "no opinion".

At 6 games it moved 1.3% of players, which is indistinguishable from not
existing. At 3 it fires far more often, and its weight is set accordingly small
-- **5% of the total**, taken proportionally from every other component in the
role. Below 3 games it is exactly zero: no opinion, rather than a guess.

Worth remembering why the gate exists at all: on held-out data this component
measured a Spearman correlation of −0.010 with actual performance, and ungated
it swung a real account 20 points across maps on three games of evidence. It is
in the score because map familiarity is real and readers expect it, not because
it has been shown to predict anything.

---

## 9. What the score does not use

Rank itself, account level, party, win rate, agent pick rate, playtime, and the
opponent-strength adjustment in `rating/adjust.py` (which exists but is switched
off). Rank enters only as the comparison group; it is never a component.

---

## 10. Building it

1. **Recover ability casts.** Add four columns to `match_players`, teach
   `store/normalize.py` to read `ability_casts`, re-parse `raw_response`.
   Measure the resulting coverage before anything depends on it.
2. **Per-agent ability norms**, with the 300-sample fallback.
3. **Role-aware components** in `rating/potential.py`: the blend in section 6,
   then the shrinkage, per role.
4. **Per-role percentile tables** in `PerfIndex` — 4 sets rather than 1. All 40
   (role, band) cells have at least 2,000 player-matches, so there is enough to
   fit on.
5. **Rebuild the index** (`tools/build_perf_index.py`) and re-validate.

**How it gets judged.** `tools/validate_potential.py` measures top-1 rate: how
often the score picks the player who actually had the best game, out of five.
"Best game" is the ten-part match rating from `rating/rating.py`, not raw ACS.
The current score gets **30.5%** against 20% by chance.

Before building anything, it is worth knowing how much room there is. Three
measurements, all on the test period:

**One match is mostly noise.** Across 4,000 players with 8+ matches, only
**16% of the variation** in a single match rating is the player; 84% is the
night they had. A player's own match-to-match spread (sd 0.191) is more than
twice the spread between players (sd 0.085). Per role it is 19-21%.

**So top-1 has a low ceiling.** Simulating a score that knows every player's
true long-run level *exactly*:

| how much of a match is the player | top-1 ceiling |
|---|---|
| 16% (measured, all players) | 35.9% |
| 20% (Duelists, Initiators) | 38.4% |
| 21% (Controllers, Sentinels) | 38.9% |
| 100% (hypothetical, no noise) | 97.9% |

A perfect score gets about 38%. The current one gets 30.5%. The entire prize
for every future improvement, combined, is about eight points.

**But the room is not spread evenly across roles.** Correlation between the
current score and what a player actually did in the next match, against the
ceiling for that role (√ICC):

| role | current | ceiling | share of what is achievable |
|---|---|---|---|
| Duelist | 0.273 | 0.447 | **61%** |
| Sentinel | 0.186 | 0.458 | 41% |
| Controller | 0.150 | 0.458 | 33% |
| Initiator | 0.141 | 0.436 | **32%** |

This is the real case for role-specific formulas, and it is stronger than the
one this spec opened with. A single ACS-led formula already captures most of
what can be captured for Duelists, and roughly a third of it for Initiators and
Controllers. The score is not equally good for everyone -- it is good at
Duelists and weak at supports, which is exactly what you would expect from a
formula built out of fragging stats.

**What to expect per role, stated before building so it cannot be rationalised
afterwards:**

- **Duelist** — no improvement. It is already at 61% of its ceiling on stats
  that suit it. First bloods add a little genuinely new information; the rest is
  re-weighting what is already there.
- **Sentinel** — small improvement plausible. First deaths at a negative weight
  is information the current score does not use at all.
- **Controller, Initiator** — where the gains are, if there are any. Assists and
  KAST are barely represented in the current formula, and Controllers average
  nearly double a Duelist's assists per round.
- **Overall top-1** — flat, within noise. The aggregate is dominated by
  Duelists, who are the most-played role and the one already handled well.

**The measured baseline, before anything changed.** 1,500 teams, test period,
`python tools/validate_potential.py --json`:

| role | players | rho | when they had the best game, named |
|---|---|---|---|
| Duelist | 2,882 | +0.191 | **34.0%** |
| Sentinel | 1,626 | +0.171 | 28.6% |
| Controller | 1,554 | +0.172 | 24.5% |
| Initiator | 1,438 | +0.129 | **19.5%** |

Overall top-1 on this run was 28.8%, against 30.3% for career ACS alone and
20.3% for the shuffled control.

The last column is the one that should be uncomfortable. When an Initiator
genuinely had the best game on their team, the current score names them 19.5%
of the time -- which is what picking at random would do. For Duelists it is
34.0%. The score is not "modestly accurate for everyone"; it is useful for
Duelists and no better than guessing for Initiators, and the aggregate hid
that completely.

**The result, measured once it was built.** 1,500 test-period teams, both
scores on the same teams, ability data 99.3% recovered
(`python tools/compare_role_score.py`):

| | top-1 |
|---|---|
| the old score | 29.7% |
| **the per-role score** | **27.7%** |
| career ACS alone | 29.7% |
| shuffled (control) | 20.5% |

| role | rho old | rho new | named old | named new |
|---|---|---|---|---|
| Duelist | +0.127 | +0.129 | 33.3% | 32.7% |
| Sentinel | +0.162 | +0.136 | 32.2% | 21.6% |
| Controller | +0.115 | +0.067 | 23.4% | 27.8% |
| Initiator | +0.213 | +0.157 | 24.9% | 18.7% |

**It fails criterion 1 and criterion 2.** Two points of top-1 is twice the
standard error, and the three roles the change was built for all got worse. The
weights in section 4 are the ones measured here; they should not ship as they
stand.

**Why, measured rather than guessed.** Two properties of each input, on 3,500
test-period player-matches:

| | ACS | ADR | K/D | KAST | assists | abilities | HS% |
|---|---|---|---|---|---|---|---|
| predicts next match (Controller) | +0.120 | +0.129 | +0.066 | +0.051 | −0.013 | **−0.015** | +0.074 |
| predicts next match (Initiator) | +0.253 | +0.246 | +0.193 | +0.098 | +0.021 | **−0.009** | +0.153 |
| is a stable trait (Controller) | +0.262 | +0.280 | +0.202 | +0.128 | +0.157 | **+0.507** | +0.657 |
| is a stable trait (Initiator) | +0.193 | +0.183 | +0.086 | +0.036 | +0.130 | **+0.508** | +0.629 |

Ability casting is among the *most* stable things about a player -- more stable
than their ACS -- and predicts performance at zero. That combination is worse
than a noisy input. Noise averages out; a stable non-signal ranks the same
players above others every match, for a reason unconnected to how well they
play. At 17-20% of the weight, a fifth of the score was being spent sorting
people by playstyle.

KAST has the opposite problem: it matters inside a match (it is 22% of the
yardstick) but barely persists between them -- +0.036 for Initiators and
Sentinels. There is little stable signal there to weight.

Both findings survive a change of yardstick. Measured against *winning* rather
than against the rating, within the same agent, casts correlate +0.13 to +0.19
raw -- and +0.02 to +0.04 once kills and deaths are accounted for. Cast counts
measure being alive, not contributing, which is what a count of button presses
was always at risk of measuring: a drone that spots three players and a drone
thrown at a wall are identical in this data.

**Success criteria, agreed in advance:**

1. Overall top-1 must not drop by more than one standard error (about 1 point
   at 1,500 teams).
2. Initiator and Controller correlation should rise from 0.14/0.15. Anything
   above 0.20 is a real gain.
3. Per-role top-1 gets reported separately from now on. A single aggregate
   number hid the fact that the score is far weaker for supports than for
   Duelists, and that is the kind of thing this project should not hide.

**One inconsistency to fix while here.** The validation harness judges "who
played best" by the ten-part match rating; the post-match comparison in
`live/review.py` judges it by ACS. Those disagree, most for the roles that frag
least. They should use the same definition, and it should be the rating.

---

## 11. Open items

- **Ability coverage** across the 76,956 stored matches is unmeasured.
- **The per-role yardstick.** Sections above judge every role by the same
  ten-part rating. If that rating undersells what a Controller does, then a
  Controller formula tuned against it inherits the same blind spot. Worth
  deciding whether each role should also be measured against what it is
  supposed to influence -- KAST and assists for the support roles, entry
  success for Duelists.
- **The map weight (5%)** was not specified; it is my proposal.
- **The role-neutral fallback** for agent select was not specified either.
- **Abilities at 17-19%** rest on the weakest evidence in the spec.
