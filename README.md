# valorant-winrate-predictor

Predicts each team's win probability the moment you load into a VALORANT match,
using only what is knowable before the first round — then explains the
prediction in plain English.

> **Status: Phases 0–6 and 9 complete.** The live path has been run against
> real matches and replayed end to end over 2,500 held-out ones, where it
> tracks the training path to within about one standard error
> ([docs/MODEL-CHOICE.md](docs/MODEL-CHOICE.md)). The results below are
> measured on a held-out test set, not estimated. The browser dashboard and
> coaching layer (Phases 7–8) are not built yet.

---

## The problem

When ten players load into a match, the outcome is not a coin flip — but it is
close to one. Rank-based matchmaking exists specifically to make it close. The
question is whether the residual signal, after matchmaking has done its job, is
large enough to measure.

A naive approach compares average K/D. That fails, because K/D is mostly a
function of rank, and matchmaking already equalised rank. The signal has to come
from what matchmaking *does not* account for:

- A player who is significantly better on this specific map
- A player on an agent they perform well or badly on
- The **interaction** — someone mediocre on Ascent generally, but strong on
  Ascent specifically as Jett
- Team composition: role balance, missing controller, duelist stacks
- Party structure — a five-stack coordinates in ways five solo queuers do not
- Skill *variance* within a team, which the average hides entirely
- Recent form, and how long the current session has been running

## Approach

```
  HenrikDev API              Riot local client API
  (bulk match history)       (live: the 10 PUUIDs, agents, teams)
         |                              |
         v                              v
  +----------------+            +----------------+
  | rate-limited   |            | lockfile auth  |
  | snowball crawl |            | ws match watch |
  +----------------+            +----------------+
         |                              |
         v                              |
  +---------------------------+         |
  | temporal store            |         |
  | "what did we know about   |<--------+
  |  player X as of time T"   |
  +---------------------------+
         |
         v
  +----------------+     +------------------+     +-------------------+
  | player rating  |---->| time-gated       |---->| calibrated model  |
  | opponent-adj.  |     | feature builder  |     | + SHAP            |
  +----------------+     +------------------+     +-------------------+
                                                          |
                                            +-------------+-------------+
                                            v                           v
                                   +----------------+          +----------------+
                                   | live dashboard |          | Claude coach   |
                                   | localhost      |          | grounded in    |
                                   |                |          | attributions   |
                                   +----------------+          +----------------+
```

### Player rating

tracker.gg's developer program does not cover VALORANT, so Tracker Score is not
available to third-party developers. Rather than work around that, this project
builds its own rating: a composite of ACS, ADR, first-blood and first-death
rate, clutch rate, trade participation, and multi-kill rate — normalised
*within rank band and map*, then adjusted for the average rank of the opposing
team. Validated by split-half reliability across each player's history, and by
whether it out-predicts raw ACS on a player's next match.

### Avoiding the obvious trap

Match data is a time series, and the failure mode for this kind of project is
leakage — computing a feature from information that did not exist when the match
started. It inflates results, which is exactly why it goes uncaught.

Countermeasures, enforced in tests rather than by discipline:

- Every feature is derived only from rows with a strictly earlier timestamp,
  through a single temporal query layer
- Time-ordered train/validation/test splits; no random k-fold
- Empirical-Bayes shrinkage on every rate feature, so a 3-game sample at 100%
  does not read as a 100% player
- A shuffled-target check: retrain on shuffled labels and assert AUC collapses
  to 0.5

## Results

<!-- results:start -->
Measured on a held-out, time-ordered test set of **7,677 matches** never touched during training or tuning. 29,040 training matches; 52 features.

| Model | Log loss | AUC | Accuracy |
|---|---|---|---|
| Logistic + margin blend | 0.6875 | 0.557 | 53.3% |
| **Logistic regression (52 features)** | **0.6875** | 0.557 | **53.4% ± 1.1%** |
| Margin regression | 0.6876 | 0.556 | 53.4% |
| Gradient boosting | 0.6884 | 0.554 | 53.8% |
| Best player's rank (1 feature) | 0.6916 | 0.533 | 52.5% |
| Player rating alone (1 feature) | 0.6921 | 0.527 | 51.9% |
| Average rank — *the baseline to beat* | 0.6926 | 0.519 | 51.7% |
| Coin flip | 0.6931 | 0.500 | 50.5% |

**Bold is the shipped model.** Logistic regression (52 features) is not the lowest log loss, but 5 models finish within one standard error (0.0013) of the best, and among those the simplest one ships. The ranking at the top of this table flips between runs, because the gaps are smaller than the noise.

**Leakage check: 25 independent label shuffles, mean AUC 0.5008 ± 0.0121, 0 of 25 above the 0.55 alarm threshold.** One draw is not a test — a single shuffle has a standard deviation near 0.012, so any one of them can land anywhere and mean nothing.

Split by how many of the ten players had prior history: **5-6** 0.551, **7-8** 0.543, **9-10** 0.572. Not monotonic — see the retraction below.
<!-- results:end -->

Why the linear model rather than the gradient booster, and what else was tried
and rejected — Bradley-Terry player strength, symmetric augmentation, the
coverage threshold — is in [docs/MODEL-CHOICE.md](docs/MODEL-CHOICE.md), with
every null result recorded rather than quietly dropped.

### It beats rank where rank tells you nothing

The honest test is the subset where both teams have the same average rank —
where matchmaking did its job, and anything left is genuine residual rather
than rank in disguise. On those 1,403 test matches:

| Model | Log loss | AUC | Accuracy |
|---|---|---|---|
| Model | **0.6871** | **0.560** | **53.0% ± 2.6%** |
| Coin flip | 0.6931 | 0.500 | 49.3% |
| Avg rank | 0.6932 | 0.499 | 49.3% |

Rank scores AUC 0.499 there — by construction, it has nothing left to say. The
model still reaches 0.560, and the accuracy interval excludes 50%. So the
features are contributing signal beyond rank, not rediscovering it.

### The bug that was hiding all of this

An earlier version of this README reported 53.4% accuracy, AUC 0.549, and
concluded that a 52-feature pipeline was statistically indistinguishable from a
single feature — and that on equal-rank matches nothing beat a coin flip.

That was true, and it was caused by a bug. **The player rating had no
shrinkage.** Every rate feature was carefully shrunk toward a prior, because a
3-game sample at 100% is not a 100% player. The rating itself was not — and
**62% of players in this dataset have exactly one prior match**, so a rating
averaged over a single game was being trusted exactly as much as one averaged
over fifty. The signal was there the whole time, buried under small-sample
noise in the project's most important feature.

Fixing it, measured on the same data before and after (a snapshot from the day
of the fix — the live figures above move as the crawl grows):

| | Before | After |
|---|---|---|
| Log loss | 0.6897 | 0.6857 |
| AUC | 0.549 | 0.568 |
| Accuracy | 53.4% | 55.2% |
| Equal-rank AUC | ~0.53 | 0.560 |

The log-loss standard error is 0.0020, so a 0.0040 gain is two standard errors
— a real improvement rather than a lucky run. It also separated models that had
been tied: the full feature set now clearly beats the single-feature baseline,
where before they were level.

The lesson is not "we found a bug". It is that **the failure looked exactly
like an honest negative result** — plausible numbers, a clean leakage check,
and a tidy story about matchmaking being too good. It survived several rounds
of review, including my own writing it up as a finding.

## Setup

Python 3.11+, and a [HenrikDev API key](https://api.henrikdev.xyz/dashboard/)
(free, generated through their Discord). Node is optional and only used to run
the dashboard page's own tests.

```bash
git clone https://github.com/Fahran-Alam/valorant-winrate-predictor
cd valorant-winrate-predictor
python -m venv .venv && .venv/Scripts/activate      # source .venv/bin/activate on POSIX
pip install -e .
cp .env.example .env                                 # then fill in HENRIK_API_KEY
python -m valwr.check                                # smoke test: key, network, schema
```

### Collect a dataset

Nothing works without match history, and there is no shipped database — it
holds other players' statistics and is not redistributable. You build your own.

```bash
python -m valwr.collect.seed --self          # seed the frontier from your own account
crawl.bat                                    # supervised crawl; Ctrl+C to stop
```

The crawler snowballs outward from your seed: it fetches a player's recent
matches, stores all ten players from each, and queues the ones it has not seen.
It self-limits to the API's 30 requests a minute and collects roughly 1,400
matches an hour. **A few hours gets you a usable model; overnight gets a good
one.** Progress is logged to `data/crawl.log`.

On Windows, register `watchdog.bat` with Task Scheduler if you want it to
survive closing the window. One default matters: Task Scheduler stops a task
after three days unless you tell it not to.

### Train

```bash
python -m valwr.model.train --rebuild        # build features, fit, select, save
python tools/build_perf_index.py             # population reference for the 0-100 score
python tools/validate_potential.py --write-index   # measure and record its hit rate
```

`train` fits ten candidates, picks by the one-standard-error rule, and writes
`models/model.joblib` plus a full report to `reports/results.json`. The README
results block above is generated from that file, so it cannot drift.

### Run it

```bash
python tools/fetch_agent_art.py    # agent portraits and map splashes, once (~11 MB)

dashboard.bat                      # live view, localhost only
phone.bat                          # same, readable from a phone on your wifi
python -m valwr.dash --demo        # invented match, to see the page without playing
python -m valwr.dash --match <id>  # replay a finished game as it would have looked
```

The live view needs VALORANT running on the same machine — it reads the local
client API to learn who is in your lobby. `dashboard.bat` runs a preflight check
first and tells you in plain language if anything is missing.

### Tests

```bash
pytest -q                          # 311 tests
python tools/audit.py              # re-derives documented claims, reports drift
```

The audit is worth running after any retrain: it catches figures in the docs
that the new numbers have made stale, artefacts older than the model they
describe, and constants that have drifted apart.

## Scope and limitations

- **NA / PC / competitive queue only.** Regional metas differ enough that a
  model trained on one does not transfer cleanly.
- **Pre-match data only.** No in-round or economy state; this predicts at the
  loading screen, not during play.
- **The live component only works on the machine running the game**, since it
  reads the local client API.
- **Player data is not redistributed.** The collected database stays local and
  is not committed.

## Ethics and Terms of Service

The local client integration is strictly read-only — it reads match state and
never writes, never automates agent selection, and never touches process memory.
See [docs/ETHICS-AND-TOS.md](docs/ETHICS-AND-TOS.md).

Not affiliated with or endorsed by Riot Games.

## Licence

MIT
