# valorant-winrate-predictor

Predicts each team's win probability the moment you load into a VALORANT match,
using only what is knowable before the first round — then explains the
prediction in plain English.

> **Status: in development.** Phase 0 of 8. Results below are placeholders and
> will be filled in with real measured numbers at Phase 5. Nothing here claims
> a result it has not measured.

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

<!-- Filled in at Phase 5. Do not populate until measured on the held-out
     time-split test set. -->

| Model | AUC | Log loss | Brier | Accuracy |
|---|---|---|---|---|
| Coin flip | 0.500 | 0.693 | 0.250 | 50.0% |
| Higher average rank wins | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Logistic regression | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Gradient boosting | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

**Expected range: AUC 0.60–0.68, accuracy 58–64%.**

That is the honest ceiling for pre-match prediction from public stats, and it is
stated up front deliberately. Matchmaking is designed to make matches even; a
model that beats a rank baseline by a few points of log loss on this problem is
doing real work. A repository claiming 90% accuracy on pre-match VALORANT
prediction has undetected leakage, not a breakthrough.

## Setup

Requires a [HenrikDev API key](https://api.henrikdev.xyz/dashboard/) (generated
via their Discord) and, for the coaching layer, an Anthropic API key.

```bash
git clone https://github.com/<user>/valorant-winrate-predictor
cd valorant-winrate-predictor
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env      # fill in HENRIK_API_KEY
python -m valwr.check     # smoke test
```

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
