# Roadmap

Nine phases, planned before any code existed, each with acceptance criteria
that had to pass before the next began. This page keeps the plan and records
what was actually built against it — including where the build deliberately
went a different way, and why.

| Phase | | Status |
|---|---|---|
| 0 | Scaffold and environment | Done |
| 1 | Collector | Done |
| 2 | Normalisation and the temporal store | Done |
| 3 | Player rating | Done — one of its three checks no longer passes; see below |
| 4 | Feature engineering | Done |
| 5 | Model | Done — logistic regression ships, not gradient boosting |
| 6 | Live client integration | Done |
| 7 | Dashboard | Done |
| 8 | LLM coach | **Not built** |
| 9 | Backtest and write-up | Done, except the optional Riot production key |

The sequencing rule held: **the project looks complete at every stopping
point.** It stopped after Phase 7 and 9 with a validated model, a live tool and
a write-up, and nothing half-built depends on Phase 8.

---

## Phase 0 — Scaffold and environment

Set up the package, dependencies, config, and reference data.

- venv, dependencies, `.env` from `.env.example`
- Verify the ML wheels exist for the local Python
- Pull agents, maps, tiers, seasons from valorant-api.com into reference tables
- Smoke-test the HenrikDev key against your own account

**Done when:** `python -m valwr.check` prints your account, your last 5 matches,
and counts of 20+ agents and 10+ maps from the reference tables.

**As built.** Dependencies live in `pyproject.toml` (`pip install -e .`).
LightGBM installed cleanly on Python 3.14, and XGBoost and SHAP turned out not
to be needed at all — see Phase 5.

## Phase 1 — Collector

The rate-limited snowball crawler.

- Every HenrikDev call goes through one shared limiter
- `429` handling: honour `Retry-After`, back off
- Seed the frontier from your own PUUID (and optionally the leaderboard)
- For each PUUID: fetch matchlist, store raw, extract the other 9 PUUIDs, queue
  the unseen ones
- **Rank-stratified**: track `tier_band`, deprioritise over-represented bands
- Fully resumable — frontier state lives in the database, not memory

**Done when:** run for 30 minutes, kill it mid-run, restart — no duplicate
matches, no lost frontier entries, no stuck `fetching` rows, and the rank
distribution is spread across bands.

**As built.** The limiter is a fixed window reconciled against the server's own
`x-ratelimit-remaining`, not a token bucket: a fresh matchlist costs about ten
quota units, which three limiter designs all confirmed
([API-NOTES.md](API-NOTES.md)). Seeding from your own account reached Iron
through Immortal at 7–12% per band, so leaderboard seeding was unnecessary
([DATA.md](DATA.md)).

## Phase 2 — Normalisation and the temporal store

Raw JSON → queryable tables, plus the leakage firewall.

- Parse `raw_response` into `matches`, `match_players`, `players`
- Idempotent: re-parsing must not duplicate
- Build `valwr/store/temporal.py` — the `as_of` query layer

**Done when:** no orphan rows, every match has exactly 10 players or is
explicitly flagged, and `player_history(puuid, as_of)` provably returns nothing
at or after `as_of`.

**As built.** Every history query is also restricted to competitive matches,
inside the temporal layer, so no caller can forget it. The filter is a
correlated `EXISTS` against the matches table: the `IN (SELECT …)` form was 534
times slower.

## Phase 3 — Player rating

A composite of ACS, ADR, first-blood and first-death rate, clutch rate, trade
participation and multi-kills, normalised within rank band and map, adjusted
for the opposing team's rank.

**Done when:** three validations pass — the rating relates to rank as designed,
split-half reliability is decent, and it out-predicts raw ACS on a player's
next match.

**As built, and re-measured on 758,920 player-match rows.** The rating is
independent of rank by design (r = +0.008) and moderately stable (split-half
reliability 0.54). **It does not out-predict raw ACS** — the two tie, z = −1.36,
and alone ACS is marginally ahead at predicting match results. An early run on
a few hundred players said the rating won; that did not survive more data. It
stays in the model as one of 52 features. Details in [MODELING.md](MODELING.md).

## Phase 4 — Feature engineering

The full vector from [MODELING.md](MODELING.md), strictly time-gated.

- Per-player: skill, shrunk history, map, agent and map × agent, form
- Empirical-Bayes shrinkage on every rate
- Team aggregations including **standard deviation**, not just mean
- Composition, party structure, off-role, match context
- Antisymmetric team-pair representation

**Done when:** the leakage audit in `test/test_leakage.py` passes, and swapping
teams produces exactly `1 - p`.

**As built.** 52 team-difference features. Swapping the teams negates the
vector exactly, and the shipped model mirrors to machine precision.

## Phase 5 — Model

Climb the baseline ladder, calibrate, attribute.

- Coin flip → rank baseline → logistic regression → gradient boosting
- Time-ordered splits; test slice touched once
- Log loss, Brier, AUC, accuracy, calibration error, reliability diagram
- **Shuffled-target check must collapse to ~0.5**

**Done when:** gradient boosting beats the rank baseline on held-out log loss,
the reliability diagram is close to the diagonal, and the shuffled-target check
passes.

**As built — three deliberate departures.**

- **Logistic regression ships, not gradient boosting.** The booster does beat
  the rank baseline (0.6882 against 0.6928), but so do three simpler models,
  all within one standard error of each other. The simplest of those ships.
  The booster also fails the mirror test: it gives two identical teams
  different odds. [MODEL-CHOICE.md](MODEL-CHOICE.md) has every model and every
  metric.
- **No calibration layer.** Isotonic recalibration, fitted on validation, made
  held-out log loss worse (0.6893 against 0.6874). The raw model is already
  calibrated to within a point — expected calibration error 0.008.
- **No SHAP.** For a linear model, each feature's contribution is exactly its
  weight times its standardised value, and those contributions sum to the
  prediction. That is what the dashboard shows. SHAP would approximate a number
  that can be computed exactly.

## Phase 6 — Live client integration

Detect a real match and resolve its ten players.

- Lockfile parse, basic auth against `127.0.0.1`, TLS verification off for
  localhost only
- Roster from pregame and core-game: 10 PUUIDs, agents, teams
- Cache first, own team first, degrade to partial predictions
- **Read-only.** See [ETHICS-AND-TOS.md](ETHICS-AND-TOS.md).

**Done when:** load into a real or custom match and the ten players resolve,
with a prediction before the first round — including the degraded path.

**As built.** The client is polled every five seconds rather than subscribed
to over its websocket; it is simpler and just as fast at this cadence. The
shard comes from `/riotclient/region-locale`, not the process command line. A
player whose newest stored match is over two hours old is refetched, your own
account always is, and two matchlist pages are fetched so a "last 20" really
is the last twenty.

## Phase 7 — Dashboard

FastAPI and a websocket, vanilla JS, no build step, bound to `127.0.0.1`.

**Done when:** the server is running, the browser is open, you load into a
match, and the page updates live without a refresh.

**As built**, in `valwr/dash/`, with more than the brief asked for:

- both teams' odds, the verdict, and the factors moving the prediction
- all ten players by team, with rank, party (duo, trio…), a 0–100 score,
  career ACS, K/D, last-20 K/D and headshot rate — competitive games only
- a card per player: score breakdown, career against recent form, record on
  this map, last matches, and how fresh the data is
- the map's own art as the background
- `--demo` (invented players), `--match <id>` (replay a finished game as it
  would have looked), and `phone.bat` (read it from a phone on your wifi)
- refuses other sites and DNS rebinding; see [SECURITY.md](../SECURITY.md)

[DASHBOARD.md](DASHBOARD.md) explains every element. An interactive demo runs
on GitHub Pages.

## Phase 8 — LLM coach

Grounded natural-language coaching: the prediction, its attributions, both
compositions and the map, passed to a language model, with a system prompt that forbids
inventing statistics and a check that every number it quotes is in its input.

**Not built.** Most predictions fall between 40% and 60%, and the features a
coach would most naturally advise on — map and agent history — measured close
to zero. A fluent coach explaining a near coin flip risks the exact failure the
brief warns about: confident advice resting on noise. The dashboard already
explains each prediction in plain language. If it is built, the constraints
above stand: players anonymised in its input, and no number it was not given.

## Phase 9 — Backtest and write-up

- Replay historical matches through the full live path, end to end
- README results: reliability diagram, what the model relies on, the
  equal-rank result, honest limitations
- Optionally apply to Riot for a production key and RSO

**As built.** `tools/backtest_live.py` replays held-out matches exactly as the
client delivers them; it found and closed a leak before producing a number.
The README carries the calibration chart, the importance chart and the
equal-rank result. The production-key application was not made.

---

## Running the crawler unattended

`python -m valwr.collect.supervise --hours 10` restarts the crawl on any
failure and logs to `data/crawl.log`. `crawl.bat` wraps it; `watch.bat` shows
live progress; `stop.bat` stops it.

Launch it **windowless** (`pythonw.exe`, `-WindowStyle Hidden`) for overnight
runs. An early attempt died at 03:52 with nothing in the Windows event log --
no sleep, no crash, no network event -- and the likeliest explanation was
simply that its console window got closed. A window that does not exist cannot
be closed by accident, and `watch.bat` provides the visibility instead.

A watchdog runs from Task Scheduler every 10 minutes and restarts the crawler
if nothing has been fetched recently. Liveness is read from the database, not
the process table -- a hung process holding a PID while fetching nothing is
just as dead as a missing one.

One Task Scheduler default silently defeats this on a laptop:
`DisallowStartIfOnBatteries` is **True** unless overridden, so an unplugged
machine never runs the task -- exactly the unattended case it exists for. Pass
`-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries`.

The crawler also blocks system sleep itself via `SetThreadExecutionState`, so
it does not depend on anything being played to keep the machine awake.

Two crawlers must never run at once: they share one API quota and will spend it
twice, earning 429s. Check with `stop.bat` before starting a new one.

## Deliberately out of scope

- **Other regions.** Metas differ; one region done properly beats three done
  badly.
- **In-round prediction.** Economy and round state are a different, much larger
  project.
- **A hosted public version.** It would mean serving other players' data from a
  server — see [ETHICS-AND-TOS.md](ETHICS-AND-TOS.md). The public demo uses
  invented players for exactly this reason.
- **Any form of gameplay automation.** Not a scope decision; a ban-safety one.
