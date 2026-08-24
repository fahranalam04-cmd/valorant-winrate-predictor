# Roadmap

Nine phases. Each is one Claude Code session, with a prompt in `prompts/` and
acceptance criteria that must actually pass before moving on.

The sequencing rule: **the project should look complete at every stopping
point.** If you stop after Phase 5, you have a validated model and a README with
real results — a finished portfolio piece. If you stop after Phase 7, you have a
live tool. Nothing is left as a half-built dependency of something else.

Budget: 1–2 months at an evenings-and-weekends pace.

---

## Phase 0 — Scaffold and environment

Set up the package, dependencies, config, and reference data.

- venv, `requirements.txt`, `.env` from `.env.example`
- **Verify LightGBM / XGBoost / SHAP wheels exist for Python 3.14.** They may
  not yet. If any fail, recreate the venv on Python 3.12 and note it in the
  README — do not spend the session fighting a source build.
- Pull agents, maps, tiers, seasons from valorant-api.com into reference tables
- Smoke-test the HenrikDev key against your own account
- **Apply for the Enhanced key now.** Approval takes days and triples crawl
  throughput; the lag is the bottleneck, not the code.

**Done when:** `python -m valwr.check` prints your account, your last 5 matches,
and counts of 20+ agents and 10+ maps from the reference tables.

## Phase 1 — Collector

The rate-limited snowball crawler.

- Shared token-bucket limiter — every HenrikDev call goes through it
- `429` handling: honour `Retry-After`, exponential backoff
- Seed the frontier from the leaderboard **and** your own PUUID
- For each PUUID: fetch matchlist, store raw, extract the other 9 PUUIDs, queue
  the unseen ones
- **Rank-stratified**: track `tier_band`, deprioritise over-represented bands
- Fully resumable — frontier state lives in the database, not memory
- Log the rank distribution at session end

**Done when:** run for 30 minutes, `kill -9` mid-run, restart — no duplicate
matches, no lost frontier entries, no stuck `fetching` rows, and the rank
distribution is spread across bands rather than concentrated at the top.

## Phase 2 — Normalisation and the temporal store

Raw JSON → queryable tables, plus the leakage firewall.

- Parse `raw_response` into `matches`, `match_players`, `players`
- Idempotent: re-parsing must not duplicate
- Build `valwr/store/temporal.py` — the `as_of` query layer
- Benchmark the hot query; denormalise `started_at` onto `match_players` if the
  join is slow

**Done when:** assertion suite passes — no orphan rows, every match has exactly
10 players or is explicitly flagged, per-player timestamps are monotonic, and
`player_history(puuid, as_of)` provably returns nothing at or after `as_of`.

## Phase 3 — Player rating

The Tracker Score replacement, and the headline component of the project.

- Composite of ACS, ADR, first-blood/first-death rate, headshot %, clutch rate,
  trade participation, multi-kill rate
- Normalised **within rank band and within map** — 200 ACS in Iron is not 200
  ACS in Immortal
- Adjusted for the average rank of the opposing team
- Keep it interpretable: a weighted z-score composite, with the weights
  documented and justified

**Done when:** three validations pass — the rating correlates with rank
(sanity), split-half reliability across a player's history is decent (it is
measuring something stable, not noise), and it out-predicts raw ACS on a
player's next-match performance (it is measuring something *useful*).

## Phase 4 — Feature engineering

The full vector from [MODELING.md](MODELING.md), strictly time-gated.

- Per-player: skill, shrunk history, map/agent/map×agent, form
- Empirical-Bayes shrinkage on every rate, prior weight tuned on validation
- Team aggregations including **standard deviation**, not just mean
- Composition, party structure, off-role, match context
- Antisymmetric team-pair representation

**Done when:** the leakage audit in `test/test_leakage.py` passes — for sampled
matches, every feature value traces to source rows strictly earlier than the
target — and swapping teams produces exactly `1 - p` on the feature level.

## Phase 5 — Model

Climb the baseline ladder, calibrate, attribute.

- Coin flip → rank baseline → logistic regression → gradient boosting
- Time-ordered splits; test slice touched once
- Isotonic calibration fit on validation
- Log loss, Brier, AUC, accuracy, ECE, reliability diagram
- SHAP: global importance plot, per-match attributions
- **Shuffled-target check must collapse to ~0.5**
- Fill in the README results table with measured numbers

**Done when:** gradient boosting beats the rank baseline on log loss on the
held-out test set, the reliability diagram is close to the diagonal, and the
shuffled-target check passes. This is the point at which the project is a
complete portfolio piece.

## Phase 6 — Live client integration

Detect a real match and resolve its ten players.

- Lockfile parse, basic auth against `127.0.0.1`, TLS verification off for
  localhost only
- Entitlements + access token; region from the `-ares-deployment` process arg
- Websocket subscription, watching pregame and core-game URI prefixes
- `Pregame_GetMatch` / `CoreGame_FetchMatch` → 10 PUUIDs, agents, teams
- **Handle the rate-limit squeeze**: cache-first, own-team-first priority,
  degrade to partial-data predictions with a wider confidence band, pre-warm
  between matches
- **Read-only.** See [ETHICS-AND-TOS.md](ETHICS-AND-TOS.md).

**Done when:** load into a real or custom match and the ten players resolve,
with a prediction produced before the first round — including the degraded path
when some players are uncached.

## Phase 7 — Dashboard

FastAPI + websocket, vanilla JS front end, bound to `127.0.0.1`.

- Two team win probabilities, prominent
- Per-player cards: rating, rank, map and agent history, resolved/unresolved
- Top positive and negative factors from SHAP, in plain language
- Confidence indicator reflecting how much data was available
- No build step

**Done when:** `uvicorn` running, browser open, load into a match, and the page
updates live without a refresh.

## Phase 8 — Claude coach

Grounded natural-language coaching.

- Load the `claude-api` skill first for current model IDs and patterns
- Input: prediction, SHAP attributions, both comps, map, per-player summaries
- **System prompt forbids inventing statistics** — reason only over supplied
  numbers. A coach that fabricates plausible VALORANT stats is worse than none,
  because it is convincing.
- Two outputs: strategic advice, and a plain-English explanation of the
  prediction
- One cached call per match; cache by `match_id`

**Done when:** given a real match, it returns coherent advice that cites only
numbers actually present in the prediction payload — verified by checking a
sample of its claims against the input.

## Phase 9 — Backtest and writeup (stretch)

- Replay historical matches through the full live path, end to end
- README results section: reliability diagram, feature importances, the
  equal-rank subset result, honest limitations
- Optionally apply to Riot for a production key + RSO, now that a prototype
  exists to show them

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
  server — see [ETHICS-AND-TOS.md](ETHICS-AND-TOS.md).
- **Any form of gameplay automation.** Not a scope decision; a ban-safety one.
