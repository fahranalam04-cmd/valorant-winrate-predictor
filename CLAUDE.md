# CLAUDE.md

Briefing for Claude Code. Read this before touching anything.

## What this is

A system that predicts each team's win probability the moment you load into a
VALORANT match, using only pre-match data about the ten players, then explains
the prediction in plain English.

It is **not** a K/D comparison. It models per-player skill (an
opponent-adjusted rating built from raw match data), map-specific and
agent-specific history, the map × agent interaction, team composition, party
structure, and recent form — then combines them into a calibrated probability.

Built as a portfolio project. That means the engineering around the model
matters more than the model: honest validation, no leakage, calibrated output,
and a README that states its own limitations.

## Status

**Phases 0, 1 and 2 are done.** Environment, schema and reference data pass;
the collector crawls, stratifies by rank band and survives being killed; raw
JSON normalises idempotently and `store/temporal.py` enforces the `as_of`
firewall. Next is Phase 3 (the player rating metric).

Work through `prompts/` in order — each file is one Claude Code session. Do not
skip ahead; later phases assume earlier acceptance criteria actually pass.

See `docs/ROADMAP.md` for all nine phases.

## Stack

Python 3.14 (fall back to 3.12 if ML wheels are missing — Phase 0 checks this).
SQLite for storage. FastAPI + websockets for the live dashboard, with a vanilla
JS front end. No build step on the front end; keep it that way.

```
valwr/          the package
  check.py      Phase 0 smoke test
  collect/      crawler, rate limiter, frontier queue
  store/        schema, normalisation, the temporal query layer
  rating/       the player rating metric
  features/     time-gated feature builder
  model/        baselines, training, calibration, SHAP
  live/         lockfile auth, websocket match detection
  web/          FastAPI app + static front end
  coach/        Claude API layer
docs/           specs — read these before implementing a phase
prompts/        one prompt per phase
test/           pytest
data/           SQLite DB, gitignored
models/         trained artefacts, gitignored
```

## Run it

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows/Git Bash
pip install -r requirements.txt
cp .env.example .env        # then fill in HENRIK_API_KEY
python -m valwr.check              # smoke test
python -m valwr.collect --minutes 60   # crawl (resumable, ctrl-c safe)
python -m valwr.store.normalize        # raw -> normalised tables
pytest                             # after ANY change to valwr/
```

## Data sources — the real ones

Full endpoint reference with parameters and response shapes is in
`docs/API-NOTES.md`. **Read it before writing any HTTP call.** Do not invent
endpoints; every one in that file was verified against live documentation.

| Source | Auth | Use |
|---|---|---|
| HenrikDev API | `Authorization: <key>` header | bulk match history, leaderboard, MMR |
| valorant-api.com | none | agent/map/rank metadata, pulled once |
| Riot local client | lockfile basic auth, localhost | live match detection, the 10 PUUIDs |

**tracker.gg is not a data source.** Their developer program does not cover
VALORANT — TRN state they are not permitted to grant it and redirect developers
to Riot. Tracker Score is therefore unobtainable, which is why `valwr/rating/`
builds its own. Do not add a tracker.gg scraper; it violates their ToS and
would break constantly.

**Riot's official VALORANT API is not a data source either**, at least not yet.
Riot does not issue personal keys for VALORANT; production keys require a
working prototype, RSO integration, and a multi-week review. Revisit after
Phase 8.

## Rules that exist for a reason

**1. The rate limit is the binding constraint. Trust the server's count, not
your own.** Every call goes through the limiter in `valwr/collect/limiter.py`,
wired into `HenrikClient` so a caller cannot forget it. The limiter reconciles
against the `x-ratelimit-remaining` header after every response and only ever
revises *down*.

Do not replace this with a client-side token bucket sized to the stated 30
req/min. That was the first implementation, and it earned three 429s while
averaging under 3 req/min: a `size=10` matchlist costs ~2 quota units, not 1,
and the bucket started full so the opening requests fired unspaced into a
rolling window that still held the previous run's calls. Real sustained
throughput on the basic tier is **~3.4 req/min**. Cache aggressively; a
player's history from two hours ago is fine.

**2. Never compute a feature from data that did not exist yet.**
This is the bug class that silently ruins the whole project, because it makes
results look *better*. Every feature for a match must be derived only from rows
with a strictly earlier timestamp. `valwr/store/temporal.py` is the only
sanctioned way to ask "what did we know about player X as of time T" — use it.
Never write a feature query that filters on player alone. The full trap list is
in `docs/MODELING.md`; the leakage audit in `test/test_leakage.py` must pass.

**3. Time-based splits only. Never random k-fold.**
Match data is a time series. Shuffling it trains on the future and tests on the
past. Train on the earliest slice, validate on the middle, test on the latest.

**4. Shrink every rate feature toward a prior.**
A player with 3 games at 100% is not a 100% win rate player. All rate features
(win rate, map win rate, agent win rate, map × agent) use empirical-Bayes
shrinkage toward the population mean, weighted by sample size. Raw rates on
sparse cells will dominate the model and generalise to nothing.

**5. If the model looks great, it is broken.**
Realistic performance for this problem is **AUC 0.60–0.68, accuracy 58–64%**.
Above ~0.75 means leakage. The `shuffled target` check in
`test/test_leakage.py` retrains on shuffled labels and asserts AUC collapses to
~0.5 — if it does not, a feature is carrying outcome information. Investigate
before celebrating.

**6. Beat the baselines in order, and keep them in the repo.**
Coin flip → *higher average rank wins* → logistic regression → gradient
boosting. The rank baseline is the honest bar and it is stronger than it looks.
A model that does not beat it is not a model. `valwr/model/baselines.py` stays
in the codebase permanently; every eval reports against it.

**7. Calibration matters more than accuracy here.**
The product outputs "58%". That has to *mean* 58% — of matches predicted at
58%, about 58% should be won. Report log loss, Brier score, and a reliability
diagram alongside AUC. Calibrate on a held-out slice.

**8. The local client API is read-only. This is a ban-safety rule.**
Never POST to it. Never automate agent selection — instalockers are what
actually gets accounts banned. No memory reading, no injection, no DLLs, no
overlay hooking. The dashboard is a separate browser window, not an overlay.
See `docs/ETHICS-AND-TOS.md`.

**9. Other players' data stays on this machine.**
The crawler collects real people's match histories. Never commit the database,
never publish a dataset of PUUIDs, never expose a public endpoint that looks up
someone else's stats. `data/` is gitignored and stays that way.

**10. The coach explains the model. It does not invent analysis.**
`valwr/coach/` passes the prediction plus its SHAP attributions to Claude. The
system prompt forbids inventing statistics — the model may only reason over
supplied numbers. An LLM that makes up plausible-sounding VALORANT stats is
worse than no coach at all, because it is convincing.

**11. Secrets never enter git.**
`.env` is gitignored; only `.env.example` is tracked. No key literal in any
committed file, notebook output, or test fixture.

## Style

Plain Python, minimal dependencies. Comments explain *why*, not *what*. Prefer
fixing a cause over adding a guard. Small functions with real names over clever
one-liners. If a phase's acceptance criteria are ambiguous, ask before building.
