# Phase 0 — Scaffold and environment

Read `CLAUDE.md` and `docs/API-NOTES.md` before starting.

## Goal

A working Python environment, the package skeleton, reference data loaded, and a
smoke test that proves the HenrikDev key works. Nothing else.

## Build

**1. Environment**

Create a venv and install `requirements.txt`.

**Critically: verify that `lightgbm`, `xgboost`, and `shap` actually install and
import on this machine's Python 3.14.2.** These ship compiled wheels and may not
have 3.14 builds yet. If any fail:

- Do **not** attempt a source build — that is a rabbit hole, not a task
- Recreate the venv on Python 3.12 instead
- Note the Python version requirement in `README.md` under Setup

Report which Python version the project ended up on.

**2. Package skeleton**

Create the `valwr/` layout described in `CLAUDE.md`, with `__init__.py` files
and empty-but-importable modules. Do not implement anything beyond Phase 0.

**3. Config loading**

`valwr/config.py` — load `.env` via `python-dotenv`. Expose typed settings:
`HENRIK_API_KEY`, `HENRIK_TIER`, `REGION`, `PLATFORM`, `RIOT_NAME`, `RIOT_TAG`,
`DATABASE_PATH`.

Fail loudly and helpfully on a missing key — print what is missing and where to
get it, not a `KeyError` traceback. This is the first thing that will break for
anyone cloning the repo.

**4. Database bootstrap**

`valwr/store/schema.py` — create the SQLite file and the tables in `docs/DATA.md`
(raw, normalised, frontier, reference layers). Idempotent: running it twice must
not error or duplicate.

**5. Reference data**

`valwr/store/reference.py` — pull from valorant-api.com (no key needed) into
`ref_agents`, `ref_maps`, `ref_tiers`, `ref_seasons`. Exact endpoints are in
`docs/API-NOTES.md`.

The agent→role mapping matters most; it drives every composition feature later.
Do not hardcode roles — read them from the API response.

**6. Smoke test**

`valwr/check.py`, runnable as `python -m valwr.check`. It should:

- Resolve `RIOT_NAME`/`RIOT_TAG` to a PUUID
- Fetch and print your last 5 matches (map, agent, K/D/A, result)
- Print counts from the reference tables
- Print which rate limit tier is configured and the resulting requests/minute

Make the failure modes readable: a bad key, a wrong Riot ID, and a network error
should each produce a clear message.

## Constraints

- Every HenrikDev call goes through one place, even now — a thin client in
  `valwr/collect/client.py` with the `Authorization` header set from config.
  Phase 1 will add the rate limiter behind the same interface.
- No hardcoded keys anywhere. `.env` only.
- `git status` must show no `.env`, no `data/*.db`.

## Acceptance criteria

`python -m valwr.check` prints your account, your last 5 real matches, and
counts of 20+ agents and 10+ maps. Running `schema.py` twice is a no-op.

## Do not build yet

No crawler, no frontier logic, no feature code, no model. If the smoke test
works and the tables exist, the phase is done.

## Also

Remind me to apply for the HenrikDev **Enhanced** key (90 req/min) at
<https://api.henrikdev.xyz/dashboard/> if I have not already — approval takes
days, and it triples crawl throughput in Phase 1.
