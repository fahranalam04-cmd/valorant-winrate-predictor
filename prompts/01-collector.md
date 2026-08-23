# Phase 1 — Collector

Read `CLAUDE.md`, `docs/API-NOTES.md`, and `docs/DATA.md` before starting.
Phase 0 must pass.

## Goal

A polite, resumable, rank-stratified crawler that fills `raw_response` with
competitive match data without ever tripping the rate limit.

## Context you need

HenrikDev Basic is **30 requests per minute**. The maintainer states the API is
not intended for large analytics projects. This is the binding constraint on the
whole project, and the crawler's design is mostly a response to it.

The efficiency lever: one matchlist call returns up to 10 full matches, each
containing all 10 players. A single request can therefore yield 100 player-match
rows and discover up to 90 new PUUIDs.

## Build

**1. Rate limiter — `valwr/collect/limiter.py`**

Token bucket, configured from `HENRIK_TIER` (basic 30/min, enhanced 90/min).
Every HenrikDev call in the project goes through it. Leave headroom — target
~90% of the stated limit, because bursts at exactly the limit will trip it.

On `429`: honour `Retry-After` if present, exponential backoff otherwise. Log it
prominently — a 429 means the limiter is misconfigured, not that the API is
being difficult.

**2. Frontier — `valwr/collect/frontier.py`**

Backed by the `frontier` table, not memory. State machine:
`pending → fetching → done | failed`.

Crash recovery on startup: reset any `fetching` row older than ~5 minutes back
to `pending` — that is a crashed worker. Cap `attempts`; a PUUID that fails
three times goes to `failed` with the error, and does not block the crawl.

**3. Seeding — `valwr/collect/seed.py`**

Two sources, both needed:
- The leaderboard endpoint (skip `is_anonymized` entries — no usable PUUID)
- My own PUUID from config

Leaderboard alone gives an Immortal/Radiant dataset that will not transfer to my
own lobbies. My history alone is too narrow. Both together, then stratification
does the rest.

**4. Crawl loop — `valwr/collect/crawl.py`**

For each `pending` PUUID: fetch matchlist filtered to competitive, store the raw
response verbatim, extract the other 9 PUUIDs per match, enqueue unseen ones.

**Rank stratification.** Track `tier_band` per discovered player. When selecting
the next PUUID to fetch, deprioritise over-represented bands. Without this the
crawl drifts upward — high-rank players have more games and get discovered more
often — and the dataset ends up unusable for predicting my own matches.

Explain the stratification approach you choose before implementing it; there are
several reasonable ones and I want to understand the trade-off.

**5. CLI**

`python -m valwr.collect --minutes 30` — run for a bounded time, then stop
cleanly.

At session end, log: requests made, matches stored (new vs duplicate), PUUIDs
discovered, frontier depth, and **the rank distribution across bands**. That
last one is the health check — if it is concentrated at the top, stratification
is broken.

## Constraints

- Store raw responses verbatim. Never parse-then-discard; API calls are the
  expensive resource and parsing is free. Phase 2 does the parsing.
- Never bypass the limiter.
- No key rotation, no proxies, no parallel accounts. See
  `docs/ETHICS-AND-TOS.md`.
- Single-threaded is fine — the rate limit dominates, so concurrency buys
  nothing here and costs correctness.

## Acceptance criteria

1. Run for 30 minutes. It stores matches and never hits a 429.
2. `kill -9` it mid-run, restart. No duplicate matches, no lost frontier
   entries, no rows stuck in `fetching`.
3. The end-of-session rank distribution is spread across bands, not concentrated
   in the top two.

Test #2 by actually doing it, not by reading the code.

## Do not build yet

No parsing into normalised tables — that is Phase 2. `raw_response` only.
