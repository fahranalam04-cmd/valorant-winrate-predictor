# Data Model

SQLite. One file, `data/valwr.db`. Chosen because the whole dataset fits
comfortably, it needs no server, and a single file is trivially backed up before
a risky migration.

The central design idea: **raw responses are kept forever, normalised tables are
derived and rebuildable.** API calls are the expensive, rate-limited resource.
Parsing is cheap. If the schema turns out wrong at Phase 4, the fix is a
re-parse, not a re-crawl.

---

## Layer 1 — raw

```sql
CREATE TABLE raw_response (
  id            INTEGER PRIMARY KEY,
  endpoint      TEXT NOT NULL,      -- e.g. 'v4/by-puuid/matches'
  params        TEXT NOT NULL,      -- JSON of path+query params
  fetched_at    INTEGER NOT NULL,   -- unix seconds
  status        INTEGER NOT NULL,
  body          TEXT NOT NULL       -- verbatim JSON
);
CREATE INDEX idx_raw_endpoint_fetched ON raw_response(endpoint, fetched_at);
```

Never mutate a row here. Never delete one to "clean up". This table is the
project's insurance policy against every parsing mistake made downstream.

## Layer 2 — normalised

```sql
CREATE TABLE matches (
  match_id      TEXT PRIMARY KEY,
  started_at    INTEGER NOT NULL,   -- unix seconds; the temporal anchor
  map           TEXT NOT NULL,
  mode          TEXT NOT NULL,
  queue         TEXT,
  region        TEXT NOT NULL,
  season        TEXT,               -- episode/act, for the patch-era feature
  rounds_red    INTEGER,
  rounds_blue   INTEGER,
  winner        TEXT,               -- 'Red' | 'Blue' | 'Draw' | NULL if unknown
  ingested_at   INTEGER NOT NULL
);
CREATE INDEX idx_matches_started ON matches(started_at);

CREATE TABLE match_players (
  match_id      TEXT NOT NULL REFERENCES matches(match_id),
  puuid         TEXT NOT NULL,
  team          TEXT NOT NULL,      -- 'Red' | 'Blue'
  agent         TEXT NOT NULL,
  party_id      TEXT,               -- premade detection
  tier          INTEGER,            -- rank at time of match
  account_level INTEGER,
  score         INTEGER,
  kills         INTEGER,
  deaths        INTEGER,
  assists       INTEGER,
  headshots     INTEGER,
  bodyshots     INTEGER,
  legshots      INTEGER,
  damage_dealt  INTEGER,
  damage_taken  INTEGER,
  PRIMARY KEY (match_id, puuid)
);
CREATE INDEX idx_mp_puuid ON match_players(puuid);

CREATE TABLE players (
  puuid           TEXT PRIMARY KEY,
  name            TEXT,
  tag             TEXT,
  current_tier    INTEGER,
  last_seen_at    INTEGER,
  history_fetched_at INTEGER        -- NULL = history never pulled
);
```

### The index that matters

```sql
CREATE INDEX idx_mp_puuid_time ON match_players(puuid, match_id);
```

Every feature query is "rows for player X, in matches before time T". Without a
composite index reaching `started_at`, that becomes a full scan per player, ten
times per prediction. Phase 2 should benchmark this — if the join is slow,
denormalise `started_at` onto `match_players` and index `(puuid, started_at)`.
Duplicating one column to make the hot query fast is the right call here.

## Layer 3 — crawler state

```sql
CREATE TABLE frontier (
  puuid         TEXT PRIMARY KEY,
  discovered_at INTEGER NOT NULL,
  tier_band     INTEGER,            -- for stratified sampling
  state         TEXT NOT NULL,      -- 'pending' | 'fetching' | 'done' | 'failed'
  attempts      INTEGER DEFAULT 0,
  last_error    TEXT
);
CREATE INDEX idx_frontier_state ON frontier(state, tier_band);
```

Crash safety comes from this table being the single source of truth for crawl
progress. A `fetching` row older than a few minutes is a crashed worker — reset
it to `pending` on startup. Do not hold the frontier in memory.

## Layer 4 — reference

Pulled once from valorant-api.com: `ref_agents` (uuid, name, **role**),
`ref_maps`, `ref_tiers` (numeric ordering ← the thing that makes ranks
comparable), `ref_seasons` (act boundaries).

Never hardcode a role table. Agents get reworked and reassigned.

---

## The temporal store — the leakage firewall

`valwr/store/temporal.py` is the **only** sanctioned way to read player history
for feature building. Its entire job is making leakage hard to write by
accident.

Every function takes an `as_of` timestamp and filters `started_at < as_of`,
strictly. Not `<=` — a match cannot inform a prediction about itself.

```python
def player_history(puuid: str, as_of: int, limit: int | None = None) -> list[Row]:
    """Matches for `puuid` that had already finished before `as_of`.

    `as_of` is the *start* time of the match being predicted. Strict `<`:
    a match never informs a prediction about itself.
    """
```

The rule enforced in review: **no feature code may query `match_players`
directly.** If a feature needs data, it goes through this module. A single
chokepoint that always demands `as_of` is far more reliable than remembering to
add a time filter in twenty different places.

### Why this design and not a feature store

The obvious alternative is precomputing rolling aggregates per player. It is
faster, and it is how you would do this at scale. It is rejected here because
rolling aggregates make leakage nearly invisible: an off-by-one in a window
boundary silently includes the target match, and nothing about the code looks
wrong. Recomputing from an explicit `as_of` is slower but auditable — you can
point at any feature value and trace it to source rows with earlier timestamps,
which is exactly what `test/test_leakage.py` does.

Revisit only if feature building becomes the bottleneck, and only behind the
same interface.

---

## Data volume expectations

At 30 req/min with `size=10`, one request yields up to 10 matches. Overlap is
heavy — competitive lobbies pull from a shared pool, so the same match arrives
via multiple players.

Rough expectation: **40,000–80,000 unique matches** over a few days of
intermittent crawling, giving 400k–800k player-match rows. That is ample; the
constraint on this project is feature quality, not sample size.

Set a target and stop. More data does not fix a leaky feature pipeline.

## Sampling bias — the thing to actively manage

Seeding purely from the leaderboard produces an Immortal/Radiant dataset, and a
model trained on it will not transfer to Fahran's own lobbies. Seeding purely
from one player's history produces a dataset centred on that player's rank, too
narrow to generalise.

The crawler does both, and **stratifies**: track `tier_band` per discovered
player, and when one band is over-represented in the frontier, deprioritise it.
Log the resulting rank distribution at the end of every crawl session — if it is
not reasonably spread across bands, the sampling is broken, and every downstream
result inherits the problem.

This is worth getting right for its own sake, and it is also a direct answer to
the "how did you handle sampling bias" interview question.
