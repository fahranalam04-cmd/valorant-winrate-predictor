# Phase 2 — Normalisation and the temporal store

Read `CLAUDE.md` and `docs/DATA.md` before starting. Phase 1 must pass and
`raw_response` should hold real data.

## Goal

Turn raw JSON into queryable tables, and build the single query layer that makes
leakage hard to write by accident. **This phase is the foundation the model's
credibility rests on** — if the temporal store is wrong, every result after it
is meaningless, and it will look *better*, not worse.

## Build

**1. Parser — `valwr/store/normalize.py`**

`raw_response` → `matches`, `match_players`, `players`.

Must be **idempotent**: re-parsing the same raw rows updates rather than
duplicates. Use upserts keyed on `match_id` and `(match_id, puuid)`. This gets
run repeatedly as parsing bugs are found, so it has to be safe to re-run.

Handle the real-world mess: matches with fewer than 10 players (disconnects),
missing fields on older records, anonymised players, deathmatch or custom modes
that slipped through the filter. Flag rather than silently drop — a
`data_quality` column or a flags table — so you can audit what was excluded.

`started_at` is the temporal anchor for everything downstream. Verify it parses
to a sane unix timestamp and not, say, a string or milliseconds.

**2. The temporal store — `valwr/store/temporal.py`**

The only sanctioned way for feature code to read player history.

```python
def player_history(puuid: str, as_of: int, limit: int | None = None) -> list[Row]:
    """Matches for `puuid` that had already finished before `as_of`.

    `as_of` is the *start* time of the match being predicted. Strict `<`:
    a match never informs a prediction about itself.
    """
```

Add the variants features will need — history filtered by map, by agent, by
map+agent, and a last-N accessor — but **every one takes `as_of` as a required
positional argument.** No defaults. Making it impossible to forget is the entire
design goal.

Read the "Why this design and not a feature store" section of `docs/DATA.md`
before proposing an optimisation here.

**3. Performance**

Benchmark `player_history` — it runs ~10 times per prediction and across the
whole training set during feature building.

If the `matches`/`match_players` join is slow, denormalise `started_at` onto
`match_players` and index `(puuid, started_at)`. Duplicating one column to make
the hot query fast is the right trade here; say so in a comment.

Report the measured timing before and after.

**4. Tests — `test/test_store.py`**

- No orphan `match_players` rows
- Every match has exactly 10 players, or is flagged
- Per-player timestamps are monotonic
- Re-running the parser changes no row counts
- **`player_history(puuid, as_of)` never returns a row with
  `started_at >= as_of`** — property-test this across many sampled players and
  timestamps, not one example

## Constraints

- Never mutate or delete `raw_response`.
- Feature code must never query `match_players` directly. Enforce by convention
  now; Phase 4's audit checks it.

## Acceptance criteria

`pytest test/test_store.py` passes, including the property test on `as_of`.
Report how many matches and player-match rows normalised, and how many rows were
flagged and why.

## Do not build yet

No features, no rating, no model. Parsing and the query layer only.
