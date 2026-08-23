"""Database schema.

Mirrors docs/DATA.md. Four layers: raw responses (never mutated), normalised
tables (derived, rebuildable), crawler state, and static reference data.

Idempotent -- running this twice is a no-op.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
-- Layer 1: raw. Never mutated, never deleted. API calls are the expensive,
-- rate-limited resource; parsing is free. This is the insurance policy against
-- every parsing mistake made downstream.
--
-- `body` is zlib-compressed JSON, not text. A v4 matchlist response measured
-- ~450 KB per match uncompressed, which is ~23 GB at 50k matches; zlib takes
-- that to 7.4% (~1.7 GB). Compressing rather than trimming keeps the response
-- verbatim, which is the whole point of this layer. Read via store.raw.
CREATE TABLE IF NOT EXISTS raw_response (
  id            INTEGER PRIMARY KEY,
  endpoint      TEXT NOT NULL,
  params        TEXT NOT NULL,
  fetched_at    INTEGER NOT NULL,
  status        INTEGER NOT NULL,
  body          BLOB NOT NULL
);

-- Layer 2: normalised.
CREATE TABLE IF NOT EXISTS matches (
  match_id      TEXT PRIMARY KEY,
  started_at    INTEGER NOT NULL,
  map           TEXT NOT NULL,
  mode          TEXT NOT NULL,
  queue         TEXT,
  region        TEXT NOT NULL,
  season        TEXT,
  rounds_red    INTEGER,
  rounds_blue   INTEGER,
  winner        TEXT,
  data_quality  TEXT,
  ingested_at   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS match_players (
  match_id      TEXT NOT NULL REFERENCES matches(match_id),
  puuid         TEXT NOT NULL,
  team          TEXT NOT NULL,
  agent         TEXT NOT NULL,
  party_id      TEXT,
  tier          INTEGER,
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
  -- Denormalised from `matches`. Every feature query is "player X's matches
  -- before time T", and carrying these here turns that into one index range
  -- scan instead of a join that fans out per match. Duplicating three columns
  -- to make the hot path fast is the right trade; see docs/DATA.md.
  started_at    INTEGER,
  map           TEXT,
  won           INTEGER,        -- 1/0 from this player's perspective, NULL if undecided
  PRIMARY KEY (match_id, puuid)
);
-- The index the whole feature pipeline leans on.

CREATE TABLE IF NOT EXISTS players (
  puuid              TEXT PRIMARY KEY,
  name               TEXT,
  tag                TEXT,
  current_tier       INTEGER,
  last_seen_at       INTEGER,
  history_fetched_at INTEGER
);

-- Layer 3: crawler state. The single source of truth for crawl progress --
-- deliberately not held in memory, so a crash loses nothing.
CREATE TABLE IF NOT EXISTS frontier (
  puuid         TEXT PRIMARY KEY,
  discovered_at INTEGER NOT NULL,
  tier_band     INTEGER,
  state         TEXT NOT NULL DEFAULT 'pending',
  attempts      INTEGER NOT NULL DEFAULT 0,
  claimed_at    INTEGER,
  last_error    TEXT
);

-- Match ids the crawler has already seen. Crawler bookkeeping, not the
-- normalised `matches` table: it answers "is this new?" without decompressing
-- every stored body, and it survives Phase 2 re-parses.
CREATE TABLE IF NOT EXISTS crawl_seen_match (
  match_id      TEXT PRIMARY KEY,
  first_seen_at INTEGER NOT NULL,
  tier_band     INTEGER
);

-- Layer 4: reference data from valorant-api.com. No key required.
CREATE TABLE IF NOT EXISTS ref_agents (
  uuid          TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  role          TEXT
);

CREATE TABLE IF NOT EXISTS ref_maps (
  uuid          TEXT PRIMARY KEY,
  name          TEXT NOT NULL
);

-- `tier` is the numeric ordering that makes ranks comparable. Resolve names
-- through this table rather than assuming the encoding is stable across acts.
CREATE TABLE IF NOT EXISTS ref_tiers (
  tier          INTEGER PRIMARY KEY,
  name          TEXT NOT NULL,
  division      TEXT
);

CREATE TABLE IF NOT EXISTS ref_seasons (
  uuid          TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  type          TEXT,
  parent_uuid   TEXT,
  start_time    TEXT,
  end_time      TEXT
);
"""


INDEXES = """
CREATE INDEX IF NOT EXISTS idx_raw_endpoint_fetched
  ON raw_response(endpoint, fetched_at);
CREATE INDEX IF NOT EXISTS idx_matches_started ON matches(started_at);
CREATE INDEX IF NOT EXISTS idx_mp_puuid ON match_players(puuid);
-- The index the whole feature pipeline leans on.
CREATE INDEX IF NOT EXISTS idx_mp_puuid_time ON match_players(puuid, started_at);
CREATE INDEX IF NOT EXISTS idx_mp_puuid_map ON match_players(puuid, map, started_at);
CREATE INDEX IF NOT EXISTS idx_mp_puuid_agent ON match_players(puuid, agent, started_at);
CREATE INDEX IF NOT EXISTS idx_frontier_state ON frontier(state, tier_band);
"""

# Columns added after a table first shipped. CREATE TABLE IF NOT EXISTS will
# not add them to an existing database, and dropping the table would throw
# away a crawl that cost hours of rate-limited requests.
MIGRATIONS: dict[str, dict[str, str]] = {
    "match_players": {
        "started_at": "INTEGER",
        "map": "TEXT",
        "won": "INTEGER",
    },
    "matches": {
        "data_quality": "TEXT",
    },
}


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Add any columns missing from an existing database. Idempotent."""
    applied = []
    for table, columns in MIGRATIONS.items():
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue  # table not created yet; the schema script will handle it
        for name, decl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                applied.append(f"{table}.{name}")
    conn.commit()
    return applied


def connect(database_path: Path) -> sqlite3.Connection:
    """Open the database, creating its directory if needed."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    # Concurrent readers alongside the crawler's writer.
    conn.execute("PRAGMA journal_mode=WAL")
    # WAL allows many readers but only one writer. The normaliser and the
    # crawler both write, so without this a concurrent run fails outright
    # instead of waiting its turn.
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_all(conn: sqlite3.Connection) -> None:
    """Create tables, apply column migrations, then build indexes.

    Order matters: an index on a column a migration is about to add would
    fail on an existing database.
    """
    conn.executescript(SCHEMA)
    migrate(conn)
    conn.executescript(INDEXES)
    conn.commit()


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row count per table, for the smoke test and crawl summaries."""
    tables = [
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    return {t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"] for t in tables}
