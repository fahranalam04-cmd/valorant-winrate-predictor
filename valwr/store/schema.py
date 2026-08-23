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
CREATE TABLE IF NOT EXISTS raw_response (
  id            INTEGER PRIMARY KEY,
  endpoint      TEXT NOT NULL,
  params        TEXT NOT NULL,
  fetched_at    INTEGER NOT NULL,
  status        INTEGER NOT NULL,
  body          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_raw_endpoint_fetched
  ON raw_response(endpoint, fetched_at);

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
CREATE INDEX IF NOT EXISTS idx_matches_started ON matches(started_at);

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
  PRIMARY KEY (match_id, puuid)
);
CREATE INDEX IF NOT EXISTS idx_mp_puuid ON match_players(puuid);

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
CREATE INDEX IF NOT EXISTS idx_frontier_state ON frontier(state, tier_band);

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


def connect(database_path: Path) -> sqlite3.Connection:
    """Open the database, creating its directory if needed."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    # Concurrent readers alongside the crawler's writer.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_all(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
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
