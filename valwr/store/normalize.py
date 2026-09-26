"""Raw JSON -> normalised tables.

Idempotent by design: this gets re-run every time a parsing bug is found, so
every write is an upsert keyed on natural ids. Nothing here re-fetches; it
reads the compressed bodies in `raw_response`, which is exactly why that layer
is kept verbatim.

Real data is messy -- disconnects, incomplete matches, anonymised players. The
policy is to flag rather than silently drop, so `matches.data_quality` records
why a row is suspect and Phase 4 can decide what to exclude.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from valwr.rating import components

EXPECTED_ROSTER = 10


class ParseError(ValueError):
    pass


def parse_started_at(value: Any) -> int:
    """ISO 8601 -> unix seconds.

    `metadata.started_at` is a string like '2026-08-23T05:39:55.948Z', but
    `matches.started_at` is INTEGER because it anchors every time-gated
    feature. Getting this wrong does not raise -- it silently corrupts the
    leakage firewall -- so it raises loudly here instead.
    """
    if isinstance(value, (int, float)):
        # Some endpoints return epoch; tolerate but normalise ms -> s.
        return int(value / 1000) if value > 1e11 else int(value)
    if not isinstance(value, str) or not value:
        raise ParseError(f"unparseable started_at: {value!r}")
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError as e:
        raise ParseError(f"unparseable started_at: {value!r}") from e


def _name(obj: Any) -> str | None:
    if isinstance(obj, dict):
        return obj.get("name")
    return obj if isinstance(obj, str) else None


def parse_match(m: dict) -> tuple[dict, list[dict], list[str]]:
    """One match object -> (match row, player rows, quality flags)."""
    md = m.get("metadata") or {}
    match_id = md.get("match_id")
    if not match_id:
        raise ParseError("match has no match_id")

    flags: list[str] = []

    teams = m.get("teams") or []
    rounds_by_team = {
        t.get("team_id"): (t.get("rounds") or {}).get("won") for t in teams
    }
    winners = [t.get("team_id") for t in teams if t.get("won")]
    winner = winners[0] if len(winners) == 1 else None
    if winner is None:
        flags.append("no_single_winner")

    if md.get("is_completed") is False:
        flags.append("incomplete")

    queue = md.get("queue") or {}

    match_row = {
        "match_id": match_id,
        "started_at": parse_started_at(md.get("started_at")),
        "map": _name(md.get("map")) or "?",
        # `mode` is what you filter on ('competitive'); `queue` keeps the
        # broader family ('Standard') for context.
        "mode": queue.get("id") or "?",
        "queue": queue.get("mode_type"),
        "region": md.get("region") or "?",
        "season": (md.get("season") or {}).get("short"),
        "rounds_red": rounds_by_team.get("Red"),
        "rounds_blue": rounds_by_team.get("Blue"),
        "winner": winner,
        "ingested_at": int(time.time()),
    }

    # Round- and kill-derived components. Computed here so the expensive raw
    # body is walked once at parse time rather than on every feature build.
    comps = components.match_components(m)

    players = m.get("players") or []
    if len(players) != EXPECTED_ROSTER:
        flags.append(f"roster_{len(players)}")

    player_rows = []
    seen: set[str] = set()
    for p in players:
        puuid = p.get("puuid")
        if not puuid or puuid in seen:
            # Anonymised or duplicated entries carry no usable identity.
            flags.append("missing_or_duplicate_puuid")
            continue
        seen.add(puuid)
        stats = p.get("stats") or {}
        dmg = stats.get("damage") or {}
        # The API names the two basic slots `ability1`/`ability2`; the columns
        # use `ability_1`/`ability_2` to read as slots rather than as one
        # run-on word. A player who cast nothing is 0, not NULL -- NULL here
        # means the response predates the field or omitted it.
        casts = p.get("ability_casts")
        casts = casts if isinstance(casts, dict) else {}
        player_rows.append({
            "match_id": match_id,
            "puuid": puuid,
            "team": p.get("team_id") or "?",
            "agent": _name(p.get("agent")) or "?",
            "party_id": p.get("party_id"),
            "tier": (p.get("tier") or {}).get("id"),
            "account_level": p.get("account_level"),
            "score": stats.get("score"),
            "kills": stats.get("kills"),
            "deaths": stats.get("deaths"),
            "assists": stats.get("assists"),
            "headshots": stats.get("headshots"),
            "bodyshots": stats.get("bodyshots"),
            "legshots": stats.get("legshots"),
            "damage_dealt": dmg.get("dealt"),
            "damage_taken": dmg.get("received"),
            "ability_grenade": casts.get("grenade"),
            "ability_1": casts.get("ability1"),
            "ability_2": casts.get("ability2"),
            "ability_ultimate": casts.get("ultimate"),
            # Denormalised from the match so history queries need no join.
            "started_at": match_row["started_at"],
            "map": match_row["map"],
            "won": None if winner is None else int(p.get("team_id") == winner),
            "_name": p.get("name"),
            "_tag": p.get("tag"),
            **comps.get(puuid, components.blank()),
        })

    match_row["data_quality"] = ",".join(sorted(set(flags))) or None
    return match_row, player_rows, flags


MATCH_COLS = ["match_id", "started_at", "map", "mode", "queue", "region", "season",
              "rounds_red", "rounds_blue", "winner", "data_quality", "ingested_at"]

PLAYER_COLS = ["match_id", "puuid", "team", "agent", "party_id", "tier",
               "account_level", "score", "kills", "deaths", "assists",
               "headshots", "bodyshots", "legshots", "damage_dealt", "damage_taken",
               "started_at", "map", "won",
               "rounds_played", "first_bloods", "first_deaths", "multikills",
               "trade_kills", "traded_deaths", "kast_rounds", "clutches",
               "plants", "defuses",
               "ability_grenade", "ability_1", "ability_2", "ability_ultimate"]


def upsert_match(conn: sqlite3.Connection, row: dict) -> None:
    cols = ",".join(MATCH_COLS)
    ph = ",".join("?" * len(MATCH_COLS))
    updates = ",".join(f"{c}=excluded.{c}" for c in MATCH_COLS if c != "match_id")
    conn.execute(
        f"INSERT INTO matches ({cols}) VALUES ({ph}) "
        f"ON CONFLICT(match_id) DO UPDATE SET {updates}",
        [row[c] for c in MATCH_COLS],
    )


def upsert_players(conn: sqlite3.Connection, rows: list[dict]) -> None:
    cols = ",".join(PLAYER_COLS)
    ph = ",".join("?" * len(PLAYER_COLS))
    updates = ",".join(f"{c}=excluded.{c}" for c in PLAYER_COLS
                       if c not in ("match_id", "puuid"))
    conn.executemany(
        f"INSERT INTO match_players ({cols}) VALUES ({ph}) "
        f"ON CONFLICT(match_id, puuid) DO UPDATE SET {updates}",
        [[r[c] for c in PLAYER_COLS] for r in rows],
    )
    # `players` tracks the latest identity we have seen for each puuid.
    conn.executemany(
        "INSERT INTO players (puuid, name, tag, current_tier, last_seen_at) "
        "VALUES (?,?,?,?,?) ON CONFLICT(puuid) DO UPDATE SET "
        "name=excluded.name, tag=excluded.tag, "
        "current_tier=excluded.current_tier, last_seen_at=excluded.last_seen_at",
        [(r["puuid"], r["_name"], r["_tag"], r["tier"], int(time.time())) for r in rows],
    )


# The two kinds of response that carry whole match objects: a player's
# matchlist, whose `data` is a list of them, and one match fetched by id -- the
# settler's -- whose `data` is a single match. Reading only the first left
# every match that arrived through the second out of any re-parse: a Swiftplay
# game the settler fetched kept empty spike columns after a backfill had
# filled every other row in the table.
MATCH_ENDPOINTS = ("%matches%", "%/match/%")


def iter_matches(conn: sqlite3.Connection) -> Iterator[dict]:
    """Every match object in every stored response that carries one."""
    from valwr.store import raw
    for _, doc in raw.iter_responses(conn, MATCH_ENDPOINTS):
        data = doc.get("data")
        yield from ([data] if isinstance(data, dict) else data or [])


# Commit every N matches rather than once at the end. SQLite allows a single
# writer, so one long transaction holds the lock for the whole run and starves
# the crawler -- which is exactly what happened: 59 consecutive crawler runs
# died with "database is locked" while a normalise pass held the lock.
COMMIT_EVERY = 200


def ingest(conn: sqlite3.Connection, payload: dict) -> dict[str, int]:
    """Parse and store one matchlist response, immediately.

    `normalize_all` reparses every response ever cached, which is the right
    tool for a backfill and far too heavy for a live fetch -- it would walk
    thousands of blobs to store ten matches.

    This exists because the live path silently did not store anything at all.
    `HenrikClient.matches()` fetches and caches the raw body; it does *not*
    normalise. `resolve` called it and then asked whether the player had
    history, on the strength of a comment claiming "the crawler normalises
    inline" -- true of the crawler, false of the client. So every live fetch
    spent an API call, wrote a blob nothing read, and left the player exactly
    as unknown as before. Local history sat frozen for twelve days while
    looking live.
    """
    stats = {"matches": 0, "players": 0, "errors": 0}
    for m in (payload or {}).get("data") or []:
        try:
            match_row, player_rows, _ = parse_match(m)
        except ParseError:
            stats["errors"] += 1
            continue
        upsert_match(conn, match_row)          # matches before their players
        if player_rows:
            upsert_players(conn, player_rows)
        stats["matches"] += 1
        stats["players"] += len(player_rows)
    conn.commit()
    return stats


class OutOfSpace(RuntimeError):
    """Stopped before the disk filled. Everything parsed so far is committed."""


def _free_bytes(conn: sqlite3.Connection) -> int:
    """Free space on whichever drive the database actually lives on."""
    row = conn.execute("PRAGMA database_list").fetchone()
    path = Path(row[2]) if row and row[2] else Path.cwd()
    return shutil.disk_usage(path.parent).free


def checkpoint(conn: sqlite3.Connection) -> None:
    """Fold the write-ahead log back into the database.

    SQLite cannot do this while a reader holds a snapshot open, so during a
    long re-parse the log grows without bound -- 14.9 GB here, which filled a
    476 GB disk and killed the run three times with errors that pointed
    everywhere except the cause. Checkpointing as it goes keeps it flat.
    """
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.OperationalError:
        pass            # a reader holds a snapshot; the next pass will get it


# Stop while there is still room to recover. Below this the checkpoint itself
# cannot run, which is what turns a full disk into a puzzle.
FREE_SPACE_FLOOR = 4 * 1024 ** 3

# How often to fold the log back in, counted in committed batches.
CHECKPOINT_EVERY = 10


def normalize_all(conn: sqlite3.Connection, verbose: bool = True) -> dict[str, int]:
    stats = {"matches": 0, "players": 0, "flagged": 0, "errors": 0}
    checkpoint(conn)
    batches = 0
    for m in iter_matches(conn):
        try:
            match_row, player_rows, flags = parse_match(m)
        except ParseError:
            stats["errors"] += 1
            continue
        # Matches must land before their players -- foreign keys are on.
        upsert_match(conn, match_row)
        if player_rows:
            upsert_players(conn, player_rows)
        stats["matches"] += 1
        stats["players"] += len(player_rows)
        if flags:
            stats["flagged"] += 1
        if stats["matches"] % COMMIT_EVERY == 0:
            conn.commit()          # release the write lock; let the crawler in
            batches += 1
            if batches % CHECKPOINT_EVERY == 0:
                checkpoint(conn)
                free = _free_bytes(conn)
                if free < FREE_SPACE_FLOOR:
                    conn.commit()
                    raise OutOfSpace(
                        f"stopping at {stats['matches']:,} matches: "
                        f"{free / 1024 ** 3:.1f} GB free, below the "
                        f"{FREE_SPACE_FLOOR / 1024 ** 3:.0f} GB floor. "
                        f"Everything parsed so far is committed. Free some "
                        f"space and re-run: it starts over, and re-parsing "
                        f"overwrites rows rather than duplicating them.")
        if verbose and stats["matches"] % 500 == 0:
            print(f"  normalised {stats['matches']} matches")
    conn.commit()
    return stats


def main(argv=None) -> int:
    """python -m valwr.store.normalize -- reparse everything in raw_response."""
    import argparse
    from valwr import config
    from valwr.store import schema

    ap = argparse.ArgumentParser(prog="valwr.store.normalize")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    schema.create_all(conn)

    stats = normalize_all(conn, verbose=not args.quiet)
    print(f"\nmatches parsed   {stats['matches']:,}")
    print(f"player rows      {stats['players']:,}")
    print(f"flagged          {stats['flagged']:,}")
    print(f"parse errors     {stats['errors']:,}")
    counts = schema.table_counts(conn)
    print(f"\nstored: {counts['matches']:,} matches, "
          f"{counts['match_players']:,} player rows, "
          f"{counts['players']:,} players")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
