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

import sqlite3
import time
from datetime import datetime
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
               "trade_kills", "traded_deaths", "kast_rounds", "clutches"]


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


def iter_matches(conn: sqlite3.Connection) -> Iterator[dict]:
    """Every match object across every stored matchlist response."""
    from valwr.store import raw
    for _, doc in raw.iter_responses(conn, "%matches%"):
        for m in doc.get("data") or []:
            yield m


def normalize_all(conn: sqlite3.Connection, verbose: bool = True) -> dict[str, int]:
    stats = {"matches": 0, "players": 0, "flagged": 0, "errors": 0}
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
