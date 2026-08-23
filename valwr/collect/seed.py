"""Seeding the crawl frontier.

Which seeds you choose decides which population the model learns, because
matchmaking bounds reachability: a snowball from a Gold player reaches roughly
Silver..Diamond and never Iron or Radiant, since those players never share a
lobby with anyone who shares a lobby with you.

That makes leaderboard seeding a real decision rather than free extra data. It
injects a pure Immortal/Radiant cluster that does not connect to a mid-ladder
seed, giving a bimodal sample with a hole in the middle. Useful if you want to
model the whole ladder; actively counterproductive if you want to predict your
own games. Hence `--seed`.
"""

from __future__ import annotations

import sqlite3

from valwr.collect import frontier
from valwr.collect.client import HenrikClient


def seed_self(conn: sqlite3.Connection, client: HenrikClient, name: str, tag: str) -> str:
    """Seed from your own account. Returns your puuid."""
    data = client.account(name, tag).get("data", {})
    puuid = data.get("puuid")
    if not puuid:
        raise RuntimeError(f"could not resolve {name}#{tag}")
    frontier.enqueue(conn, puuid, None)
    conn.commit()
    return puuid


def seed_leaderboard(
    conn: sqlite3.Connection, client: HenrikClient, region: str, platform: str, limit: int = 200
) -> int:
    """Seed from the top of the ladder.

    Skips `is_anonymized` entries -- they carry no usable puuid and would
    otherwise poison the frontier with rows that can never be fetched.
    """
    resp = client.leaderboard(region, platform)
    data = resp.get("data")
    players = data.get("players", data) if isinstance(data, dict) else data or []

    items: list[tuple[str, int | None]] = []
    for p in players[:limit]:
        if p.get("is_anonymized") or p.get("is_banned"):
            continue
        puuid = p.get("puuid")
        if puuid:
            items.append((puuid, p.get("tier")))
    return frontier.enqueue_many(conn, items)
