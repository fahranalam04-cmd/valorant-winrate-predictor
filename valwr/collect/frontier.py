"""The crawl frontier.

Backed by the `frontier` table rather than memory, so a crash loses nothing.
State machine: pending -> fetching -> done | failed.
"""

from __future__ import annotations

import sqlite3
import time

MAX_ATTEMPTS = 3
STALE_SECONDS = 300  # a 'fetching' row older than this is a crashed worker

UNRANKED_BAND = 0


def band_of(tier: int | None) -> int:
    """Tier id -> rank band.

    Tier ids run 0 (unranked) then 3..27 in threes: Iron 3-5, Bronze 6-8,
    Silver 9-11, ... Immortal 24-26, Radiant 27. Integer division by three
    gives a stable band without hardcoding names, which get reworked.
    """
    if not tier:
        return UNRANKED_BAND
    return tier // 3


def enqueue(conn: sqlite3.Connection, puuid: str, tier: int | None = None) -> bool:
    """Add a PUUID if unseen. Returns True if it was new."""
    cur = conn.execute(
        "INSERT INTO frontier (puuid, discovered_at, tier_band) VALUES (?,?,?) "
        "ON CONFLICT(puuid) DO NOTHING",
        (puuid, int(time.time()), band_of(tier)),
    )
    return cur.rowcount > 0


def enqueue_many(conn: sqlite3.Connection, items: list[tuple[str, int | None]]) -> int:
    new = sum(enqueue(conn, puuid, tier) for puuid, tier in items)
    conn.commit()
    return new


def recover_stale(conn: sqlite3.Connection, stale_seconds: int = STALE_SECONDS) -> int:
    """Return crashed workers' claims to the queue. Run on startup."""
    cutoff = int(time.time()) - stale_seconds
    cur = conn.execute(
        "UPDATE frontier SET state='pending', claimed_at=NULL "
        "WHERE state='fetching' AND (claimed_at IS NULL OR claimed_at < ?)",
        (cutoff,),
    )
    conn.commit()
    return cur.rowcount


def claim(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """Claim the next PUUID: least-crawled band first, highest leverage within it.

    Two objectives, in priority order.

    **Band stratification** (`p.done ASC`) keeps rank coverage even, so the
    dataset is not silently dominated by whichever bracket the seeds favoured.

    **Coverage leverage** (`l.n DESC`) is the fix for a mistake worth recording.
    The first version picked the oldest pending player in the least-crawled
    band, which optimises breadth -- and breadth was never the constraint. Each
    request fetches one player but discovers roughly sixty more, so the frontier
    outran the fetches at 1:56 and 86% of known players sat on a single match
    with no prior history. Features need *history*, so only 1.3% of collected
    matches had all ten players covered: 3,223 matches, 42 of them trainable.

    Leverage counts how many already-collected matches a pending player appears
    in. Fetching a player who shows up in five of our matches adds history to
    five matches at once, instead of one. Same request budget, several times the
    usable output.
    """
    row = conn.execute(
        """
        WITH progress AS (
          SELECT tier_band,
                 SUM(state='done')    AS done,
                 SUM(state='pending') AS pending
          FROM frontier GROUP BY tier_band
        ),
        leverage AS (
          SELECT puuid, COUNT(DISTINCT match_id) AS n
          FROM match_players GROUP BY puuid
        )
        SELECT f.* FROM frontier f
        JOIN progress p ON p.tier_band = f.tier_band
        LEFT JOIN leverage l ON l.puuid = f.puuid
        WHERE f.state='pending' AND p.pending > 0
        ORDER BY p.done ASC, COALESCE(l.n, 0) DESC, f.discovered_at ASC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    conn.execute(
        "UPDATE frontier SET state='fetching', claimed_at=? WHERE puuid=?",
        (int(time.time()), row["puuid"]),
    )
    conn.commit()
    return row


def coverage_summary(conn: sqlite3.Connection) -> dict[str, int]:
    """How many matches are actually trainable -- the metric that matters.

    Match count alone is misleading: a match whose players have no prior
    history carries no features.
    """
    rows = conn.execute(
        """
        SELECT SUM(CASE WHEN EXISTS (
                 SELECT 1 FROM match_players p2
                 WHERE p2.puuid = mp.puuid AND p2.started_at < mp.started_at
               ) THEN 1 ELSE 0 END) AS covered
        FROM match_players mp GROUP BY mp.match_id
        """
    ).fetchall()
    covered = [r["covered"] or 0 for r in rows]
    return {
        "matches": len(covered),
        "full_10": sum(1 for c in covered if c >= 10),
        "usable_8": sum(1 for c in covered if c >= 8),
        "partial_5": sum(1 for c in covered if c >= 5),
    }


def release(conn: sqlite3.Connection, puuid: str) -> None:
    """Hand a claim back unfetched, without charging an attempt.

    Used when we stop for our own reasons (deadline, ctrl-c). The puuid did
    nothing wrong, so it must not accumulate toward MAX_ATTEMPTS.
    """
    conn.execute(
        "UPDATE frontier SET state='pending', claimed_at=NULL WHERE puuid=?", (puuid,)
    )
    conn.commit()


def complete(conn: sqlite3.Connection, puuid: str) -> None:
    conn.execute(
        "UPDATE frontier SET state='done', claimed_at=NULL WHERE puuid=?", (puuid,)
    )
    conn.commit()


def fail(conn: sqlite3.Connection, puuid: str, error: str) -> None:
    """Record a failure. Gives up after MAX_ATTEMPTS so one bad PUUID
    cannot block the crawl indefinitely."""
    conn.execute(
        "UPDATE frontier SET attempts = attempts + 1, last_error = ?, claimed_at = NULL, "
        "state = CASE WHEN attempts + 1 >= ? THEN 'failed' ELSE 'pending' END "
        "WHERE puuid = ?",
        (error[:500], MAX_ATTEMPTS, puuid),
    )
    conn.commit()


def counts_by_state(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        r["state"]: r["n"]
        for r in conn.execute("SELECT state, COUNT(*) n FROM frontier GROUP BY state")
    }


def counts_by_band(conn: sqlite3.Connection) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {}
    for r in conn.execute(
        "SELECT tier_band, state, COUNT(*) n FROM frontier GROUP BY tier_band, state"
    ):
        out.setdefault(r["tier_band"], {})[r["state"]] = r["n"]
    return out
