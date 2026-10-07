"""The game's own Performance Score, read from your client after a match.

    python -m valwr.live.client_scores        # backfill recorded matches

Patch 13.06 replaced combat score with Performance Score (0-500), and no public
API carries it. The End of Game screen shows it, and that screen is built from
the client's ``pd/match-details/v1/matches/{id}`` -- which holds it for every
player, under scrambled field names. Which one is the score was settled by
matching against the screen: on one match the four 0-500 candidates read 215,
358, 167 and 102 for the account owner, whose screen said 215.

So ``scores.TempValueF`` is Performance Score. It is kept with its decimals
(215.10 for that 215) and shown rounded, as the game does. A scrambled name is
the kind of thing a patch can move, so every value is checked to be a number in
0-500 before it is trusted; anything else is reported as a changed format and
nothing is stored, rather than storing the wrong column under the right name.

Read-only: one GET per match, to Riot's own server with the running game's
session, for matches you played. Collected while the game is open -- the
session exists only then -- after each match, and for any recorded match still
missing one.
"""

from __future__ import annotations

import json
import sqlite3
import time

# The obfuscated field, and where the game's own up/down breakdown sits.
PS_FIELD = "TempValueF"
BREAKDOWN = ("TempValueL", ("TempValueP", "TempValueQ"))
PS_RANGE = (0, 500)
# How many matches one pass will ask about. A pass runs once a minute while
# the game is open, so a backlog clears in a few minutes without a burst.
PER_PASS = 5


class FormatChanged(RuntimeError):
    """The client's match-details no longer look the way this expects."""


class NotScored(RuntimeError):
    """A match the game gives no Performance Score -- Team Deathmatch did not.
    Not a format change: the field is absent for everyone, not malformed."""


# Matches the client answered with nothing usable -- not held, or not scored --
# so a pass a minute does not ask about them again for as long as this process
# runs. Only ever a handful of ids.
_NO_ANSWER: set[str] = set()


def parse(data: dict) -> list[tuple[str, float, dict]]:
    """(puuid, performance score, breakdown) for every player in the match."""
    players = data.get("players") or []
    if not players:
        raise FormatChanged("no player list in match-details")
    if all((p.get("scores") or {}).get(PS_FIELD) is None for p in players):
        raise NotScored("no player in this match has a Performance Score")
    out = []
    for p in players:
        scores = p.get("scores") or {}
        value = scores.get(PS_FIELD)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not PS_RANGE[0] <= value <= PS_RANGE[1]):
            raise FormatChanged(
                f"scores.{PS_FIELD} is {value!r}, not a 0-500 Performance "
                f"Score; the client's format has changed")
        outer, inner = BREAKDOWN
        parts = {}
        for key in inner:
            got = (scores.get(outer) or {}).get(key)
            if isinstance(got, dict):
                parts.update({k: v for k, v in got.items() if isinstance(v, str)})
        out.append((p.get("subject"), float(value), parts))
    if any(not puuid for puuid, _, _ in out):
        raise FormatChanged("a player in match-details has no subject")
    return out


def shown(value: float | None) -> int | None:
    """The whole number the End of Game screen shows: 215.10 is 215."""
    return None if value is None else int(value + 0.5)


def store(conn: sqlite3.Connection, match_id: str,
          rows: list[tuple[str, float, dict]], now: int | None = None) -> int:
    now = int(now or time.time())
    conn.executemany(
        "INSERT INTO client_scores (match_id, puuid, performance_score, "
        "breakdown, fetched_at) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(match_id, puuid) DO UPDATE SET "
        "performance_score = excluded.performance_score, "
        "breakdown = excluded.breakdown, fetched_at = excluded.fetched_at",
        [(match_id, puuid, ps, json.dumps(parts, sort_keys=True), now)
         for puuid, ps, parts in rows])
    conn.commit()
    return len(rows)


def scores_for(conn: sqlite3.Connection, match_id: str) -> dict[str, float]:
    """puuid -> Performance Score for one match; empty when not collected."""
    return {r[0]: r[1] for r in conn.execute(
        "SELECT puuid, performance_score FROM client_scores WHERE match_id = ?",
        (match_id,))}


def missing(conn: sqlite3.Connection, limit: int | None = None) -> list[str]:
    """Recorded bomb-mode matches with no Performance Scores yet, newest first.

    Only standard bomb defusal: the other modes carry no score to ask for.
    """
    sql = ("SELECT match_id FROM live_predictions WHERE standard_mode = 1 "
           "AND match_id NOT IN (SELECT DISTINCT match_id FROM client_scores) "
           "ORDER BY made_at DESC")
    args: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        args = (limit,)
    return [r[0] for r in conn.execute(sql, args)]


def fetch(session, match_id: str) -> dict | None:
    """match-details for one match, or None if the client does not have it.

    Through roster._get, so it verifies Riot's certificate, reads a 404 as
    "not there", and raises SessionExpired for a session that has aged out.
    """
    from valwr.live import roster
    return roster._get(session, f"{session.pd}/match-details/v1/matches/{match_id}")


def collect(conn: sqlite3.Connection, session, limit: int | None = PER_PASS,
            now: int | None = None) -> dict[str, int]:
    """Ask the client about recorded matches still missing their scores.

    Afterwards the stored verdicts are re-judged, because "played best" means
    Performance Score wherever the game's own numbers exist.
    """
    out = {"stored": 0, "unavailable": 0, "not_scored": 0, "format_changed": 0}
    todo = [m for m in missing(conn) if m not in _NO_ANSWER]
    for match_id in todo[:limit] if limit is not None else todo:
        data = fetch(session, match_id)
        if data is None:
            _NO_ANSWER.add(match_id)
            out["unavailable"] += 1
            continue
        try:
            rows = parse(data)
        except NotScored:
            _NO_ANSWER.add(match_id)
            out["not_scored"] += 1
            continue
        except FormatChanged:
            out["format_changed"] += 1
            continue
        store(conn, match_id, rows, now)
        out["stored"] += 1
    if out["stored"]:
        from valwr.live import outcomes
        outcomes.rescore(conn)
    return out


def main(argv=None) -> int:
    from valwr import config
    from valwr.live import lockfile
    from valwr.live import session as S
    from valwr.store import schema

    try:
        sess = S.build()
    except lockfile.ClientNotRunning as e:
        print(f"  VALORANT is not running -- {e}")
        print("  Start the game (the menus are enough) and run this again.")
        return 1
    conn = schema.connect(config.load(require_key=False).database_path)
    schema.create_all(conn)
    todo = len(missing(conn))
    print(f"  {todo} recorded bomb-mode match(es) without Performance Scores")
    got = collect(conn, sess, limit=None)
    print(f"  stored {got['stored']}, not held by the client {got['unavailable']}"
          f", not scored by the game {got['not_scored']}, unreadable "
          f"{got['format_changed']}")
    if got["format_changed"]:
        print("  The client's format has changed; nothing unreadable was stored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
