"""Who on your team is on a run: three or more results the same, this session.

Shown as what happened, and never fed into the score or the prediction. It was
measured first: over 787,460 competitive games, the win rate after a run of
losses or wins stayed at 50% within its margin -- 52.2% after three losses in
a row -- because matchmaking answers a run with easier or harder lobbies. The
badge was wanted anyway, and docs/DASHBOARD.md carries the measurement.

A run only counts inside one sitting: a break of two hours between one game
ending and the next beginning ends it, so last night's losses are not tonight's
streak. A draw ends it too.

A run must also be current to be worth showing, and what is stored can be a
game behind. A teammate whose newest stored game began under 55 minutes ago
cannot have finished another since -- no game fits in that time -- so their
record is complete. Anyone whose newest game is older may have played one we
do not hold; their first page is fetched again (at the lowest priority of the
lookup, see live/resolve.py), and until it has answered, no badge is shown
rather than one that may be stale.
"""

from __future__ import annotations

import sqlite3

from valwr.store import temporal

SHOWN_FROM = 3
# Start-to-start gaps stand in for breaks, so a game's own length comes off
# first: two games that began 2.5 hours apart had a break of under two.
SESSION_BREAK_SECONDS = 2 * 3600
GAME_SECONDS = 40 * 60
# The shortest a competitive game plus the queue for the next one can take is
# well over half an hour, and two of them a good deal more. A stored game that
# began this recently is the player's latest.
COMPLETE_WITHIN_SECONDS = 55 * 60
LOOK_BACK = 20                       # keep in step with resolve.FORM_WINDOW


def _break_between(later: int, earlier: int) -> bool:
    return later - earlier - GAME_SECONDS > SESSION_BREAK_SECONDS


def may_be_missing_a_game(conn: sqlite3.Connection, puuid: str,
                          as_of: int) -> bool:
    """Could they have finished a game since the newest one we hold, inside
    this session? Then their first page is worth fetching again."""
    rows = temporal.player_history(conn, puuid, as_of, limit=1)
    if not rows:
        return False
    return as_of - rows[0]["started_at"] > COMPLETE_WITHIN_SECONDS


def current(conn: sqlite3.Connection, puuid: str, as_of: int,
            fetched: bool) -> dict | None:
    """{"result": "won" | "lost", "count": n} for a run of SHOWN_FROM or more
    this session, or None.

    `fetched` is whether their first page answered during this match. Without
    it, only a record recent enough to be complete is read.
    """
    rows = temporal.player_history(conn, puuid, as_of, limit=LOOK_BACK)
    if not rows:
        return None
    newest = rows[0]["started_at"]
    if not fetched and as_of - newest > COMPLETE_WITHIN_SECONDS:
        return None
    if _break_between(as_of, newest):
        return None                          # no session going: nothing ran
    result, count, later = None, 0, newest
    for row in rows:
        if _break_between(later, row["started_at"]) or row["won"] is None:
            break
        this = "won" if row["won"] else "lost"
        if result not in (None, this):
            break
        result, count, later = this, count + 1, row["started_at"]
    return {"result": result, "count": count} if count >= SHOWN_FROM else None
