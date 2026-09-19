"""What the live view predicted, and what actually happened.

The held-out test set says how the model does on matches collected alongside
its training data. This says how it does on *your* matches, which is the only
number you can check for yourself -- and the only one that stays true as the
meta moves.

Three steps, each plain:

1. **Record.** As a match loads, the dashboard stores the prediction and the
   state behind it. The first sight is what counts: a prediction made at the
   loading screen is the claim being tested, so a later poll never overwrites
   it. The exception is agent select, where the enemy team is hidden -- that
   one is upgraded when the match proper starts.
2. **Settle.** A few minutes after the match ends, one API call fetches it, and
   it is stored like any other match. The result is then read back from the
   normal tables rather than from the response, so a settled prediction is
   scored against exactly the data the model trains on.
3. **Score.** Predicted against actual, per match and in aggregate.

None of this becomes training data. Scoring the model on matches you played and
then training on them is a loop that flatters itself.
"""

from __future__ import annotations

import json
import sqlite3
import time

from valwr.store import normalize

# A prediction is recorded as the match *loads*, so the clock starts before a
# single round is played. A competitive match runs 25 to 45 minutes, which is
# the fact the first version of this got wrong: it began asking three minutes
# in and gave up after six tries a minute apart, so the whole retry budget was
# spent inside the first ten minutes of a match that had not finished. Every
# recorded match failed to settle, and the dashboard showed no comparison ever
# -- with a log full of 404s that looked like a broken endpoint rather than a
# question asked far too early.
FIRST_ATTEMPT_SECONDS = 20 * 60

# Each further attempt waits half as long again, so the retries spread out
# instead of hammering a match that is still being played. Twelve attempts
# reach about 28 hours, which covers a long session and a match the API is slow
# to publish.
ATTEMPT_BACKOFF = 1.5
MAX_ATTEMPTS = 12


def due_after(attempts: int) -> float:
    """How long after recording an attempt number is worth making."""
    return FIRST_ATTEMPT_SECONDS * (ATTEMPT_BACKOFF ** max(attempts, 0))

COLUMNS = (
    "match_id, made_at, phase, map, mode, standard_mode, is_custom, own_puuid, "
    "own_team, win_probability, own_probability, coverage, confidence, model, "
    "state_json"
)


def record(conn: sqlite3.Connection, state: dict, now: int | None = None) -> bool:
    """Store the prediction for this match. True when something was written.

    False for a match already recorded, and for a state with no prediction --
    there is nothing to score in that case.
    """
    pred = state.get("prediction")
    if not pred:
        return False
    match_id = state["match_id"]
    seen = conn.execute(
        "SELECT phase FROM live_predictions WHERE match_id = ?",
        (match_id,)).fetchone()
    if seen is not None and not (seen["phase"] == "pregame"
                                 and state.get("phase") != "pregame"):
        return False

    values = (
        match_id, int(now or time.time()), state.get("phase"), state.get("map"),
        state.get("mode"), int(bool(state.get("standard_mode"))),
        int(bool(state.get("is_custom"))),
        next((p["puuid"] for p in state.get("players", []) if p.get("is_you")),
             None),
        state.get("own_team"), pred.get("win_probability"),
        pred.get("own_probability"), state.get("coverage"),
        state.get("confidence"), state.get("model"), json.dumps(state),
    )
    placeholders = ",".join("?" * len(values))
    conn.execute(
        "INSERT INTO live_predictions (" + COLUMNS + ") "
        "VALUES (" + placeholders + ") "
        "ON CONFLICT(match_id) DO UPDATE SET "
        "phase=excluded.phase, map=excluded.map, mode=excluded.mode, "
        "standard_mode=excluded.standard_mode, is_custom=excluded.is_custom, "
        "own_team=excluded.own_team, win_probability=excluded.win_probability, "
        "own_probability=excluded.own_probability, coverage=excluded.coverage, "
        "confidence=excluded.confidence, state_json=excluded.state_json",
        values)
    conn.commit()
    return True


def pending(conn: sqlite3.Connection, now: int | None = None) -> list[str]:
    """Matches whose result is worth asking for, oldest first.

    Each match backs off on its own schedule, so one that is still being played
    is not asked about every minute until its budget is gone.
    """
    now = int(now or time.time())
    out = []
    for r in conn.execute(
            "SELECT match_id, made_at, attempts FROM live_predictions "
            "WHERE settled_at IS NULL AND attempts < ? ORDER BY made_at",
            (MAX_ATTEMPTS,)):
        if now - r["made_at"] >= due_after(r["attempts"]):
            out.append(r["match_id"])
    return out


def actual_best(conn: sqlite3.Connection, match_id: str,
                team: str | None) -> str | None:
    """Who actually had the highest combat score on that team."""
    if not team:
        return None
    row = conn.execute(
        "SELECT puuid FROM match_players WHERE match_id = ? AND team = ? "
        "AND rounds_played > 0 "
        "ORDER BY CAST(score AS REAL) / rounds_played DESC LIMIT 1",
        (match_id, team)).fetchone()
    return row["puuid"] if row else None


def top_pick(state: dict, team: str | None) -> str | None:
    """Whom the score put first on that team, when it rated enough of it.

    The same ordering the scoreboard used, so what is scored afterwards is the
    pick the player actually saw at the top of the list.
    """
    from valwr.rating.role_score import standing
    rated = [p for p in state.get("players", [])
             if p.get("team") == team and standing(p) is not None]
    if len(rated) < 2:
        return None
    return max(rated, key=standing)["puuid"]


def settle(conn: sqlite3.Connection, client, region: str, match_id: str,
           now: int | None = None) -> str:
    """Fetch and score one finished match.

    Returns 'settled', 'waiting' (the API does not have it yet) or 'error'.
    Attempts are counted, so a match that will never appear stops being asked
    about rather than costing a call every minute forever.
    """
    now = int(now or time.time())
    row = conn.execute("SELECT * FROM live_predictions WHERE match_id = ?",
                       (match_id,)).fetchone()
    if row is None:
        return "error"

    note = None
    # The crawler may already have collected this match through somebody's
    # history. It is the same data the API would return, stored the same way,
    # so fetching it again would spend a rate-limited call to learn nothing.
    known = conn.execute(
        "SELECT winner FROM matches WHERE match_id = ? AND winner IS NOT NULL",
        (match_id,)).fetchone()
    if known is None and client is not None:
        try:
            payload = client.match(region, match_id)
            data = (payload or {}).get("data")
            # ingest takes a matchlist; one match is a list of one.
            normalize.ingest(conn,
                             {"data": [data] if isinstance(data, dict) else data})
        except Exception as e:                       # noqa: BLE001
            note = f"{type(e).__name__}: {e}"[:200]

    played = conn.execute(
        "SELECT winner, rounds_blue, rounds_red FROM matches WHERE match_id = ?",
        (match_id,)).fetchone()
    if played is None or played["winner"] is None:
        conn.execute("UPDATE live_predictions SET attempts = attempts + 1, "
                     "last_error = ? WHERE match_id = ?", (note, match_id))
        conn.commit()
        return "error" if note else "waiting"

    winner = played["winner"]
    own_team = row["own_team"]
    own_won = None if winner not in ("Blue", "Red") else int(winner == own_team)
    p_own = row["own_probability"]
    correct = brier = None
    if own_won is not None and p_own is not None:
        correct = int((p_own >= 0.5) == bool(own_won))
        brier = (p_own - own_won) ** 2

    state = json.loads(row["state_json"])
    picked = top_pick(state, own_team)
    best = actual_best(conn, match_id, own_team) if picked else None
    hit = None if not (picked and best) else int(picked == best)

    conn.execute(
        "UPDATE live_predictions SET settled_at = ?, winner = ?, "
        "rounds_blue = ?, rounds_red = ?, own_won = ?, correct = ?, brier = ?, "
        "top_pick_hit = ?, attempts = attempts + 1, last_error = NULL "
        "WHERE match_id = ?",
        (now, winner, played["rounds_blue"], played["rounds_red"], own_won,
         correct, brier, hit, match_id))
    conn.commit()
    return "settled"


def unstick(conn: sqlite3.Connection, now: int | None = None) -> int:
    """Give back the attempts the old schedule burned. Returns how many.

    Before the backoff existed, every retry for a match was spent inside the
    first ten minutes of a game that runs forty, so predictions were left with
    a full attempt count and no result -- permanently, since the budget was
    gone. Those rows are still recoverable: the matches finished long ago and
    the result is there for the asking.

    A row is only cleared when it is older than a match can last and still has
    nothing recorded against it, so this cannot rescue a match that genuinely
    does not exist -- that one simply spends its attempts again, slowly.
    """
    now = int(now or time.time())
    cutoff = now - int(due_after(1))
    return conn.execute(
        "UPDATE live_predictions SET attempts = 0, last_error = NULL "
        "WHERE settled_at IS NULL AND attempts > 0 AND made_at < ?",
        (cutoff,)).rowcount


def settle_pending(conn: sqlite3.Connection, client, region: str,
                   now: int | None = None) -> dict[str, int]:
    """Try every match that is due. Safe to call on a timer."""
    # `client` may be None -- a match the crawler already holds settles from
    # the database alone, with no call to make. A missing database is the one
    # thing that stops this cold, and the live view hands one over only once
    # its context is open.
    out = {"settled": 0, "waiting": 0, "error": 0}
    if conn is None:
        return out
    for match_id in pending(conn, now):
        out[settle(conn, client, region, match_id, now)] += 1
    return out
