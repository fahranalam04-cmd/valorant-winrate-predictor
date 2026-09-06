"""Rebuild the dashboard for a match that already happened.

    python -m valwr.dash --match <match_id>

The live view is gone the moment the match starts, so there is no way to look
at what the model actually said about a game you remember. This reconstructs
that screen from stored history.

**It is a genuine replay, not a re-scoring.** `as_of` is the match's own start
time and every read goes through `store/temporal.py`, which filters
`started_at < as_of` strictly -- so the scores, the prediction and every card
are built from exactly what was knowable at the loading screen, and the match
being replayed cannot inform its own prediction. Feeding a match its own
result would make the model look extraordinary and mean nothing.

The one thing the live view cannot have is the outcome, so the replay carries
it: `outcome` says who actually won and whether the prediction was right.
"""

from __future__ import annotations

from valwr.live import predict as LP
from valwr.live import state as ST
from valwr.live.resolve import Resolution
from valwr.live.roster import LiveMatch, LivePlayer


class NoSuchMatch(LookupError):
    """Named a match id that is not in the database."""


def _agent_ids(conn) -> dict[str, str]:
    return {r["name"]: r["uuid"]
            for r in conn.execute("SELECT uuid, name FROM ref_agents")}


def replay_state(conn, match_id: str, bundle: dict, index, own_puuid: str
                 ) -> dict:
    """The same dictionary `poll_once` returns, for a finished match."""
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM match_players WHERE match_id = ?", (match_id,))]
    if not rows:
        raise NoSuchMatch(f"no match {match_id} in the database")

    head = conn.execute(
        "SELECT started_at, map, mode, winner FROM matches WHERE match_id = ?",
        (match_id,)).fetchone()
    if head is None:
        raise NoSuchMatch(f"no match {match_id} in the database")

    # The loading screen for this match. Strictly before, so the match cannot
    # see itself -- store/temporal.py enforces that for every read below.
    as_of = int(head["started_at"])
    ids = _agent_ids(conn)

    match = LiveMatch(
        match_id=match_id, phase="coregame", map_name=head["map"],
        mode="BombGameMode",
        players=[LivePlayer(puuid=r["puuid"], team=r["team"],
                            agent_id=ids.get(r["agent"] or ""),
                            agent=r["agent"] or "?") for r in rows])

    # Every player in a stored match is by definition known: their history is
    # whatever preceded this game. Nothing is fetched.
    resolution = Resolution(known={r["puuid"] for r in rows})

    ctx = ST.LiveContext(
        conn=conn, bundle=bundle, index=index,
        session=type("S", (), {"puuid": own_puuid})(), client=None,
        settings=type("C", (), {"region": "na", "platform": "pc"})(),
        deadline=0.0)

    prediction = LP.predict(conn, match, bundle, resolution, own_puuid,
                            as_of=as_of)
    own_team = match.team_of(own_puuid) or "Blue"

    mine = next((r for r in rows if r["puuid"] == own_puuid), None)
    won = bool(mine["won"]) if mine and mine["won"] is not None else None
    called = None
    if prediction is not None and won is not None:
        called = (prediction.own_probability >= 0.5) == won

    state = {
        "match_id": match_id, "phase": "coregame", "is_custom": False,
        "standard_mode": True, "map": head["map"], "mode": head["mode"],
        "as_of": as_of, "own_team": own_team,
        "enemy_team": "Red" if own_team == "Blue" else "Blue",
        "team_sizes": {"Blue": match.team_size("Blue"),
                       "Red": match.team_size("Red")},
        "coverage": resolution.coverage, "confidence": resolution.confidence,
        "fetched": 0, "model": bundle.get("best", "?"),
        "warnings": [],
        "players": ST._player_rows(ctx, match, as_of),
        "prediction": None,
    }
    if prediction is not None:
        state["prediction"] = {
            "own_probability": round(prediction.own_probability, 4),
            "win_probability": round(prediction.win_probability, 4),
            "factors": [{"name": n, "value": round(v, 4)}
                        for n, v in prediction.factors],
        }

    # The only field a live poll cannot have. Everything above was knowable
    # before the first round; this is what happened afterwards.
    state["outcome"] = {
        "winner": head["winner"], "you_won": won, "called_it": called,
        "scoreline": (f"{mine['kills']}/{mine['deaths']}/{mine['assists']}"
                      if mine else None),
        "your_agent": mine["agent"] if mine else None,
    }
    return state
