"""One live poll, as plain data.

The terminal view and the browser dashboard must never disagree about what is
happening in a match. The only reliable way to guarantee that is for both to
render the *same* dictionary rather than each assembling its own -- which is
the identical argument `live/predict.py` makes for sharing `build_match` with
training, and for the same reason: a second copy of the logic fails silently,
with every number still present and still plausible.

So this module owns the poll, and returns JSON-serialisable state. Nothing here
formats anything. `live/__main__.py` prints it; `dash/server.py` sends it over a
websocket.

Read-only throughout, like everything else under `live/`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from valwr import config
from valwr.collect.client import HenrikClient
from valwr.collect.limiter import TokenBucket
from valwr.live import lockfile, predict as P, resolve as R, roster
from valwr.live import session as S
from valwr.rating import potential as pot
from valwr.store import schema

DEFAULT_DEADLINE = 25.0


class NotReady(RuntimeError):
    """Something the live view needs is missing, with a readable reason."""


@dataclass
class LiveContext:
    """Everything a poll needs, opened once and reused."""
    conn: Any
    bundle: dict
    index: Any                  # pot.PerfIndex | None
    session: Any
    client: Any                 # HenrikClient | None
    settings: Any
    deadline: float = DEFAULT_DEADLINE

    @property
    def model_name(self) -> str:
        return self.bundle.get("best", "?")

    def close(self) -> None:
        if self.client is not None:
            self.client.close()


def open_context(no_fetch: bool = False,
                 deadline: float = DEFAULT_DEADLINE) -> LiveContext:
    """Open the database, model and client session, or explain what is missing."""
    if not lockfile.game_is_running():
        raise NotReady("VALORANT is not running -- start the game and try again.")

    settings = config.load(require_key=False)
    conn = schema.connect(settings.database_path)
    model_path = settings.database_path.parent.parent / "models" / "model.joblib"
    if not model_path.exists():
        raise NotReady(f"no model at {model_path}; run python -m valwr.model.train")

    import joblib
    bundle = joblib.load(model_path)

    # Optional: without it the per-player scores are simply absent, which the
    # renderers show as "--" rather than inventing a number.
    try:
        index = pot.PerfIndex.load()
    except FileNotFoundError:
        index = None

    client = None
    if not no_fetch:
        full = config.load()
        client = HenrikClient(full.henrik_api_key, conn=conn,
                              limiter=TokenBucket(full.requests_per_minute))

    return LiveContext(conn=conn, bundle=bundle, index=index, session=S.build(),
                       client=client, settings=settings, deadline=deadline)


def agents_by_id(conn) -> dict[str, str]:
    return {r["uuid"].lower(): r["name"]
            for r in conn.execute("SELECT uuid, name FROM ref_agents")}


def gamertags(conn, puuids: list[str]) -> dict[str, str]:
    """puuid -> "name#tag" for whoever the store already holds.

    The client exposes names only through a PUT on its name-service endpoint,
    which docs/ETHICS-AND-TOS.md rules out without exception, so an
    unrecognised player keeps a short PUUID rather than being looked up.
    """
    if not puuids:
        return {}
    q = ",".join("?" * len(puuids))
    out: dict[str, str] = {}
    for r in conn.execute(
            f"SELECT puuid, name, tag FROM players WHERE puuid IN ({q})",
            puuids):
        if r["name"]:
            out[r["puuid"]] = f"{r['name']}#{r['tag']}" if r["tag"] else r["name"]
    return out


def _warnings(match, own_puuid: str) -> list[str]:
    """Where the model's training distribution stops being a safe assumption."""
    out = []
    if not match.is_standard_mode:
        out.append(f"{match.mode} is not bomb defusal. The model only ever saw "
                   f"standard 5v5, so a win probability here means nothing.")
    elif match.is_custom and not match.is_even_5v5:
        out.append(f"Uneven teams ({match.team_size('Blue')}v"
                   f"{match.team_size('Red')}). Team features are averages, so "
                   f"this still computes -- but the model was trained on 5v5.")
    if match.team_of(own_puuid) is None and match.players:
        out.append("You are not on either team (spectating or coaching); "
                   "percentages are from Team Blue's side.")
    if match.phase == "pregame":
        out.append("Enemy team is hidden during agent select. It fills in once "
                   "the match starts.")
    return out


def _player_rows(ctx: LiveContext, match, as_of: int) -> list[dict]:
    """Every player in the lobby, scored where we can and honest where we cannot."""
    names = gamertags(ctx.conn, [p.puuid for p in match.players])
    roles = ctx.bundle.get("roles") or {}
    rows = []
    for p in match.players:
        entry = {
            "puuid": p.puuid,
            "name": names.get(p.puuid, p.puuid[:8]),
            "known_name": p.puuid in names,
            "agent": p.agent,
            # The same UUID as ref_agents.uuid and as valorant-api's, so the
            # page addresses the bundled artwork directly. None during agent
            # select, before a pick is locked.
            "agent_id": p.agent_id,
            "role": roles.get(p.agent),
            "team": p.team,
            "is_you": p.puuid == ctx.session.puuid,
            "score": None, "reason": "no history", "flag": None,
            # Lifted out of `detail` so the scoreboard row does not have to
            # reach into the breakdown for the numbers it prints on every line.
            "career": None,
        }
        if ctx.index is not None:
            got = pot.detail(ctx.conn, p.puuid, as_of, match.map_name or "?",
                             ctx.bundle["norms"], ctx.index)
            if got is not None:
                entry["score"] = got["score"]
                entry["reason"] = got["reason"]
                entry["flag"] = got["flag"]
                # The whole card, so a reader can audit the number rather than
                # take it on trust -- and so the freshness line is always
                # available. A score computed from two-week-old history looked
                # identical to a live one before this.
                entry["detail"] = got
                entry["career"] = got["career"]
        rows.append(entry)
    # Best first; unscored last rather than dropped -- they are in the lobby
    # whether or not we know anything about them, and saying so is the point.
    rows.sort(key=lambda r: (r["score"] is not None, r["score"] or 0),
              reverse=True)
    return rows


def poll_once(ctx: LiveContext) -> dict | None:
    """The current match as plain data, or None when not in one.

    Everything a renderer needs and nothing it has to compute for itself.
    """
    match = roster.current(ctx.session, agents_by_id(ctx.conn))
    if match is None:
        return None

    as_of = int(time.time())
    resolution = R.resolve(ctx.conn, match, ctx.session.puuid, as_of,
                           client=ctx.client, deadline_seconds=ctx.deadline,
                           region=ctx.settings.region,
                           platform=ctx.settings.platform)
    prediction = P.predict(ctx.conn, match, ctx.bundle, resolution,
                           ctx.session.puuid, as_of=as_of)

    own_team = match.team_of(ctx.session.puuid) or "Blue"
    state: dict = {
        "match_id": match.match_id,
        "phase": match.phase,
        "is_custom": match.is_custom,
        "standard_mode": match.is_standard_mode,
        "map": match.map_name,
        "mode": match.mode,
        "as_of": as_of,
        "own_team": own_team,
        "enemy_team": "Red" if own_team == "Blue" else "Blue",
        "team_sizes": {"Blue": match.team_size("Blue"),
                       "Red": match.team_size("Red")},
        "coverage": resolution.coverage,
        "confidence": resolution.confidence,
        "fetched": resolution.fetched,
        "model": ctx.model_name,
        "warnings": _warnings(match, ctx.session.puuid),
        "players": _player_rows(ctx, match, as_of),
        "prediction": None,
    }
    if prediction is not None:
        state["prediction"] = {
            "own_probability": round(prediction.own_probability, 4),
            "win_probability": round(prediction.win_probability, 4),
            "factors": [{"name": n, "value": round(v, 4)}
                        for n, v in prediction.factors],
        }
    return state
