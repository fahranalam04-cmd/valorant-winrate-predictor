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

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any

from valwr import config
from valwr.collect.client import HenrikClient
from valwr.collect.limiter import TokenBucket
from valwr.live import comp, lockfile, picks, predict as P, resolve as R, roster, streak
from valwr.live import session as S
from valwr.rating import potential as pot
from valwr.rating import role_score as rs
from valwr.rating import roleindex
from valwr.rating import ranks
from valwr.store import temporal
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
    role_index: Any = None      # roleindex.RoleIndex | None
    # What the model's probabilities have meant in testing, for the page to
    # set beside each one; None when results.json is not this model's.
    record: dict | None = None
    deadline: float = DEFAULT_DEADLINE
    # Lookups already answered in the current match. The poll runs every few
    # seconds and must not repeat them; see resolve.Resolution.completed.
    work_match: str | None = None
    work_done: set = field(default_factory=set)
    work_failed: set = field(default_factory=set)
    # When this match was first seen. The streak badge is judged from here,
    # not from each poll: nobody in this match can have played another game
    # since it began, and measuring from "now" made a teammate's complete
    # record look a game behind six minutes into the match.
    work_since: int = 0
    # Per-match results that do not change within it (see _assemble).
    cache: dict = field(default_factory=dict)

    @property
    def model_name(self) -> str:
        return self.bundle.get("best", "?")

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    def refresh_session(self) -> None:
        """Build a new client session in place, or say the game has gone."""
        try:
            self.session = S.build()
        except lockfile.ClientNotRunning as e:
            raise NotReady(f"lost the game client -- {e}") from e


def open_context(no_fetch: bool = False,
                 deadline: float = DEFAULT_DEADLINE) -> LiveContext:
    """Open the database, model and client session, or explain what is missing."""
    if not lockfile.game_is_running():
        raise NotReady("VALORANT is not running -- start the game and try again.")

    settings = config.load(require_key=False)
    conn = schema.connect(settings.database_path)
    model_path = settings.models_path / "model.joblib"
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

    # The per-role score. Without it the card falls back to the old single
    # formula rather than losing its number entirely -- a fresh clone that has
    # not fitted one yet still shows a scoreboard.
    try:
        role_index = roleindex.RoleIndex.load()
    except FileNotFoundError:
        role_index = None

    client = None
    if not no_fetch:
        full = config.load()
        client = HenrikClient(full.henrik_api_key, conn=conn,
                              limiter=TokenBucket(full.requests_per_minute))

    from valwr.model import calibration
    return LiveContext(conn=conn, bundle=bundle, index=index,
                       role_index=role_index, session=S.build(),
                       client=client, settings=settings, deadline=deadline,
                       record=calibration.track_record(bundle))


def display_map(conn, reported: str | None) -> str | None:
    """The map's real name, given what the client called it.

    The client reports an internal codename -- Plummet for Summit, Jam for
    Lotus -- and 25 of the 26 maps have one that differs from the name people
    use. Left as-is, the live header named a map nobody recognises and the
    page looked for artwork under that name, so no background ever painted.
    Anything not in the table is passed through unchanged.
    """
    if not reported:
        return reported
    row = conn.execute(
        "SELECT name FROM ref_maps WHERE lower(path) = lower(?)",
        (reported,)).fetchone()
    return row["name"] if row else reported


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


PARTY_LABELS = {2: "duo", 3: "trio", 4: "four", 5: "five-stack"}


def parties(conn, match, as_of: int, exact: dict | None = None) -> list[dict]:
    """Who queued together, as groups.

    Two sources, and the view is told which it got. A finished match records
    party ids outright, so a replay is exact. A live lobby has no row yet and
    the client will not say who the enemy queued with, so it is inferred from
    whether two players have entered a party together before -- 99.8%
    precise, 53% recall, measured. Absence of a group therefore means "not
    established", never "solo", and the page says so.
    """
    out = []
    for team in ("Blue", "Red"):
        side = [p.puuid for p in match.players if p.team == team]
        # Union-find over the side: pair evidence merges two players into one
        # group, so a trio is found from its three pairs without special-casing.
        parent = {p: p for p in side}

        # `parent` is passed rather than closed over: the closure is only used
        # inside this iteration, so capturing it works, but a linter is right
        # that a loop variable bound by a nested function is a trap waiting
        # for the day someone moves the definition out of the loop.
        def find(x, parent=parent):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, a in enumerate(side):
            for b in side[i + 1:]:
                together = (exact.get(a) is not None and exact.get(a) == exact.get(b)
                            if exact is not None
                            else temporal.times_partied(conn, a, b, as_of) > 0)
                if together:
                    parent[find(a)] = find(b)

        groups: dict[str, list[str]] = {}
        for p in side:
            groups.setdefault(find(p), []).append(p)
        for members in groups.values():
            if len(members) > 1:
                out.append({
                    "members": members, "team": team, "size": len(members),
                    "label": PARTY_LABELS.get(len(members), f"{len(members)}-stack"),
                    "source": "exact" if exact is not None else "inferred",
                })
    return out


def _player_rows(ctx: LiveContext, match, as_of: int,
                 fetched: set = frozenset(),
                 since: int | None = None) -> list[dict]:
    """Every player in the lobby, scored where we can and honest where we cannot.

    `fetched` holds the lookups that have answered during this match, and
    `since` is when the match was first seen.
    """
    names = gamertags(ctx.conn, [p.puuid for p in match.players])
    roles = ctx.bundle.get("roles") or {}
    # For what the page shows -- a role tag, the team's strip, likely roles --
    # an agent released since the model was trained still has a role.
    # Scoring keeps the model's own map.
    shown = display_roles(ctx.conn, roles)
    own_team = match.team_of(ctx.session.puuid)
    since = since or as_of
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
            "role": shown.get(p.agent),
            "team": p.team,
            "is_you": p.puuid == ctx.session.puuid,
            # Agent select: locked, hovering, or neither -- and for anyone who
            # has not locked, the role their last twenty games point to.
            "selection": p.selection,
            "likely": (comp.likely_role(ctx.conn, p.puuid, as_of, shown)
                       if match.phase == "pregame"
                       and (not p.agent_id or p.selection == "selected")
                       else None),
            "score": None, "reason": "no history", "flag": None,
            # Rank as of their most recent stored match. It can lag a climb,
            # which is why the card prints it beside the data's freshness.
            "rank": ranks.describe(temporal.current_tier(ctx.conn, p.puuid,
                                                         as_of)),
            # Lifted out of `detail` so the scoreboard row does not have to
            # reach into the breakdown for the numbers it prints on every line.
            "career": None, "recent": None,
            # Three or more the same this session, for your own team only.
            # See live/streak.py for why a badge and nothing more.
            "streak": (streak.current(ctx.conn, p.puuid, since,
                                      fetched=(p.puuid, 0) in fetched)
                       if own_team and p.team == own_team else None),
        }
        if ctx.index is not None:
            got = pot.detail(ctx.conn, p.puuid, as_of, match.map_name or "?",
                             ctx.bundle["norms"], ctx.index)
            # The score, its components and the one-line reason come from the
            # per-role tables; everything else on the card -- career, form,
            # this map, freshness, the above-rank flag -- is the same data
            # either way and is left alone.
            if got is not None and ctx.role_index is not None:
                scored = rs.describe(ctx.conn, p.puuid, as_of,
                                     match.map_name or "?", roles.get(p.agent),
                                     p.agent, roles, ctx.role_index)
                if scored is not None:
                    got.update({k: scored[k] for k in
                                ("score", "raw", "reason", "components")})
                    got["role_score"] = {k: scored[k] for k in
                                         ("role", "weights_from", "role_games",
                                          "ability_games")}
                    # The map block is written by the old score, which gates
                    # the map at a different number of games. Left alone, the
                    # card explains a threshold the number in front of it did
                    # not use.
                    if got.get("map"):
                        got["map"]["gate"] = scored["map_gate"]
                        got["map"]["counts_toward_score"] = scored["map_counts"]
            if got is not None:
                entry["score"] = got["score"]
                # Ordered by the cross-role figure, displayed as the
                # within-role percentile. See rating/role_score.standing.
                entry["raw"] = got.get("raw")
                entry["reason"] = got["reason"]
                entry["flag"] = got["flag"]
                # The whole card, so a reader can audit the number rather than
                # take it on trust -- and so the freshness line is always
                # available. A score computed from two-week-old history looked
                # identical to a live one before this.
                entry["detail"] = got
                entry["career"] = got["career"]
                entry["recent"] = got["recent"]
        rows.append(entry)
    # Best first; unscored last rather than dropped -- they are in the lobby
    # whether or not we know anything about them, and saying so is the point.
    # `standing` is the raw cross-role number, not the 0-100 beside the name:
    # the percentile is within a role and sorting by it drops the fact that
    # some roles top their team far more often than others.
    rows.sort(key=lambda r: (rs.standing(r) is not None, rs.standing(r) or 0),
              reverse=True)
    return rows


def current_match(ctx: LiveContext):
    """The roster, rebuilding the session once if it has expired.

    The context is opened once and polled for hours, but the session inside it
    is short lived: tokens age out after about an hour and a client restart
    moves the port. Without this, the first match worked and every later one
    reported an error until the dashboard was restarted by hand.
    """
    try:
        return roster.current(ctx.session, agents_by_id(ctx.conn))
    except S.SessionExpired:
        ctx.refresh_session()
        try:
            return roster.current(ctx.session, agents_by_id(ctx.conn))
        except S.SessionExpired as e:
            # A fresh session that also fails is not a session problem.
            raise NotReady(f"the game client is not answering -- {e}") from e


def poll_once(ctx: LiveContext, on_progress=None) -> dict | None:
    """The current match as plain data, or None when not in one.

    Everything a renderer needs and nothing it has to compute for itself.

    `on_progress`, if given, is handed a complete state every time the lookup
    learns something: first with everything already stored, before a single
    request is made, then again as each player lands. Agent select leaves
    about a minute to lock in, and this used to show nothing until the last
    lookup finished -- up to 25 seconds -- although most of the lobby was
    usually in the database the whole time.
    """
    match = current_match(ctx)
    if match is None:
        return None
    match = dataclasses.replace(
        match, map_name=display_map(ctx.conn, match.map_name))

    as_of = int(time.time())
    if ctx.work_match != match.match_id:
        ctx.work_match, ctx.work_since = match.match_id, as_of
        ctx.work_done, ctx.work_failed, ctx.cache = set(), set(), {}

    def progress(res):
        # With nothing left to fetch, the final state is built the moment
        # this returns; painting it twice would only cost a rebuild.
        if not res.remaining:
            return
        try:
            on_progress(_assemble(ctx, match, res, as_of))
        except Exception:                           # noqa: BLE001
            # A partial paint is a courtesy; the lookup it reports on is not,
            # and must not die with it. A real fault in _assemble surfaces
            # anyway, in the final state built from the same function below.
            pass

    own_team = match.team_of(ctx.session.puuid)
    resolution = R.resolve(ctx.conn, match, ctx.session.puuid, as_of,
                           client=ctx.client, deadline_seconds=ctx.deadline,
                           region=ctx.settings.region,
                           platform=ctx.settings.platform,
                           on_progress=progress if on_progress else None,
                           already=ctx.work_done,
                           teammates_first=match.phase == "pregame",
                           refresh_last=[
                               p.puuid for p in match.players
                               if own_team and p.team == own_team
                               and streak.may_be_missing_a_game(
                                   ctx.conn, p.puuid, ctx.work_since)])
    ctx.work_done |= resolution.completed
    ctx.work_failed |= resolution.failed
    return _assemble(ctx, match, resolution, as_of)


def display_roles(conn, roles: dict[str, str]) -> dict[str, str]:
    """The model's agent-to-role map, filled in from ref_agents for agents
    released since it was trained -- a teammate on a new agent otherwise
    left their role marked open."""
    try:
        ref = {r[0]: r[1] for r in conn.execute(
            "SELECT name, role FROM ref_agents WHERE role IS NOT NULL")}
    except Exception:                               # noqa: BLE001
        ref = {}
    return {**ref, **roles}


def _your_picks(ctx: LiveContext, match, as_of: int) -> dict | None:
    """Your picks, built once per match. They read your whole history, and
    agent select repaints every second or two; only a refresh of your own
    games can change them, so that is part of the key."""
    me = ctx.session.puuid
    key = (match.match_id, match.map_name, R.newest_match(ctx.conn, me, as_of))
    cache = ctx.cache
    held = cache.get("your_picks")
    if held is None or held[0] != key:
        held = (key, picks.your_picks(
            ctx.conn, me, as_of, match.map_name, ctx.bundle.get("norms"),
            display_roles(ctx.conn, ctx.bundle.get("roles") or {})))
        cache["your_picks"] = held
    return held[1]


def _assemble(ctx: LiveContext, match, resolution, as_of: int) -> dict:
    """The state for one moment of a lookup: partial or final, same shape."""
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
        "parties": parties(ctx.conn, match, as_of),
        "players": _player_rows(
            ctx, match, as_of, since=ctx.work_since or as_of,
            fetched=((ctx.work_done | resolution.completed)
                     - (ctx.work_failed | resolution.failed))),
        # Who is still being looked up. A player with no card data is either
        # still coming or has no competitive history at all, and the page has
        # to know which before it tells you something about them.
        "lookup": {"pending": sorted(resolution.pending),
                   "remaining": len(resolution.remaining)},
        # Agent select is where you pick, so it is the only phase that asks.
        "your_picks": (_your_picks(ctx, match, as_of)
                       if match.phase == "pregame" else None),
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
