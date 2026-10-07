"""Phase 6 tests: lockfile auth, roster parsing, and the resolution deadline.

Most of what can go wrong here is silent. A launcher mistaken for a running
game, a placeholder version header, an XMPP region used as a game shard -- none
of those raise anything obvious; they produce an opaque 400 or a host that does
not resolve, several layers away from the cause. Each was hit for real while
building this, and each has a test.
"""

from __future__ import annotations

import types

import pytest

from valwr.live import lockfile, resolve as R, roster
from valwr.live.roster import LiveMatch, LivePlayer


# --- lockfile ---------------------------------------------------------

def test_lockfile_parses_the_five_colon_separated_fields(tmp_path):
    f = tmp_path / "lockfile"
    f.write_text("Riot Client:32840:52385:sUp3rSecret:https")
    lock = lockfile.read(f)
    assert (lock.name, lock.pid, lock.port, lock.protocol) == (
        "Riot Client", 32840, 52385, "https")
    assert lock.base == "https://127.0.0.1:52385"


def test_auth_header_uses_the_literal_username_riot(tmp_path):
    import base64
    f = tmp_path / "lockfile"
    f.write_text("Riot Client:1:2:pw:https")
    header = lockfile.read(f).auth_header["Authorization"]
    assert header.startswith("Basic ")
    decoded = base64.b64decode(header.removeprefix("Basic ")).decode()
    assert decoded == "riot:pw"


def test_a_missing_lockfile_says_the_client_is_not_running(tmp_path):
    with pytest.raises(lockfile.ClientNotRunning, match="not running"):
        lockfile.read(tmp_path / "absent")


def test_a_malformed_lockfile_is_rejected_with_its_field_count(tmp_path):
    f = tmp_path / "lockfile"
    f.write_text("only:three:fields")
    with pytest.raises(lockfile.ClientNotRunning, match="3 fields"):
        lockfile.read(f)


def test_launcher_only_functions_do_not_count_as_the_game_running():
    """The lockfile exists whenever the Riot Client launcher is up, which is
    most of the time. With the launcher alone, /help exposes seven functions
    and every endpoint this project needs 404s. Trusting the file's existence
    was a real mistake made while building this.
    """
    assert lockfile.LAUNCHER_ONLY >= {"Exit", "Help", "Subscribe"}
    launcher = set(lockfile.LAUNCHER_ONLY)
    assert not (launcher - lockfile.LAUNCHER_ONLY), "launcher alone: not running"
    with_game = launcher | {"GetPregameV1Player"}
    assert with_game - lockfile.LAUNCHER_ONLY, "game functions present: running"


def test_placeholder_client_version_is_rejected():
    """external-sessions carries a host_app entry whose version is the literal
    string "0". Sending it as X-Riot-ClientVersion earns an opaque 400 that
    names nothing."""
    from valwr.live import session as S
    assert "host_app" in S.PLACEHOLDER_SESSIONS
    assert S.MIN_VERSION_LENGTH > 1, "'0' must not pass the length check"


# --- roster -----------------------------------------------------------

def match_with(n_blue=5, n_red=5, locked=10):
    players = []
    for i in range(n_blue):
        players.append(LivePlayer(f"b{i}", "Blue",
                                  "aid" if len(players) < locked else None))
    for i in range(n_red):
        players.append(LivePlayer(f"r{i}", "Red",
                                  "aid" if len(players) < locked else None))
    return LiveMatch("m1", "coregame", "Ascent", "Standard", players)


def test_map_name_is_extracted_from_the_asset_path():
    assert roster._map_name("/Game/Maps/Ascent/Ascent") == "Ascent"
    assert roster._map_name(None) is None
    assert roster._map_name("") is None


def test_locked_in_counts_only_players_who_have_picked():
    assert match_with(locked=10).locked_in == 10
    assert match_with(locked=3).locked_in == 3


def test_team_of_finds_a_player_and_tolerates_a_stranger():
    m = match_with()
    assert m.team_of("b0") == "Blue"
    assert m.team_of("r0") == "Red"
    assert m.team_of("nobody") is None


def test_agent_uuids_resolve_to_names():
    m = LiveMatch("m", "coregame", "Ascent", None,
                  [LivePlayer("p1", "Blue", "ABC-123")])
    out = roster.resolve_agents(m, {"abc-123": "Jett"})
    assert out.players[0].agent == "Jett"


def test_an_unknown_agent_uuid_does_not_crash_the_roster():
    m = LiveMatch("m", "coregame", "Ascent", None,
                  [LivePlayer("p1", "Blue", "not-in-table")])
    out = roster.resolve_agents(m, {})
    assert out.players[0].agent == roster.UNKNOWN_AGENT


# --- resolution -------------------------------------------------------

def test_own_team_is_fetched_before_the_enemy():
    """Under a deadline the ordering decides what you end up knowing."""
    m = match_with()
    order = R.order_for_fetching(m, "b2")
    assert order[:5] == ["b0", "b1", "b2", "b3", "b4"]
    assert set(order[5:]) == {"r0", "r1", "r2", "r3", "r4"}


def test_confidence_tracks_how_much_of_the_lobby_is_known():
    def conf(n):
        r = R.Resolution(known={f"p{i}" for i in range(n)})
        return r.confidence
    assert conf(10) == "high"
    assert conf(9) == "high"
    assert conf(7) == "moderate"
    assert conf(5) == "low"
    assert conf(2) == "very low"


def test_cache_only_resolution_never_touches_the_api(tmp_path):
    """The dashboard's first paint must not spend quota."""
    from valwr.store import schema
    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)
    out = R.resolve(conn, match_with(), "b0", as_of=2_000_000_000, client=None)
    assert out.fetched == 0
    assert out.coverage == 0
    assert len(out.unknown) == 10


def test_resolution_stops_at_the_deadline_rather_than_finishing(tmp_path):
    """A fetch that does not finish in time is the normal case, not an error.
    Ten uncached players would take four minutes against a 30-second window."""
    from valwr.store import schema

    class SlowClient:
        def __init__(self):
            self.calls = 0

        def matches(self, *a, **kw):
            self.calls += 1
            import time
            time.sleep(0.05)
            return {"data": []}

    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)
    client = SlowClient()
    out = R.resolve(conn, match_with(), "b0", as_of=2_000_000_000,
                    client=client, deadline_seconds=0.12)
    assert client.calls < 10, "must stop at the deadline, not resolve everyone"
    assert out.seconds >= 0


def test_rate_limiting_ends_fetching_without_raising(tmp_path):
    from valwr.collect.client import RateLimited
    from valwr.store import schema

    class Limited:
        def matches(self, *a, **kw):
            raise RateLimited(60.0)

    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)
    out = R.resolve(conn, match_with(), "b0", as_of=2_000_000_000,
                    client=Limited(), deadline_seconds=5.0)
    assert out.fetched == 0
    assert out.coverage == 0, "no quota means the cached answer ships"


# --- custom games ------------------------------------------------------
# Customs differ from queued matches in ways that break assumptions safe for
# competitive: modes without bomb defusal, teams that are not 5v5, and a local
# player who may not be on a team at all.

def _coregame(players, **extra):
    """A core-game payload in the shape the client actually returns."""
    body = {"MatchID": "m1", "MapID": "/Game/Maps/Ascent/Ascent",
            "ModeID": "/Game/GameModes/Bomb/BombGameMode", "Players": players}
    body.update(extra)
    return body


def _player(puuid, team, **extra):
    p = {"Subject": puuid, "TeamID": team, "CharacterID": "agent-uuid"}
    p.update(extra)
    return p


class _StubSession:
    """Just enough session for fetch() to build its URL."""
    glz = "http://stub"
    puuid = "me"


def _parse(body):
    """Run roster.fetch's coregame branch against a canned payload."""
    import valwr.live.roster as mod
    saved = mod._get
    mod._get = lambda session, url: body
    try:
        return mod.fetch(session=_StubSession(), match_id="m1",
                         phase="coregame")
    finally:
        mod._get = saved


def test_a_custom_game_is_recognised_as_one():
    m = _parse(_coregame([_player(f"p{i}", "Blue") for i in range(5)]
                         + [_player(f"q{i}", "Red") for i in range(5)],
                         ProvisioningFlowID="CustomGame"))
    assert m.is_custom
    assert m.is_even_5v5
    assert m.team_size("Blue") == 5 and m.team_size("Red") == 5


def test_a_queued_match_is_not_flagged_as_custom():
    m = _parse(_coregame([_player("p", "Blue"), _player("q", "Red")],
                         ProvisioningFlowID="Matchmaking"))
    assert not m.is_custom


def test_uneven_custom_teams_are_reported_not_rejected():
    """3v5 still computes -- team features are averages -- but is not 5v5.

    The output warns rather than refusing, because a scrim with uneven sides
    is a real thing people run and the ranking is still informative.
    """
    m = _parse(_coregame([_player(f"p{i}", "Blue") for i in range(3)]
                         + [_player(f"q{i}", "Red") for i in range(5)],
                         ProvisioningFlowID="CustomGame"))
    assert m.team_size("Blue") == 3
    assert not m.is_even_5v5
    assert len(m.players) == 8


def test_non_bomb_modes_are_flagged_as_not_comparable():
    """The model only ever saw bomb defusal; deathmatch has no teams to speak of."""
    dm = _parse(_coregame([_player("p", "Blue")],
                          ModeID="/Game/GameModes/Deathmatch/DeathmatchGameMode"))
    assert not dm.is_standard_mode
    bomb = _parse(_coregame([_player("p", "Blue")]))
    assert bomb.is_standard_mode


def test_a_missing_mode_is_treated_as_standard():
    """Absent on some responses. Refusing to predict then would break ordinary
    competitive matches to guard against an unusual one."""
    m = _parse(_coregame([_player("p", "Blue")], ModeID=None))
    assert m.is_standard_mode


def test_a_spectator_has_no_team():
    m = _parse(_coregame([_player("p", "Blue"), _player("q", "Red")],
                         ProvisioningFlowID="CustomGame"))
    assert m.team_of("someone-else") is None


def test_a_competitive_match_triggers_none_of_the_custom_warnings():
    """The custom-game handling must not change what a queued match prints.

    Pregame exposes only AllyTeam, so the enemy count is structurally zero.
    That is not an uneven lobby and must not be reported as one -- the uneven
    warning is gated on is_custom for exactly this reason.
    """
    m = _parse(_coregame([_player(f"p{i}", "Blue") for i in range(5)]
                         + [_player(f"q{i}", "Red") for i in range(5)],
                         ProvisioningFlowID="Matchmaking"))
    assert not m.is_custom
    assert m.is_standard_mode
    assert m.is_even_5v5


def test_competitive_pregame_is_not_mistaken_for_an_uneven_lobby():
    from valwr.live.roster import LiveMatch, LivePlayer
    ally = [LivePlayer(f"p{i}", "Blue", "x", "Jett") for i in range(5)]
    m = LiveMatch("m", "pregame", "Ascent", "BombGameMode", ally,
                  flow="Matchmaking")
    assert not m.is_custom          # so the uneven-teams warning stays silent
    assert m.team_size("Red") == 0  # structural, not a real 5v0


def test_the_roster_read_is_time_gated(tmp_path):
    """Rank and level must come from before the match, not after it.

    `_roster_rows` took each player's most recent row with no `as_of` filter.
    Live that is harmless -- nothing later than now exists -- so it survived.
    In a replay it reads the player's *future* rank, which would let a backtest
    quietly flatter itself. Found by building the end-to-end replay, which is
    what that harness is for.
    """
    from valwr.live import predict as LP
    from valwr.live.roster import LiveMatch, LivePlayer
    from valwr.store import schema

    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)
    for mid, ts in (("m1", 100), ("m2", 300)):
        conn.execute(
            "INSERT INTO matches (match_id, started_at, map, mode, queue, "
            "region, season, rounds_red, rounds_blue, winner, data_quality, "
            f"ingested_at) VALUES ('{mid}', {ts}, 'Ascent', 'competitive', "
            "'Standard', 'na', 's', 9, 13, 'Blue', NULL, 0)")
    for mid, ts, tier in (("m1", 100, 5), ("m2", 300, 25)):
        conn.execute(
            "INSERT INTO match_players (match_id, puuid, team, agent, "
            "party_id, tier, account_level, score, kills, deaths, assists, "
            "headshots, bodyshots, legshots, damage_dealt, damage_taken, "
            "started_at, map, won, rounds_played) VALUES "
            f"('{mid}', 'p', 'Blue', 'Jett', NULL, {tier}, {tier * 10}, "
            f"1, 1, 1, 1, 1, 1, 1, 1, 1, {ts}, 'Ascent', 1, 20)")
    conn.commit()

    match = LiveMatch("live", "coregame", "Ascent", "BombGameMode",
                      [LivePlayer("p", "Blue", "x", "Jett")])
    # As of 200 only the tier-5 row exists; the tier-25 row is in the future.
    assert LP._roster_rows(match, conn, 200)[0]["tier"] == 5
    assert LP._roster_rows(match, conn, 400)[0]["tier"] == 25


# --- the dashboard -----------------------------------------------------

def test_the_dashboard_websocket_accepts_a_connection():
    """It did not, and nothing short of connecting would have shown it.

    `server.py` uses `from __future__ import annotations`, which turns every
    annotation into a string. FastAPI resolves those against the *module*
    namespace, so with `WebSocket` imported inside `build_app` the name was not
    there to resolve -- FastAPI fell back to treating the `socket` parameter as
    a query parameter and rejected every handshake with 403 Forbidden and
    "loc: ['query', 'socket'], Field required".

    The page loaded fine, the route was registered, `build_app` succeeded and
    the unit tests passed. Only an actual handshake failed.
    """
    from fastapi.testclient import TestClient
    from valwr.dash.server import build_app

    # A real local address: the server refuses TestClient's default
    # "testserver" host, as it refuses any domain name (DNS rebinding).
    with TestClient(build_app(no_fetch=True),
                    base_url="http://127.0.0.1:8787") as client:
        with client.websocket_connect("ws://127.0.0.1:8787/ws") as ws:
            # Sent before the first poll, so the page is never blank while
            # ten players are being resolved.
            assert ws.receive_json()["status"] == "working"
            msg = ws.receive_json()
            # With VALORANT closed this is the error branch, which is fine --
            # the point is that the handshake completed at all.
            assert msg["status"] in ("error", "lobby", "match")


def test_the_dashboard_serves_its_page():
    from fastapi.testclient import TestClient
    from valwr.dash.server import build_app

    # A real local address: the server refuses TestClient's default
    # "testserver" host, as it refuses any domain name (DNS rebinding).
    with TestClient(build_app(no_fetch=True),
                    base_url="http://127.0.0.1:8787") as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "valwr live" in r.text


def test_the_dashboard_binds_localhost_only():
    """docs/ETHICS-AND-TOS.md: never expose an endpoint that looks up players.

    Binding 0.0.0.0 would put the live view on the local network. The constant
    is asserted rather than trusted because it is one character from being
    wrong and nothing else would catch it.
    """
    from valwr.dash import server
    assert server.HOST == "127.0.0.1"


def test_the_dashboard_exposes_no_other_routes():
    """The page, the socket, and a directory of agent PNGs. Nothing else.

    Asserted as an exact set rather than a subset, so adding a route has to be
    a decision someone makes here on purpose. `/agents` was added that way: it
    serves Riot's artwork, takes no query, names no player and reveals nothing
    about anyone, which is what docs/ETHICS-AND-TOS.md actually forbids -- an
    endpoint that looks a player up.
    """
    from valwr.dash.server import AGENTS, MAPS, build_app
    paths = {r.path for r in build_app(no_fetch=True).routes
             if hasattr(r, "path")}
    # /m/{match_id} serves a match this dashboard itself predicted, and 404s
    # for anything else; /results and /api/scorecard take no parameters and
    # report on those same recorded matches. None of them can be asked about
    # a player, which is what docs/ETHICS-AND-TOS.md forbids.
    expected = {"/", "/ws", "/m/{match_id}", "/results", "/api/scorecard"}
    # Both are static directories of Riot's own art, mounted only once the
    # files exist. Neither takes a parameter or names a player.
    if AGENTS.is_dir():
        expected.add("/agents")
    if MAPS.is_dir():
        expected.add("/maps")
    assert paths == expected


# --- freshness ---------------------------------------------------------

def _tiny_db(tmp_path, rows):
    """A store holding `rows` of (match_id, puuid, started_at)."""
    from valwr.store import schema
    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)
    seen = set()
    for mid, puuid, ts in rows:
        if mid not in seen:
            conn.execute(
                "INSERT INTO matches (match_id, started_at, map, mode, queue, "
                "region, season, rounds_red, rounds_blue, winner, "
                "data_quality, ingested_at) VALUES "
                f"('{mid}', {ts}, 'Ascent', 'competitive', 'Standard', 'na', "
                "'s', 9, 13, 'Blue', NULL, 0)")
            seen.add(mid)
        conn.execute(
            "INSERT INTO match_players (match_id, puuid, team, agent, "
            "party_id, tier, account_level, score, kills, deaths, assists, "
            "headshots, bodyshots, legshots, damage_dealt, damage_taken, "
            "started_at, map, won, rounds_played) VALUES "
            f"('{mid}', '{puuid}', 'Blue', 'Jett', NULL, 15, 100, 4000, 15, "
            f"15, 5, 5, 5, 5, 3000, 3000, {ts}, 'Ascent', 1, 20)")
    conn.commit()
    return conn


def test_a_two_week_old_account_is_stale(tmp_path):
    """The reported bug: a score frozen for twelve days while looking live.

    `has_history` was the only test, and it is true of a single August row, so
    the account was marked known and never refetched.
    """
    from valwr.live import resolve as R
    now = 2_000_000_000
    old = now - 12 * 86400
    conn = _tiny_db(tmp_path, [(f"m{i}", "p", old - i * 3600) for i in range(8)])
    assert R.has_history(conn, "p", now), "still known -- that was never wrong"
    assert R.is_stale(conn, "p", now), "but two weeks old must count as stale"


def test_an_account_played_an_hour_ago_is_not_stale(tmp_path):
    from valwr.live import resolve as R
    now = 2_000_000_000
    conn = _tiny_db(tmp_path,
                    [(f"m{i}", "p", now - 3600 - i * 3600) for i in range(8)])
    assert not R.is_stale(conn, "p", now)


def test_thin_history_counts_as_stale(tmp_path):
    """Two stored matches is mostly prior, so it is worth a refresh."""
    from valwr.live import resolve as R
    now = 2_000_000_000
    conn = _tiny_db(tmp_path, [("m0", "p", now - 600), ("m1", "p", now - 1200)])
    assert R.is_stale(conn, "p", now)


def test_a_player_with_no_history_is_unknown_not_stale(tmp_path):
    """They cost the same fetch but mean different things on screen."""
    from valwr.live import resolve as R
    conn = _tiny_db(tmp_path, [("m0", "other", 1_999_000_000)])
    assert not R.has_history(conn, "nobody", 2_000_000_000)
    assert not R.is_stale(conn, "nobody", 2_000_000_000)


def test_fetching_actually_stores_what_it_fetched(tmp_path):
    """The bug that made every live fetch pointless.

    `HenrikClient.matches()` fetches and caches the raw body; it does not
    normalise. `resolve` called it and then asked whether the player now had
    history, trusting a comment that said "the crawler normalises inline" --
    true of the crawler, false of the client. So a fetch spent an API call,
    wrote a blob nothing read, and left the player exactly as unknown.

    Nothing caught it: the call succeeded, the response was stored, no
    exception was raised, and coverage still looked plausible because the
    crawler had independently collected most players.
    """
    from valwr.live import resolve as R
    from valwr.live.roster import LiveMatch, LivePlayer

    conn = _tiny_db(tmp_path, [("seed", "me", 1_999_000_000)])
    payload = {"data": [{
        "metadata": {"match_id": "new1", "started_at": "2026-09-01T00:00:00Z",
                     "map": {"name": "Ascent"}, "queue": {"id": "competitive"},
                     "region": "na", "cluster": "na", "rounds_played": 20,
                     "season": {"short": "e1a1"}},
        "players": [{"puuid": "stranger", "team_id": "Blue", "name": "S",
                     "tag": "1", "account_level": 100,
                     "agent": {"name": "Jett"},
                     "tier": {"id": 15},
                     "stats": {"score": 4000, "kills": 15, "deaths": 15,
                               "assists": 5, "headshots": 5, "bodyshots": 5,
                               "legshots": 5, "damage": {"dealt": 3000,
                                                         "received": 3000}}}],
        "teams": [{"team_id": "Blue", "won": True, "rounds": {"won": 13, "lost": 7}},
                  {"team_id": "Red", "won": False, "rounds": {"won": 7, "lost": 13}}],
    }]}

    class Stub:
        def matches(self, *a, **k):
            return payload

    match = LiveMatch("live", "coregame", "Ascent", "BombGameMode",
                      [LivePlayer("me", "Blue", "x", "Jett"),
                       LivePlayer("stranger", "Red", "x", "Jett")])
    before = R.has_history(conn, "stranger", 2_000_000_000)
    R.resolve(conn, match, "me", 2_000_000_000, client=Stub(),
              deadline_seconds=5)
    after = R.has_history(conn, "stranger", 2_000_000_000)
    assert not before
    assert after, "a fetched player must be queryable afterwards"


def test_the_context_and_the_poll_share_one_thread(monkeypatch):
    """A SQLite connection belongs to the thread that opened it.

    The context was opened on the event-loop thread and polled on an
    `asyncio.to_thread` worker, so every poll raised ProgrammingError --
    "SQLite objects created in a thread can only be used in that same thread"
    -- and the page displayed that forever. The websocket test above could not
    catch it: it accepts `status == "error"`, which it has to, because
    VALORANT is normally closed when the suite runs. So the whole dashboard
    was dead with a green suite.
    """
    import threading
    from fastapi.testclient import TestClient
    from valwr.dash import server as S

    seen = {}

    class _Ctx:
        index = None
        conn = None
        client = None            # no client means no settling calls
        settings = type("S", (), {"region": "na"})()

        def close(self):
            pass

    def fake_open(**kw):
        seen["opened"] = threading.get_ident()
        return _Ctx()

    def fake_poll(ctx, on_progress=None):
        seen["polled"] = threading.get_ident()
        return None

    monkeypatch.setattr(S.st, "open_context", fake_open)
    monkeypatch.setattr(S.st, "poll_once", fake_poll)
    with TestClient(S.build_app(no_fetch=True),
                    base_url="http://127.0.0.1:8787") as client:
        with client.websocket_connect("ws://127.0.0.1:8787/ws") as ws:
            assert ws.receive_json()["status"] == "working"
            assert ws.receive_json()["status"] == "lobby"

    assert seen["opened"] == seen["polled"], \
        "opened on one thread and polled on another: SQLite refuses that"
    assert seen["opened"] != threading.get_ident(), \
        "the poll must stay off the event loop"


def test_launching_the_dashboard_does_not_die_on_a_missing_import():
    """`main()` starts a thread that opens the browser once the port is up,
    using `threading` and `time`. Neither was imported, so starting the
    dashboard the normal way raised NameError before it ever bound -- and
    nothing imported `main`, so nothing noticed.
    """
    from valwr.dash import server as S
    assert hasattr(S, "threading") and hasattr(S, "time")


def test_your_own_account_is_refreshed_even_when_it_looks_current(tmp_path):
    """Finish a game, requeue five minutes later: your own last-20 must
    include the game you just played.

    The staleness rule cannot help here -- the account is minutes old, so it
    is "current" by any threshold. It is the one account nothing else keeps
    up to date, it costs a single call, and it is the row you actually read.
    The comment claimed this was unconditional while the code gated it.
    """
    from valwr.live import resolve as R
    now = 2_000_000_000
    # Eight matches, the newest five minutes old: recent enough and deep
    # enough that neither the staleness threshold nor the thin-history rule
    # would ask for a refetch.
    conn = _tiny_db(tmp_path, [(f"m{i}", "me", now - 300 - i * 3600)
                               for i in range(8)])
    assert R.has_history(conn, "me", now)
    assert not R.is_stale(conn, "me", now), "fresh by every rule we have"

    calls = []

    class _Client:
        def matches(self, region, platform, puuid, size, mode, start=0):
            calls.append((puuid, start))
            return {"data": []}

    match = LiveMatch("live", "coregame", "Ascent", "BombGameMode",
                      [LivePlayer("me", "Blue", "x", "Jett")])
    R.resolve(conn, match, "me", now, client=_Client(), deadline_seconds=5)
    assert calls and calls[0] == ("me", 0),         "the local account must be fetched first and regardless"


def test_a_player_who_played_two_hours_ago_counts_as_stale(tmp_path):
    """19.3% of players queue again within the hour. A threshold that called
    them current left their last-20 several games behind while looking fine."""
    from valwr.live import resolve as R
    now = 2_000_000_000
    conn = _tiny_db(tmp_path, [(f"m{i}", "p", now - 3 * 3600 - i * 60)
                               for i in range(8)])
    assert R.has_history(conn, "p", now)
    assert R.is_stale(conn, "p", now), "three hours old must be refetched"
    assert R.STALE_AFTER_SECONDS <= 3 * 3600


def test_enough_pages_are_fetched_to_fill_the_form_window(tmp_path):
    """One page is ten matches; the form figure on every row is twenty.

    Asking for one page and calling the result a "last 20" pads it out with
    whatever older games happen to be stored, which is a different sample from
    the player's actual last twenty. Measured against tracker.gg on one real
    account, that was a 0.914 where the truth was 0.952 -- six of their last
    twenty games were simply never requested.
    """
    from valwr.live import resolve as R
    now = 2_000_000_000
    conn = _tiny_db(tmp_path, [("m0", "me", now - 300)])
    calls = []

    class _Client:
        def matches(self, region, platform, puuid, size, mode, start=0):
            calls.append((puuid, start))
            return {"data": []}

    match = LiveMatch("live", "coregame", "Ascent", "BombGameMode",
                      [LivePlayer("me", "Blue", "x", "Jett"),
                       LivePlayer("them", "Red", "x", "Sova")])
    R.resolve(conn, match, "me", now, client=_Client(), deadline_seconds=30)

    assert R.HISTORY_PAGES >= 2, "one page cannot cover a twenty-game window"
    pages = [s for p, s in calls if p == "me"]
    assert pages == [0, 10], f"own history not paged: {pages}"
    # The stranger has no stored history at all, so they need depth too.
    assert ("them", 0) in calls and ("them", 10) in calls


def test_the_form_window_matches_the_one_the_score_uses(tmp_path):
    """Two modules name this number. If they drift, the live path fetches a
    different window from the one the row prints."""
    from valwr.live import resolve as R
    from valwr.rating import potential as P
    assert R.FORM_WINDOW == P.RECENT_GAMES


def test_the_dashboard_still_defaults_to_localhost():
    """`--host` exists so the lobby can be read on a phone on the same wifi.

    It must stay opt-in. The default binding is the one thing standing between
    "the match I am in" and "anything on this network", and the state it
    serves carries other players' gamertags and statistics.
    """
    import inspect

    from valwr.dash import server
    assert server.HOST == "127.0.0.1"
    src = inspect.getsource(server.main)
    assert 'ap.add_argument("--host", default=HOST' in src, (
        "--host must default to the localhost constant")
    assert "host=args.host" in src, "the server must honour --host"
    assert 'if args.host == HOST:' in src, (
        "widening the binding must be announced, not silent")


# --- surviving more than one match -------------------------------------
# The context is opened once and polled for hours. The session inside it is
# not durable: the client's access token ages out after about an hour, and
# restarting VALORANT gives the lockfile a new port and password. Before this,
# the first match worked and every later one showed an error until the
# dashboard was restarted by hand.

class _Resp:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("should have been handled before this")


def _session():
    from valwr.live.session import Session
    return Session(puuid="p", shard="na", access_token="t",
                   entitlements_token="e", client_version="v")


@pytest.mark.parametrize("status", [400, 401, 403])
def test_an_aged_out_session_is_reported_as_expired_not_as_a_failure(monkeypatch, status):
    from valwr.live import session as S
    monkeypatch.setattr(roster.httpx, "get", lambda *a, **k: _Resp(status))
    with pytest.raises(S.SessionExpired):
        roster._get(_session(), "https://glz/whatever")


def test_a_client_that_moved_is_reported_as_expired(monkeypatch):
    """The lockfile port changes on every restart, so the old one stops answering."""
    from valwr.live import session as S

    def refuse(*a, **k):
        raise roster.httpx.ConnectError("connection refused")

    monkeypatch.setattr(roster.httpx, "get", refuse)
    with pytest.raises(S.SessionExpired):
        roster._get(_session(), "https://glz/whatever")


def test_not_in_a_match_is_still_not_an_error(monkeypatch):
    monkeypatch.setattr(roster.httpx, "get", lambda *a, **k: _Resp(404))
    assert roster._get(_session(), "https://glz/whatever") is None


def _ctx(session="old"):
    from valwr.live import state as st
    return st.LiveContext(conn=None, bundle={}, index=None, session=session,
                          client=None, settings=None)


def test_an_expired_session_is_rebuilt_once_and_the_poll_carries_on(monkeypatch):
    from valwr.live import session as S
    from valwr.live import state as st

    seen = []

    def current(session, agents=None):
        seen.append(session)
        if len(seen) == 1:
            raise S.SessionExpired("HTTP 401")
        return "the roster"

    monkeypatch.setattr(st.roster, "current", current)
    monkeypatch.setattr(st, "agents_by_id", lambda conn: {})
    monkeypatch.setattr(st.S, "build", lambda: "new")

    ctx = _ctx()
    assert st.current_match(ctx) == "the roster"
    assert seen == ["old", "new"], "it must retry with the session it just built"
    assert ctx.session == "new", "and keep it for the next poll"


def test_a_client_that_has_closed_is_reported_as_not_ready(monkeypatch):
    """Rebuilding cannot help if the game is gone; the caller reopens instead."""
    from valwr.live import session as S
    from valwr.live import state as st
    from valwr.live.lockfile import ClientNotRunning

    def expired(*a, **k):
        raise S.SessionExpired("HTTP 401")

    def gone():
        raise ClientNotRunning("no lockfile")

    monkeypatch.setattr(st.roster, "current", expired)
    monkeypatch.setattr(st, "agents_by_id", lambda conn: {})
    monkeypatch.setattr(st.S, "build", gone)
    with pytest.raises(st.NotReady, match="lost the game client"):
        st.current_match(_ctx())


def test_a_fresh_session_that_also_fails_is_not_retried_forever(monkeypatch):
    from valwr.live import session as S
    from valwr.live import state as st

    def expired(*a, **k):
        raise S.SessionExpired("HTTP 401")

    monkeypatch.setattr(st.roster, "current", expired)
    monkeypatch.setattr(st, "agents_by_id", lambda conn: {})
    monkeypatch.setattr(st.S, "build", lambda: "new")
    with pytest.raises(st.NotReady, match="not answering"):
        st.current_match(_ctx())


def test_the_dashboard_reopens_its_context_instead_of_staying_stuck(monkeypatch):
    """A poll that loses the client must not strand the page until a restart."""
    from fastapi.testclient import TestClient

    from valwr.dash import server as DS

    opened = []

    class Ctx:
        index = None
        role_index = None
        conn = None
        client = None
        settings = type("S", (), {"region": "na"})()

        def close(self):
            pass

    def open_context(**kw):
        opened.append(1)
        return Ctx()

    polls = []

    def poll_once(ctx, on_progress=None):
        polls.append(1)
        if len(polls) == 1:
            raise DS.st.NotReady("lost the game client -- no lockfile.")
        return None

    monkeypatch.setattr(DS.st, "open_context", open_context)
    monkeypatch.setattr(DS.st, "poll_once", poll_once)
    monkeypatch.setattr(DS, "POLL_SECONDS", 0.01)

    with TestClient(DS.build_app(no_fetch=True),
                    base_url="http://127.0.0.1:8787") as client:
        with client.websocket_connect("ws://127.0.0.1:8787/ws") as ws:
            assert ws.receive_json()["status"] == "working"
            first = ws.receive_json()
            assert first["status"] == "error" and "Retrying" in first["message"]
            # The next tick recovers on its own.
            assert ws.receive_json()["status"] == "lobby"

    assert len(opened) == 2, "the context must be rebuilt, not reused"


# --- launching a second copy -------------------------------------------
# The usual mistake is double-clicking dashboard.bat while the first window is
# still open. Uvicorn's bind failure is a raw WinError 10048 that reads like a
# crash, so the two cases are told apart before it gets that far.

def test_a_port_in_use_is_reported_as_such():
    import socket

    from valwr.dash import server as DS

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        assert not DS.port_free("127.0.0.1", taken)
        assert DS.free_port("127.0.0.1", taken) == taken + 1
    assert DS.port_free("127.0.0.1", taken), "and free again once released"


def _fake_uvicorn(monkeypatch):
    """Stands in for uvicorn so main() can be driven without binding."""
    import sys
    import types

    seen = {}

    class Server:
        started = False

        def __init__(self, config):
            seen.update(config)

        def run(self):
            seen["ran"] = True

    module = types.ModuleType("uvicorn")
    module.Config = lambda app, host=None, port=None, log_level=None: {
        "host": host, "port": port}
    module.Server = Server
    monkeypatch.setitem(sys.modules, "uvicorn", module)
    return seen


def test_a_second_launch_opens_the_dashboard_already_running(monkeypatch):
    from valwr.dash import server as DS

    seen = _fake_uvicorn(monkeypatch)
    opened = []
    monkeypatch.setattr(DS, "port_free", lambda host, port: False)
    monkeypatch.setattr(DS, "dashboard_at", lambda port, host=DS.HOST: True)
    monkeypatch.setattr(DS.webbrowser, "open", opened.append)

    assert DS.main([]) == 0
    assert opened == [f"http://{DS.HOST}:{DS.PORT}/"]
    assert "ran" not in seen, "a second server must not be started"


def test_a_port_held_by_something_else_moves_to_the_next_one(monkeypatch):
    from valwr.dash import server as DS

    seen = _fake_uvicorn(monkeypatch)
    monkeypatch.setattr(DS, "port_free",
                        lambda host, port: port != DS.PORT)
    monkeypatch.setattr(DS, "dashboard_at", lambda port, host=DS.HOST: False)

    assert DS.main(["--no-browser"]) == 0
    assert seen["port"] == DS.PORT + 1
    assert seen.get("ran") is True


def test_a_stale_client_version_is_recoverable_and_says_so(monkeypatch):
    """X-Riot-ClientVersion comes from the running game and changes when it
    updates. The glz endpoints answer a stale one with a bare 400, which used
    to surface as an HTTPStatusError and strand the live view."""
    from valwr.live import session as S
    monkeypatch.setattr(roster.httpx, "get", lambda *a, **k: _Resp(400))
    with pytest.raises(S.SessionExpired, match="client version"):
        roster._get(_session(), "https://glz/core-game/v1/players/p")


# --- the map the client names is not the map people know ----------------
# Riot ships most maps under a codename: Summit is "Plummet", Lotus is "Jam",
# Breeze is "Foxtrot". 25 of the 26 differ. The live view printed whatever the
# client said and looked for artwork under that name, so every live match but
# one showed an unfamiliar name on a blank background.

def _maps_db(tmp_path, rows):
    from valwr.store import schema
    conn = schema.connect(tmp_path / "maps.db")
    schema.create_all(conn)
    conn.executemany(
        "INSERT INTO ref_maps (uuid, name, path) VALUES (?,?,?)", rows)
    conn.commit()
    return conn


def test_a_codename_resolves_to_the_name_people_use(tmp_path):
    from valwr.live import state as st
    conn = _maps_db(tmp_path, [("u1", "Summit", "Plummet"),
                               ("u2", "Lotus", "Jam")])
    assert st.display_map(conn, "Plummet") == "Summit"
    assert st.display_map(conn, "plummet") == "Summit", "matched case-insensitively"
    assert st.display_map(conn, "Jam") == "Lotus"


def test_an_unknown_map_is_passed_through_rather_than_blanked(tmp_path):
    """A map that ships before the reference table is refreshed still names
    itself, and the page falls back to no background rather than breaking."""
    from valwr.live import state as st
    conn = _maps_db(tmp_path, [("u1", "Summit", "Plummet")])
    assert st.display_map(conn, "SomethingNew") == "SomethingNew"
    assert st.display_map(conn, None) is None
    assert st.display_map(conn, "") == ""


def test_the_poll_reports_the_resolved_map(tmp_path, monkeypatch):
    from valwr.live import predict as P
    from valwr.live import resolve as R
    from valwr.live import state as st

    conn = _maps_db(tmp_path, [("u1", "Summit", "Plummet")])
    match = LiveMatch("m1", "coregame", "Plummet", "Bomb",
                      [LivePlayer("me", "Blue", "aid")])
    monkeypatch.setattr(st, "current_match", lambda ctx: match)
    monkeypatch.setattr(st, "_player_rows", lambda *a, **k: [])
    monkeypatch.setattr(st, "parties", lambda *a, **k: [])
    monkeypatch.setattr(R, "resolve", lambda *a, **k: R.Resolution(known=set()))
    monkeypatch.setattr(P, "predict", lambda *a, **k: None)

    class Sess:
        puuid = "me"

    import types
    ctx = st.LiveContext(conn=conn, bundle={"best": "logistic regression"},
                         index=None, session=Sess(), client=None,
                         settings=types.SimpleNamespace(region="na",
                                                        platform="pc"))
    assert st.poll_once(ctx)["map"] == "Summit"


def test_every_map_the_api_ships_carries_its_codename():
    """load_maps used to keep only maps with coordinates, which dropped the
    range and the deathmatch arenas -- and the live view follows you there."""
    from valwr.store import reference

    class FakeClient:
        def get(self, url, params=None):
            class R:
                status_code = 200

                @staticmethod
                def raise_for_status():
                    pass

                @staticmethod
                def json():
                    return {"data": [
                        {"uuid": "u1", "displayName": "Summit",
                         "mapUrl": "/Game/Maps/Plummet/Plummet",
                         "coordinates": "x"},
                        {"uuid": "u2", "displayName": "The Range",
                         "mapUrl": "/Game/Maps/Range/Range"},
                        {"uuid": "u3", "mapUrl": "/Game/Maps/Nameless/Nameless"},
                    ]}
            return R()

    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    from valwr.store import schema
    schema.create_all(conn)
    assert reference.load_maps(conn, FakeClient()) == 2, "the nameless one is skipped"
    got = {r["name"]: r["path"] for r in conn.execute("SELECT name, path FROM ref_maps")}
    assert got == {"Summit": "Plummet", "The Range": "Range"}
    assert reference.codename(None) is None


# --- a tab per match, and the record behind it --------------------------

def _recorded_db(tmp_path, monkeypatch):
    """A dashboard whose database holds one recorded, settled match."""
    import importlib.util
    import pathlib as _pathlib

    from valwr.dash import server as DS
    from valwr.live import outcomes
    from valwr.store import schema

    spec = importlib.util.spec_from_file_location(
        "test_outcomes", _pathlib.Path(__file__).with_name("test_outcomes.py"))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    path = tmp_path / "rec.db"
    conn = schema.connect(path)
    schema.create_all(conn)
    outcomes.record(conn, helpers.state(), now=1000)
    outcomes.settle(conn, helpers.FakeAPI(helpers.finished()), "na", "m1",
                    now=2000)
    conn.close()
    # A connection per call: these routes run on worker threads, and SQLite
    # refuses a connection opened on another one.
    monkeypatch.setattr(DS, "_open_db", lambda: schema.connect(path))
    return DS, path


def test_a_match_this_dashboard_never_saw_is_not_served(tmp_path, monkeypatch):
    """The pinned page must not become a way to look anything else up."""
    from fastapi.testclient import TestClient

    DS, _ = _recorded_db(tmp_path, monkeypatch)
    with TestClient(DS.build_app(no_fetch=True),
                    base_url="http://127.0.0.1:8787") as client:
        assert client.get("/m/m1").status_code == 200
        assert client.get("/m/somebody-elses-match").status_code == 404


def test_a_pinned_tab_shows_that_match_and_its_result(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    DS, _ = _recorded_db(tmp_path, monkeypatch)
    with TestClient(DS.build_app(no_fetch=True),
                    base_url="http://127.0.0.1:8787") as client:
        with client.websocket_connect("ws://127.0.0.1:8787/ws?pinned=m1") as ws:
            msg = ws.receive_json()

    assert msg["status"] == "match"
    assert msg["state"]["map"] == "Sunset", "the state as it was recorded"
    assert msg["review"]["settled"] is True
    assert msg["review"]["actual"]["winner"] == "Blue"
    assert msg["review"]["correct"] == 1


def test_the_scorecard_endpoint_reports_the_record(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    DS, _ = _recorded_db(tmp_path, monkeypatch)
    with TestClient(DS.build_app(no_fetch=True),
                    base_url="http://127.0.0.1:8787") as client:
        card = client.get("/api/scorecard").json()
        assert client.get("/results").status_code == 200

    assert card["recorded"] == 1 and card["competitive"]["n"] == 1
    assert card["insights"], "it always says what the record supports"


def test_each_new_match_opens_its_own_tab_once(tmp_path, monkeypatch):
    """Instead of a tab being replaced by the next match, each match gets one."""
    from fastapi.testclient import TestClient

    from valwr.dash import server as DS

    opened = []
    seen = []

    class Ctx:
        index = None
        role_index = None
        conn = None
        client = None
        settings = type("S", (), {"region": "na"})()

        def close(self):
            pass

    def poll(ctx, on_progress=None):
        seen.append(1)
        return {"match_id": "m1" if len(seen) < 3 else "m2", "players": [],
                "warnings": [], "prediction": None}

    monkeypatch.setattr(DS.st, "open_context", lambda **kw: Ctx())
    monkeypatch.setattr(DS, "poll_and_record", poll)
    monkeypatch.setattr(DS.outcomes, "settle_pending", lambda *a, **k: None)
    monkeypatch.setattr(DS.webbrowser, "open", opened.append)
    monkeypatch.setattr(DS, "POLL_SECONDS", 0.01)

    app = DS.build_app(no_fetch=True)
    app.state.base_url = "http://127.0.0.1:8787/"
    with TestClient(app, base_url="http://127.0.0.1:8787") as client:
        with client.websocket_connect("ws://127.0.0.1:8787/ws") as ws:
            for _ in range(4):
                ws.receive_json()

    assert opened == ["http://127.0.0.1:8787/m/m1",
                      "http://127.0.0.1:8787/m/m2"], "one tab per match, once each"



@pytest.fixture(autouse=True)
def _isolated_port_file(tmp_path, monkeypatch):
    """No test may read or write the real dashboard-port file.

    It lives beside the database, so without this the suite both inherits
    whatever port the user's last real run chose -- which made one test fail
    on a developer machine and pass in CI -- and writes into their data
    directory, changing which port their next dashboard binds.
    """
    from valwr.dash import server as S
    monkeypatch.setattr(S, "port_file", lambda: tmp_path / "dashboard-port")


# --- coming back to the same port --------------------------------------

def test_the_port_the_last_run_used_is_preferred(tmp_path, monkeypatch):
    """Every per-match tab is an absolute address with a port in it.

    A run that binds somewhere else leaves every one of those tabs dead --
    ERR_CONNECTION_REFUSED, with nothing the page can do about it, because
    nothing is listening to serve it.
    """
    from valwr.dash import server as S
    path = tmp_path / "dashboard-port"
    monkeypatch.setattr(S, "port_file", lambda: path)

    assert S.remembered_port() is None, "nothing remembered on a fresh install"
    assert S.preferred_port(None, None) == S.PORT

    S.remember_port(8790)
    assert S.remembered_port() == 8790
    assert S.preferred_port(None, S.remembered_port()) == 8790


def test_an_explicit_port_beats_the_remembered_one():
    from valwr.dash import server as S
    assert S.preferred_port(9001, 8790) == 9001


@pytest.mark.parametrize("junk", ["", "  ", "not-a-port", "80", "99999"])
def test_a_damaged_port_file_is_ignored(tmp_path, monkeypatch, junk):
    """A truncated or hand-edited file must not stop the dashboard starting,
    and must not send it at a privileged or impossible port."""
    from valwr.dash import server as S
    path = tmp_path / "dashboard-port"
    path.write_text(junk, encoding="utf-8")
    monkeypatch.setattr(S, "port_file", lambda: path)
    assert S.remembered_port() is None
    assert S.preferred_port(None, S.remembered_port()) == S.PORT


def test_remembering_a_port_never_fails_the_launch(tmp_path, monkeypatch):
    """Best effort: an unwritable directory is not a reason not to start."""
    from valwr.dash import server as S
    monkeypatch.setattr(S, "port_file",
                        lambda: tmp_path / "nope" / "deeper" / "dashboard-port")
    S.remember_port(8788)
    monkeypatch.setattr(S, "port_file", lambda: None)
    S.remember_port(8788)
    assert S.remembered_port() is None


def test_a_browser_that_will_not_open_cannot_take_down_the_live_view(monkeypatch,
                                                                    capsys):
    """Opening the per-match tab runs inside the poll loop.

    A browser that refuses -- or throws, which it does on a machine with none
    registered -- would otherwise kill the websocket and take the live view
    with it, for the sake of a convenience window.
    """
    from valwr.dash import server as S

    def boom(url):
        raise OSError("no browser here")
    monkeypatch.setattr(S.webbrowser, "open", boom)
    assert S.open_match_tab("http://127.0.0.1:8787/", "m1") is False
    said = capsys.readouterr().out
    assert "m/m1" in said, "the URL has to be printed so it is still reachable"

    monkeypatch.setattr(S.webbrowser, "open", lambda url: False)
    assert S.open_match_tab("http://127.0.0.1:8787/", "m1") is False
    assert "m/m1" in capsys.readouterr().out

    opened = []
    monkeypatch.setattr(S.webbrowser, "open", lambda url: opened.append(url) or True)
    assert S.open_match_tab("http://127.0.0.1:8787/", "m2") is True
    assert opened == ["http://127.0.0.1:8787/m/m2"]


def test_results_are_collected_without_a_browser_or_the_game(monkeypatch):
    """Settling must not depend on a websocket or on VALORANT running.

    It used to live inside the websocket loop, which needed a tab open, and the
    loop gave up before reaching it whenever the game was closed. Finishing a
    match and quitting the game is the most ordinary thing a player does, and
    it was exactly the case where the result was never fetched: the prediction
    sat unsettled and the comparison never appeared.
    """
    import time as _time

    from fastapi.testclient import TestClient

    from valwr.dash import server as DS

    calls = []
    monkeypatch.setattr(DS, "settle_tick",
                        lambda no_fetch=False: (calls.append(no_fetch),
                                                {"settled": 0})[1])
    monkeypatch.setattr(DS, "SETTLE_EVERY_SECONDS", 0.02)
    # No websocket is ever opened, and no live context exists.
    with TestClient(DS.build_app(no_fetch=True, settle=True)):
        for _ in range(100):
            if calls:
                break
            _time.sleep(0.02)
    assert calls, "nothing collected results with no tab open"
    assert calls[0] is True, "the no-fetch flag has to reach it"


def test_the_demo_never_collects_results(monkeypatch):
    """The demo touches no database, so it must not start a settler either."""
    import time as _time

    from fastapi.testclient import TestClient

    from valwr.dash import server as DS

    calls = []
    monkeypatch.setattr(DS, "settle_tick",
                        lambda no_fetch=False: calls.append(1))
    monkeypatch.setattr(DS, "SETTLE_EVERY_SECONDS", 0.02)
    with TestClient(DS.build_app(demo=True, settle=True)):
        _time.sleep(0.15)
    assert calls == []


def test_the_recorded_matches_do_not_need_a_live_context(monkeypatch, tmp_path):
    """Read with their own connection, because the moment the game closes
    there is no context -- and that is when they matter most."""
    from valwr.dash import server as DS
    from valwr.live import outcomes
    from valwr.store import schema

    db = tmp_path / "s.db"
    conn = schema.connect(db)
    schema.create_all(conn)
    state = {"match_id": "m1", "phase": "coregame", "map": "Split",
             "mode": "Bomb", "standard_mode": True, "is_custom": False,
             "own_team": "Blue", "coverage": 8, "confidence": "moderate",
             "model": "logistic regression", "players": [],
             "prediction": {"own_probability": 0.6, "win_probability": 0.6}}
    outcomes.record(conn, state, now=1000)
    conn.commit()
    conn.close()

    monkeypatch.setattr(DS, "review", DS.review)
    monkeypatch.setattr("valwr.config.load",
                        lambda require_key=True: types.SimpleNamespace(
                            database_path=db, region="na"))
    got = DS.recent_rows()
    assert [r["match_id"] for r in got] == ["m1"]
    assert got[0]["map"] == "Split" and got[0]["settled"] is False


def test_building_an_app_does_not_reach_the_real_database(monkeypatch):
    """Constructing an app must have no side effects on the live database.

    It used to. The results collector started with the app, so every test that
    built one opened the user's real database and wrote to it -- holding the
    write lock long enough to kill two full backfills mid-run with
    "database is locked". Only `main` asks for the collector now.
    """
    import time as _time

    from fastapi.testclient import TestClient

    from valwr.dash import server as DS

    calls = []
    monkeypatch.setattr(DS, "settle_tick",
                        lambda no_fetch=False: calls.append(1))
    monkeypatch.setattr(DS, "SETTLE_EVERY_SECONDS", 0.01)
    with TestClient(DS.build_app(no_fetch=True)):
        _time.sleep(0.15)
    assert calls == [], "an app built for a test collected results"


# --- agent select ------------------------------------------------------
# Agent select allows about a minute to lock in, and the API allows roughly
# one lookup every two and a half seconds. What the screen shows inside that
# minute is decided by what gets fetched first and what never gets repeated.

class _Recorder:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def matches(self, region, platform, puuid, size, mode, start=0):
        self.calls.append((puuid, start))
        if self.fail:
            raise self.fail
        return {"data": []}


def _pregame(*teammates):
    return LiveMatch("live", "pregame", "Ascent", "BombGameMode",
                     [LivePlayer("me", "Blue", "x", None)]
                     + [LivePlayer(t, "Blue", "x", None) for t in teammates])


def test_agent_select_looks_up_teammates_before_you(tmp_path):
    """You know how you play. The four strangers are what the screen is for."""
    now = 2_000_000_000
    conn = _tiny_db(tmp_path, [("m0", "me", now - 300)])
    client = _Recorder()
    R.resolve(conn, _pregame("t1", "t2"), "me", now, client=client,
              deadline_seconds=30, teammates_first=True)
    first_own = client.calls.index(("me", 0))
    assert {("t1", 0), ("t2", 0)} <= set(client.calls[:first_own]), (
        f"teammates must come first in agent select: {client.calls}")
    assert ("me", 0) in client.calls, "your own refresh still happens, last"


def test_outside_agent_select_your_own_account_still_comes_first(tmp_path):
    now = 2_000_000_000
    conn = _tiny_db(tmp_path, [("m0", "me", now - 300)])
    client = _Recorder()
    R.resolve(conn, _pregame("t1"), "me", now, client=client,
              deadline_seconds=30)
    assert client.calls[0] == ("me", 0)


def test_a_lookup_that_answered_is_not_repeated_in_the_same_match(tmp_path):
    """The dashboard polls every few seconds. Anyone short of twenty stored
    games -- or with no competitive history at all -- was re-fetched on every
    one of those polls, spending the quota agent select needs on answers the
    last poll already had."""
    now = 2_000_000_000
    conn = _tiny_db(tmp_path, [("m0", "me", now - 300)])
    first = _Recorder()
    out = R.resolve(conn, _pregame("t1"), "me", now, client=first,
                    deadline_seconds=30, teammates_first=True)
    assert out.completed == set(first.calls) and first.calls

    second = _Recorder()
    R.resolve(conn, _pregame("t1"), "me", now, client=second,
              deadline_seconds=30, teammates_first=True,
              already=out.completed)
    assert second.calls == [], f"repeated lookups: {second.calls}"


def test_the_first_report_comes_before_any_lookup(tmp_path):
    """The first paint: everyone already stored is shown immediately, with
    the players still to come named, rather than after the slowest lookup."""
    now = 2_000_000_000
    conn = _tiny_db(tmp_path, [(f"m{i}", "known", now - 600 - i)
                               for i in range(8)])
    client = _Recorder()
    seen = []

    def progress(res):
        seen.append((len(client.calls), set(res.known), set(res.pending)))

    R.resolve(conn, _pregame("known", "stranger"), "me", now, client=client,
              deadline_seconds=30, teammates_first=True, on_progress=progress)
    calls_then, known, pending = seen[0]
    assert calls_then == 0, "the first report must not wait on the API"
    assert "known" in known
    assert "stranger" in pending, "the page must be told who is still coming"


def test_a_lookup_cut_off_by_the_deadline_stays_pending(tmp_path):
    now = 2_000_000_000
    conn = _tiny_db(tmp_path, [])
    out = R.resolve(conn, _pregame("t1", "t2"), "me", now, client=_Recorder(),
                    deadline_seconds=0, teammates_first=True)
    assert {"t1", "t2", "me"} <= out.pending
    assert not out.completed


def test_running_out_of_quota_leaves_the_lookup_for_the_next_poll(tmp_path):
    """Rate limiting is temporary. Marking the lookup done would stop the next
    poll from retrying it, and the player would stay blank all match."""
    from valwr.collect.client import RateLimited
    now = 2_000_000_000
    conn = _tiny_db(tmp_path, [])
    out = R.resolve(conn, _pregame("t1"), "me", now,
                    client=_Recorder(fail=RateLimited(60.0)),
                    deadline_seconds=30, teammates_first=True)
    assert ("t1", 0) not in out.completed
    assert "t1" in out.pending


def _agent_select_ctx(tmp_path, monkeypatch, match):
    """A live context around a real store and a recording client, with the
    scoring stubbed out -- these tests are about when states are sent."""
    import types

    from valwr.live import predict as P
    from valwr.live import state as st
    conn = _tiny_db(tmp_path, [(f"k{i}", "known", 1_999_990_000 - i)
                               for i in range(8)])
    monkeypatch.setattr(st, "current_match", lambda ctx: match)
    monkeypatch.setattr(st, "display_map", lambda conn, m: m)
    monkeypatch.setattr(st, "_player_rows", lambda *a, **k: [])
    monkeypatch.setattr(st, "parties", lambda *a, **k: [])
    monkeypatch.setattr(P, "predict", lambda *a, **k: None)
    monkeypatch.setattr(st.time, "time", lambda: 2_000_000_000)

    class Sess:
        puuid = "me"

    client = _Recorder()
    ctx = st.LiveContext(conn=conn, bundle={"best": "lr"}, index=None,
                         session=Sess(), client=client, deadline=30,
                         settings=types.SimpleNamespace(region="na",
                                                        platform="pc"))
    return ctx, client


def test_agent_select_paints_before_the_first_lookup(tmp_path, monkeypatch):
    """The page used to show nothing until every lookup had finished -- up to
    25 seconds of a 60-second pick -- although most of the lobby was usually
    stored all along."""
    from valwr.live import state as st
    ctx, client = _agent_select_ctx(tmp_path, monkeypatch,
                                    _pregame("known", "stranger"))
    sent = []
    final = st.poll_once(
        ctx, on_progress=lambda s: sent.append((len(client.calls), s)))

    calls_then, first = sent[0]
    assert calls_then == 0, "the first paint must not wait on the API"
    assert set(first) == set(final), "a partial state has the final shape"
    assert "stranger" in first["lookup"]["pending"]
    assert final["lookup"] == {"pending": [], "remaining": 0}


def test_the_next_poll_repeats_no_lookup(tmp_path, monkeypatch):
    from valwr.live import state as st
    ctx, client = _agent_select_ctx(tmp_path, monkeypatch,
                                    _pregame("known", "stranger"))
    st.poll_once(ctx)
    first = len(client.calls)
    assert first, "the first poll should have looked players up"
    st.poll_once(ctx)
    assert len(client.calls) == first, (
        f"second poll repeated lookups: {client.calls[first:]}")


def test_a_new_match_is_looked_up_afresh(tmp_path, monkeypatch):
    """Remembered lookups belong to one match. Carrying them into the next
    would skip your own refresh -- the bug that froze a score for 12 days."""
    import dataclasses

    from valwr.live import state as st
    match = _pregame("known")
    ctx, client = _agent_select_ctx(tmp_path, monkeypatch, match)
    st.poll_once(ctx)
    before = len(client.calls)
    monkeypatch.setattr(st, "current_match",
                        lambda c: dataclasses.replace(match, match_id="next"))
    st.poll_once(ctx)
    assert ("me", 0) in client.calls[before:]


@pytest.mark.parametrize("flags", [["--demo", "pregame"], ["--demo"],
                                   ["--match", "m1"]])
def test_a_demo_or_replay_does_not_move_the_live_dashboard(tmp_path,
                                                            monkeypatch, flags):
    """A demo on a side port recorded that port as the dashboard's, so the
    next dashboard.bat quietly moved to it -- and every match tab left open
    from the live one pointed at nothing. Only the live view opens those
    tabs, so only it may say where the dashboard lives."""
    import uvicorn

    from valwr import config
    from valwr.dash import server as S
    path = tmp_path / "dashboard-port"
    path.write_text("8787", encoding="utf-8")
    monkeypatch.setattr(S, "port_file", lambda: path)
    monkeypatch.setattr(S, "port_free", lambda host, port: True)

    def no_database(**kw):
        raise RuntimeError("tests must not open the real database")
    monkeypatch.setattr(config, "load", no_database)

    class _Server:
        def __init__(self, cfg):
            self.started = True

        def run(self):
            pass
    monkeypatch.setattr(uvicorn, "Server", _Server)

    assert S.main([*flags, "--port", "8788", "--no-browser"]) == 0
    assert path.read_text(encoding="utf-8") == "8787"


def test_the_live_dashboard_still_remembers_its_port(tmp_path, monkeypatch):
    import uvicorn

    from valwr import config
    from valwr.dash import server as S
    path = tmp_path / "dashboard-port"
    monkeypatch.setattr(S, "port_file", lambda: path)
    monkeypatch.setattr(S, "port_free", lambda host, port: True)

    def no_database(**kw):
        raise RuntimeError("tests must not open the real database")
    monkeypatch.setattr(config, "load", no_database)

    class _Server:
        def __init__(self, cfg):
            self.started = True

        def run(self):
            pass
    monkeypatch.setattr(uvicorn, "Server", _Server)

    assert S.main(["--port", "8790", "--no-browser", "--no-tabs"]) == 0
    assert path.read_text(encoding="utf-8") == "8790"


def test_the_poll_is_quick_enough_for_agent_select_and_no_quicker():
    """Agent select allows about a minute to lock in; at five seconds an
    average 2.5 of it passed before the page noticed. Every poll is a request
    to Riot's servers made with the player's session, and the project's own
    rules allow polling "every few seconds" -- so there is a floor as well as a
    ceiling. The terminal view answers the same question and keeps the same
    cadence, which is what the docs promise."""
    from valwr.dash import server as DS
    from valwr.live import __main__ as LV
    assert 2.0 <= DS.POLL_SECONDS <= 3.0
    assert LV.POLL_SECONDS == DS.POLL_SECONDS


def test_your_picks_are_offered_in_agent_select_and_not_in_game(tmp_path, monkeypatch):
    """Agent select is where you pick; in game the question is settled."""
    import dataclasses

    from valwr.live import state as st
    match = _pregame("known")
    ctx, _ = _agent_select_ctx(tmp_path, monkeypatch, match)
    sentinel = {"map": "Ascent", "games": 9, "agents": []}
    monkeypatch.setattr(st.picks, "your_picks", lambda *a, **k: sentinel)
    assert st.poll_once(ctx)["your_picks"] == sentinel
    monkeypatch.setattr(st, "current_match", lambda c: dataclasses.replace(
        match, phase="coregame"))
    assert st.poll_once(ctx)["your_picks"] is None


# --- the team's roles in agent select --------------------------------------

def _pregame_payload(*players):
    return {"MatchID": "m1", "MapID": "/Game/Maps/Ascent/Ascent",
            "ModeID": "/Game/GameModes/Bomb/BombGameMode",
            "AllyTeam": {"TeamID": "Blue", "Players": list(players)}}


def _fetch_pregame(monkeypatch, body):
    monkeypatch.setattr(roster, "_get", lambda session, url: body)
    return roster.fetch(_StubSession(), "m1", "pregame")


def test_agent_select_tells_a_hovered_agent_from_a_locked_one(monkeypatch):
    """A hover can still change; the team's roles are read from both, and the
    page must not show one as the other."""
    m = _fetch_pregame(monkeypatch, _pregame_payload(
        {"Subject": "a", "CharacterID": "x", "CharacterSelectionState": "locked"},
        {"Subject": "b", "CharacterID": "y", "CharacterSelectionState": "selected"},
        {"Subject": "c", "CharacterID": "", "CharacterSelectionState": ""},
        {"Subject": "d", "CharacterID": "z"}))
    got = {p.puuid: (p.agent_id, p.selection) for p in m.players}
    assert got == {"a": ("x", "locked"), "b": ("y", "selected"),
                   "c": (None, None), "d": ("z", None)}


def test_naming_the_agents_keeps_everything_else_about_the_match():
    """resolve_agents rebuilt the match from five named fields and dropped the
    rest, so every live poll lost `flow` and `coaches`: a custom game read as
    a queued one and its warnings never showed."""
    m = LiveMatch("m", "pregame", "Ascent", "BombGameMode",
                  [LivePlayer("a", "Blue", "x", selection="selected")],
                  flow="CustomGame", coaches=1)
    r = roster.resolve_agents(m, {"x": "Jett"})
    assert r.is_custom and r.coaches == 1
    assert r.players[0].agent == "Jett" and r.players[0].selection == "selected"


def test_a_likely_role_is_read_from_the_last_twenty_games(tmp_path):
    from valwr.live import comp
    now = 2_000_000_000
    conn = _tiny_db(tmp_path, [(f"m{i}", "p", now - 100 - i) for i in range(25)])
    # _tiny_db plays every game on Jett; recast the five oldest as Omen, so
    # they fall outside the twenty that count.
    conn.execute("UPDATE match_players SET agent='Omen' WHERE match_id IN "
                 "('m20','m21','m22','m23','m24')")
    roles = {"Jett": "Duelist", "Omen": "Controller"}
    assert comp.likely_role(conn, "p", now, roles) == {
        "role": "Duelist", "games": 20, "of": 20}
    assert comp.likely_role(conn, "nobody", now, roles) is None


def test_a_tied_likely_role_goes_to_the_one_played_most_recently(tmp_path):
    from valwr.live import comp
    now = 2_000_000_000
    conn = _tiny_db(tmp_path, [(f"m{i}", "p", now - 100 - i) for i in range(4)])
    conn.execute("UPDATE match_players SET agent='Omen' WHERE match_id IN ('m0','m1')")
    roles = {"Jett": "Duelist", "Omen": "Controller"}
    assert comp.likely_role(conn, "p", now, roles)["role"] == "Controller"
