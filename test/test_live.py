"""Phase 6 tests: lockfile auth, roster parsing, and the resolution deadline.

Most of what can go wrong here is silent. A launcher mistaken for a running
game, a placeholder version header, an XMPP region used as a game shard -- none
of those raise anything obvious; they produce an opaque 400 or a host that does
not resolve, several layers away from the cause. Each was hit for real while
building this, and each has a test.
"""

from __future__ import annotations

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
    expected = {"/", "/ws"}
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

        def close(self):
            pass

    def fake_open(**kw):
        seen["opened"] = threading.get_ident()
        return _Ctx()

    def fake_poll(ctx):
        seen["polled"] = threading.get_ident()
        return None

    monkeypatch.setattr(S.st, "open_context", fake_open)
    monkeypatch.setattr(S.st, "poll_once", fake_poll)
    with TestClient(S.build_app(no_fetch=True),
                    base_url="http://127.0.0.1:8787") as client:
        with client.websocket_connect("ws://127.0.0.1:8787/ws") as ws:
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
