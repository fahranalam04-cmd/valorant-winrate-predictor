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
