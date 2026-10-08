"""Three or more results the same this session, and only when it is current."""

from __future__ import annotations

import pytest

from valwr.live import resolve as R
from valwr.live import streak as K
from valwr.live.roster import LiveMatch, LivePlayer
from valwr.store import schema

NOW = 2_000_000_000
MIN = 60
HOUR = 3600


def store(tmp_path, games, puuid="t1"):
    """`games` as (minutes before NOW the game began, won) -- won None is a
    draw. Bomb-mode competitive, as the history query requires."""
    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)
    for i, (ago, won) in enumerate(games):
        mid, ts = f"m{i}", NOW - ago * MIN
        conn.execute(
            "INSERT INTO matches (match_id, started_at, map, mode, queue, region,"
            " season, rounds_red, rounds_blue, winner, data_quality, ingested_at)"
            " VALUES (?, ?, 'Ascent', 'competitive', 'Standard', 'na', 's', 9,"
            " 13, 'Blue', NULL, 0)", (mid, ts))
        conn.execute(
            "INSERT INTO match_players (match_id, puuid, team, agent, tier, "
            "score, kills, deaths, assists, headshots, bodyshots, legshots, "
            "damage_dealt, damage_taken, started_at, map, won, rounds_played) "
            "VALUES (?, ?, 'Blue', 'Jett', 15, 4000, 15, 15, 5, 5, 5, 5, 3000, "
            "3000, ?, 'Ascent', ?, 20)", (mid, puuid, ts, won))
    conn.commit()
    return conn


# Games about 45 minutes apart, the newest just finished.
EVENING = [30, 75, 120, 165, 210]


def run(tmp_path, results, fetched=False, starts=EVENING):
    conn = store(tmp_path, list(zip(starts, results)))
    return K.current(conn, "t1", NOW, fetched=fetched)


def test_three_losses_this_session_is_a_run(tmp_path):
    assert run(tmp_path, [0, 0, 0, 1, 1]) == {"result": "lost", "count": 3}


def test_and_so_is_a_winning_one_counted_to_its_end(tmp_path):
    assert run(tmp_path, [1, 1, 1, 1, 0]) == {"result": "won", "count": 4}


def test_two_is_not_shown(tmp_path):
    assert run(tmp_path, [0, 0, 1, 0, 0]) is None


def test_a_draw_ends_the_run(tmp_path):
    assert run(tmp_path, [0, 0, None, 0, 0]) is None


def test_a_two_hour_break_ends_the_session(tmp_path):
    """Last night's losses are not tonight's streak. The 4h gap is between
    the second and third games."""
    starts = [30, 75, 75 + 4 * 60, 75 + 5 * 60, 75 + 6 * 60]
    assert run(tmp_path, [0, 0, 0, 0, 0], starts=starts) is None


def test_a_break_under_two_hours_does_not(tmp_path):
    """Start to start includes the game itself: 2h30m between starts is a
    break of under two hours."""
    starts = [30, 30 + 150, 30 + 300]
    assert run(tmp_path, [0, 0, 0], starts=starts, fetched=True) == {
        "result": "lost", "count": 3}


def test_a_run_from_hours_ago_with_no_session_since_is_not_shown(tmp_path):
    starts = [5 * 60, 5 * 60 + 45, 5 * 60 + 90]
    assert run(tmp_path, [0, 0, 0], starts=starts, fetched=True) is None


def test_a_record_that_may_be_a_game_behind_shows_nothing_until_fetched(tmp_path):
    """Newest stored game 80 minutes ago: they could have finished one since,
    and a badge built on the games before it could be wrong."""
    conn = store(tmp_path, [(80, 0), (125, 0), (170, 0)])
    assert K.current(conn, "t1", NOW, fetched=False) is None
    assert K.current(conn, "t1", NOW, fetched=True) == {
        "result": "lost", "count": 3}


@pytest.mark.parametrize("ago, missing", [(30, False), (54, False),
                                          (56, True), (110, True)])
def test_who_may_be_missing_a_game(tmp_path, ago, missing):
    conn = store(tmp_path, [(ago, 0)])
    assert K.may_be_missing_a_game(conn, "t1", NOW) is missing


def test_nobody_stored_is_not_missing_anything(tmp_path):
    conn = store(tmp_path, [])
    assert K.may_be_missing_a_game(conn, "t1", NOW) is False
    assert K.current(conn, "t1", NOW, fetched=True) is None


def test_no_game_fits_inside_the_complete_window():
    """The rule the window rests on: under COMPLETE_WITHIN_SECONDS since a
    game began, it cannot have ended and another been played."""
    shortest_game_and_queue = 25 * MIN
    assert K.COMPLETE_WITHIN_SECONDS < 2 * shortest_game_and_queue + 10 * MIN


# --- the lookup: fetched again, but after everything else ----------------

class _Recorder:
    def __init__(self):
        self.calls = []

    def matches(self, region, platform, puuid, size, mode, start=0):
        self.calls.append((puuid, start))
        return {"data": []}


def _lobby(tmp_path, phase="pregame"):
    """You, a stranger, and two known teammates -- one a game behind. Once the
    match has loaded, an enemy who is a game behind as well."""
    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)
    rows = [("me", 20), ("behind", 80), ("current", 20), ("rival", 80)]
    for puuid, ago in rows:
        for k in range(R.FORM_WINDOW):         # deep enough for no page two
            mid = f"{puuid}{k}"
            ts = NOW - (ago + 45 * k) * MIN
            conn.execute(
                "INSERT INTO matches (match_id, started_at, map, mode, queue, "
                "region, season, rounds_red, rounds_blue, winner, data_quality,"
                " ingested_at) VALUES (?, ?, 'Ascent', 'competitive', "
                "'Standard', 'na', 's', 9, 13, 'Blue', NULL, 0)", (mid, ts))
            conn.execute(
                "INSERT INTO match_players (match_id, puuid, team, agent, "
                "started_at, map, won, rounds_played) VALUES "
                "(?, ?, 'Blue', 'Jett', ?, 'Ascent', 0, 20)", (mid, puuid, ts))
    conn.commit()
    match = LiveMatch("live", phase, "Ascent", "BombGameMode", [
        LivePlayer(p, "Blue", "x", None)
        for p in ("me", "stranger", "behind", "current")]
        + ([LivePlayer("rival", "Red", "x", None)] if phase == "coregame"
           else []))
    return conn, match


def test_a_teammate_a_game_behind_is_fetched_after_everything_else(tmp_path):
    conn, match = _lobby(tmp_path)
    client = _Recorder()
    R.resolve(conn, match, "me", NOW, client=client, deadline_seconds=30,
              teammates_first=True, refresh_last=["behind"])
    assert client.calls[-1] == ("behind", 0), client.calls
    assert client.calls.index(("stranger", 0)) < client.calls.index(("me", 0)) \
        < client.calls.index(("behind", 0)), "after the ratings and your own"


def test_the_refresh_is_not_repeated_within_the_match(tmp_path):
    conn, match = _lobby(tmp_path)
    first = _Recorder()
    out = R.resolve(conn, match, "me", NOW, client=first, deadline_seconds=30,
                    refresh_last=["behind"])
    again = _Recorder()
    R.resolve(conn, match, "me", NOW, client=again, deadline_seconds=30,
              already=out.completed, refresh_last=["behind"])
    assert ("behind", 0) not in again.calls


def test_a_stranger_is_not_queued_twice(tmp_path):
    """Nobody stored has nothing to be behind on; their first page is already
    the second item on the list."""
    conn, match = _lobby(tmp_path)
    client = _Recorder()
    R.resolve(conn, match, "me", NOW, client=client, deadline_seconds=30,
              refresh_last=["stranger"])
    assert client.calls.count(("stranger", 0)) == 1


def test_out_of_time_it_waits_for_the_next_poll(tmp_path):
    """The badge shows late rather than costing a rating its lookup."""
    conn, match = _lobby(tmp_path)
    out = R.resolve(conn, match, "me", NOW, client=_Recorder(),
                    deadline_seconds=0, teammates_first=True,
                    refresh_last=["behind"])
    assert ("behind", 0) in out.remaining


@pytest.mark.parametrize("phase", ["pregame", "coregame"])
def test_the_live_poll_asks_again_for_a_teammate_a_game_behind(
        tmp_path, monkeypatch, phase):
    """And for nobody else: not a teammate whose record is complete, and not
    an enemy, who carries no badge."""
    import types

    from valwr.live import predict as P
    from valwr.live import state as st
    conn, match = _lobby(tmp_path, phase)
    monkeypatch.setattr(st, "current_match", lambda ctx: match)
    monkeypatch.setattr(st, "display_map", lambda conn, m: m)
    monkeypatch.setattr(st, "_player_rows", lambda *a, **k: [])
    monkeypatch.setattr(st, "parties", lambda *a, **k: [])
    monkeypatch.setattr(st.picks, "your_picks", lambda *a, **k: None)
    monkeypatch.setattr(P, "predict", lambda *a, **k: None)
    monkeypatch.setattr(st.time, "time", lambda: NOW)
    client = _Recorder()
    ctx = st.LiveContext(conn=conn, bundle={"best": "lr"}, index=None,
                         session=types.SimpleNamespace(puuid="me"),
                         client=client, deadline=30,
                         settings=types.SimpleNamespace(region="na",
                                                        platform="pc"))
    st.poll_once(ctx)
    assert client.calls[-1] == ("behind", 0), client.calls
    assert ("current", 0) not in client.calls
    assert ("rival", 0) not in client.calls


def test_a_known_teammate_being_refreshed_is_not_being_looked_up(tmp_path):
    """Their card is full. The team bar used to say "looking up" for them."""
    conn, match = _lobby(tmp_path)
    out = R.resolve(conn, match, "me", NOW, client=_Recorder(),
                    deadline_seconds=0, teammates_first=True,
                    refresh_last=["behind"])
    assert ("behind", 0) in out.remaining
    assert out.pending == {"stranger"}


def test_a_refused_refresh_is_not_an_answer(tmp_path):
    from valwr.collect.client import HenrikError

    class Refuses(_Recorder):
        def matches(self, region, platform, puuid, size, mode, start=0):
            if puuid == "behind":
                raise HenrikError("404")
            return super().matches(region, platform, puuid, size, mode, start)
    conn, match = _lobby(tmp_path)
    out = R.resolve(conn, match, "me", NOW, client=Refuses(),
                    deadline_seconds=30, refresh_last=["behind"])
    assert ("behind", 0) in out.completed, "not asked again"
    assert ("behind", 0) in out.failed, "but it told us nothing"


def _poll(tmp_path, monkeypatch, client, clock, match=None):
    """poll_once on a real store, with scoring stubbed and the clock set."""
    import types

    from valwr.live import predict as P
    from valwr.live import state as st
    if match is None:
        conn, match = _lobby(tmp_path, "coregame")
    else:
        conn = match[0]
        match = match[1]
    monkeypatch.setattr(st, "current_match", lambda ctx: match)
    monkeypatch.setattr(st, "display_map", lambda conn, m: m)
    monkeypatch.setattr(st, "parties", lambda *a, **k: [])
    monkeypatch.setattr(P, "predict", lambda *a, **k: None)
    monkeypatch.setattr(st.time, "time", lambda: clock[0])
    ctx = st.LiveContext(conn=conn, bundle={"best": "lr"}, index=None,
                         session=types.SimpleNamespace(puuid="me"),
                         client=client, deadline=30,
                         settings=types.SimpleNamespace(region="na",
                                                        platform="pc"))
    return ctx, conn, match


def test_a_record_complete_when_the_match_began_stays_complete_during_it(
        tmp_path, monkeypatch):
    """Judged from each poll's clock, a teammate whose last game began 50
    minutes before the match looked a game behind six minutes in: the badge
    vanished and a lookup was spent mid-match on a game nobody could have
    played -- they were in this one."""
    from valwr.live import state as st
    # Twenty stored, so nothing about their depth asks for a fetch either.
    conn = store(tmp_path, [(50 + 45 * k, 0 if k < 3 else 1) for k in range(20)],
                 puuid="ally")
    match = LiveMatch("live", "coregame", "Ascent", "BombGameMode", [
        LivePlayer("me", "Blue", "x", None), LivePlayer("ally", "Blue", "x", None)])
    clock = [NOW]
    client = _Recorder()
    ctx, _, _ = _poll(tmp_path, monkeypatch, client, clock, (conn, match))
    badge = lambda s: next(p["streak"] for p in s["players"]          # noqa: E731
                           if p["puuid"] == "ally")
    assert badge(st.poll_once(ctx)) == {"result": "lost", "count": 3}
    clock[0] = NOW + 30 * MIN
    assert badge(st.poll_once(ctx)) == {"result": "lost", "count": 3}
    assert ("ally", 0) not in client.calls


def test_no_badge_from_a_record_whose_refresh_was_refused(tmp_path,
                                                          monkeypatch):
    """Their newest stored game began 80 minutes before the match: they may
    have played one since. The refresh that would say was refused, so the
    stored run of losses may be out of date, and is not shown."""
    from valwr.collect.client import HenrikError
    from valwr.live import state as st

    class Refuses(_Recorder):
        def matches(self, region, platform, puuid, size, mode, start=0):
            super().matches(region, platform, puuid, size, mode, start)
            raise HenrikError("404")
    conn = store(tmp_path, [(80 + 45 * k, 0 if k < 3 else 1) for k in range(20)],
                 puuid="ally")
    match = LiveMatch("live", "pregame", "Ascent", "BombGameMode", [
        LivePlayer("me", "Blue", "x", None), LivePlayer("ally", "Blue", "x", None)])
    monkeypatch.setattr(st.picks, "your_picks", lambda *a, **k: None)
    client = Refuses()
    ctx, _, _ = _poll(tmp_path, monkeypatch, client, [NOW], (conn, match))
    got = st.poll_once(ctx)
    assert ("ally", 0) in client.calls, "the refresh was asked for"
    assert next(p["streak"] for p in got["players"]
                if p["puuid"] == "ally") is None


def test_a_new_match_starts_the_clock_again(tmp_path, monkeypatch):
    from valwr.live import state as st
    clock = [NOW]
    ctx, conn, match = _poll(tmp_path, monkeypatch, _Recorder(), clock)
    st.poll_once(ctx)
    assert ctx.work_since == NOW
    clock[0] = NOW + 45 * MIN
    st.poll_once(ctx)
    assert ctx.work_since == NOW, "the same match keeps its start"
    import dataclasses
    later = dataclasses.replace(match, match_id="next")
    monkeypatch.setattr(st, "current_match", lambda ctx: later)
    st.poll_once(ctx)
    assert ctx.work_since == NOW + 45 * MIN and ctx.work_failed == set()


def test_your_picks_are_built_once_a_match(tmp_path, monkeypatch):
    """They read your whole history; agent select repaints every second or
    two. A refresh of your own games is the one thing that changes them."""
    from valwr.live import state as st
    conn, match = _lobby(tmp_path)
    clock = [NOW]
    ctx, _, _ = _poll(tmp_path, monkeypatch, _Recorder(), clock, (conn, match))
    built = []
    monkeypatch.setattr(st.picks, "your_picks",
                        lambda *a, **k: built.append(1) or {"agents": []})
    for _ in range(3):
        st.poll_once(ctx)
    assert len(built) == 1
    conn.execute("INSERT INTO match_players (match_id, puuid, team, agent, "
                 "started_at, map, won, rounds_played) VALUES "
                 "('me0', 'me', 'Blue', 'Jett', ?, 'Ascent', 1, 20) "
                 "ON CONFLICT DO UPDATE SET started_at = excluded.started_at",
                 (NOW - 5 * MIN,))
    st.poll_once(ctx)
    assert len(built) == 2, "your own newest game moved"


def test_an_agent_newer_than_the_model_still_has_a_role(tmp_path):
    """Locked on an agent the model's role map predates, a teammate filled no
    role at all, and the strip called theirs open."""
    import types

    from valwr.live import state as st
    conn = store(tmp_path, [(30, 1)], puuid="ally")
    conn.execute("INSERT INTO ref_agents (uuid, name, role) VALUES "
                 "('u-new', 'Newcomer', 'Sentinel')")
    conn.commit()
    match = LiveMatch("live", "pregame", "Ascent", "BombGameMode", [
        LivePlayer("me", "Blue", "x", None),
        LivePlayer("ally", "Blue", "u-new", "Newcomer", "locked")])
    ctx = types.SimpleNamespace(conn=conn, bundle={"roles": {"Jett": "Duelist"}},
                                index=None, role_index=None,
                                session=types.SimpleNamespace(puuid="me"))
    rows = {r["puuid"]: r for r in st._player_rows(ctx, match, NOW)}
    assert rows["ally"]["role"] == "Sentinel"
    assert st.display_roles(conn, {"Newcomer": "Duelist"})["Newcomer"] == \
        "Duelist", "the model's own map wins where it has the agent"


# --- on the page's state ---------------------------------------------------

def test_only_your_own_team_carries_a_streak(tmp_path, monkeypatch):
    import types

    from valwr.live import state as st
    conn = store(tmp_path, [(30, 0), (75, 0), (120, 0)], puuid="ally")
    for i, ago in enumerate((30, 75, 120)):
        conn.execute(
            "INSERT INTO match_players (match_id, puuid, team, agent, "
            "started_at, map, won, rounds_played) VALUES "
            "(?, 'enemy', 'Red', 'Jett', ?, 'Ascent', 0, 20)",
            (f"m{i}", NOW - ago * MIN))
    conn.commit()
    match = LiveMatch("live", "coregame", "Ascent", "BombGameMode", [
        LivePlayer("me", "Blue", "x", None), LivePlayer("ally", "Blue", "x", None),
        LivePlayer("enemy", "Red", "x", None)])
    ctx = types.SimpleNamespace(conn=conn, bundle={}, index=None,
                                role_index=None,
                                session=types.SimpleNamespace(puuid="me"))
    rows = {r["puuid"]: r for r in st._player_rows(ctx, match, NOW)}
    assert rows["ally"]["streak"] == {"result": "lost", "count": 3}
    assert rows["enemy"]["streak"] is None, "the badge is for your team"
    assert rows["me"]["streak"] is None


def test_a_teammate_a_game_behind_is_badged_once_their_page_has_answered(
        tmp_path):
    import types

    from valwr.live import state as st
    conn = store(tmp_path, [(80, 0), (125, 0), (170, 0)], puuid="ally")
    match = LiveMatch("live", "pregame", "Ascent", "BombGameMode", [
        LivePlayer("me", "Blue", "x", None), LivePlayer("ally", "Blue", "x", None)])
    ctx = types.SimpleNamespace(conn=conn, bundle={}, index=None,
                                role_index=None,
                                session=types.SimpleNamespace(puuid="me"))

    def badge(fetched):
        rows = st._player_rows(ctx, match, NOW, fetched=fetched)
        return next(r["streak"] for r in rows if r["puuid"] == "ally")
    assert badge(set()) is None, "shown before the refresh could be stale"
    assert badge({("ally", 0)}) == {"result": "lost", "count": 3}
    assert badge({("ally", 10)}) is None, "only the first page is their latest"
