"""The game's own Performance Score, read from the client's match-details."""

from __future__ import annotations

import pytest

from valwr.live import client_scores as C
from valwr.store import schema


def details(*values, field=C.PS_FIELD):
    """match-details in the shape the client returns, one player per value."""
    return {"players": [
        {"subject": f"p{i}",
         "scores": ({field: v} if v is not None else {}) | {
             "TempValueL": {"TempValueP": {"damage": "up", "trades": "neutral"},
                            "TempValueQ": {"assists": "double_up"}}}}
        for i, v in enumerate(values)]}


def test_scores_keep_their_decimals_and_show_as_the_game_rounds_them():
    """215.102 is what the client holds for a screen that said 215."""
    rows = C.parse(details(215.10227560133734, 500, 0.0))
    assert [ps for _, ps, _ in rows] == [pytest.approx(215.102, abs=1e-3), 500.0, 0.0]
    assert C.shown(215.10227560133734) == 215
    assert C.shown(214.5) == 215 and C.shown(None) is None


def test_the_breakdown_is_the_games_own_arrows():
    _, _, parts = C.parse(details(215.0))[0]
    assert parts == {"damage": "up", "trades": "neutral", "assists": "double_up"}


@pytest.mark.parametrize("bad", [612.0, -3.0, "215", True])
def test_a_value_that_cannot_be_a_performance_score_is_not_stored(bad):
    """The field name is scrambled and a patch could move it. Storing the wrong
    column under the right name would be worse than storing nothing."""
    with pytest.raises(C.FormatChanged):
        C.parse(details(215.0, bad))


def test_a_player_with_no_score_is_left_out_and_the_rest_kept():
    """Someone who never connected has no Performance Score. Treating the
    whole match as unreadable for it asked about that match every minute
    forever and kept the other nine scores out."""
    rows = C.parse(details(215.0, None, 300.0))
    assert [(p, ps) for p, ps, _ in rows] == [("p0", 215.0), ("p2", 300.0)]


def test_a_mode_the_game_does_not_score_is_not_a_changed_format():
    """Team Deathmatch has no Performance Score for anyone. Reporting that as
    a format change was a false alarm."""
    with pytest.raises(C.NotScored):
        C.parse(details(None, None, None))


@pytest.fixture
def conn(tmp_path):
    c = schema.connect(tmp_path / "t.db")
    schema.create_all(c)
    for mid, made, standard in (("new", 300, 1), ("old", 100, 1), ("tdm", 200, 0)):
        c.execute("INSERT INTO live_predictions (match_id, made_at, phase, "
                  "standard_mode, state_json) VALUES (?, ?, 'coregame', ?, '{}')",
                  (mid, made, standard))
    c.commit()
    C._NO_ANSWER.clear()
    C._UNREADABLE.clear()
    yield c
    C._NO_ANSWER.clear()
    C._UNREADABLE.clear()


def test_only_bomb_mode_matches_without_scores_are_asked_about(conn):
    assert C.missing(conn) == ["new", "old"]
    C.store(conn, "new", [("p0", 215.1, {})], now=1)
    assert C.missing(conn) == ["old"]
    assert C.scores_for(conn, "new") == {"p0": pytest.approx(215.1)}


def test_collecting_stores_what_the_client_holds(conn, monkeypatch):
    monkeypatch.setattr(C, "fetch", lambda s, mid: details(215.1, 330.0))
    got = C.collect(conn, object(), limit=None)
    assert got["stored"] == 2
    assert C.scores_for(conn, "old") == {"p0": pytest.approx(215.1), "p1": 330.0}


def test_a_match_with_no_answer_is_not_asked_about_every_minute(conn, monkeypatch):
    asked = []

    def nothing(session, mid):
        asked.append(mid)
        return None
    monkeypatch.setattr(C, "fetch", nothing)
    C.collect(conn, object(), limit=None)
    C.collect(conn, object(), limit=None)
    assert sorted(asked) == ["new", "old"], f"asked again: {asked}"


def test_a_changed_format_is_asked_about_once_a_run(conn, monkeypatch):
    """Not every minute -- five unreadable matches, newest first, used to
    crowd out every older one -- but again after a restart, since an update
    to this code is what would fix it."""
    asked = []

    def unreadable(session, mid):
        asked.append(mid)
        return details(999.0)
    monkeypatch.setattr(C, "fetch", unreadable)
    assert C.collect(conn, object(), limit=None)["format_changed"] == 2
    assert C.collect(conn, object(), limit=None)["format_changed"] == 0
    assert C.wanted(conn) == []
    C._UNREADABLE.clear()                        # a restart
    assert C.collect(conn, object(), limit=None)["format_changed"] == 2
    assert len(asked) == 4


def test_a_match_still_being_played_is_not_asked_about(conn, monkeypatch):
    """Mid-game the server has nothing yet, and "nothing" used to be
    remembered for the whole run -- so the match just played never got its
    score. It waits until it is settled, or long enough over."""
    import time as _time
    now = int(_time.time())
    conn.execute("INSERT INTO live_predictions (match_id, made_at, phase, "
                 "standard_mode, state_json) VALUES ('live', ?, 'coregame', 1, "
                 "'{}')", (now - 600,))
    conn.commit()
    assert "live" not in C.missing(conn, now=now)
    conn.execute("UPDATE live_predictions SET settled_at = ? WHERE match_id = "
                 "'live'", (now,))
    assert C.missing(conn, now=now)[0] == "live"
    conn.execute("UPDATE live_predictions SET settled_at = NULL WHERE "
                 "match_id = 'live'")
    assert "live" in C.missing(conn, now=now + C.FINISHED_AFTER_SECONDS)


def test_a_server_error_ends_the_pass_and_keeps_what_was_stored(conn,
                                                                monkeypatch):
    import httpx

    from valwr.live import outcomes
    answers = iter([details(215.0, 330.0), httpx.HTTPStatusError(
        "503", request=httpx.Request("GET", "https://pd"), response=httpx.Response(503))])

    def fetch(session, mid):
        got = next(answers)
        if isinstance(got, Exception):
            raise got
        return got
    rescored = []
    monkeypatch.setattr(C, "fetch", fetch)
    monkeypatch.setattr(outcomes, "rescore", lambda c: rescored.append(1))
    got = C.collect(conn, object(), limit=None)
    assert got["stored"] == 1 and got["server_error"] == 1
    assert rescored == [1], "the stored match must still be re-judged"


def test_what_was_stored_is_rejudged_even_if_the_session_expires(conn,
                                                                 monkeypatch):
    from valwr.live import outcomes
    from valwr.live import session as S
    answers = iter([details(215.0, 330.0), S.SessionExpired("aged out")])

    def fetch(session, mid):
        got = next(answers)
        if isinstance(got, Exception):
            raise got
        return got
    rescored = []
    monkeypatch.setattr(C, "fetch", fetch)
    monkeypatch.setattr(outcomes, "rescore", lambda c: rescored.append(1))
    with pytest.raises(S.SessionExpired):
        C.collect(conn, object(), limit=None)
    assert rescored == [1]
