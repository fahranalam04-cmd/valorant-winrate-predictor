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


def test_a_field_missing_for_some_players_is_a_changed_format():
    with pytest.raises(C.FormatChanged):
        C.parse(details(215.0, None, 300.0))


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
    yield c
    C._NO_ANSWER.clear()


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


def test_a_changed_format_is_asked_about_again(conn, monkeypatch):
    """Unlike "not held", a format change may be fixed by an update to this
    code, so it is not written off."""
    monkeypatch.setattr(C, "fetch", lambda s, mid: details(999.0))
    assert C.collect(conn, object(), limit=None)["format_changed"] == 2
    assert C.collect(conn, object(), limit=None)["format_changed"] == 2
