"""Tests for live/state.py -- the one dictionary both views render.

`poll_once` is the reason the terminal and the browser cannot disagree: they
format the same dict rather than each assembling one. That makes it
load-bearing, and until now nothing tested it.

Three things here break quietly. The warnings are an if/elif chain, so a new
case can mask an existing one. The dict crosses a websocket, so a value that
is not JSON-serialisable fails in the browser and nowhere else -- no
traceback, the dashboard simply stops updating. And unscored players are meant
to stay in the list rather than vanish, which is the whole point of showing a
lobby honestly.
"""

from __future__ import annotations

import json

from valwr.live import state as ST
from valwr.live.roster import LiveMatch, LivePlayer

ME = "me-puuid"

# Exactly what poll_once promises. Equality rather than a subset is deliberate:
# the dashboard page and live/render.py both index these by name, so a rename
# that only touches one of them is a bug this test is here to catch.
DOCUMENTED_KEYS = {
    "match_id", "phase", "is_custom", "standard_mode", "map", "mode", "as_of",
    "own_team", "enemy_team", "team_sizes", "coverage", "confidence",
    "fetched", "model", "warnings", "parties", "players", "prediction",
}

CARD_KEYS = {"puuid", "name", "known_name", "agent", "role", "team", "is_you",
             "score", "reason", "flag", "rank"}


# The keys `_player_rows` lifts out of `pot.detail`. Kept in one place because
# a stub of this shape has already gone stale twice, each time failing in an
# unrelated sorting test rather than saying what was actually missing.
def _detail(score):
    return {"score": score, "reason": "r", "flag": None,
            "career": None, "recent": None}


def _rank():
    from valwr.rating import ranks
    return ranks.describe(16)


class _Stub:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _match(phase="coregame", mode="BombGameMode", flow=None,
           blue=5, red=5, include_me=True):
    players = [LivePlayer(ME if (i == 0 and include_me) else f"b{i}",
                          "Blue", "aid", "Jett") for i in range(blue)]
    players += [LivePlayer(f"r{i}", "Red", "aid", "Sova") for i in range(red)]
    return LiveMatch("m1", phase, "Ascent", mode, players, flow)


def _db(tmp_path, named=(), history=()):
    from valwr.store import schema
    conn = schema.connect(tmp_path / "s.db")
    schema.create_all(conn)
    conn.execute("INSERT INTO ref_agents (uuid, name, role) "
                 "VALUES ('aid', 'Jett', 'Duelist')")
    for puuid, name, tag in named:
        conn.execute("INSERT INTO players (puuid, name, tag) VALUES (?, ?, ?)",
                     (puuid, name, tag))
    for i, puuid in enumerate(history):
        mid = f"h{i}"
        conn.execute(
            "INSERT INTO matches (match_id, started_at, map, mode, queue, "
            "region, season, rounds_red, rounds_blue, winner, data_quality, "
            "ingested_at) VALUES (?, 1000, 'Ascent', 'competitive', "
            "'Standard', 'na', 's', 9, 13, 'Blue', NULL, 0)", (mid,))
        conn.execute(
            "INSERT INTO match_players (match_id, puuid, team, agent, "
            "party_id, tier, account_level, score, kills, deaths, assists, "
            "headshots, bodyshots, legshots, damage_dealt, damage_taken, "
            "started_at, map, won, rounds_played) VALUES (?, ?, 'Blue', "
            "'Jett', NULL, 15, 100, 4000, 15, 15, 5, 5, 5, 5, 3000, 3000, "
            "1000, 'Ascent', 1, 20)", (mid, puuid))
    conn.commit()
    return conn


def _ctx(conn, index=None, role_index=None):
    return ST.LiveContext(
        conn=conn,
        bundle={"best": "logistic regression",
                "roles": {"Jett": "Duelist", "Sova": "Initiator"},
                "norms": {}},
        index=index, role_index=role_index, session=_Stub(puuid=ME),
        client=None,
        settings=_Stub(region="na", platform="pc"), deadline=1.0)


# --- the per-role score, as the live view actually assembles it ---------

def _perf_index():
    """A small stand-in for the old index, built here rather than loaded.

    Loading models/perf_index.json would make these tests pass or fail on
    whether a model happens to be fitted on this machine.
    """
    from valwr.rating import potential as pot
    return pot.PerfIndex(
        means={"acs": 200.0, "rating": 1.0, "kd": 1.0, "map_edge": 0.0},
        stds={"acs": 40.0, "rating": 0.15, "kd": 0.25, "map_edge": 0.05},
        quantiles=[i / 50.0 - 1.0 for i in range(101)], as_of=900, n=1000)


def _scored_ctx(conn, role_index=None):
    """A context whose cards can actually be built: real norms, real index."""
    from valwr.rating.normalize import build_norms
    ctx = _ctx(conn, index=_perf_index(), role_index=role_index)
    ctx.bundle["norms"] = build_norms(conn, 2000)
    return ctx


def _role_index():
    """A reference population wide enough to z-score against."""
    from valwr.rating import roleindex
    from valwr.rating.role_score import RoleComponents
    samples = []
    for i in range(400):
        step = (i % 20) / 20.0
        for role, agent, acs in (("Duelist", "Jett", 180 + 80 * step),
                                 ("Initiator", "Sova", 170 + 70 * step)):
            samples.append(RoleComponents(
                role=role, agent=agent,
                values={"acs": acs, "adr": 120 + 50 * step, "kd": 0.8 + 0.5 * step,
                        "kda": 1.1 + 0.6 * step, "assists": 0.2 + 0.2 * step,
                        "kast": 0.6 + 0.2 * step, "fb": 0.08 + 0.06 * step,
                        "fd": 0.08 + 0.06 * step, "hs": 0.18 + 0.1 * step,
                        "abilities": 1.5 + 0.8 * step},
                counts={}, map_edge=0.0, n_games=30, n_role_games=30,
                n_map_games=0, n_ability_games=30, tier=15, account_level=100))
    return roleindex.fit(samples, as_of=900)


def test_the_live_card_is_scored_by_the_players_role(tmp_path):
    """The whole point of the change: a Jett and a Sova are not scored on the
    same things, and the card has to say which table produced the number."""
    conn = _db(tmp_path, history=[ME] * 6)
    conn.execute("INSERT INTO ref_agents (uuid, name, role) "
                 "VALUES ('sid', 'Sova', 'Initiator')")
    conn.commit()
    rows = ST._player_rows(_scored_ctx(conn, _role_index()), _match(), 2000)
    mine = next(r for r in rows if r["is_you"])
    card = mine["detail"]
    assert card["role_score"]["role"] == "Duelist"
    assert card["role_score"]["weights_from"] == "Duelist"
    keys = {c["key"] for c in card["components"]}
    # Components the old single formula never had, and one it had that a
    # Duelist no longer scores on.
    assert {"kast", "abilities", "fb", "adr"} <= keys
    assert "rating" not in keys
    assert "assists" not in keys, "Duelists are not scored on assists"


def test_a_supporting_role_is_scored_on_different_components(tmp_path):
    conn = _db(tmp_path, history=[ME] * 6)
    conn.execute("INSERT INTO ref_agents (uuid, name, role) "
                 "VALUES ('sid', 'Sova', 'Initiator')")
    conn.commit()
    match = LiveMatch("m1", "coregame", "Ascent", "BombGameMode",
                      [LivePlayer(ME, "Blue", "sid", "Sova")], None)
    rows = ST._player_rows(_scored_ctx(conn, _role_index()), match, 2000)
    keys = {c["key"] for c in rows[0]["detail"]["components"]}
    assert "kda" in keys and "kd" not in keys, "Initiators use (K+A)/D"
    assert "fb" not in keys, "only Duelists score opening kills"


def test_without_a_role_index_the_card_still_scores(tmp_path):
    """A fresh clone has not fitted one. The old formula carries the card
    rather than the score vanishing."""
    conn = _db(tmp_path, history=[ME] * 6)
    rows = ST._player_rows(_scored_ctx(conn), _match(), 2000)
    mine = next(r for r in rows if r["is_you"])
    assert mine["score"] is not None
    assert "role_score" not in (mine["detail"] or {})


# --- warnings ----------------------------------------------------------

def test_a_non_bomb_mode_says_the_probability_means_nothing():
    assert any("means nothing" in w
               for w in ST._warnings(_match(mode="Deathmatch"), ME))


def test_uneven_custom_teams_are_reported_with_their_sizes():
    w = ST._warnings(_match(flow="CustomGame", blue=4, red=6), ME)
    assert any("4v6" in x for x in w)


def test_an_even_custom_produces_no_size_warning():
    assert not [x for x in ST._warnings(_match(flow="CustomGame"), ME)
                if "Uneven" in x]


def test_a_non_standard_mode_masks_the_size_warning_on_purpose():
    """The chain is if/elif. When the mode is not bomb defusal the prediction
    is meaningless whatever the team sizes are, so a second line about them
    would be noise. Asserted so that reordering the chain has to be a choice.
    """
    w = ST._warnings(
        _match(mode="Deathmatch", flow="CustomGame", blue=4, red=6), ME)
    assert not [x for x in w if "Uneven" in x]
    assert any("means nothing" in x for x in w)


def test_a_spectator_is_told_whose_side_the_numbers_are_from():
    assert any("not on either team" in w
               for w in ST._warnings(_match(include_me=False), ME))


def test_pregame_says_the_enemy_team_is_still_hidden():
    assert any("Enemy team is hidden" in w
               for w in ST._warnings(_match(phase="pregame"), ME))


def test_an_ordinary_competitive_match_warns_about_nothing():
    assert ST._warnings(_match(), ME) == []


# --- player rows -------------------------------------------------------

def test_without_an_index_every_player_is_honestly_unscored(tmp_path):
    rows = ST._player_rows(_ctx(_db(tmp_path)), _match(), 1000)
    assert len(rows) == 10
    assert all(r["score"] is None and r["reason"] == "no history" for r in rows)


def test_a_known_gamertag_is_used_and_a_stranger_keeps_a_short_puuid(tmp_path):
    conn = _db(tmp_path, named=[(ME, "tester", "0000")])
    rows = {r["puuid"]: r for r in ST._player_rows(_ctx(conn), _match(), 1000)}
    assert rows[ME]["name"] == "tester#0000" and rows[ME]["known_name"]
    assert rows["r0"]["name"] == "r0" and not rows["r0"]["known_name"]


def test_exactly_one_row_is_marked_as_you(tmp_path):
    rows = ST._player_rows(_ctx(_db(tmp_path)), _match(), 1000)
    assert [r["is_you"] for r in rows].count(True) == 1


def test_scored_players_sort_first_and_unscored_are_kept(tmp_path,
                                                         monkeypatch):
    """Unscored players stay in the list. They are in the lobby whether or not
    anything is known about them, and dropping them would quietly understate
    how much of the prediction is guesswork."""
    scores = {"b1": 80, "b2": 40}
    monkeypatch.setattr(ST.pot, "detail", lambda c, pu, ao, mp, nm, ix: (
        _detail(scores[pu]) if pu in scores else None))
    rows = ST._player_rows(_ctx(_db(tmp_path), index=object()), _match(), 1000)
    assert len(rows) == 10
    assert [r["puuid"] for r in rows[:2]] == ["b1", "b2"]
    assert all(r["score"] is None for r in rows[2:])


# --- the poll ----------------------------------------------------------

def test_not_being_in_a_match_is_none_not_an_empty_state(tmp_path,
                                                         monkeypatch):
    monkeypatch.setattr(ST.roster, "current", lambda s, a: None)
    assert ST.poll_once(_ctx(_db(tmp_path))) is None


def test_the_poll_returns_the_keys_both_renderers_read(tmp_path, monkeypatch):
    conn = _db(tmp_path, named=[(ME, "tester", "0000")], history=[ME])
    monkeypatch.setattr(ST.roster, "current", lambda s, a: _match())
    monkeypatch.setattr(ST.P, "predict", lambda *a, **k: None)
    st = ST.poll_once(_ctx(conn))
    assert set(st) == DOCUMENTED_KEYS
    assert st["prediction"] is None
    assert (st["own_team"], st["enemy_team"]) == ("Blue", "Red")
    assert st["team_sizes"] == {"Blue": 5, "Red": 5}
    assert st["model"] == "logistic regression"
    assert all(CARD_KEYS <= set(r) for r in st["players"])


def test_the_poll_is_json_serialisable(tmp_path, monkeypatch):
    """dash/server.py sends this over a websocket. A set, a numpy float or a
    dataclass raises inside the send and nowhere the terminal would show it --
    the browser simply stops receiving updates.
    """
    conn = _db(tmp_path, named=[(ME, "tester", "0000")], history=[ME])
    monkeypatch.setattr(ST.roster, "current", lambda s, a: _match())
    monkeypatch.setattr(ST.P, "predict", lambda *a, **k: None)
    json.dumps(ST.poll_once(_ctx(conn)))


def test_a_prediction_is_rounded_and_still_serialisable(tmp_path, monkeypatch):
    from valwr.live.predict import Prediction
    pred = Prediction(win_probability=0.5123456, own_team="Blue",
                      own_probability=0.4876544, coverage=3, confidence="low",
                      model="logistic regression",
                      factors=[("rank_diff", 0.123456789)])
    monkeypatch.setattr(ST.roster, "current", lambda s, a: _match())
    monkeypatch.setattr(ST.P, "predict", lambda *a, **k: pred)
    st = ST.poll_once(_ctx(_db(tmp_path)))
    assert st["prediction"]["own_probability"] == 0.4877
    assert st["prediction"]["factors"] == [{"name": "rank_diff",
                                            "value": 0.1235}]
    json.dumps(st)


def test_a_spectator_gets_blues_perspective_rather_than_a_crash(tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr(ST.roster, "current",
                        lambda s, a: _match(include_me=False))
    monkeypatch.setattr(ST.P, "predict", lambda *a, **k: None)
    st = ST.poll_once(_ctx(_db(tmp_path)))
    assert st["own_team"] == "Blue"
    assert any("not on either team" in w for w in st["warnings"])


# --- the demo match ----------------------------------------------------

def test_the_demo_match_has_exactly_the_shape_poll_once_returns():
    """`--demo` renders through the real page, so it must not drift.

    If it were allowed to carry its own approximate shape, the demo would keep
    working while the live view broke -- which is the worst possible failure
    for the one screen used to check the live view by eye.
    """
    from valwr.dash.demo import demo_state
    s = demo_state()
    assert set(s) == DOCUMENTED_KEYS
    assert len(s["players"]) == 10
    assert {p["team"] for p in s["players"]} == {"Blue", "Red"}
    for p in s["players"]:
        assert CARD_KEYS <= set(p)
        assert "career" in p and "agent_id" in p


def test_the_demo_match_invents_everyone_in_it():
    """It must never ship a real gamertag or PUUID -- docs/ETHICS-AND-TOS.md
    forbids publishing either, and this file is committed."""
    from valwr.dash import demo
    s = demo.demo_state()
    assert all(p["puuid"].startswith("demo-") for p in s["players"])
    # A real PUUID is a 32-4-4-4-12 hex UUID; nothing here may look like one.
    import re
    blob = repr(s)
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                         r"[0-9a-f]{4}-[0-9a-f]{12}", blob), "looks like a PUUID"


def test_the_demo_match_is_json_serialisable():
    from valwr.dash.demo import demo_state
    json.dumps(demo_state())


def test_a_live_row_carries_the_agent_uuid_for_its_artwork(tmp_path,
                                                           monkeypatch):
    """The page builds /agents/<uuid>-icon.png straight from this field."""
    conn = _db(tmp_path)
    monkeypatch.setattr(ST.roster, "current", lambda s, a: _match())
    monkeypatch.setattr(ST.P, "predict", lambda *a, **k: None)
    st = ST.poll_once(_ctx(conn))
    assert all("agent_id" in p for p in st["players"])
    assert st["players"][0]["agent_id"] == "aid"


def test_the_stub_detail_carries_everything_a_row_lifts_from_it():
    """Guards the helper above against the real `detail()` growing past it.

    When `_player_rows` starts reading a new block, this fails and names it,
    instead of a sorting test failing with a bare KeyError.
    """
    import inspect
    src = inspect.getsource(ST._player_rows)
    stub = set(_detail(50))
    for key in ("career", "recent", "score", "reason", "flag"):
        assert f'got["{key}"]' not in src or key in stub, (
            f"_player_rows reads got[{key!r}]; add it to _detail()")


def test_a_replayed_match_has_the_shape_the_page_renders(tmp_path):
    """`--match` rebuilds a finished game as the dashboard would have shown
    it, so it has to carry everything poll_once does -- plus the one thing a
    live poll cannot have, which is what actually happened.
    """
    from valwr.dash import replay
    conn = _db(tmp_path, named=[(ME, "tester", "0000")], history=[ME])
    # One stored match, replayed. Its own row must not inform its prediction.
    row = conn.execute("SELECT match_id FROM matches").fetchone()
    st = replay.replay_state(
        conn, row["match_id"],
        {"best": "logistic regression", "roles": {}, "norms": {},
         "columns": [], "model": None},
        None, ME)
    assert DOCUMENTED_KEYS <= set(st)
    assert set(st) - DOCUMENTED_KEYS == {"outcome"}
    assert st["as_of"] == 1000, "as_of must be the match's own start time"
    assert st["outcome"]["your_agent"] == "Jett"
    json.dumps(st)


def test_replaying_a_match_that_does_not_exist_says_so(tmp_path):
    import pytest as _pytest
    from valwr.dash import replay
    conn = _db(tmp_path)
    with _pytest.raises(replay.NoSuchMatch):
        replay.replay_state(conn, "nope", {"best": "x"}, None, ME)


# --- rank and parties --------------------------------------------------

def test_a_players_rank_comes_from_their_most_recent_match(tmp_path):
    """Rank is not something the loading screen hands us; what the stored data
    has is the tier a player was in each game. The newest is the answer, and
    it can lag a climb -- which is why the card prints it beside freshness."""
    from valwr.rating import ranks
    from valwr.store import temporal
    conn = _db(tmp_path, history=[ME])
    conn.execute("UPDATE match_players SET tier = 16 WHERE puuid = ?", (ME,))
    conn.commit()
    assert temporal.current_tier(conn, ME, 2000) == 16
    assert ranks.name(16) == "Platinum 2"
    assert ranks.short(16) == "P2"
    assert ranks.describe(16)["division"] == "Platinum"
    assert ranks.describe(0)["ranked"] is False


def test_a_trio_is_found_from_its_pairs(tmp_path):
    """Grouping is union-find over pairwise evidence, so three players who
    have each queued with each other collapse into one trio rather than three
    duos -- and A-with-B plus B-with-C is enough even without A-with-C."""
    import valwr.live.state as S
    conn = _db(tmp_path)
    match = _match()
    calls = []

    def fake(c, a, b, as_of):
        calls.append((a, b))
        return 1 if {a, b} in ({"b1", "b2"}, {"b2", "b3"}) else 0

    import valwr.store.temporal as T
    old = T.times_partied
    T.times_partied = fake
    try:
        groups = S.parties(conn, match, 1000)
    finally:
        T.times_partied = old

    blue = [g for g in groups if g["team"] == "Blue"]
    assert len(blue) == 1, f"expected one group, got {blue}"
    assert set(blue[0]["members"]) == {"b1", "b2", "b3"}
    assert blue[0]["label"] == "trio" and blue[0]["size"] == 3
    assert blue[0]["source"] == "inferred"


def test_solo_players_produce_no_group(tmp_path):
    """A missing marker means "not established", never "definitely solo" --
    the inference catches about half of real parties."""
    import valwr.live.state as S
    import valwr.store.temporal as T
    conn = _db(tmp_path)
    old = T.times_partied
    T.times_partied = lambda *a: 0
    try:
        assert S.parties(conn, _match(), 1000) == []
    finally:
        T.times_partied = old


def test_exact_parties_beat_inference_when_the_match_is_stored(tmp_path):
    """A finished match records party ids outright, so a replay never guesses.
    Passing them in must also stop `times_partied` being consulted at all."""
    import valwr.live.state as S
    import valwr.store.temporal as T
    conn = _db(tmp_path)
    called = []
    old = T.times_partied
    T.times_partied = lambda *a: called.append(a) or 1
    try:
        groups = S.parties(conn, _match(), 1000,
                           exact={"b1": "P", "b2": "P", "r0": "Q", "r1": "Q"})
    finally:
        T.times_partied = old
    assert not called, "exact parties must not fall back to inference"
    assert {g["source"] for g in groups} == {"exact"}
    assert sorted(len(g["members"]) for g in groups) == [2, 2]


def test_the_card_explains_the_map_gate_the_score_actually_used(tmp_path):
    """Two scores, two thresholds, one card.

    `potential.detail` writes the map block using its own six-game gate while
    the per-role score gates at three. Left alone, the card could say "not
    counted -- 4 games on this map, 6 needed" underneath a number that had
    counted those four games.
    """
    from valwr.rating import role_score as RS
    conn = _db(tmp_path, history=[ME] * 6)
    rows = ST._player_rows(_scored_ctx(conn, _role_index()), _match(), 2000)
    mine = next(r for r in rows if r["is_you"])
    block = mine["detail"]["map"]
    assert block["gate"] == RS.MIN_MAP_GAMES
    # Six games on Ascent, and the fixture plays Ascent every time.
    assert block["counts_toward_score"] is True


def test_the_scoreboard_is_ordered_by_standing_not_by_the_number_shown(
        tmp_path, monkeypatch):
    """The list order and the 0-100 beside it answer different questions.

    The percentile is within a role; the order has to work across the lobby, or
    a Sentinel who is good for a Sentinel outranks a Duelist who is more likely
    to top the scoreboard. The two are forced to disagree here, because a
    fixture where they happen to agree proves nothing.
    """
    from valwr.rating import role_score as RS
    conn = _db(tmp_path, history=[ME, "b1"])
    conn.execute("INSERT INTO ref_agents (uuid, name, role) "
                 "VALUES ('sid', 'Sova', 'Initiator')")
    conn.commit()

    forced = {ME: {"score": 60, "raw": 0.90}, "b1": {"score": 90, "raw": 0.10}}
    real = RS.describe

    def fake(conn_, puuid, *a, **kw):
        got = real(conn_, puuid, *a, **kw)
        if got is not None and puuid in forced:
            got.update(forced[puuid])
        return got
    monkeypatch.setattr(ST.rs, "describe", fake)

    match = LiveMatch("m1", "coregame", "Ascent", "BombGameMode",
                      [LivePlayer(ME, "Blue", "aid", "Jett"),
                       LivePlayer("b1", "Blue", "sid", "Sova")], None)
    rows = ST._player_rows(_scored_ctx(conn, _role_index()), match, 2000)

    assert [r["puuid"] for r in rows] == [ME, "b1"], "ordered by the percentile"
    assert rows[0]["score"] < rows[1]["score"], "and the display disagrees"
    assert [RS.standing(r) for r in rows] == [0.90, 0.10]
