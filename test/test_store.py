"""Phase 2 tests: normalisation, integrity, and the leakage firewall."""

from __future__ import annotations

import pytest

from valwr.store import normalize, schema, temporal


@pytest.fixture
def conn(tmp_path):
    c = schema.connect(tmp_path / "t.db")
    schema.create_all(c)
    return c


def make_match(match_id="m1", started="2026-08-23T05:39:55.948Z", n=10,
               winner="Blue", map_name="Sunset", agent="Jett"):
    return {
        "metadata": {"match_id": match_id, "started_at": started,
                     "map": {"name": map_name}, "region": "na",
                     "queue": {"id": "competitive", "mode_type": "Standard"},
                     "season": {"short": "e11a5"}, "is_completed": True},
        "teams": [{"team_id": "Red", "won": winner == "Red", "rounds": {"won": 12}},
                  {"team_id": "Blue", "won": winner == "Blue", "rounds": {"won": 14}}],
        "players": [{"puuid": f"p{i}", "name": f"n{i}", "tag": "NA1",
                     "team_id": "Blue" if i < 5 else "Red",
                     "agent": {"name": agent}, "tier": {"id": 13},
                     "party_id": None, "account_level": 100,
                     "stats": {"score": 200, "kills": 15, "deaths": 14, "assists": 5,
                               "headshots": 10, "bodyshots": 20, "legshots": 2,
                               "damage": {"dealt": 3000, "received": 3100}}}
                    for i in range(n)],
    }


# --- timestamp parsing ------------------------------------------------

def test_iso_timestamp_is_parsed_to_epoch():
    """started_at is an ISO string but the column is INTEGER.

    Silently mis-parsing this corrupts the leakage firewall without raising,
    which is the worst kind of bug, so it must fail loudly instead.
    """
    assert normalize.parse_started_at("2026-08-23T05:39:55.948Z") == 1787463595


def test_epoch_milliseconds_are_normalised():
    assert normalize.parse_started_at(1787463595948) == 1787463595


@pytest.mark.parametrize("bad", [None, "", "not-a-date", {}])
def test_unparseable_timestamps_raise(bad):
    with pytest.raises(normalize.ParseError):
        normalize.parse_started_at(bad)


# --- parsing ----------------------------------------------------------

def test_parse_extracts_match_and_players():
    row, players, flags = normalize.parse_match(make_match())
    assert row["match_id"] == "m1"
    assert row["map"] == "Sunset"
    assert row["mode"] == "competitive"
    assert row["winner"] == "Blue"
    assert len(players) == 10
    assert flags == []


def test_winner_determines_per_player_won():
    _, players, _ = normalize.parse_match(make_match(winner="Blue"))
    blue = [p for p in players if p["team"] == "Blue"]
    red = [p for p in players if p["team"] == "Red"]
    assert all(p["won"] == 1 for p in blue)
    assert all(p["won"] == 0 for p in red)


def test_short_roster_is_flagged_not_dropped():
    row, players, flags = normalize.parse_match(make_match(n=8))
    assert "roster_8" in flags
    assert row["data_quality"] and "roster_8" in row["data_quality"]
    assert len(players) == 8, "flagged, but still parsed"


def test_draw_is_flagged_and_leaves_won_null():
    row, players, flags = normalize.parse_match(make_match(winner=None))
    assert "no_single_winner" in flags
    assert all(p["won"] is None for p in players)


# --- integrity --------------------------------------------------------

def _ingest(conn, *matches):
    for m in matches:
        row, players, _ = normalize.parse_match(m)
        normalize.upsert_match(conn, row)
        normalize.upsert_players(conn, players)
    conn.commit()


def test_reparsing_does_not_duplicate(conn):
    """The parser is re-run whenever a bug is found, so it must be idempotent."""
    _ingest(conn, make_match())
    first = schema.table_counts(conn)
    _ingest(conn, make_match())
    assert schema.table_counts(conn) == first


def test_no_orphan_player_rows(conn):
    _ingest(conn, make_match("m1"), make_match("m2"))
    orphans = conn.execute(
        "SELECT COUNT(*) n FROM match_players mp "
        "LEFT JOIN matches m ON m.match_id = mp.match_id WHERE m.match_id IS NULL"
    ).fetchone()["n"]
    assert orphans == 0


def test_every_match_has_ten_players_or_is_flagged(conn):
    _ingest(conn, make_match("full"), make_match("short", n=7))
    for r in conn.execute(
        "SELECT m.match_id, m.data_quality, COUNT(mp.puuid) n FROM matches m "
        "JOIN match_players mp ON mp.match_id = m.match_id GROUP BY m.match_id"
    ):
        assert r["n"] == 10 or r["data_quality"], f"{r['match_id']} short but unflagged"


# --- the leakage firewall (the point of this phase) -------------------

def test_history_excludes_the_match_being_predicted(conn):
    """Strict `<`. A match must never inform a prediction about itself."""
    _ingest(conn, make_match("m1", started="2026-08-01T00:00:00Z"))
    at = normalize.parse_started_at("2026-08-01T00:00:00Z")
    assert temporal.player_history(conn, "p0", at) == []
    assert len(temporal.player_history(conn, "p0", at + 1)) == 1


def test_history_never_returns_rows_at_or_after_as_of(conn):
    """Property test across many players and cut points, not one example."""
    stamps = [f"2026-08-{d:02d}T00:00:00Z" for d in range(1, 21)]
    _ingest(conn, *[make_match(f"m{i}", started=t) for i, t in enumerate(stamps)])

    epochs = [normalize.parse_started_at(t) for t in stamps]
    for puuid in ("p0", "p3", "p9"):
        for as_of in epochs + [e + 1 for e in epochs] + [0, 2**31]:
            for row in temporal.player_history(conn, puuid, as_of):
                assert row["started_at"] < as_of, (
                    f"leak: {row['started_at']} >= as_of {as_of}")


def test_record_counts_only_decided_matches(conn):
    _ingest(conn,
            make_match("w", started="2026-08-01T00:00:00Z", winner="Blue"),
            make_match("l", started="2026-08-02T00:00:00Z", winner="Red"),
            make_match("d", started="2026-08-03T00:00:00Z", winner=None))
    rec = temporal.record(conn, "p0", 2**31)   # p0 is Blue
    assert (rec.games, rec.wins) == (2, 1), "draw must not count"


def test_shrinkage_pulls_small_samples_toward_the_prior():
    """Three games at 100% is not a 100% player."""
    tiny = temporal.Record(games=3, wins=3)
    assert tiny.win_rate == 1.0
    assert tiny.shrunk(0.5, 20) < 0.62

    big = temporal.Record(games=300, wins=300)
    assert big.shrunk(0.5, 20) > 0.93


def test_map_and_agent_history_are_filtered(conn):
    _ingest(conn,
            make_match("a", started="2026-08-01T00:00:00Z", map_name="Ascent", agent="Jett"),
            make_match("b", started="2026-08-02T00:00:00Z", map_name="Sunset", agent="Jett"),
            make_match("c", started="2026-08-03T00:00:00Z", map_name="Ascent", agent="Omen"))
    late = 2**31
    assert len(temporal.player_history_on_map(conn, "p0", late, "Ascent")) == 2
    assert len(temporal.player_history_on_agent(conn, "p0", late, "Jett")) == 2
    assert len(temporal.player_history_map_agent(conn, "p0", late, "Ascent", "Jett")) == 1


# --- migrations -------------------------------------------------------

def test_migrate_is_idempotent(conn):
    assert schema.migrate(conn) == []
    schema.create_all(conn)
    assert schema.migrate(conn) == []
