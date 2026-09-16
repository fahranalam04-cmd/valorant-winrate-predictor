"""Per-role components, measured from history.

The measurement half of the per-role score. What matters here is not that the
averages are arithmetically right -- it is that the three things that make them
more than averages actually hold: weighted sums rather than means of ratios,
a separate denominator for ability casts, and role history blended in by how
much of it there is.
"""

from __future__ import annotations

import pytest

from valwr.rating import role_score as R
from valwr.store import schema, temporal

DAY = 86400
AS_OF = 100 * DAY

AGENTS = {"Jett": "Duelist", "Reyna": "Duelist", "Omen": "Controller",
          "Sage": "Sentinel"}


@pytest.fixture
def conn(tmp_path):
    c = schema.connect(tmp_path / "t.db")
    schema.create_all(c)
    for i, (name, role) in enumerate(AGENTS.items()):
        c.execute("INSERT INTO ref_agents (uuid, name, role) VALUES (?,?,?)",
                  (f"u{i}", name, role))
    c.commit()
    return c


def play(conn, mid, *, started, agent="Jett", map_name="Sunset", rounds=20,
         score=4000, kills=15, deaths=15, assists=5, kast=14, damage=2800,
         fb=2, fd=2, headshots=10, bodyshots=20, legshots=2, casts=None,
         puuid="p1"):
    """One stored match for one player. `casts` of None means not recorded."""
    conn.execute(
        "INSERT OR REPLACE INTO matches (match_id, started_at, map, mode, region,"
        " rounds_red, rounds_blue, winner, ingested_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (mid, started, map_name, temporal.COMPETITIVE, "na", 13, 11, "Blue", 0))
    g, a1, a2, ult = casts if casts else (None, None, None, None)
    conn.execute(
        "INSERT OR REPLACE INTO match_players (match_id, puuid, team, agent,"
        " tier, account_level, score, kills, deaths, assists, headshots,"
        " bodyshots, legshots, damage_dealt, damage_taken, started_at, map, won,"
        " rounds_played, first_bloods, first_deaths, multikills, trade_kills,"
        " traded_deaths, kast_rounds, clutches, ability_grenade, ability_1,"
        " ability_2, ability_ultimate) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mid, puuid, "Blue", agent, 13, 100, score, kills, deaths, assists,
         headshots, bodyshots, legshots, damage, 2900, started, map_name, 1,
         rounds, fb, fd, 1, 2, 2, kast, 0, g, a1, a2, ult))
    conn.commit()


def measure(conn, role="Duelist", map_name="Sunset", role_means=None, agent=None):
    return R.measure(conn, "p1", AS_OF, map_name, role, AGENTS,
                     role_means=role_means, agent=agent)


# --- the three things that make this more than an average ---------------

def test_ratios_come_from_summed_counts_not_averaged_ratios(conn):
    """One match with a single death must not carry a career K/D of 14.

    Averaging per-match ratios is the obvious implementation and it puts
    anyone who once went 14/1 at the top of the lobby forever. Summing kills
    over summed deaths cannot do that.
    """
    play(conn, "m1", started=AS_OF - DAY, kills=14, deaths=1)
    play(conn, "m2", started=AS_OF - 2 * DAY, kills=10, deaths=20)
    play(conn, "m3", started=AS_OF - 3 * DAY, kills=10, deaths=20)
    got = measure(conn)
    # Averaged ratios would be (14 + 0.5 + 0.5) / 3 = 5.0. Summed, and with
    # recency weighting, it stays near one.
    assert got.values["kd"] < 1.5
    assert 0.6 < got.values["kd"] < 1.2


def test_assists_lift_kda_above_kd(conn):
    play(conn, "m1", started=AS_OF - DAY, kills=10, deaths=10, assists=5)
    got = measure(conn)
    assert got.values["kd"] == pytest.approx(1.0)
    assert got.values["kda"] == pytest.approx(1.5)


def test_ability_casts_are_divided_only_by_rounds_that_recorded_them(conn):
    """Half a player's history predates the columns.

    Dividing recovered casts by every round they have ever played would halve
    the ability component for anyone with matches on both sides of that line --
    and it would look like a quiet, plausible number rather than a bug.
    """
    play(conn, "m1", started=AS_OF - DAY, rounds=20, casts=(10, 10, 10, 10))
    play(conn, "m2", started=AS_OF - DAY, rounds=20, casts=None)
    got = measure(conn)
    assert got.values["abilities"] == pytest.approx(2.0), "40 casts over 20 rounds"
    assert got.n_ability_games == 1
    assert got.n_games == 2
    assert got.counts["abilities"] == 1


def test_a_player_with_no_recorded_casts_has_no_ability_value(conn):
    """None, not zero. Zero would say they cast nothing all match."""
    play(conn, "m1", started=AS_OF - DAY, casts=None)
    got = measure(conn)
    assert got.values["abilities"] is None
    assert got.values["acs"] is not None


def test_role_history_takes_over_as_it_accumulates(conn):
    """91% of player-role pairs have under five games, so the blend is the
    normal path, not an edge case."""
    # Poor on the role being scored, strong everywhere else.
    for i in range(2):
        play(conn, f"r{i}", started=AS_OF - (i + 1) * DAY, agent="Omen", score=6000)
    play(conn, "d0", started=AS_OF - 5 * DAY, agent="Jett", score=2000)
    thin = measure(conn, role="Duelist").values["acs"]

    for i in range(1, 20):
        play(conn, f"d{i}", started=AS_OF - (5 + i) * DAY, agent="Jett", score=2000)
    thick = measure(conn, role="Duelist").values["acs"]

    assert thick < thin, "more Duelist history should pull toward Duelist form"
    assert thick == pytest.approx(100.0, abs=8), "2000 over 20 rounds"


def test_a_role_never_played_falls_back_to_everything_they_have(conn):
    play(conn, "m1", started=AS_OF - DAY, agent="Jett", score=4000)
    got = measure(conn, role="Sentinel")
    assert got.n_role_games == 0
    assert got.values["acs"] == pytest.approx(200.0)


# --- shrinkage ----------------------------------------------------------

def test_a_thin_player_is_pulled_toward_their_role_average(conn):
    play(conn, "m1", started=AS_OF - DAY, score=6000)     # 300 ACS, one game
    means = {"acs": 200.0}
    alone = measure(conn).values["acs"]
    shrunk = measure(conn, role_means=means).values["acs"]
    assert alone == pytest.approx(300.0)
    # One game against a prior worth four: (300 + 200*4) / 5 = 220.
    assert shrunk == pytest.approx(220.0)


def test_fitting_the_population_sees_unshrunk_values(conn):
    """The role average cannot be built from numbers already pulled toward it."""
    play(conn, "m1", started=AS_OF - DAY, score=6000)
    assert measure(conn, role_means=None).values["acs"] == pytest.approx(300.0)


# --- the map component --------------------------------------------------

def test_the_map_component_says_nothing_below_three_games(conn):
    for i in range(2):
        play(conn, f"a{i}", started=AS_OF - (i + 1) * DAY, map_name="Lotus", score=6000)
    play(conn, "b0", started=AS_OF - 9 * DAY, map_name="Sunset", score=2000)
    assert measure(conn, map_name="Lotus").map_edge == 0.0
    assert measure(conn, map_name="Lotus").n_map_games == 2


def test_the_map_component_fires_at_three_games(conn):
    for i in range(3):
        play(conn, f"a{i}", started=AS_OF - (i + 1) * DAY, map_name="Lotus", score=6000)
    for i in range(3):
        play(conn, f"b{i}", started=AS_OF - (10 + i) * DAY, map_name="Sunset", score=2000)
    got = measure(conn, map_name="Lotus")
    assert got.n_map_games == 3
    assert got.map_edge > 0, "they are better on Lotus than overall"


# --- the firewall and the edges -----------------------------------------

def test_a_match_at_the_cutoff_is_not_visible(conn):
    """Strict `<`. The match being predicted cannot inform its own score."""
    play(conn, "now", started=AS_OF, score=9000)
    assert measure(conn) is None
    play(conn, "before", started=AS_OF - 1, score=4000)
    assert measure(conn).values["acs"] == pytest.approx(200.0)


def test_no_history_means_no_components(conn):
    assert measure(conn) is None


def test_recent_matches_count_for_more(conn):
    play(conn, "old", started=AS_OF - 120 * DAY, score=6000)
    play(conn, "new", started=AS_OF - DAY, score=2000)
    got = measure(conn)
    assert got.values["acs"] < 150, "four half-lives should mostly discount the old one"


def test_every_component_the_weights_name_is_measured(conn):
    """A weight naming a component nothing computes would be silently ignored,
    which is the failure mode this project keeps finding."""
    from valwr.rating import roles
    play(conn, "m1", started=AS_OF - DAY, casts=(4, 4, 4, 4))
    got = measure(conn)
    for name in roles.components_used():
        if name == "map_edge":
            continue
        assert name in got.values, name
        assert got.values[name] is not None, name
