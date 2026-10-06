"""Your best picks: how well you play each agent, not how often you win on it."""

from __future__ import annotations

import pytest

from valwr.live import picks
from valwr.store import schema

NOW = 2_000_000_000


@pytest.fixture
def history(tmp_path, monkeypatch):
    """Write games for "me" and fix each game's match impact by its id, so the
    tests are about ranking and shrinking rather than about the rating."""
    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)
    impacts = {}

    def add(n, agent, impact, won=True, map_name="Ascent"):
        for _ in range(n):
            mid = f"m{len(impacts)}"
            impacts[mid] = impact
            ts = NOW - 1000 - len(impacts) * 60
            conn.execute(
                "INSERT INTO matches (match_id, started_at, map, mode, queue, "
                "region, season, rounds_red, rounds_blue, winner, data_quality, "
                "ingested_at) VALUES (?, ?, ?, 'competitive', 'Standard', 'na', "
                "'s', 9, 13, 'Blue', NULL, 0)", (mid, ts, map_name))
            conn.execute(
                "INSERT INTO match_players (match_id, puuid, team, agent, "
                "party_id, tier, account_level, score, kills, deaths, assists, "
                "headshots, bodyshots, legshots, damage_dealt, damage_taken, "
                "started_at, map, won, rounds_played) VALUES (?, 'me', 'Blue', ?, "
                "NULL, 15, 100, 4000, 15, 15, 5, 5, 5, 5, 3000, 3000, ?, ?, ?, 20)",
                (mid, agent, ts, map_name, 1 if won else 0))
        conn.commit()

    monkeypatch.setattr(picks.rating, "match_impact",
                        lambda row, norms=None: impacts[row["match_id"]])
    return conn, add


def _picks(conn, map_name="Ascent"):
    return picks.your_picks(conn, "me", NOW, map_name, None,
                            {"Jett": "Duelist", "Omen": "Controller",
                             "Sova": "Initiator", "Sage": "Sentinel"})


def _names(group):
    return [a["agent"] for a in group]


def test_ranked_by_how_well_you_play_not_by_how_often_you_win(history):
    """A win rate is half your teammates'. Jett here wins every game while
    playing below par; Omen loses every game while playing well."""
    conn, add = history
    add(8, "Jett", 0.90, won=True)
    add(8, "Omen", 1.20, won=False)
    got = _picks(conn)["agents"]
    assert _names(got) == ["Omen", "Jett"]
    assert got[0]["wins"] == 0 and got[1]["losses"] == 0


def test_three_games_makes_an_agent_established(history):
    """Your threshold: three or more games are ranked as your picks; one or
    two sit underneath -- shown, not dropped."""
    conn, add = history
    add(3, "Omen", 1.0)
    add(2, "Sova", 1.0)
    add(1, "Sage", 1.0)
    got = _picks(conn)
    assert _names(got["agents"]) == ["Omen"]
    assert sorted(_names(got["few"])) == ["Sage", "Sova"], "a single game still shows"


def test_established_picks_are_pulled_toward_your_average(history):
    """Five virtual games at your own average stand beside each ranked agent's
    real ones: three games count for three eighths of themselves, fifteen for
    three quarters."""
    import statistics
    conn, add = history
    add(15, "Omen", 1.20)
    add(3, "Sova", 1.60)
    add(15, "Jett", 0.90)
    got = {a["agent"]: a for a in _picks(conn)["agents"]}
    games = [1.20] * 15 + [1.60] * 3 + [0.90] * 15
    mean, spread = statistics.mean(games), statistics.pstdev(games)
    for agent, n, value in (("Omen", 15, 1.20), ("Sova", 3, 1.60)):
        unpulled = (value - mean) / spread
        assert got[agent]["vs_usual"] == pytest.approx(
            unpulled * n / (n + picks.SHRINK_GAMES), abs=0.01), agent


def test_a_game_or_two_says_what_it_showed(history):
    """Pulled like the rest, one game would land five sixths of the way back
    to "your usual" -- disregarded by another name. Underneath, it is shown
    as it was, with the game count beside it."""
    import statistics
    conn, add = history
    add(10, "Omen", 1.0)
    add(1, "Sage", 1.5)
    add(2, "Sova", 0.6)
    got = {a["agent"]: a for a in _picks(conn)["few"]}
    games = [1.0] * 10 + [1.5] + [0.6] * 2
    mean, spread = statistics.mean(games), statistics.pstdev(games)
    assert got["Sage"]["vs_usual"] == pytest.approx((1.5 - mean) / spread, abs=0.01)
    assert got["Sage"]["label"] == "well above your usual"
    assert got["Sova"]["label"] == "well below your usual"
    assert _names(_picks(conn)["few"]) == ["Sage", "Sova"], "ranked too"


def test_the_map_record_counts_only_this_map(history):
    conn, add = history
    add(3, "Omen", 1.0, won=True, map_name="Ascent")
    add(4, "Omen", 1.0, won=False, map_name="Bind")
    omen = _picks(conn)["agents"][0]
    assert omen["games"] == 7 and (omen["wins"], omen["losses"]) == (3, 4)
    assert omen["here"] == {"games": 3, "wins": 3, "losses": 0}
    assert omen["role"] == "Controller"


def test_labels_read_in_units_of_your_own_spread():
    assert picks.label(0.5) == "well above your usual"
    assert picks.label(0.2) == "above your usual"
    assert picks.label(0.0) == "your usual"
    assert picks.label(-0.2) == "below your usual"
    assert picks.label(-0.5) == "well below your usual"


def test_nothing_is_shown_without_history_to_judge_by(history):
    conn, _ = history
    assert _picks(conn) is None


def test_each_section_is_capped_best_first(history):
    conn, add = history
    for i in range(10):
        add(4, f"E{i}", 1.0 + i / 10)
    for i in range(8):
        add(1, f"F{i}", 1.0 + i / 10)
    got = _picks(conn)
    assert len(got["agents"]) == picks.SHOWN_ESTABLISHED
    assert len(got["few"]) == picks.SHOWN_FEW
    assert _names(got["agents"])[:2] == ["E9", "E8"]
    assert _names(got["few"])[:2] == ["F7", "F6"]
