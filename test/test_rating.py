"""Phase 3 tests: derived components and the rating composite."""

from __future__ import annotations

import pytest

from valwr.rating import adjust
from valwr.rating.components import TRADE_WINDOW_MS, match_components, per_round_rates


def kill(rnd, killer, victim, t, assists=()):
    return {"round": rnd, "time_in_round_in_ms": t,
            "killer": {"puuid": killer}, "victim": {"puuid": victim},
            "assistants": [{"puuid": a} for a in assists]}


def match(kills, n_rounds=1, ceremonies=None, winners=None):
    rounds = [{"id": i,
               "ceremony": (ceremonies or {}).get(i, "CeremonyDefault"),
               "winning_team": (winners or {}).get(i, "Blue"),
               "stats": []} for i in range(n_rounds)]
    players = [{"puuid": f"b{i}", "team_id": "Blue"} for i in range(5)]
    players += [{"puuid": f"r{i}", "team_id": "Red"} for i in range(5)]
    return {"rounds": rounds, "kills": kills, "players": players}


# --- first bloods -----------------------------------------------------

def test_first_kill_of_round_is_a_first_blood_and_first_death():
    c = match_components(match([kill(0, "b0", "r0", 5000), kill(0, "b1", "r1", 9000)]))
    assert c["b0"]["first_bloods"] == 1
    assert c["r0"]["first_deaths"] == 1
    assert c["b1"]["first_bloods"] == 0, "only the first kill counts"
    assert c["r1"]["first_deaths"] == 0


def test_first_blood_uses_time_not_array_order():
    """Kill arrays are not guaranteed sorted; ordering by time is the fix."""
    c = match_components(match([kill(0, "b1", "r1", 9000), kill(0, "b0", "r0", 5000)]))
    assert c["b0"]["first_bloods"] == 1
    assert c["b1"]["first_bloods"] == 0


def test_first_bloods_and_first_deaths_always_balance():
    c = match_components(match(
        [kill(0, "b0", "r0", 1000), kill(1, "r2", "b3", 2000)], n_rounds=2))
    assert sum(v["first_bloods"] for v in c.values()) == 2
    assert sum(v["first_deaths"] for v in c.values()) == 2


# --- trades -----------------------------------------------------------

def test_kill_avenging_a_teammate_within_the_window_is_a_trade():
    c = match_components(match([
        kill(0, "r0", "b0", 1000),                        # r0 kills a Blue
        kill(0, "b1", "r0", 1000 + TRADE_WINDOW_MS - 1),  # b1 avenges b0
    ]))
    assert c["b1"]["trade_kills"] == 1
    assert c["b0"]["traded_deaths"] == 1


def test_revenge_outside_the_window_is_not_a_trade():
    c = match_components(match([
        kill(0, "r0", "b0", 1000),
        kill(0, "b1", "r0", 1000 + TRADE_WINDOW_MS + 1),
    ]))
    assert c["b1"]["trade_kills"] == 0
    assert c["b0"]["traded_deaths"] == 0


def test_killing_someone_who_killed_an_enemy_is_not_a_trade():
    """A trade avenges a TEAMMATE. Killing an enemy who fragged another enemy
    is not one, and treating it as such would inflate every duelist."""
    c = match_components(match([
        kill(0, "r0", "r1", 1000),        # r0 team-kills a Red
        kill(0, "b1", "r0", 2000),        # b1 kills r0 -- avenges nobody of theirs
    ]))
    assert c["b1"]["trade_kills"] == 0


# --- KAST -------------------------------------------------------------

def test_surviving_the_round_counts_toward_kast():
    c = match_components(match([kill(0, "b0", "r0", 1000)]))
    assert c["b4"]["kast_rounds"] == 1, "never died, so survived"
    assert c["r0"]["kast_rounds"] == 0, "died, no kill, no assist, not traded"


def test_kill_assist_and_trade_each_count_toward_kast():
    c = match_components(match([
        kill(0, "b0", "r0", 1000, assists=["b1"]),
        kill(0, "r1", "b2", 2000),
        kill(0, "b3", "r1", 2500),        # trades b2
    ]))
    assert c["b0"]["kast_rounds"] == 1   # kill
    assert c["b1"]["kast_rounds"] == 1   # assist
    assert c["b2"]["kast_rounds"] == 1   # died but was traded
    assert c["r0"]["kast_rounds"] == 0


def test_kast_never_exceeds_rounds_played():
    c = match_components(match(
        [kill(i, "b0", "r0", 1000) for i in range(3)], n_rounds=3))
    for stats in c.values():
        assert stats["kast_rounds"] <= stats["rounds_played"]


# --- multikills and clutches -----------------------------------------

def test_three_kills_in_a_round_is_a_multikill():
    c = match_components(match(
        [kill(0, "b0", f"r{i}", 1000 + i * 100) for i in range(3)]))
    assert c["b0"]["multikills"] == 1


def test_two_kills_is_not_a_multikill():
    c = match_components(match(
        [kill(0, "b0", f"r{i}", 1000 + i * 100) for i in range(2)]))
    assert c["b0"]["multikills"] == 0


def test_clutch_is_credited_to_the_sole_survivor():
    """Inferred: ceremony marks that a clutch happened, not who won it."""
    kills = [kill(0, "r0", f"b{i}", 1000 + i * 100) for i in range(4)]  # b0..b3 die
    c = match_components(match(kills, ceremonies={0: "CeremonyClutch"},
                               winners={0: "Blue"}))
    assert c["b4"]["clutches"] == 1


def test_ambiguous_clutch_is_skipped_not_guessed():
    c = match_components(match([kill(0, "r0", "b0", 1000)],
                               ceremonies={0: "CeremonyClutch"}, winners={0: "Blue"}))
    assert sum(v["clutches"] for v in c.values()) == 0, "4 survivors -> ambiguous"


# --- rates ------------------------------------------------------------

def test_per_round_rates_use_the_real_round_count():
    r = per_round_rates({"rounds_played": 20, "score": 4000, "damage_dealt": 2800,
                         "kills": 15, "deaths": 14, "assists": 5, "kast_rounds": 14,
                         "headshots": 10, "bodyshots": 30, "legshots": 10})
    assert r["acs"] == 200.0
    assert r["adr"] == 140.0
    assert r["kast"] == 0.7
    assert r["hs_pct"] == 0.2


def test_zero_rounds_yields_none_not_a_division_error():
    assert per_round_rates({"rounds_played": 0})["acs"] is None


# --- opponent adjustment ---------------------------------------------

def test_facing_stronger_opposition_raises_the_rating():
    assert adjust.adjust(1.0, gap=3.0) > 1.0
    assert adjust.adjust(1.0, gap=-3.0) < 1.0
    assert adjust.adjust(1.0, gap=None) == 1.0


def test_opponent_gap_is_enemy_minus_own_tier():
    rows = [{"puuid": "me", "team": "Blue", "tier": 13}]
    rows += [{"puuid": f"e{i}", "team": "Red", "tier": 16} for i in range(5)]
    assert adjust.opponent_gap(rows, "me") == pytest.approx(3.0)
