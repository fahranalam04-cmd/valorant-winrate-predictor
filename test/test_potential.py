"""The 0-100 potential score, and the table that prints it.

Everything here runs against synthetic players in an in-memory database, so
these tests need neither the real database nor a fitted index.
"""

from __future__ import annotations

import json

import pytest

from valwr.live.__main__ import _display_width, _fit
from valwr.rating import potential as P
from valwr.sandbox import profiles, world


def an_index() -> P.PerfIndex:
    """A hand-built index, so the tests do not depend on a fitted one."""
    return P.PerfIndex(
        means={"acs": 212.0, "rating": 1.0, "kd": 1.05, "map_edge": 0.0},
        stds={"acs": 20.0, "rating": 0.06, "kd": 0.13, "map_edge": 0.02},
        quantiles=[i / 100.0 - 1.0 for i in range(201)],   # -1.0 .. 1.0
        as_of=0, n=201)


# --- the score ---------------------------------------------------------

def test_percentile_is_bounded_and_monotone():
    idx = an_index()
    assert idx.percentile(-99.0) == 0
    assert idx.percentile(99.0) == 100
    assert 0 <= idx.percentile(0.0) <= 100
    assert idx.percentile(-0.5) < idx.percentile(0.0) < idx.percentile(0.5)


def test_a_player_with_no_history_scores_nothing():
    """None, never a number.

    Inventing a score for someone we know nothing about is the failure this
    project keeps guarding against; the caller prints "--" instead.
    """
    conn = world.new_connection()
    norms = _norms(conn)
    assert P.measure(conn, "nobody", 1_800_000_000, "Ascent", norms) is None


def test_map_edge_is_zero_without_map_history():
    """The map component is a delta, so "no history here" means "no opinion".

    An absolute map rating would restate overall skill, which `rating` already
    carries; the delta has to vanish when there is nothing to compare.
    """
    conn, as_of, norms = _one_player(map_played="Ascent")
    c = P.measure(conn, "p", as_of, "Icebox", norms)     # never played Icebox
    assert c is not None
    assert c.n_map_games == 0
    assert c.map_edge == pytest.approx(0.0, abs=1e-12)


def test_components_come_back_populated():
    conn, as_of, norms = _one_player(map_played="Ascent")
    c = P.measure(conn, "p", as_of, "Ascent", norms)
    assert c.n_games > 0 and c.n_map_games > 0
    for name in P.WEIGHTS:
        assert isinstance(getattr(c, name), float)


def test_weights_are_a_convex_combination():
    """They should sum to 1, or the composite silently changes scale."""
    assert sum(P.WEIGHTS.values()) == pytest.approx(1.0)


def test_explain_picks_the_most_unusual_component():
    """Ranked by raw z, not weighted contribution.

    Weighting let ACS -- the heaviest weight -- win nearly every row, so the
    column read "high combat score" for four players out of five.
    """
    idx = an_index()
    c = P.Components(rating=1.0, acs=212.0, kd=1.05, map_edge=0.10,
                     n_games=50, n_map_games=20)      # map_edge is +5 sd
    assert "map" in P.explain(idx, c)


def test_thin_history_is_called_out():
    idx = an_index()
    c = P.Components(rating=1.2, acs=280.0, kd=1.6, map_edge=0.0,
                     n_games=2, n_map_games=0)
    assert "2 game" in P.explain(idx, c)


def test_index_round_trips_through_json():
    idx = an_index()
    back = json.loads(idx.to_json())
    assert back["means"]["acs"] == idx.means["acs"]
    assert len(back["quantiles"]) == len(idx.quantiles)


# --- the table ---------------------------------------------------------

def test_display_width_counts_wide_glyphs_as_two():
    assert _display_width("abc") == 3
    assert _display_width("테스트샘플") == 10         # 5 glyphs, 10 columns
    assert _display_width("테스트#abcd") == 11


def test_fit_pads_to_exact_display_width():
    """A wide-glyph name must occupy the same columns as an ASCII one.

    Character-count padding under-pads wide glyphs and the table's columns
    walk out of line, which is what the first version did.
    """
    for name in ("short", "테스트샘플#abcd", "a" * 40, "샘플" * 20, ""):
        assert _display_width(_fit(name, 20)) == 20


# --- helpers -----------------------------------------------------------

def _norms(conn):
    from valwr.rating.normalize import build_norms
    return build_norms(conn, 2_000_000_000)


def _one_player(map_played: str):
    as_of = 1_800_000_000
    conn = world.new_connection()
    world.write_player(conn, "p", profiles.get("average"), as_of,
                       map_played, "Jett")
    return conn, as_of, _norms(conn)


# --- against synthetic ground truth ------------------------------------
# These need the fitted index and the model bundle, so they skip rather than
# fail on a fresh clone where models/ has not been built.

def _bundle_and_index():
    from valwr.rating import potential as RP
    from valwr.sandbox import predictor as pred
    try:
        bundle = pred.load_bundle()
        index = RP.PerfIndex.load()
    except (pred.MissingModel, FileNotFoundError) as e:
        pytest.skip(str(e))
    return bundle, index


def test_the_score_ranks_the_ability_ladder_correctly():
    """The one check real data cannot supply.

    Every archetype is derived from a single ability parameter, so the correct
    order is known before the score runs. On live matches there is no such
    truth to compare against -- only how often the ranking happens to be right.
    """
    from valwr.sandbox import potential as sp, scenarios
    bundle, index = _bundle_and_index()
    scored = sp.score_scenario(scenarios.get("potential_skill_ladder"),
                               bundle, index, team="Blue")
    assert sp.ordering_matches(scored, [
        "elite", "strong", "above_average", "below_average", "weak"])


def test_thin_history_does_not_win_on_a_perfect_win_rate():
    """Three games at 100% must not outrank a 600-game veteran.

    Shrinkage is the guard, and the score ignores win rate outright -- so the
    gap across the whole team stays small rather than crowning the small
    sample.
    """
    from valwr.sandbox import potential as sp, scenarios
    bundle, index = _bundle_and_index()
    scored = sp.score_scenario(scenarios.get("potential_thin_history"),
                               bundle, index, team="Blue")
    assert sp.spread(scored) < 15


def test_role_bias_is_present_and_bounded():
    """A known defect, pinned so it cannot drift unnoticed.

    Four archetypes of identical ability, differing only by the ACS and K/D
    their role actually averages, score 52 points apart. The assertion is
    deliberately two-sided: the gap is real and must not be papered over, and
    it must not silently grow either. If a future change genuinely removes the
    bias, this test should fail and be deleted on purpose.
    """
    from valwr.sandbox import potential as sp, scenarios
    bundle, index = _bundle_and_index()
    scored = sp.score_scenario(scenarios.get("potential_role_bias"),
                               bundle, index, team="Blue")
    by_name = {p.profile: p.score for p in scored}
    assert by_name["duelist_main"] > by_name["initiator_main"]
    gap = by_name["duelist_main"] - by_name["initiator_main"]
    assert 30 <= gap <= 70, f"role gap moved to {gap} points"


def test_unknown_players_are_ranked_last_not_dropped():
    from valwr.sandbox import potential as sp, scenarios
    bundle, index = _bundle_and_index()
    scored = sp.score_scenario(scenarios.get("coverage_a_0_of_5"),
                               bundle, index, team="Blue")
    assert len(scored) == 5
    ranked = sp.ranked(scored)
    known = [p.known for p in ranked]
    assert known == sorted(known, reverse=True)


def test_kd_reaches_the_synthetic_world():
    """world.py used to pin kills and deaths equal, so K/D was always 1.0.

    That left the K/D component of the score untestable. The profile now
    declares it; this asserts it actually survives into the database.
    """
    from valwr.sandbox import world
    as_of = 1_800_000_000
    conn = world.new_connection()
    world.write_player(conn, "hi", profiles.get("duelist_main"), as_of,
                       "Ascent", "Reyna")
    world.write_player(conn, "lo", profiles.get("average"), as_of,
                       "Ascent", "Jett")
    got = {}
    for puuid in ("hi", "lo"):
        k, d = conn.execute(
            "SELECT SUM(kills), SUM(deaths) FROM match_players WHERE puuid=?",
            (puuid,)).fetchone()
        got[puuid] = k / d
    assert got["hi"] > got["lo"]
    assert got["lo"] == pytest.approx(1.0, abs=0.01)


# --- the above-rank flag -----------------------------------------------

def _flag_index(cut=1.0):
    idx = an_index()
    return P.PerfIndex(means=idx.means, stds=idx.stds, quantiles=idx.quantiles,
                       as_of=0, n=201, flag_cut=cut)


def _comp(rating, level, games=60):
    return P.Components(rating=rating, acs=212.0, kd=1.05, map_edge=0.0,
                        n_games=games, n_map_games=10, tier=10,
                        account_level=level)


def test_a_young_account_playing_above_its_rank_is_flagged():
    idx = _flag_index()
    f = P.above_rank(idx, _comp(rating=1.12, level=22))   # z = +2.0
    assert f.flagged and "above their rank" in f.note


def test_an_established_account_is_not_flagged_however_good():
    """The discriminating case, and the reason the second condition exists.

    `elite` scores a higher band-relative z than `smurf_like` -- measured
    +2.56 against +2.36 -- and must still not flag. A very good player on a
    mature account is not an anomaly, they are just good.
    """
    idx = _flag_index()
    assert not P.above_rank(idx, _comp(rating=1.30, level=400)).flagged


def test_high_rank_with_ordinary_output_is_not_flagged():
    """rank_only_smurf: the rank says a lot, the performance says nothing."""
    idx = _flag_index()
    assert not P.above_rank(idx, _comp(rating=1.00, level=20)).flagged


def test_a_young_account_playing_badly_is_not_flagged():
    idx = _flag_index()
    assert not P.above_rank(idx, _comp(rating=0.90, level=14)).flagged


def test_games_played_does_not_gate_the_flag():
    """n_games counts what the crawler collected, not what the player played.

    Measured on the training period, 99.8% of players fall under 40 games and
    the median is 3, so gating on it fired for 9,033 of 9,067 accounts and
    made the second condition inert. Only account level gates now.
    """
    idx = _flag_index()
    thin = P.above_rank(idx, _comp(rating=1.12, level=400, games=2))
    assert not thin.flagged, "a mature account must not flag on thin history"


def test_a_missing_account_level_never_flags():
    """Absent level means we cannot judge the account is young."""
    idx = _flag_index()
    c = P.Components(rating=1.30, acs=212.0, kd=1.05, map_edge=0.0,
                     n_games=5, n_map_games=1, tier=10, account_level=None)
    assert not P.above_rank(idx, c).flagged


def test_the_flag_survives_a_json_round_trip():
    idx = _flag_index(cut=1.7)
    back = P.PerfIndex(**{**json.loads(idx.to_json())})
    assert back.flag_cut == 1.7


def test_smurf_archetype_flags_and_rank_only_smurf_does_not():
    """Against the real calibrated index, on synthetic ground truth."""
    from valwr.sandbox import world
    bundle, index = _bundle_and_index()
    as_of = 1_800_000_000
    got = {}
    for name in ("smurf_like", "rank_only_smurf", "elite", "new_player"):
        conn = world.new_connection()
        world.write_player(conn, "p", profiles.get(name), as_of, "Ascent", "Jett")
        p = P.evaluate(conn, "p", as_of, "Ascent", bundle["norms"], index)
        got[name] = p.flag.flagged
        conn.close()
    assert got["smurf_like"], "the designed smurf must flag"
    assert not got["rank_only_smurf"], "high rank, ordinary output is not a smurf"
    assert not got["elite"], "an established strong player is not an anomaly"
    assert not got["new_player"], "young but not performing above rank"
