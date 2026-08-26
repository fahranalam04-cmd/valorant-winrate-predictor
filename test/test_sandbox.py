"""Sandbox invariants.

These are the mechanical guarantees -- determinism, valid domains, finite
numbers, symmetry, feature completeness, offline operation. They fail the
build.

Directional expectations are deliberately absent here. "A stronger team should
be favoured" is a claim about the *model*, and a trained model disagreeing with
it is a finding to read, not a build to break. Those live in the CLI report as
MODEL WARNINGs.

The full 343-scenario catalog and Monte Carlo runs are CLI-only; this suite
takes a representative sample so `pytest` stays fast.
"""

from __future__ import annotations

import math

import pytest

from valwr.features import player as pf
from valwr.features import team as tf
from valwr.sandbox import profiles, runner, scenarios, sweeps, variance, world
from valwr.sandbox import predictor as pred
from valwr.sandbox.schema import (InvalidProfile, MatchScenario, PlayerProfile,
                                  TeamProfile, VarianceSpec)

SAMPLE = ("fair_match", "single_smurf", "bad_map", "good_map",
          "five_stack_vs_solos", "hot_vs_cold_form", "coverage_a_0_of_5",
          "shrinkage_3_at_100", "edge_wr_one", "dominance_extreme")


@pytest.fixture(scope="module")
def bundle():
    try:
        return pred.load_bundle()
    except pred.MissingModel:
        pytest.skip("no trained model bundle present")


@pytest.fixture(scope="module")
def model(bundle):
    return pred.shipped(bundle)


def sample_scenarios():
    return [scenarios.get(n) for n in SAMPLE]


# --- construction is deterministic ------------------------------------

def test_the_catalog_is_stable_across_rebuilds():
    """Static scenarios are benchmarks; rebuilding must not change them."""
    first = {s.name: s for s in scenarios.build()}
    second = {s.name: s for s in scenarios.build()}
    assert first.keys() == second.keys()
    for name, a in first.items():
        b = second[name]
        assert [p.acs for p in a.team_a.players] == [p.acs for p in b.team_a.players]
        assert a.map_name == b.map_name


def test_every_scenario_validates():
    for s in scenarios.CATALOG + sweeps.generated():
        s.validate()


def test_every_team_has_five_players():
    for s in scenarios.CATALOG:
        assert len(s.team_a.players) == 5
        assert len(s.team_b.players) == 5


def test_features_are_deterministic(bundle):
    s = scenarios.get("single_smurf")
    a = runner.features_for(s, bundle)
    b = runner.features_for(s, bundle)
    assert a == b


# --- profile validation -----------------------------------------------

def test_impossible_profiles_are_rejected():
    with pytest.raises(InvalidProfile):
        PlayerProfile(name="bad", win_rate=1.7).validate()
    with pytest.raises(InvalidProfile):
        PlayerProfile(name="bad", games=-3).validate()
    with pytest.raises(InvalidProfile):
        PlayerProfile(name="bad", kast=-0.2).validate()
    with pytest.raises(InvalidProfile):
        # a subset larger than its set
        PlayerProfile(name="bad", games=10, map_games=20).validate()


def test_all_shipped_profiles_are_valid():
    profiles.validate_all()


# --- the world reproduces what a profile declares ----------------------

def test_declared_history_round_trips(bundle):
    """Counts must be exact; rates only as exact as integers allow.

    A 3-game player cannot have a 0.486 win rate -- the achievable values are
    thirds. So the tolerance is the quantisation bound 1/(2n), not a constant.
    """
    from valwr.store import temporal
    as_of = runner.AS_OF
    for prof in profiles.ALL.values():
        if not prof.has_history:
            continue
        s = MatchScenario(
            name="t", category="t", description="",
            team_a=TeamProfile(players=(prof,) * 5, agents=("Jett",) * 5),
            team_b=TeamProfile(players=(profiles.AVERAGE,) * 5,
                               agents=("Jett",) * 5))
        conn, _ = world.build_world(s, as_of)
        try:
            overall = temporal.record(conn, "blue0", as_of)
            on_map = temporal.record_on_map(conn, "blue0", as_of, "Ascent")
            assert overall.games == prof.games, prof.name
            assert on_map.games == prof.map_games, prof.name
            bound = 1.0 / (2 * prof.games) + 1e-9
            assert abs((overall.win_rate or 0) - prof.win_rate) <= bound, prof.name
        finally:
            conn.close()


def test_a_player_with_no_history_is_neutral_not_terrible(bundle):
    """Unknown must mean "we do not know", never "bad".

    Checked at the player level rather than by comparing two scenarios: a
    profile with zero games should land on the population prior, and must not
    come back as a 0.0 win rate, which would read as a player who never wins.
    """
    from valwr.rating.normalize import build_norms
    conn = world.new_connection()
    try:
        norms = bundle["norms"]
        prior = bundle["prior_rate"]
        p = pf.build(conn, "nobody", runner.AS_OF, "Ascent", "Jett",
                     None, None, norms, prior)
        assert p.games == 0
        assert p.values["wr"] == pytest.approx(prior), (
            "an unknown player's win rate must be the prior, not zero")
        assert p.values["rating"] == pytest.approx(1.0), (
            "an unknown player's rating must be neutral, not zero")
    finally:
        conn.close()


def test_unknown_players_do_not_drag_a_team_below_a_weak_one(bundle):
    """A lobby of strangers should not look worse than a lobby of known-bad
    players -- absence of evidence is not evidence of weakness."""
    unknown = runner.features_for(scenarios.get("coverage_none_vs_full"), bundle)
    known_weak = runner.features_for(
        MatchScenario(name="weak_vs_full", category="t", description="",
                      team_a=TeamProfile(players=(profiles.WEAK,) * 5,
                                         agents=("Jett",) * 5),
                      team_b=TeamProfile(players=(profiles.AVERAGE,) * 5,
                                         agents=("Jett",) * 5)), bundle)
    assert unknown["d_wr_mean"] > known_weak["d_wr_mean"]


# --- numbers stay finite and in range ---------------------------------

def test_no_feature_is_nan_or_infinite(bundle):
    for s in sample_scenarios():
        for key, value in runner.features_for(s, bundle).items():
            assert math.isfinite(value), f"{s.name}.{key} = {value}"


def test_boundary_scenarios_do_not_crash(bundle, model):
    edges = [s for s in scenarios.CATALOG if "boundary" in s.tags]
    assert edges, "expected boundary scenarios"
    for s in edges:
        p = model.predict_proba(runner.features_for(s, bundle))
        assert 0.0 <= p <= 1.0 and math.isfinite(p), s.name


def test_probabilities_stay_in_range(bundle, model):
    for s in sample_scenarios():
        p = model.predict_proba(runner.features_for(s, bundle))
        assert 0.0 <= p <= 1.0


# --- symmetry ----------------------------------------------------------

def test_mirroring_negates_the_feature_vector_exactly(bundle):
    """The representation is a difference, so this has no numerical excuse."""
    for s in sample_scenarios():
        assert runner.antisymmetry_error(s, bundle) < 1e-9, s.name


def test_mirrored_probabilities_sum_to_one(bundle, model):
    """Within a documented tolerance, not exactly.

    StandardScaler subtracts non-zero training means from all 52 columns, so
    the standardised vector does not simply negate even though the raw one
    does. Measured error is ~3e-4 for the shipped linear model.
    """
    for s in sample_scenarios():
        r = runner.run(s, model, bundle)
        assert r.mirror_error < 1e-3, f"{s.name}: {r.mirror_error}"


def test_identical_teams_give_a_zero_difference_vector(bundle):
    features = runner.features_for(scenarios.get("fair_match"), bundle)
    for key, value in features.items():
        assert abs(value) < 1e-9, f"{key} = {value} for identical teams"


# --- generated coverage ------------------------------------------------

def test_every_production_feature_is_covered():
    """Adding a production feature without sandbox coverage fails here.

    This is the guard against the sandbox quietly falling behind the model.
    """
    missing = sweeps.missing_features()
    assert not missing, (
        f"{len(missing)} production features have no sandbox coverage: "
        f"{sorted(missing)}. Add a knob in sweeps.KNOBS, or list it in "
        f"ROSTER_LEVEL with the category that exercises it.")


def test_every_player_feature_has_a_knob():
    missing = [f for f in pf.FEATURE_NAMES if f not in sweeps.KNOBS]
    assert not missing, f"no sweep knob for {missing}"


def test_sweep_scenarios_change_only_their_own_feature(bundle):
    """A single-factor sweep that moved several features would make its curve
    meaningless."""
    neutral = runner.features_for(
        next(s for s in sweeps.single_factor()
             if s.name == "sweep__kast__neutral"), bundle)
    high = runner.features_for(
        next(s for s in sweeps.single_factor()
             if s.name == "sweep__kast__very_high"), bundle)
    moved = {k for k in neutral if abs(neutral[k] - high[k]) > 1e-6}
    assert moved, "the knob moved nothing at all"
    assert all("kast" in k or "rating" in k for k in moved), (
        f"kast sweep also moved unrelated features: "
        f"{sorted(k for k in moved if 'kast' not in k and 'rating' not in k)}")


# --- variance -----------------------------------------------------------

def test_variance_is_reproducible_for_a_seed():
    s = scenarios.get("single_smurf")
    a = [p.acs for r in variance.sample(s, 5, seed=7) for p in r.team_a.players]
    b = [p.acs for r in variance.sample(s, 5, seed=7) for p in r.team_a.players]
    assert a == b


def test_different_seeds_produce_different_samples():
    s = scenarios.get("single_smurf")
    a = [p.acs for r in variance.sample(s, 5, seed=1) for p in r.team_a.players]
    b = [p.acs for r in variance.sample(s, 5, seed=2) for p in r.team_a.players]
    assert a != b


def test_variance_respects_every_domain_bound():
    for name in ("fair_match", "edge_wr_one", "edge_kast_zero", "single_smurf"):
        for realisation in variance.sample(scenarios.get(name), 12, seed=3):
            realisation.validate()          # raises on any out-of-domain value
            for team in (realisation.team_a, realisation.team_b):
                for p in team.players:
                    assert p.games >= 0 and isinstance(p.games, int)
                    assert 0.0 <= p.win_rate <= 1.0
                    assert 0.0 <= p.kast <= 1.0
                    assert p.map_games <= p.games
                    assert p.map_agent_games <= min(p.map_games, p.agent_games)


def test_scenario_identity_survives_variance():
    """`single_smurf` stops being that scenario if the smurf is not the best
    player, so realisations are rejection-sampled against a guard."""
    s = scenarios.get("single_smurf")
    assert s.variance.preserve is not None
    for realisation in variance.sample(s, 25, seed=11):
        acs = [p.acs for p in realisation.team_a.players]
        assert acs[2] == max(acs), "the carry stopped being the carry"


def test_variance_statistics_are_finite(bundle, model):
    v = runner.run_variance(scenarios.get("fair_match"), model, bundle,
                            samples=12, seed=5)
    for value in (v.mean, v.std, v.median, v.drift, v.flip_rate):
        assert math.isfinite(value)
    assert 0.0 <= v.flip_rate <= 1.0
    assert v.robustness in ("stable", "moderately sensitive",
                            "highly sensitive")


# --- isolation ----------------------------------------------------------

def test_the_world_is_always_in_memory():
    """The sandbox must never read or write the real database."""
    conn = world.new_connection()
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
        assert row["file"] in ("", None), f"sandbox opened a file: {row['file']}"
    finally:
        conn.close()


def test_no_real_puuids_appear_anywhere():
    s = scenarios.get("fair_match")
    _, roster = world.build_world(s, runner.AS_OF)
    for row in roster:
        assert row["puuid"].startswith(("blue", "red")), row["puuid"]
        assert len(row["puuid"]) < 12, "looks like a real 36-char PUUID"


def test_sandbox_modules_make_no_network_calls():
    """A grep, deliberately: the point is that no future edit adds one."""
    import pathlib
    import re
    package = pathlib.Path(world.__file__).parent
    offenders = []
    for path in package.glob("*.py"):
        code = re.sub(r'"{3}.*?"{3}', "", path.read_text(encoding="utf-8"),
                      flags=re.S)
        if re.search(r"\b(httpx|requests|urllib|socket)\b", code):
            offenders.append(path.name)
    assert not offenders, f"sandbox modules reference network libraries: {offenders}"


# --- the sample runs end to end ----------------------------------------

def test_the_sampled_scenarios_all_run(bundle, model):
    results = runner.run_many(sample_scenarios(), model, bundle)
    assert len(results) == len(SAMPLE)
    for r in results:
        assert math.isfinite(r.probability)
        assert r.features
