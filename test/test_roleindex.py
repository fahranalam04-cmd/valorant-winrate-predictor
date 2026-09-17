"""The population the per-role score is measured against.

Three reference points, and each one exists to fix something a single
population got wrong: rank band (250 ACS is not 250 ACS everywhere), agent
(Cypher casts twice what Chamber does), role (a Duelist's 82 should mean what
a Sentinel's 82 means).
"""

from __future__ import annotations

import pytest

from valwr.rating import roleindex, roles
from valwr.rating.role_score import RoleComponents

BASE = {"acs": 200.0, "adr": 140.0, "kd": 1.0, "kda": 1.4, "assists": 0.25,
        "kast": 0.71, "fb": 0.10, "fd": 0.10, "hs": 0.22, "abilities": 2.0}


def comps(role="Duelist", agent="Jett", tier=13, map_games=0, map_edge=0.0,
          **over):
    values = dict(BASE)
    values.update(over)
    return RoleComponents(
        role=role, agent=agent, values=values,
        counts={k: 20 for k in values}, map_edge=map_edge, n_games=20,
        n_role_games=20, n_map_games=map_games, n_ability_games=20,
        tier=tier, account_level=100)


def population():
    """A spread of players across two rank bands and two roles.

    Iron plays worse than Immortal in raw numbers; Sentinels score less than
    Duelists. Both facts are true of the real database and both are things the
    index has to handle without penalising anyone for them.
    """
    out = []
    for i in range(400):
        step = (i % 20) / 20.0
        out.append(comps(role="Duelist", agent="Jett", tier=5,
                         acs=120 + 60 * step, kast=0.60 + 0.1 * step,
                         abilities=1.6 + 0.5 * step))
        out.append(comps(role="Duelist", agent="Jett", tier=22,
                         acs=220 + 60 * step, kast=0.70 + 0.1 * step,
                         abilities=1.8 + 0.5 * step))
        out.append(comps(role="Sentinel", agent="Cypher", tier=22,
                         acs=180 + 40 * step, kast=0.72 + 0.1 * step,
                         abilities=3.2 + 0.6 * step))
    return out


@pytest.fixture(scope="module")
def index():
    return roleindex.fit(population(), as_of=1000)


# --- rank band ----------------------------------------------------------

def test_the_same_acs_is_worth_more_in_a_lower_band(index):
    """250 in Iron is a different performance from 250 in Immortal."""
    low = index.z("acs", 250.0, band=roleindex.band_of(5))
    high = index.z("acs", 250.0, band=roleindex.band_of(22))
    assert low > high + 1.0, "the gap should be large, not incidental"
    # A band holds every role together, so its centre sits between them.
    # What matters is that the same number is read differently by band,
    # not that either side lands on a particular sign.
    assert low > 0


def test_an_unseen_band_falls_back_to_the_population(index):
    """A band with no samples must still score, shrunk toward the global."""
    z = index.z("acs", 250.0, band=roleindex.band_of(27))
    assert z is not None and -4 < z < 4


# --- the agent ----------------------------------------------------------

def test_abilities_are_judged_against_the_same_agent(index):
    """Cypher's 3.5 casts a round and Jett's 1.9 are both ordinary.

    Judged against one population, every Cypher outranks every Jett on the
    component -- which would score the agent, not the player.
    """
    jett = index.z_ability(2.4, agent="Jett", role="Duelist")
    cypher = index.z_ability(3.8, agent="Cypher", role="Sentinel")
    assert jett > 0 and cypher > 0
    # Cypher casts far more in absolute terms but is not therefore better.
    assert index.z_ability(2.4, "Cypher", "Sentinel") < 0


def test_a_thin_agent_falls_back_to_its_role_and_says_so(index):
    """An agent nobody in the sample played cannot define its own spread."""
    assert index.z_ability(2.0, agent="BrandNewAgent", role="Duelist") is not None
    thin = roleindex.fit(population() + [comps(role="Duelist", agent="Rare")])
    assert "Rare" in thin.fell_back
    assert index.by_agent.get("Rare") is None


def test_a_player_with_no_recorded_casts_has_no_ability_z(index):
    assert index.z_ability(None, "Jett", "Duelist") is None


# --- the composite ------------------------------------------------------

def test_a_missing_component_rescales_rather_than_penalises(index):
    """Half the lobby has no ability history. They must not all sink."""
    known = comps()
    unknown = comps(abilities=None)
    a = index.composite(known, roles.weights_for("Duelist"))
    b = index.composite(unknown, roles.weights_for("Duelist"))
    assert a is not None and b is not None
    # An average player either way: dropping a component they are average at
    # should barely move them, and must not push them down.
    assert abs(a - b) < 0.5


def test_a_gated_map_contributes_exactly_nothing(index):
    """Below the threshold the map is 'no opinion', not 'average'."""
    quiet = index.composite(comps(map_games=0, map_edge=99.0),
                            roles.weights_for("Duelist"))
    same = index.composite(comps(map_games=0, map_edge=-99.0),
                           roles.weights_for("Duelist"))
    assert quiet == pytest.approx(same)


def test_a_player_with_nothing_measurable_scores_nothing(index):
    empty = comps(**{k: None for k in BASE})
    assert index.composite(empty, roles.weights_for("Duelist")) is None
    assert index.score(empty) is None


def test_first_deaths_push_a_sentinel_down(index):
    """The only negative weight in the project has to actually be negative."""
    steady = index.composite(comps(role="Sentinel", agent="Cypher", fd=0.05),
                             roles.weights_for("Sentinel"))
    dying = index.composite(comps(role="Sentinel", agent="Cypher", fd=0.20),
                            roles.weights_for("Sentinel"))
    assert dying < steady


# --- the percentile -----------------------------------------------------

def test_the_percentile_is_taken_within_the_role(index):
    """A Sentinel at the top of Sentinels scores like a Duelist at the top of
    Duelists, even though Sentinels put up smaller raw numbers."""
    top_duelist = index.score(comps(role="Duelist", agent="Jett", tier=22,
                                    acs=280, kast=0.80, abilities=2.3))
    top_sentinel = index.score(comps(role="Sentinel", agent="Cypher", tier=22,
                                     acs=220, kast=0.82, abilities=3.8))
    assert top_duelist > 80 and top_sentinel > 80
    assert abs(top_duelist - top_sentinel) < 20


def test_scores_stay_inside_the_scale(index):
    for c in (comps(acs=9000, kast=1.0), comps(acs=1, kast=0.0)):
        assert 0 <= index.score(c) <= 100


def test_an_unknown_role_still_scores(index):
    """Agent select, before anyone has locked in."""
    got = index.score(comps(role=None, agent=None))
    assert got is not None and 0 <= got <= 100


# --- storage ------------------------------------------------------------

def test_an_index_survives_a_round_trip(index, tmp_path):
    path = tmp_path / "role_index.json"
    path.write_text(index.to_json(), encoding="utf-8")
    back = roleindex.RoleIndex.load(path)
    c = comps(role="Sentinel", agent="Cypher", tier=22)
    assert back.score(c) == index.score(c)
    assert back.n == index.n and back.as_of == index.as_of
    assert back.role_means.keys() == index.role_means.keys()


def test_the_index_is_fitted_on_unshrunk_values(index):
    """The role average is what thin players are pulled toward, so it cannot
    itself be built from values that were already pulled toward it."""
    assert index.role_means["Duelist"]["acs"] == pytest.approx(200.0, abs=25)
    assert index.role_means["Sentinel"]["acs"] == pytest.approx(199.0, abs=25)


# --- the reference has to describe the score that is actually served ----

def _spread(index, samples):
    """Scores for these players, put through the live path's shrinkage."""
    from valwr.rating.role_score import shrink_values
    import dataclasses
    out = []
    for c in samples:
        live = dataclasses.replace(c, values=shrink_values(
            c.values, c.counts, index.role_means.get(c.role or "?")))
        got = index.score(live)
        if got is not None:
            out.append(got)
    return out


def _thin_population():
    """Players with little history, so the shrinkage actually bites."""
    out = []
    for i in range(600):
        step = (i % 30) / 30.0
        for role, agent, base in (("Duelist", "Jett", 140.0),
                                  ("Sentinel", "Cypher", 130.0)):
            c = comps(role=role, agent=agent, tier=15,
                      acs=base + 160 * step, adr=90 + 110 * step,
                      kd=0.5 + 1.2 * step, kda=0.8 + 1.4 * step,
                      kast=0.55 + 0.3 * step, abilities=1.0 + 2.0 * step,
                      hs=0.12 + 0.2 * step)
            out.append(dataclasses_replace_counts(c, 6))
    return out


def dataclasses_replace_counts(c, n):
    import dataclasses
    return dataclasses.replace(c, counts={k: n for k in c.values}, n_games=n)


def test_the_scale_is_not_compressed_by_shrinkage():
    """Fit and serve have to agree about shrinkage.

    The tables were once fitted on raw values while every live player was
    scored with values shrunk toward their role average. Shrinking pulls
    everyone inward, so real scores landed in a narrow band around the middle:
    measured on 4,700 players, 1% scored above 90 where a percentile should put
    10% there, and the sd was 17 instead of 29. The number still looked like a
    percentile and was not one.
    """
    samples = _thin_population()
    index = roleindex.fit(samples, as_of=1000)
    scores = _spread(index, samples)
    assert len(scores) > 500

    import statistics as st
    assert st.pstdev(scores) > 20, "scores are bunched in the middle"
    top = sum(1 for s in scores if s >= 90) / len(scores)
    bottom = sum(1 for s in scores if s <= 10) / len(scores)
    assert top > 0.04, f"only {top:.1%} of players score above 90"
    assert bottom > 0.04, f"only {bottom:.1%} of players score 10 or less"


def test_role_averages_still_come_from_raw_values():
    """The shrinkage target cannot be built from shrunk values, or each
    rebuild pulls the population a little further toward its own middle."""
    samples = _thin_population()
    index = roleindex.fit(samples, as_of=1000)
    raw = [c.values["acs"] for c in samples if c.role == "Duelist"]
    assert index.role_means["Duelist"]["acs"] == pytest.approx(
        sum(raw) / len(raw), abs=1.0)
