"""The per-role weight tables.

These are hand-set judgements rather than fitted numbers, so what a test can
check is that they say what the spec says they say -- and that the arithmetic
around them (the map component's share, the Chamber blend, the agent-select
fallback) cannot quietly drift.
"""

from __future__ import annotations

import pytest

from valwr.rating import roles

ROLES = ("Duelist", "Controller", "Initiator", "Sentinel")


@pytest.mark.parametrize("role", ROLES)
def test_every_role_totals_one_hundred(role):
    """Before the map component, each table is a clean 100%.

    Not cosmetic: the tables are meant to be read and argued with as
    percentages, and one that sums to 97 silently makes every component in it
    worth slightly more than the number printed beside it.
    """
    assert sum(abs(v) for v in roles.ROLE_WEIGHTS[role].values()) == 100


@pytest.mark.parametrize("role", ROLES)
def test_weights_are_rescaled_to_make_room_for_the_map(role):
    w = roles.weights_for(role)
    assert w["map_edge"] == pytest.approx(0.05)
    assert sum(abs(v) for v in w.values()) == pytest.approx(1.0)


def test_only_duelists_score_first_bloods():
    """Entering is the Duelist's job. For everyone else a first blood is luck
    or a mistake, and rewarding it would push supports toward playing badly."""
    assert "fb" in roles.ROLE_WEIGHTS["Duelist"]
    for role in ("Controller", "Initiator", "Sentinel"):
        assert "fb" not in roles.ROLE_WEIGHTS[role]


def test_only_sentinels_are_penalised_for_dying_first():
    assert roles.ROLE_WEIGHTS["Sentinel"]["fd"] < 0
    for role in ("Duelist", "Controller", "Initiator"):
        assert "fd" not in roles.ROLE_WEIGHTS[role]


def test_duelists_are_the_most_damage_led_and_controllers_the_least():
    """ACS and ADR correlate 0.98, so they act as one dial. How far that dial
    is turned up is the clearest statement each table makes about its role."""
    damage = {r: roles.ROLE_WEIGHTS[r].get("acs", 0) + roles.ROLE_WEIGHTS[r].get("adr", 0)
              for r in ROLES}
    assert damage["Duelist"] == max(damage.values())
    assert damage["Controller"] == min(damage.values())
    assert damage["Duelist"] >= damage["Controller"] * 1.5


def test_support_roles_lean_on_kast_and_abilities():
    """The two roles the old score failed at are the two whose contribution is
    least visible in fragging stats, so these have to outweigh damage there."""
    for role in ("Controller", "Initiator", "Sentinel"):
        w = roles.ROLE_WEIGHTS[role]
        support = w["kast"] + w["abilities"]
        damage = w.get("acs", 0) + w.get("adr", 0)
        assert support > damage, role


def test_headshots_are_the_smallest_weight_everywhere():
    """HS% tracks winning at 0.01-0.06 and is a share of hits, not of shots --
    it is the noisiest thing in the table and is weighted accordingly."""
    for role in ROLES:
        w = roles.ROLE_WEIGHTS[role]
        assert w["hs"] == min(abs(v) for v in w.values())


def test_chamber_is_scored_between_a_duelist_and_a_sentinel():
    """A Sentinel built around a rifle: 1.67 casts per round against Cypher's
    3.55. Scored as a Sentinel he loses points for playing his own kit."""
    got = roles.weights_for("Sentinel", agent="Chamber")
    duelist = roles.weights_for("Duelist")
    sentinel = roles.weights_for("Sentinel")
    for key in ("acs", "kast", "abilities"):
        lo, hi = sorted((duelist.get(key, 0), sentinel.get(key, 0)))
        assert lo < got[key] < hi, key
    assert got["fb"] > 0 and got["fd"] < 0, "he inherits both roles' extremes"
    assert sum(abs(v) for v in got.values()) == pytest.approx(1.0)


def test_another_sentinel_is_not_affected_by_the_chamber_exception():
    assert roles.weights_for("Sentinel", agent="Cypher") == roles.weights_for("Sentinel")


def test_an_unknown_agent_falls_back_to_a_neutral_set():
    """Agent select: the roster is known before anyone locks in, so a score
    shown then has no role to use. The fallback must still be a valid set
    rather than an empty one, and must not silently become a Duelist."""
    for unknown in (None, "?", "NotAnAgent"):
        w = roles.weights_for(unknown)
        assert sum(abs(v) for v in w.values()) == pytest.approx(1.0)
        assert w != roles.weights_for("Duelist")
        assert set(w) >= {"acs", "kast", "abilities", "map_edge"}


def test_the_neutral_set_carries_every_component_any_role_uses():
    """Averaging the four must not drop the components only one role has, or
    agent select would score nobody on entries or on dying first."""
    assert set(roles.NEUTRAL) == set(roles.components_used()) - {"map_edge"}
    assert roles.NEUTRAL["fd"] < 0 < roles.NEUTRAL["fb"]


def test_every_component_a_role_names_is_one_the_score_knows_about():
    known = set(roles.components_used())
    for role in ROLES:
        assert set(roles.ROLE_WEIGHTS[role]) <= known, role
