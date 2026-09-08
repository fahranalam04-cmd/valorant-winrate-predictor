"""The competitive tier table, pinned.

`ranks.py` copies Riot's numbers rather than fetching them, because a live
view should not need a web request to name a rank it already has the integer
for. The cost of copying is that it can silently fall out of step -- Ascendant
was inserted mid-life and shifted everything above it -- and a wrong table is
invisible: Gold renamed Silver still looks like a rank.

So every boundary is asserted. These come from
valorant-api.com/v1/competitivetiers, latest episode set.
"""

from __future__ import annotations

import pytest

from valwr.rating import ranks

# The first and last tier of every division, which is what a shift moves.
BOUNDARIES = {
    0: "Unranked",
    3: "Iron 1", 5: "Iron 3",
    6: "Bronze 1", 8: "Bronze 3",
    9: "Silver 1", 11: "Silver 3",
    12: "Gold 1", 14: "Gold 3",
    15: "Platinum 1", 17: "Platinum 3",
    18: "Diamond 1", 20: "Diamond 3",
    21: "Ascendant 1", 23: "Ascendant 3",
    24: "Immortal 1", 26: "Immortal 3",
    27: "Radiant",
}


@pytest.mark.parametrize("tier,expected", sorted(BOUNDARIES.items()))
def test_every_division_boundary(tier, expected):
    assert ranks.name(tier) == expected


def test_the_table_covers_every_tier_between_iron_and_radiant():
    """A hole would render as "Unranked" for a real rank."""
    for tier in range(3, 28):
        assert ranks.name(tier) != "Unranked", f"tier {tier} is missing"


def test_short_forms_are_unique():
    """They label a column too narrow for the full name, so two ranks sharing
    one would be indistinguishable exactly where it matters."""
    seen = [ranks.short(t) for t in range(3, 28)]
    assert len(set(seen)) == len(seen), "duplicate short form"


def test_divisions_group_as_expected():
    assert ranks.division(16) == "Platinum"
    assert ranks.division(27) == "Radiant"
    assert ranks.division(0) == "Unranked"


def test_unranked_and_unknown_are_not_treated_as_ranked():
    """Tier 0 is genuinely unranked; 1 and 2 are Riot's unused slots and have
    never appeared in this database."""
    for tier in (None, 0, 1, 2):
        assert ranks.describe(tier)["ranked"] is False
    assert ranks.describe(3)["ranked"] is True
