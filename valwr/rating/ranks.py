"""Competitive tier numbers, in words.

`match_players.tier` is Riot's integer, 0 to 27, and it is what the model
already uses as a feature. This turns it into the label a player recognises.

The table is Riot's own, from valorant-api.com/v1/competitivetiers, copied here
rather than fetched: it is 28 rows that change about once every two years -- the
last change was adding Ascendant -- and a live view should not depend on a web
request to name a rank it already has the number for. `test_ranks.py` pins the
boundaries, so a shift shows up as a failure rather than as Gold being quietly
renamed Silver.

Tiers 1 and 2 exist in the API as "Unused1" and "Unused2". They have never
appeared in this database and are treated as unranked.
"""

from __future__ import annotations

# tier -> (full name, short form for a narrow column)
_NAMES = {
    0: ("Unranked", "--"),
    3: ("Iron 1", "I1"), 4: ("Iron 2", "I2"), 5: ("Iron 3", "I3"),
    6: ("Bronze 1", "B1"), 7: ("Bronze 2", "B2"), 8: ("Bronze 3", "B3"),
    9: ("Silver 1", "S1"), 10: ("Silver 2", "S2"), 11: ("Silver 3", "S3"),
    12: ("Gold 1", "G1"), 13: ("Gold 2", "G2"), 14: ("Gold 3", "G3"),
    15: ("Platinum 1", "P1"), 16: ("Platinum 2", "P2"), 17: ("Platinum 3", "P3"),
    18: ("Diamond 1", "D1"), 19: ("Diamond 2", "D2"), 20: ("Diamond 3", "D3"),
    21: ("Ascendant 1", "A1"), 22: ("Ascendant 2", "A2"),
    23: ("Ascendant 3", "A3"),
    24: ("Immortal 1", "IM1"), 25: ("Immortal 2", "IM2"),
    26: ("Immortal 3", "IM3"),
    27: ("Radiant", "RAD"),
}

# The division a tier belongs to, for colouring. Radiant is its own.
_DIVISIONS = ("Unranked", "Iron", "Bronze", "Silver", "Gold", "Platinum",
              "Diamond", "Ascendant", "Immortal", "Radiant")


def name(tier: int | None) -> str:
    """`16` -> `Platinum 2`. Unknown or unranked comes back as 'Unranked'."""
    if tier is None:
        return "Unranked"
    return _NAMES.get(int(tier), ("Unranked", "--"))[0]


def short(tier: int | None) -> str:
    """`16` -> `P2`. For the scoreboard column, where there is no room."""
    if tier is None:
        return "--"
    return _NAMES.get(int(tier), ("Unranked", "--"))[1]


def division(tier: int | None) -> str:
    """`16` -> `Platinum`. The band, without the number inside it."""
    return name(tier).rsplit(" ", 1)[0] if name(tier) != "Radiant" else "Radiant"


def describe(tier: int | None) -> dict:
    """Everything a view needs about one rank, as plain data."""
    return {"tier": None if tier is None else int(tier),
            "name": name(tier), "short": short(tier),
            "division": division(tier),
            "ranked": tier is not None and int(tier) >= 3}
