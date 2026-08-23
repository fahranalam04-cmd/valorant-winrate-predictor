"""Opponent adjustment.

A 250 ACS game against Radiant opponents is worth more than 250 against
Silver. Normalising within rank band (see rating/normalize.py) handles the
player's own bracket, but not the specific lobby: matchmaking is imperfect and
teams are often mismatched.

The adjustment is deliberately simple -- linear in the rank gap between the
opposing team and the player. A more principled treatment would be Elo-like,
or a random-effects model with a per-player latent skill term. That is a
better answer and it is not worth it yet: with a few hundred matches the
linear term is already at the edge of what the data can support, and an
unidentifiable model would look more sophisticated while measuring less.
"""

from __future__ import annotations

import sqlite3

# One full tier of rank gap moves the rating by this much of a standard
# deviation. Calibrated to be modest: the gap is usually small, and
# over-correcting turns the rating into a rank proxy, which defeats the point.
TIER_GAP_COEFFICIENT = 0.06


def team_average_tier(rows: list[sqlite3.Row], team: str) -> float | None:
    tiers = [r["tier"] for r in rows if r["team"] == team and r["tier"]]
    return sum(tiers) / len(tiers) if tiers else None


def opponent_gap(rows: list[sqlite3.Row], puuid: str) -> float | None:
    """Enemy average tier minus this player's tier, in tier steps.

    Positive means the player faced stronger opposition than their own rank.
    """
    me = next((r for r in rows if r["puuid"] == puuid), None)
    if me is None or not me["tier"]:
        return None
    enemy = "Red" if me["team"] == "Blue" else "Blue"
    enemy_avg = team_average_tier(rows, enemy)
    if enemy_avg is None:
        return None
    return enemy_avg - me["tier"]


def adjust(raw_rating: float, gap: float | None) -> float:
    """Reward performance against stronger opposition, discount against weaker."""
    if gap is None:
        return raw_rating
    return raw_rating + TIER_GAP_COEFFICIENT * gap
