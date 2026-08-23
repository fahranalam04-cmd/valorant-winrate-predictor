"""The composite player rating -- this project's replacement for Tracker Score.

tracker.gg's developer programme does not cover VALORANT, so Tracker Score is
unavailable to third-party developers (docs/API-NOTES.md). Rather than work
around that, the rating is built here from raw match data.

It is deliberately a weighted z-score composite rather than a learned rating.
A learned one would likely score marginally better and could not be explained;
being able to say "damage and KAST carry the most weight, because they are the
most robust indicators of round contribution" is worth more than the fraction
of a point it costs.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from valwr.collect.frontier import band_of
from valwr.rating import adjust as adjust_mod
from valwr.rating.components import per_round_rates
from valwr.rating.normalize import Norms

# Weights on z-scored components. Rationale, since these are hand-set and an
# interviewer will reasonably ask:
#
#   adr, kast   highest. Damage is the most robust single indicator of round
#               contribution, and KAST captures consistent participation that
#               raw fragging misses. These are what analysts weight heaviest.
#   acs         correlated with adr, so weighted lower to avoid double-counting
#               the same underlying signal twice.
#   kpr, dpr    direct but noisy; deaths negative.
#   fb, fd      entry impact and its cost. Small: high variance, few per match.
#   trade       rewards playing with teammates rather than alone.
#   multikill   rewards round-swinging rounds specifically.
#   hs_pct      mechanical skill, but the noisiest of the set. Smallest weight.
WEIGHTS = {
    "adr": 0.22,
    "kast": 0.22,
    "acs": 0.12,
    "kpr": 0.10,
    "dpr": -0.10,
    "trade_rate": 0.08,
    "fb_rate": 0.06,
    "fd_rate": -0.05,
    "multikill_rate": 0.05,
    "hs_pct": 0.03,
}

# Centre of the scale. 1.0 means "average for this rank band and map", chosen
# to read like the familiar rating scales rather than a bare z-score.
BASELINE = 1.0
SCALE = 0.30


@dataclass(frozen=True)
class Rating:
    value: float
    components: dict[str, float]
    coverage: int          # how many components had data
    gap: float | None      # opponent rank gap applied

    def __float__(self) -> float:
        return self.value


def rate_performance(row: dict, norms: Norms, gap: float | None = None) -> Rating | None:
    """Rate one player's performance in one match."""
    if not row.get("rounds_played"):
        return None

    band = band_of(row.get("tier"))
    map_name = row.get("map") or "?"
    rates = per_round_rates(row)

    zs: dict[str, float] = {}
    weighted = 0.0
    weight_used = 0.0
    for comp, w in WEIGHTS.items():
        z = norms.z(comp, rates.get(comp), band, map_name)
        if z is None:
            continue
        # Clamp: a single freak match should not dominate a career average.
        z = max(-4.0, min(4.0, z))
        zs[comp] = z
        weighted += w * z
        weight_used += abs(w)

    if not weight_used:
        return None

    # Rescale by the weight actually available, so a row missing a component
    # is not silently penalised.
    normalised = weighted / weight_used * sum(abs(w) for w in WEIGHTS.values())
    value = adjust_mod.adjust(BASELINE + SCALE * normalised, gap)
    return Rating(value=value, components=zs, coverage=len(zs), gap=gap)


def rate_player_history(conn: sqlite3.Connection, puuid: str, as_of: int,
                        norms: Norms, limit: int | None = None) -> list[Rating]:
    """Rate every match this player played before `as_of`.

    Goes through the temporal layer, so the rating is subject to the same
    time gating as every other feature.
    """
    from valwr.store import temporal
    out = []
    for row in temporal.player_history(conn, puuid, as_of, limit=limit):
        r = rate_performance(dict(row), norms)
        if r is not None:
            out.append(r)
    return out


def average_rating(conn: sqlite3.Connection, puuid: str, as_of: int,
                   norms: Norms, limit: int | None = None) -> float | None:
    ratings = rate_player_history(conn, puuid, as_of, norms, limit=limit)
    return sum(r.value for r in ratings) / len(ratings) if ratings else None
