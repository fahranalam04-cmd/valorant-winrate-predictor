"""What each role is scored on.

The 0-100 score used to be one formula for everybody: ACS-led, with a composite
rating and K/D behind it. Measured per role, that formula picks a Duelist who
had their team's best game 34.0% of the time and an Initiator 19.5% of the time
-- and 20% is what picking at random does. It is not a score that is modestly
accurate for everyone. It is a score that works for the role whose contribution
shows up in fragging stats, and does nothing for the roles whose does not.

So there is one weight set per role, and the differences are the point:

  Duelist     damage-led, and the only role that scores first bloods. Entering
              is the job; the trade is the cost of doing it.
  Controller  the least damage-driven of the four. KAST and assists carry it,
              because a Controller who never fires a shot can still be the
              reason a site falls.
  Initiator   KAST and (K+A)/D lead. Setting up a kill and taking one are not
              the same contribution, and only one of them shows in K/D.
  Sentinel    like the Initiator, plus the only negative weight in the project:
              a Sentinel who dies first repeatedly is not holding anything.

Weights are hand-set, from `docs/SCORE-SPEC.md`. They are not fitted, and the
spec says plainly why: one match is 84% noise, a score with perfect knowledge
would reach about 38% top-1 against today's 30.5%, and every weighting tried in
the original sweep sat inside the error bars of ACS alone. Fitting these would
be fitting noise. They encode a claim about what each role is *for*, which is
a judgement, and judgements belong in a table a reader can argue with rather
than in coefficients nobody can.
"""

from __future__ import annotations

# Percentages, per role. Each column sums to 100 before the map component is
# folded in (see MAP_WEIGHT). Negative entries are penalties and still consume
# their share of the total.
#
#   acs       combat score per round          kast      KAST rounds / rounds
#   adr       damage per round                assists   assists per round
#   kd        kills / deaths                  fb        first bloods per round
#   kda       (kills + assists) / deaths      fd        first deaths per round
#   hs        headshots / hits                abilities casts per round
ROLE_WEIGHTS: dict[str, dict[str, float]] = {
    "Duelist":    {"acs": 23, "kd": 21, "adr": 16, "kast": 14,
                   "abilities": 11, "fb": 9, "hs": 6},
    "Controller": {"kd": 21, "kast": 20, "abilities": 18, "assists": 14,
                   "acs": 11, "adr": 10, "hs": 6},
    "Initiator":  {"kda": 24, "kast": 23, "abilities": 20, "acs": 14,
                   "adr": 13, "hs": 6},
    "Sentinel":   {"kda": 23, "kast": 20, "abilities": 19, "acs": 15,
                   "adr": 11, "hs": 6, "fd": -6},
}

# The map component sits outside the per-role tables because it is not about
# the role at all -- it is the same question for everyone, and the answer is
# usually "no opinion". 5% of the total, with every other weight scaled to make
# room, so the role tables stay readable as percentages of 100.
MAP_WEIGHT = 5.0

# Agents whose kit does not match their role's profile. Each one needs a stated
# reason, not a hunch, because every entry here is a place the score stops
# being explainable by role alone.
#
# Chamber is a Sentinel who plays like a Duelist: his kit is built around a
# rifle rather than utility, and he casts 1.67 abilities per round against
# Cypher's 3.55. Scored as a Sentinel he is penalised for the way the agent is
# designed to be played.
AGENT_BLENDS: dict[str, tuple[str, str, float]] = {
    "Chamber": ("Duelist", "Sentinel", 0.5),
}


def blend(a: dict[str, float], b: dict[str, float], t: float) -> dict[str, float]:
    """`t` of `a` and the rest of `b`, over the union of their components."""
    return {k: (1 - t) * b.get(k, 0.0) + t * a.get(k, 0.0)
            for k in sorted(set(a) | set(b))}


def _mean(sets: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for s in sets for k in s})
    return {k: sum(s.get(k, 0.0) for s in sets) / len(sets) for k in keys}


# Agent select: the roster exists but nobody has locked in, so there is no role
# to score by. The average of the four is a fallback rather than a fifth
# formula, and the page marks a score built from it as provisional.
NEUTRAL: dict[str, float] = _mean(list(ROLE_WEIGHTS.values()))


def weights_for(role: str | None, agent: str | None = None) -> dict[str, float]:
    """The weight set to score this player with, as fractions of one.

    Includes `map_edge`, and the absolute values sum to 1.0 -- so a component
    that is missing for this player can be dropped and the rest rescaled by
    what is left, the same way `rating/rating.py` handles a missing input.
    """
    table = None
    if agent and agent in AGENT_BLENDS:
        left, right, t = AGENT_BLENDS[agent]
        table = blend(ROLE_WEIGHTS[left], ROLE_WEIGHTS[right], t)
    elif role in ROLE_WEIGHTS:
        table = ROLE_WEIGHTS[role]
    else:
        table = NEUTRAL

    room = (100.0 - MAP_WEIGHT) / 100.0
    out = {k: v / 100.0 * room for k, v in table.items() if v}
    out["map_edge"] = MAP_WEIGHT / 100.0
    return out


def components_used() -> list[str]:
    """Every component name any role scores on, map included."""
    names = {k for w in ROLE_WEIGHTS.values() for k in w}
    return sorted(names | {"map_edge"})
