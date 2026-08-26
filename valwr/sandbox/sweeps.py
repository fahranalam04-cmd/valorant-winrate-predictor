"""Generated coverage: single-factor sweeps, pairwise grids, context sweeps.

The curated catalog in scenarios.py is human-readable but hand-written, so it
can fall behind the model. This module is generated *from* the production
feature lists -- `features.player.FEATURE_NAMES` and `features.team` -- which
means a newly added feature is swept automatically and, if nobody can say how
to vary it, fails the completeness test rather than passing unnoticed.

There is exactly one authoritative feature list and it lives in production.
Nothing here keeps a copy.
"""

from __future__ import annotations

from typing import Callable

from valwr.features import player as pf
from valwr.features import team as tf
from valwr.sandbox import profiles as P
from valwr.sandbox.scenarios import BALANCED, scenario, team
from valwr.sandbox.schema import MatchScenario, PlayerProfile, TeamProfile

LEVELS = ("very_low", "low", "neutral", "high", "very_high")

Knob = Callable[[PlayerProfile, str], PlayerProfile]


def _repair_counts(p: PlayerProfile) -> PlayerProfile:
    """Keep the count hierarchy consistent after a knob moves one of them.

    map_games and agent_games are subsets of games, and map_agent_games is a
    subset of both. Sweeping `games` down to 2 while leaving map_games at 12
    describes an impossible player, so the dependents are clamped rather than
    left to fail validation.
    """
    games = max(0, p.games)
    map_games = min(games, p.map_games)
    agent_games = min(games, p.agent_games)
    combo = min(map_games, agent_games, p.map_agent_games)
    if (map_games, agent_games, combo) == (p.map_games, p.agent_games,
                                           p.map_agent_games):
        return p
    return p.with_(map_games=map_games, agent_games=agent_games,
                   map_agent_games=combo)


def _numeric(field: str, values: dict[str, float], integer: bool = False) -> Knob:
    def apply(p: PlayerProfile, level: str) -> PlayerProfile:
        value = values[level]
        moved = p.with_(**{field: int(value) if integer else value},
                        name=f"{p.name}_{field}_{level}")
        return _repair_counts(moved)
    return apply


def _combo_games(values: dict[str, float]) -> Knob:
    """map_agent_games is a subset of both parents, so raising it may need
    them raised first -- clamping alone would silently cap the sweep."""
    def apply(p: PlayerProfile, level: str) -> PlayerProfile:
        want = int(values[level])
        parents = max(p.map_games, want), max(p.agent_games, want)
        moved = p.with_(map_games=min(p.games, parents[0]),
                        agent_games=min(p.games, parents[1]),
                        map_agent_games=want,
                        name=f"{p.name}_map_agent_games_{level}")
        return _repair_counts(moved)
    return apply


def _overall_wr(values: dict[str, float]) -> Knob:
    """Move the overall win rate, with every subset rate tracking it.

    Rate features are nested: map games are a subset of all games. Setting the
    overall rate while pinning the subsets forces the complement cells to the
    boundary, so what looks like a clean single-factor sweep is really
    "better overall AND catastrophically worse off-map".

    A player who simply wins more wins more everywhere, so all the rates move
    together. This is as close to isolating overall win rate as nested rates
    permit.
    """
    def apply(p: PlayerProfile, level: str) -> PlayerProfile:
        v = values[level]
        return p.with_(win_rate=v, map_win_rate=v, agent_win_rate=v,
                       map_agent_win_rate=v,
                       name=f"{p.name}_wr_{level}")
    return apply


def _subset_wr(field: str, subset_attr: str, values: dict[str, float]) -> Knob:
    """Move one subset rate, holding the complement neutral.

    The overall rate is then *derived* rather than pinned, for the same reason
    as above -- pinning it would push the complement to 0 or 1 and the sweep
    would measure that instead.
    """
    def apply(p: PlayerProfile, level: str) -> PlayerProfile:
        v = values[level]
        n = getattr(p, subset_attr)
        overall = (v * n + 0.50 * (p.games - n)) / p.games if p.games else 0.5
        return p.with_(**{field: v}, win_rate=min(1.0, max(0.0, overall)),
                       name=f"{p.name}_{field}_{level}")
    return apply


def _rating(p: PlayerProfile, level: str) -> PlayerProfile:
    """The rating is a composite, so it is moved through its own inputs.

    Nudging a `rating` field directly would bypass the rating pipeline, which
    is the thing worth exercising.
    """
    shift = {"very_low": -2.0, "low": -1.0, "neutral": 0.0,
             "high": 1.0, "very_high": 2.0}[level]
    return p.with_(acs=P.POP_ACS + 40.0 * shift, adr=P.POP_ADR + 26.0 * shift,
                   kast=min(0.95, max(0.35, P.POP_KAST + 0.045 * shift)),
                   name=f"{p.name}_rating_{level}")


def _off_role(p: PlayerProfile, level: str) -> PlayerProfile:
    """Off-role is categorical: the player's usual role versus their pick."""
    off = level in ("high", "very_high")
    return p.with_(role="Sentinel" if off else "Duelist",
                   name=f"{p.name}_offrole_{level}")


# One knob per production player feature. A feature with no entry here fails
# the completeness test -- that is the mechanism that stops the sandbox
# quietly falling behind the model.
KNOBS: dict[str, Knob] = {
    "games_played": _numeric("games", {"very_low": 2, "low": 15, "neutral": 60,
                                       "high": 200, "very_high": 600}, True),
    "rating_n": _numeric("games", {"very_low": 2, "low": 15, "neutral": 60,
                                   "high": 200, "very_high": 600}, True),
    "tier": _numeric("tier", {"very_low": 3, "low": 9, "neutral": 15,
                              "high": 21, "very_high": 27}, True),
    "account_level": _numeric("account_level",
                              {"very_low": 5, "low": 40, "neutral": 120,
                               "high": 500, "very_high": 1500}, True),
    "wr": _overall_wr({"very_low": 0.30, "low": 0.42, "neutral": 0.50,
                       "high": 0.58, "very_high": 0.70}),
    "wr_map": _subset_wr("map_win_rate", "map_games",
                         {"very_low": 0.25, "low": 0.40, "neutral": 0.50,
                          "high": 0.60, "very_high": 0.75}),
    "games_map": _numeric("map_games", {"very_low": 0, "low": 4, "neutral": 12,
                                        "high": 30, "very_high": 55}, True),
    "wr_agent": _subset_wr("agent_win_rate", "agent_games",
                           {"very_low": 0.25, "low": 0.40, "neutral": 0.50,
                            "high": 0.60, "very_high": 0.75}),
    "games_agent": _numeric("agent_games", {"very_low": 0, "low": 6,
                                            "neutral": 20, "high": 40,
                                            "very_high": 58}, True),
    "wr_map_agent": _subset_wr("map_agent_win_rate", "map_agent_games",
                               {"very_low": 0.20, "low": 0.38, "neutral": 0.50,
                                "high": 0.62, "very_high": 0.80}),
    "games_map_agent": _combo_games({"very_low": 0, "low": 2, "neutral": 5,
                                     "high": 9, "very_high": 12}),
    "wr_recent": _numeric("recent_win_rate", {"very_low": 0.20, "low": 0.35,
                                              "neutral": 0.50, "high": 0.65,
                                              "very_high": 0.85}),
    "rating": _rating,
    "acs": _numeric("acs", {"very_low": 130.0, "low": 175.0, "neutral": 212.0,
                            "high": 255.0, "very_high": 305.0}),
    "adr": _numeric("adr", {"very_low": 90.0, "low": 118.0, "neutral": 141.0,
                            "high": 168.0, "very_high": 200.0}),
    "kast": _numeric("kast", {"very_low": 0.52, "low": 0.62, "neutral": 0.712,
                              "high": 0.79, "very_high": 0.88}),
    "fb_rate": _numeric("fb_rate", {"very_low": 0.02, "low": 0.06,
                                    "neutral": 0.099, "high": 0.15,
                                    "very_high": 0.24}),
    "fd_rate": _numeric("fd_rate", {"very_low": 0.02, "low": 0.06,
                                    "neutral": 0.099, "high": 0.15,
                                    "very_high": 0.24}),
    "rating_trend": _numeric("trend", {"very_low": -0.6, "low": -0.25,
                                       "neutral": 0.0, "high": 0.25,
                                       "very_high": 0.6}),
    "days_since_last": _numeric("days_since_last",
                                {"very_low": 0.0, "low": 1.0, "neutral": 4.0,
                                 "high": 30.0, "very_high": 200.0}),
    "off_role": _off_role,
}

# Team-level features that have no single-player knob because they are
# properties of the roster, not of a player. Each names the scenario category
# that exercises it -- an allow-list with reasons, never a silent pass.
ROSTER_LEVEL: dict[str, str] = {
    "n_duelist": "composition", "n_controller": "composition",
    "n_initiator": "composition", "n_sentinel": "composition",
    "has_duelist": "composition", "has_controller": "composition",
    "has_initiator": "composition", "has_sentinel": "composition",
    "role_balance": "composition", "n_off_role": "off_role",
    "max_party": "party", "n_parties": "party", "n_grouped": "party",
    "n_with_history": "coverage",
}


def single_factor(base: PlayerProfile | None = None) -> list[MatchScenario]:
    """One scenario per (feature, level), varying Team A only.

    Team B is held at the neutral profile throughout, so any movement is
    attributable to the single feature under test.
    """
    base = base or P.AVERAGE
    out: list[MatchScenario] = []
    for feature in pf.FEATURE_NAMES:
        knob = KNOBS.get(feature)
        if knob is None:
            continue                      # caught by the completeness test
        for level in LEVELS:
            varied = knob(base, level)
            agents = BALANCED
            if feature == "off_role":
                # Everyone picks a Duelist; the knob changes their usual role.
                agents = ("Jett", "Raze", "Phoenix", "Reyna", "Neon")
            out.append(scenario(
                name=f"sweep__{feature}__{level}",
                category="sweep",
                description=f"{feature} at {level}, everything else neutral",
                a=team(varied, agents=agents), b=team(base, agents=agents),
                tags=("generated", "sweep", feature)))
    return out


# Curated pairs, not a Cartesian product. Each crosses two families whose
# interaction is plausible and worth seeing.
PAIRS: tuple[tuple[str, str], ...] = (
    ("tier", "rating"),
    ("rating", "wr_map"),
    ("wr_map", "wr_agent"),
    ("wr", "wr_recent"),
    ("rating", "games_played"),
    ("acs", "kast"),
    ("wr_map_agent", "games_map_agent"),
    ("rating", "days_since_last"),
)

GRID = ("very_low", "neutral", "very_high")


def pairwise(base: PlayerProfile | None = None) -> list[MatchScenario]:
    base = base or P.AVERAGE
    out: list[MatchScenario] = []
    for left, right in PAIRS:
        kl, kr = KNOBS.get(left), KNOBS.get(right)
        if not kl or not kr:
            continue
        for a_level in GRID:
            for b_level in GRID:
                varied = kr(kl(base, a_level), b_level)
                out.append(scenario(
                    name=f"grid__{left}_{a_level}__{right}_{b_level}",
                    category="pairwise",
                    description=f"{left}={a_level}, {right}={b_level}",
                    a=team(varied), b=team(base),
                    tags=("generated", "pairwise", left, right)))
    return out


def context_sweep(maps: tuple[str, ...] | None = None) -> list[MatchScenario]:
    """One fixed roster across every map, to expose context sensitivity."""
    from valwr.sandbox.scenarios import MAPS
    out = []
    for map_name in (maps or MAPS):
        out.append(scenario(
            name=f"context__map_{map_name.lower()}",
            category="context",
            description=f"An identical roster played on {map_name}",
            a=team(P.AVERAGE), b=team(P.AVERAGE), map_name=map_name,
            expect="even", tags=("generated", "context")))
    return out


def generated() -> list[MatchScenario]:
    return single_factor() + pairwise() + context_sweep()


def covered_features() -> set[str]:
    """Every production team feature the sandbox can actually move.

    A player-level knob covers each team aggregate derived from it, because
    moving the player statistic necessarily moves its mean/max/min/spread.
    """
    covered: set[str] = set(ROSTER_LEVEL)
    for feature in KNOBS:
        if feature in tf.SPREAD_FEATURES:
            covered.update(f"{feature}_{how}"
                           for how in ("mean", "max", "min", "std"))
        covered.add(f"{feature}_mean")
    return {name for name in tf.feature_names() if name in covered}


def missing_features() -> set[str]:
    """Production features the sandbox cannot exercise. Must be empty."""
    return set(tf.feature_names()) - covered_features()
