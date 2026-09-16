"""Per-role player components, measured from history before the match.

This is the measurement half of the per-role 0-100 score specified in
`docs/SCORE-SPEC.md`. It turns a player's past matches into the ten values the
role weight tables in `rating/roles.py` are written against. Turning those into
a score needs a fitted population, which lives with the index; nothing here
knows what "good" is.

Three things make this more than a set of averages.

**Sums, not averages of ratios.** Every component is a weighted sum over a
weighted denominator -- kills over deaths, combat score over rounds -- never
the mean of per-match ratios. A player with one match and one death otherwise
carries a career K/D of 14 into the average and outranks everyone.

**Role history is nearly always thin.** 91% of player-role pairs in the
database have fewer than five games on that role; the median is one. A formula
that needed role history would have nothing to work with, so each component
blends what the player did *on this role* with what they did overall, weighted
by how much role history exists.

**Ability casts have their own denominator.** They were not stored until
recently, so a player's older matches have NULL where their newer ones have
counts. Dividing recovered casts by every round they have ever played would
quietly halve the ability component for anyone with history on both sides of
that line.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from valwr.store import temporal

# Components built as a weighted count over weighted rounds.
PER_ROUND = ("acs", "adr", "assists", "kast", "fb", "fd", "abilities")
# Components built as a weighted count over weighted deaths.
PER_DEATH = ("kd", "kda")
# Headshots over hits, which is its own denominator again.
SHARES = ("hs",)

COMPONENTS = PER_ROUND + PER_DEATH + SHARES

# Where each component's numerator and denominator come from in a history row.
NUMERATOR = {
    "acs": ("score",), "adr": ("damage_dealt",), "assists": ("assists",),
    "kast": ("kast_rounds",), "fb": ("first_bloods",), "fd": ("first_deaths",),
    "kd": ("kills",), "kda": ("kills", "assists"), "hs": ("headshots",),
}
ABILITY_SLOTS = ("ability_grenade", "ability_1", "ability_2", "ability_ultimate")

RECENCY_HALFLIFE_DAYS = 30.0     # matches features/player.py and potential.py

# At this many games on the role, the role-specific figure carries half the
# weight; by twenty it carries 80%. Chosen in the spec rather than fitted --
# there is no target to fit it against that is not itself the thing being
# measured.
ROLE_BLEND_GAMES = 5.0

# Shrinkage toward the role average, in units of matches. Same strength the
# previous score used, and for the same reason: 53% of players in this database
# have fewer than five prior matches, and one good game should not put them at
# the top of a lobby.
PRIOR_GAMES = 4.0

# Below this many games on the map, the map component is exactly zero -- no
# opinion rather than a shrunk guess. Lowered from six, which fired for 1.3% of
# players and was therefore indistinguishable from not existing.
MIN_MAP_GAMES = 3


def _decay(as_of: int, started_at: int | None) -> float:
    if not started_at:
        return 1.0
    days = max(0.0, (as_of - started_at) / 86400.0)
    return 0.5 ** (days / RECENCY_HALFLIFE_DAYS)


@dataclass
class _Sums:
    """Weighted numerators and denominators, accumulated over matches."""
    num: dict[str, float] = field(default_factory=dict)
    rounds: float = 0.0
    deaths: float = 0.0
    hits: float = 0.0
    ability_rounds: float = 0.0     # rounds from matches that recorded casts
    games: int = 0
    ability_games: int = 0

    def add(self, row: dict, w: float) -> None:
        rounds = row.get("rounds_played") or 0
        if not rounds:
            return
        self.games += 1
        self.rounds += w * rounds
        self.deaths += w * (row.get("deaths") or 0)
        self.hits += w * sum((row.get(k) or 0) for k in
                             ("headshots", "bodyshots", "legshots"))
        for name, cols in NUMERATOR.items():
            self.num[name] = self.num.get(name, 0.0) + w * sum(
                (row.get(c) or 0) for c in cols)
        # Only matches that actually recorded casts contribute, numerator and
        # denominator together.
        if any(row.get(k) is not None for k in ABILITY_SLOTS):
            self.ability_games += 1
            self.ability_rounds += w * rounds
            self.num["abilities"] = self.num.get("abilities", 0.0) + w * sum(
                (row.get(k) or 0) for k in ABILITY_SLOTS)

    def value(self, name: str) -> float | None:
        """One component, or None when its denominator is empty."""
        if name == "abilities":
            return (self.num.get("abilities", 0.0) / self.ability_rounds
                    if self.ability_rounds else None)
        if name in PER_ROUND:
            return self.num.get(name, 0.0) / self.rounds if self.rounds else None
        if name in PER_DEATH:
            return self.num.get(name, 0.0) / self.deaths if self.deaths else None
        if name == "hs":
            return self.num.get("hs", 0.0) / self.hits if self.hits else None
        raise KeyError(name)

    def count(self, name: str) -> int:
        """Matches behind a component -- not the same for abilities."""
        return self.ability_games if name == "abilities" else self.games


@dataclass(frozen=True)
class RoleComponents:
    """What a player brings to a match, in raw units, by role."""
    role: str | None
    agent: str | None
    values: dict[str, float | None]
    counts: dict[str, int]
    map_edge: float
    n_games: int
    n_role_games: int
    n_map_games: int
    n_ability_games: int
    tier: int | None
    account_level: int | None

    @property
    def thin(self) -> bool:
        """Too little history for the number to carry much weight."""
        return self.n_games < 5


def _blend(role_sums: _Sums, all_sums: _Sums, name: str) -> tuple[float | None, int]:
    """This role's figure and the player's overall one, mixed by evidence."""
    on_role, overall = role_sums.value(name), all_sums.value(name)
    if on_role is None:
        return overall, all_sums.count(name)
    if overall is None:
        return on_role, role_sums.count(name)
    n = role_sums.count(name)
    w = n / (n + ROLE_BLEND_GAMES)
    return w * on_role + (1 - w) * overall, all_sums.count(name)


def _shrink(value: float | None, n: int, prior: float | None) -> float | None:
    """Pull a thin sample toward the role average."""
    if value is None:
        return None
    if prior is None or n <= 0:
        return value
    return (value * n + prior * PRIOR_GAMES) / (n + PRIOR_GAMES)


def measure(conn: sqlite3.Connection, puuid: str, as_of: int, map_name: str,
            role: str | None, roles_by_agent: dict[str, str],
            role_means: dict[str, float] | None = None,
            agent: str | None = None) -> RoleComponents | None:
    """Components from this player's history strictly before `as_of`.

    None for a player with no history at all. That is deliberate: a score
    invented for someone we know nothing about is the failure this project
    keeps guarding against, and the caller shows "--" instead.

    `role_means` shrinks thin players toward their role's average. Passing None
    returns unshrunk values, which is what fitting the population needs -- the
    average cannot be built out of numbers that were already pulled toward it.
    """
    history = temporal.player_history(conn, puuid, as_of)
    if not history:
        return None

    all_sums, role_sums = _Sums(), _Sums()
    map_weight = map_rounds = map_score = 0.0
    n_map = 0
    for raw in history:
        row = dict(raw)
        w = _decay(as_of, row.get("started_at"))
        all_sums.add(row, w)
        if role and roles_by_agent.get(row.get("agent") or "") == role:
            role_sums.add(row, w)
        if map_name and row.get("map") == map_name and (row.get("rounds_played") or 0):
            n_map += 1
            map_rounds += w * row["rounds_played"]
            map_score += w * (row.get("score") or 0)
            map_weight += w

    values: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for name in COMPONENTS:
        blended, n = _blend(role_sums, all_sums, name)
        counts[name] = n
        values[name] = _shrink(blended, n,
                               (role_means or {}).get(name) if role_means else None)

    # The map component is a difference against the player's own level, so it
    # is zero by construction for someone with no map history -- "no opinion",
    # never "average". Combat score is the measure because it is the one
    # component present on every row ever stored.
    map_edge = 0.0
    if n_map >= MIN_MAP_GAMES and map_rounds and all_sums.rounds:
        overall_acs = all_sums.num.get("acs", 0.0) / all_sums.rounds
        map_edge = (map_score / map_rounds) - overall_acs

    newest = dict(history[0])
    return RoleComponents(
        role=role, agent=agent, values=values, counts=counts, map_edge=map_edge,
        n_games=all_sums.games, n_role_games=role_sums.games, n_map_games=n_map,
        n_ability_games=all_sums.ability_games,
        tier=newest.get("tier"), account_level=newest.get("account_level"))


def roles_by_agent(conn: sqlite3.Connection) -> dict[str, str]:
    """Agent name -> role, from ref_agents."""
    return {r["name"]: r["role"] for r in
            conn.execute("SELECT name, role FROM ref_agents")}
