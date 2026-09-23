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

from valwr.collect.frontier import band_of
from valwr.rating import roles
from valwr.rating.potential import _band as potential_band
from valwr.store import temporal

# Components built as a weighted count over weighted rounds.
PER_ROUND = ("acs", "adr", "assists", "kast", "fb", "fd", "trades",
             "abilities", "plants", "defuses")
# Components built as a weighted count over weighted deaths.
PER_DEATH = ("kd", "kda")
# Headshots over hits, which is its own denominator again.
SHARES = ("hs",)

COMPONENTS = PER_ROUND + PER_DEATH + SHARES

# Where each component's numerator and denominator come from in a history row.
NUMERATOR = {
    "acs": ("score",), "adr": ("damage_dealt",), "assists": ("assists",),
    "kast": ("kast_rounds",), "fb": ("first_bloods",), "fd": ("first_deaths",),
    "trades": ("trade_kills",),
    "kd": ("kills",), "kda": ("kills", "assists"), "hs": ("headshots",),
    "plants": ("plants",), "defuses": ("defuses",),
}
ABILITY_SLOTS = ("ability_grenade", "ability_1", "ability_2", "ability_ultimate")

# Components whose columns were added after matches were already stored, so a
# row can legitimately hold nothing for them. Each gets its own denominator:
# dividing recovered counts by every round a player has ever played would
# quietly halve the figure for anyone with history on both sides of the line
# where the column appeared.
OPTIONAL: dict[str, tuple[str, ...]] = {
    "abilities": ABILITY_SLOTS,
    "plants": ("plants",),
    "defuses": ("defuses",),
}

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
    # Rounds and matches behind each component that carries its own
    # denominator -- see OPTIONAL.
    opt_rounds: dict[str, float] = field(default_factory=dict)
    opt_games: dict[str, int] = field(default_factory=dict)
    games: int = 0

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
            if name in OPTIONAL:
                continue
            self.num[name] = self.num.get(name, 0.0) + w * sum(
                (row.get(c) or 0) for c in cols)
        # Only matches that actually recorded the column contribute, numerator
        # and denominator together.
        for name, cols in OPTIONAL.items():
            if not any(row.get(c) is not None for c in cols):
                continue
            self.opt_games[name] = self.opt_games.get(name, 0) + 1
            self.opt_rounds[name] = self.opt_rounds.get(name, 0.0) + w * rounds
            self.num[name] = self.num.get(name, 0.0) + w * sum(
                (row.get(c) or 0) for c in cols)

    def value(self, name: str) -> float | None:
        """One component, or None when its denominator is empty."""
        if name in OPTIONAL:
            rounds = self.opt_rounds.get(name, 0.0)
            return self.num.get(name, 0.0) / rounds if rounds else None
        if name in PER_ROUND:
            return self.num.get(name, 0.0) / self.rounds if self.rounds else None
        if name in PER_DEATH:
            return self.num.get(name, 0.0) / self.deaths if self.deaths else None
        if name == "hs":
            return self.num.get("hs", 0.0) / self.hits if self.hits else None
        raise KeyError(name)

    def count(self, name: str) -> int:
        """Matches behind a component. Not the same for every one of them: the
        late-added columns are only present on the rows that recorded them."""
        return (self.opt_games.get(name, 0) if name in OPTIONAL
                else self.games)


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


def shrink_values(values: dict[str, float | None], counts: dict[str, int],
                  role_means: dict[str, float] | None) -> dict[str, float | None]:
    """Apply the thin-history shrinkage to values measured without it.

    Fitting needs both: the role average has to come from raw values, and the
    reference population then has to be built from values shrunk the same way
    a live player's are. Without this the tables describe a distribution the
    score never produces -- which showed up as a 0-100 that never went above
    90 and never below 10.
    """
    if not role_means:
        return dict(values)
    return {k: _shrink(v, counts.get(k, 0), role_means.get(k))
            for k, v in values.items()}


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
    map_rounds = map_damage = 0.0
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
            map_damage += w * (row.get("damage_dealt") or 0)

    values: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for name in COMPONENTS:
        blended, n = _blend(role_sums, all_sums, name)
        counts[name] = n
        values[name] = _shrink(blended, n,
                               (role_means or {}).get(name) if role_means else None)

    # The map component is a difference against the player's own level, so it
    # is zero by construction for someone with no map history -- "no opinion",
    # never "average". Damage per round is the measure: it is on every row ever
    # stored, and it replaced combat score when patch 13.06 removed that from
    # the game. The two correlate 0.98, so the term means what it always did.
    map_edge = 0.0
    if n_map >= MIN_MAP_GAMES and map_rounds and all_sums.rounds:
        overall_adr = all_sums.num.get("adr", 0.0) / all_sums.rounds
        map_edge = (map_damage / map_rounds) - overall_adr

    newest = dict(history[0])
    return RoleComponents(
        role=role, agent=agent, values=values, counts=counts, map_edge=map_edge,
        n_games=all_sums.games, n_role_games=role_sums.games, n_map_games=n_map,
        n_ability_games=all_sums.opt_games.get("abilities", 0),
        tier=newest.get("tier"), account_level=newest.get("account_level"))


def standing(player: dict) -> float | None:
    """How a player is ordered against the rest of the lobby.

    The 0-100 on the card is a percentile *within a role*, which is what makes
    a Sentinel's 70 mean what a Duelist's 70 means. It is the wrong thing to
    sort by: a Duelist is the best player on their team in 43.5% of matches
    and an Initiator in 14.6%, and scoring everyone against their own role
    deliberately erases that difference. Sorting by the percentile therefore
    throws away information the raw composite still has -- measured at 1.4
    points of top-1 accuracy, 28.0% against 29.4%.

    So the page shows the percentile and orders by this. Recorded matches from
    before the raw was stored fall back to the percentile, which is what they
    were ordered by at the time.
    """
    detail = player.get("detail") or {}
    raw = player.get("raw", detail.get("raw"))
    if raw is not None:
        return float(raw)
    score = player.get("score")
    return float(score) if score is not None else None


def roles_by_agent(conn: sqlite3.Connection) -> dict[str, str]:
    """Agent name -> role, from ref_agents."""
    return {r["name"]: r["role"] for r in
            conn.execute("SELECT name, role FROM ref_agents")}


# --- putting the number into words -------------------------------------

COMPONENT_LABELS = {
    "acs": "combat score", "adr": "damage per round", "kd": "kills per death",
    "trades": "trades", "plants": "spike plants", "defuses": "defuses",
    "kda": "kills and assists per death", "kast": "rounds contributed to",
    "assists": "assists per round", "fb": "opening kills",
    "fd": "opening deaths", "hs": "headshot rate",
    "abilities": "ability use", "map_edge": "map adjustment",
}

# What a high and a low value are worth saying about a player. `map_edge` is
# deliberately absent: it is an adjustment against the player's own level, not
# a trait, and narrating it produced sentences nobody could act on.
_HIGH = {
    "trades": "trades out teammates", "plants": "plants the spike",
    "defuses": "defuses under pressure",
    "acs": "high combat score", "adr": "damages every round",
    "kd": "wins duels", "kda": "in on most kills",
    "kast": "contributes nearly every round", "assists": "sets up teammates",
    "fb": "opens rounds", "fd": "dies first often",
    "hs": "headshot-heavy aim", "abilities": "uses their kit heavily",
}
_LOW = {
    "trades": "rarely trades a teammate", "plants": "rarely plants",
    "defuses": "rarely defuses",
    "acs": "low combat score", "adr": "does little damage",
    "kd": "loses duels", "kda": "little involved in kills",
    "kast": "quiet rounds", "assists": "rarely assists",
    "fb": "rarely opens a round", "fd": "rarely dies first",
    "hs": "low headshot rate", "abilities": "uses little utility",
}

# Below this many standard deviations from the population, a component is not
# worth naming -- the same threshold potential.py uses, and for the same
# reason: without it a perfectly average player gets a confident-sounding
# label on the strength of noise.
NOTABLE_Z = 0.5


def explain(index, comps, weights: dict[str, float]) -> str:
    """What is most distinctive about this player, in words.

    Ranked by raw z-score rather than by weighted contribution. Weighting made
    the biggest weight win almost every time, so the column read "high combat
    score" for four players in five -- true, and useless. What is unusual about
    a player is what a teammate wants to know, even when it is not what moved
    their score most.
    """
    zs = {}
    for name in weights:
        if name not in _HIGH:
            continue
        z = (index.z_ability(comps.values.get("abilities"), comps.agent, comps.role)
             if name == "abilities"
             else index.z(name, comps.values.get(name), band_of(comps.tier)))
        if z is not None:
            zs[name] = z
    if not zs:
        return "no history"

    name = max(zs, key=lambda k: abs(zs[k]))
    if abs(zs[name]) < NOTABLE_Z:
        if comps.n_games < 5:
            return f"middle of the pack, on only {comps.n_games} game" + (
                "s" if comps.n_games != 1 else "")
        return "middle of the pack"
    word = (_HIGH if zs[name] > 0 else _LOW)[name]
    if comps.n_games < 5:
        return f"{word}, but only {comps.n_games} game" + (
            "s" if comps.n_games != 1 else "")
    return word


def describe(conn: sqlite3.Connection, puuid: str, as_of: int, map_name: str,
             role: str | None, agent: str | None, roles_by_agent: dict[str, str],
             index) -> dict | None:
    """One player's score and the parts it is built from.

    The score alone is unauditable -- a reader cannot tell whether 68 came from
    consistent play or from three good games. This returns the components
    beside it, each with the weight its role gives it, so the number can be
    checked rather than believed.
    """
    comps = measure(conn, puuid, as_of, map_name, role, roles_by_agent,
                    role_means=index.role_means.get(role or "?"), agent=agent)
    if comps is None:
        return None

    weights = roles.weights_for(role, agent)
    raw = index.composite(comps, weights)
    band = band_of(comps.tier)

    components = []
    for name, w in sorted(weights.items(), key=lambda kv: -abs(kv[1])):
        if name == "abilities":
            value = comps.values.get("abilities")
            z = index.z_ability(value, comps.agent, comps.role)
        elif name == "map_edge":
            value = comps.map_edge
            z = (index.z("map_edge", value, band)
                 if comps.n_map_games >= MIN_MAP_GAMES else None)
        else:
            value = comps.values.get(name)
            z = index.z(name, value, band)
        entry = {
            "key": name,
            "label": COMPONENT_LABELS.get(name, name),
            "value": round(value, 3) if value is not None else None,
            "z": round(z, 2) if z is not None else None,
            # A fraction, like potential.detail emits. The page does not
            # print it today, but two providers of the same field disagreeing
            # about its units is a trap for whoever prints it next.
            "weight": round(w, 4),
            "contribution": round(w * z, 3) if z is not None else 0.0,
            "note": potential_band(z) if z is not None else "not counted",
        }
        if name == "map_edge" and comps.n_map_games < MIN_MAP_GAMES:
            entry["note"] = (f"not counted -- {comps.n_map_games} game"
                             f"{'s' if comps.n_map_games != 1 else ''} on this "
                             f"map, {MIN_MAP_GAMES} needed")
        elif name == "abilities" and value is None:
            entry["note"] = "not counted -- no match recorded their ability use"
        components.append(entry)

    return {
        "score": index.percentile(raw, role),
        "raw": round(raw, 4) if raw is not None else None,
        "reason": explain(index, comps, weights),
        "components": components,
        "role": role,
        "agent": agent,
        # Which role's table produced this, so the card can say so. Chamber is
        # scored on a blend and a reader should not have to guess why his
        # weights differ from Cypher's.
        "weights_from": ("a Duelist/Sentinel blend" if agent in roles.AGENT_BLENDS
                         else role or "no role yet"),
        "role_games": comps.n_role_games,
        "ability_games": comps.n_ability_games,
        # The gate this score actually applied. `potential.detail` fills the
        # card's map block using its own, older threshold, so without this the
        # card can say "not counted -- 4 games, 6 needed" about a map the score
        # did count.
        "map_gate": MIN_MAP_GAMES,
        "map_counts": comps.n_map_games >= MIN_MAP_GAMES,
    }
