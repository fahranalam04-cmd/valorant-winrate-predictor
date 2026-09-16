"""The population the per-role score is measured against.

`role_score.measure` says what a player did. This says what that is worth, and
it is three separate reference points rather than one:

**Rank band, for most components.** 250 ACS in Iron and 250 in Immortal are not
the same performance, so every component is a z-score inside the player's own
band, shrunk toward the global figure where a band is thin. Note that the band
is *not* split by role: a Sentinel's ACS is compared with everyone in the band,
which leaves every Sentinel with a systematically negative ACS z-score. That is
intended. The percentile step below is taken within the role, so a level
difference shared by every Sentinel cancels there.

**The agent, for abilities.** Kits are not comparable: Cypher casts 3.55 per
round, Chamber 1.67, and they are the same role. Scored against anything wider
than the agent, the ability component would reward picking Cypher. Agents with
too few samples fall back to their role, and the fallback is recorded rather
than hidden.

**The role, for the final percentile.** A score of 82 means "better than 82% of
the Sentinels", so a Sentinel's 82 and a Duelist's 82 mean the same thing. The
old single population could not claim that: it scored Duelists higher for being
Duelists.

Fitted on the training period only, like every other norm in this project. A
reference built from data the model will later be judged on is a leak.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from valwr.collect.frontier import band_of
from valwr.rating import roles
from valwr.rating.normalize import Moments
from valwr.rating.role_score import COMPONENTS, MIN_MAP_GAMES

INDEX_PATH = Path("models") / "role_index.json"

# How hard a thin band is pulled toward the global figure is owned by
# rating/normalize.py -- `Moments.shrunk_toward` applies its PRIOR_WEIGHT, and
# duplicating the constant here would create a second dial that does nothing.

# Below this many samples, an agent's own ability distribution is itself noise
# and the role's is the safer of two imperfect references.
MIN_AGENT_SAMPLE = 300


@dataclass(frozen=True)
class RoleIndex:
    """Reference population, fitted on the training period."""
    # component -> Moments, globally and per rank band
    glob: dict[str, Moments]
    by_band: dict[str, dict[int, Moments]]
    # ability casts per round, per agent and per role
    by_agent: dict[str, Moments]
    ability_by_role: dict[str, Moments]
    # raw-unit averages a thin player is shrunk toward
    role_means: dict[str, dict[str, float]]
    # sorted composites, one list per role, for the percentile
    quantiles: dict[str, list[float]]
    as_of: int = 0
    n: int = 0
    # agents that fell back to their role's ability distribution
    fell_back: tuple[str, ...] = ()
    # How often this index picks the best player out of five, measured on
    # held-out data by tools/compare_role_score.py --write-index. It travels
    # with the index that produced it for the same reason the old one does:
    # the live view used to print its accuracy as a literal, which went
    # silently wrong at the next retrain with nothing to catch it. None means
    # "not measured yet", and the renderers then say nothing rather than
    # quoting a figure that belongs to a different score.
    top1_rate: float | None = None

    # --- reading -------------------------------------------------------
    def z(self, name: str, value: float | None, band: int) -> float | None:
        """Where `value` sits in its rank band, shrunk toward the population."""
        if value is None:
            return None
        g = self.glob.get(name)
        if g is None or not g.n:
            return None
        mean, std = self.by_band.get(name, {}).get(band, Moments()).shrunk_toward(g)
        if std <= 1e-9:
            return 0.0
        return (value - mean) / std

    def z_ability(self, value: float | None, agent: str | None,
                  role: str | None) -> float | None:
        """Ability casts, against the same agent -- never anything wider."""
        if value is None:
            return None
        cell = self.by_agent.get(agent or "")
        if cell is None or cell.n < MIN_AGENT_SAMPLE:
            cell = self.ability_by_role.get(role or "")
        if cell is None or not cell.n or cell.std <= 1e-9:
            return None
        return (value - cell.mean) / cell.std

    def composite(self, comps, weights: dict[str, float]) -> float | None:
        """Weighted z-scores, rescaled by the weight actually available.

        A player missing a component -- no recorded casts, no map history -- is
        scored on what is known rather than penalised for the gap, the same way
        `rating/rating.py` handles a missing input.
        """
        band = band_of(comps.tier)
        total = sum(abs(w) for w in weights.values())
        acc = used = 0.0
        for name, w in weights.items():
            if name == "abilities":
                z = self.z_ability(comps.values.get("abilities"), comps.agent,
                                   comps.role)
            elif name == "map_edge":
                # Below the gate the map is dropped, not scored as average.
                # It is the difference between "no opinion" and "ordinary
                # here" -- and since the gate is the common case, its weight
                # has to be rescaled away rather than quietly consumed: a
                # player with nothing measurable at all would otherwise come
                # back as 0.0, which reads as a real and very bad number.
                z = (self.z("map_edge", comps.map_edge, band)
                     if comps.n_map_games >= MIN_MAP_GAMES else None)
            else:
                z = self.z(name, comps.values.get(name), band)
            if z is None:
                continue
            acc += w * z
            used += abs(w)
        if used <= 0:
            return None
        return acc / used * total

    def percentile(self, raw: float | None, role: str | None) -> int | None:
        """Where `raw` sits among players of the same role, 0-100."""
        if raw is None:
            return None
        qs = self.quantiles.get(role or "") or self.quantiles.get("__all__")
        if not qs:
            return None
        import bisect
        i = bisect.bisect_left(qs, raw)
        return max(0, min(100, round(100.0 * i / len(qs))))

    def score(self, comps) -> int | None:
        """The 0-100 a player gets, end to end."""
        weights = roles.weights_for(comps.role, comps.agent)
        return self.percentile(self.composite(comps, weights), comps.role)

    # --- storage -------------------------------------------------------
    def to_json(self) -> str:
        def pack(m: Moments) -> list:
            return [m.n, m.total, m.total_sq]
        return json.dumps({
            "glob": {k: pack(v) for k, v in self.glob.items()},
            "by_band": {k: {str(b): pack(m) for b, m in v.items()}
                        for k, v in self.by_band.items()},
            "by_agent": {k: pack(v) for k, v in self.by_agent.items()},
            "ability_by_role": {k: pack(v) for k, v in self.ability_by_role.items()},
            "role_means": self.role_means,
            "quantiles": self.quantiles,
            "as_of": self.as_of, "n": self.n,
            "fell_back": list(self.fell_back),
            "top1_rate": self.top1_rate,
        })

    @classmethod
    def load(cls, path: Path | None = None) -> "RoleIndex":
        raw = json.loads((path or INDEX_PATH).read_text(encoding="utf-8"))

        def un(v) -> Moments:
            return Moments(n=v[0], total=v[1], total_sq=v[2])
        return cls(
            glob={k: un(v) for k, v in raw["glob"].items()},
            by_band={k: {int(b): un(m) for b, m in v.items()}
                     for k, v in raw["by_band"].items()},
            by_agent={k: un(v) for k, v in raw["by_agent"].items()},
            ability_by_role={k: un(v) for k, v in raw["ability_by_role"].items()},
            role_means=raw["role_means"], quantiles=raw["quantiles"],
            as_of=raw.get("as_of", 0), n=raw.get("n", 0),
            fell_back=tuple(raw.get("fell_back", ())),
            top1_rate=raw.get("top1_rate"))


def fit(samples: list, as_of: int = 0) -> RoleIndex:
    """Build an index from measured components.

    `samples` are `RoleComponents` from the training period, measured with
    `role_means=None` -- the average cannot be built out of values that were
    already shrunk toward it.
    """
    glob: dict[str, Moments] = {}
    by_band: dict[str, dict[int, Moments]] = {}
    by_agent: dict[str, Moments] = {}
    ability_by_role: dict[str, Moments] = {}
    role_values: dict[str, dict[str, list[float]]] = {}

    for c in samples:
        band = band_of(c.tier)
        for name in COMPONENTS:
            v = c.values.get(name)
            if v is None:
                continue
            glob.setdefault(name, Moments()).add(v)
            by_band.setdefault(name, {}).setdefault(band, Moments()).add(v)
            role_values.setdefault(c.role or "?", {}).setdefault(name, []).append(v)
        # The map component is a difference against the player's own level, so
        # zero is its true centre by construction. Only gate-clearing samples
        # define its spread; the rest are exactly zero and would collapse it.
        if c.n_map_games >= MIN_MAP_GAMES:
            glob.setdefault("map_edge", Moments()).add(c.map_edge)
            by_band.setdefault("map_edge", {}).setdefault(band, Moments()).add(c.map_edge)
        casts = c.values.get("abilities")
        if casts is not None and c.agent:
            by_agent.setdefault(c.agent, Moments()).add(casts)
            ability_by_role.setdefault(c.role or "?", Moments()).add(casts)

    role_means = {role: {k: sum(v) / len(v) for k, v in comps.items()}
                  for role, comps in role_values.items()}
    fell_back = tuple(sorted(a for a, m in by_agent.items()
                             if m.n < MIN_AGENT_SAMPLE))

    index = RoleIndex(glob=glob, by_band=by_band, by_agent=by_agent,
                      ability_by_role=ability_by_role, role_means=role_means,
                      quantiles={}, as_of=as_of, n=len(samples),
                      fell_back=fell_back)

    # Percentiles come last: they are taken over the composites this same index
    # produces, so the index has to exist before they can be measured.
    quantiles: dict[str, list[float]] = {}
    for c in samples:
        raw = index.composite(c, roles.weights_for(c.role, c.agent))
        if raw is None:
            continue
        quantiles.setdefault(c.role or "?", []).append(raw)
        quantiles.setdefault("__all__", []).append(raw)
    for qs in quantiles.values():
        qs.sort()

    return RoleIndex(glob=glob, by_band=by_band, by_agent=by_agent,
                     ability_by_role=ability_by_role, role_means=role_means,
                     quantiles=quantiles, as_of=as_of, n=len(samples),
                     fell_back=fell_back)
