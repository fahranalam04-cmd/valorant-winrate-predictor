"""Who is likely to play well in *this* match, scored 0-100.

`rating.py` scores a performance that already happened. This asks the forward
question -- given everything known before the match starts, how well is this
player likely to do on this map? -- and answers it on the same 0-100 scale for
everyone, so a team can be ranked.

Four components, deliberately chosen:

    rating      the existing composite: opponent-adjusted, normalised within
                rank band and map. The single most informative thing we have.
    acs         average combat score per round.
    kd          kills over deaths, as a pooled ratio.
    map_edge    how much better (or worse) this player is *on this map* than
                they are in general.

`map_edge` is a **delta, not a level**, and that is the important design
choice. An absolute map rating would mostly restate overall skill, which
`rating` already carries, and the two would double-count. The delta isolates
what the map actually adds: a player with no history here scores 0 on it and
is ranked purely on the rest, which is the honest answer rather than a guess.

Everything is shrunk toward a prior by sample size, the same empirical-Bayes
treatment every rate feature in this project gets -- 62% of players in this
dataset have a single prior match, and without shrinkage one lucky game would
outrank a hundred consistent ones.

The 0-100 number is a **percentile against the training population**, not a
rescaled z-score. "82" means "played better than 82% of players", which is what
a reader assumes it means anyway. Breakpoints are fitted on the training period
only, for the same reason the win model fits its norms there.

This module does not touch `features/player.FEATURE_NAMES`. Adding a column
there would change the model's 52 features and force a retrain; this score is
computed alongside the model, never inside it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from valwr.rating.components import per_round_rates
from valwr.rating.normalize import Norms
from valwr.rating.rating import rate_performance
from valwr.store import temporal

# Weights on the z-scored components, chosen on the VALIDATION period by
# tools/validate_potential.py --sweep, never on test.
#
# The sweep's honest finding: ACS alone ranks a team as well as any combination
# tried (30.2% against this weighting's 29.8%, on a standard error of 0.9). The
# components are close to interchangeable for *ranking*, and a first draft that
# led with `rating` scored 29.0% -- worse than the single feature it was built
# on top of.
#
# So this is deliberately ACS-led, and the other three earn their place by
# explaining rather than by ranking: they are what `explain()` turns into "wins
# duels" or "strong on this map", which a bare ACS number cannot say. The cost
# of keeping them is inside noise; the gain is a readable reason.
WEIGHTS = {"acs": 0.45, "rating": 0.25, "kd": 0.15, "map_edge": 0.15}

# Shrinkage strengths, in units of "matches". Map history is thinner than
# overall history, so it is pulled harder toward no-opinion.
PRIOR_N_RATING = 4.0
PRIOR_N_ACS = 4.0
PRIOR_N_KD = 4.0
PRIOR_N_MAP = 6.0

RECENCY_HALFLIFE_DAYS = 30.0    # matches features/player.py
POP_KD = 1.08                   # measured; only a fallback if norms lack it

INDEX_PATH = Path("models") / "perf_index.json"


@dataclass(frozen=True)
class Components:
    """Raw, shrunk component values for one player entering one match."""
    rating: float
    acs: float
    kd: float
    map_edge: float
    n_games: int
    n_map_games: int


def _decay(as_of: int, started_at: int | None) -> float:
    age_days = max(0.0, (as_of - (started_at or as_of)) / 86400.0)
    return 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)


def _weighted(values, weights) -> float | None:
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
    if not pairs:
        return None
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else None


def _shrink(value, n, prior, weight) -> float:
    """Pull a small-sample average toward a prior, by sample size."""
    if value is None:
        return prior
    return (value * n + prior * weight) / (n + weight)


def measure(conn: sqlite3.Connection, puuid: str, as_of: int, map_name: str,
            norms: Norms) -> Components | None:
    """Component values from history strictly before `as_of`.

    Returns None for a player with no history at all. That is deliberate: a
    score invented for someone we know nothing about is exactly the failure
    this project keeps guarding against, so the caller shows "--" instead.
    """
    history = temporal.player_history(conn, puuid, as_of)
    if not history:
        return None

    ratings, accs, weights = [], [], []
    map_ratings, map_weights = [], []
    kills = deaths = 0
    for row in history:
        d = dict(row)
        w = _decay(as_of, d.get("started_at"))
        r = rate_performance(d, norms)
        value = r.value if r is not None else None
        ratings.append(value)
        accs.append(per_round_rates(d).get("acs"))
        weights.append(w)
        kills += d.get("kills") or 0
        deaths += d.get("deaths") or 0
        if map_name and d.get("map") == map_name:
            map_ratings.append(value)
            map_weights.append(w)

    n_rated = sum(1 for r in ratings if r is not None)
    n_map = sum(1 for r in map_ratings if r is not None)

    acs_cell = norms.glob.get("acs")
    acs_prior = acs_cell.mean if acs_cell is not None and acs_cell.n else 0.0

    rating = _shrink(_weighted(ratings, weights), n_rated, 1.0, PRIOR_N_RATING)
    acs = _shrink(_weighted(accs, weights), len(history), acs_prior, PRIOR_N_ACS)
    kd = _shrink(kills / deaths if deaths else None, len(history), POP_KD,
                 PRIOR_N_KD)

    # Shrunk toward *this player's own* rating, so "no history here" means
    # "no opinion" rather than "average player".
    map_mean = _weighted(map_ratings, map_weights)
    map_edge = _shrink(map_mean, n_map, rating, PRIOR_N_MAP) - rating

    return Components(rating=rating, acs=acs, kd=kd, map_edge=map_edge,
                      n_games=len(history), n_map_games=n_map)


@dataclass(frozen=True)
class PerfIndex:
    """Population reference, fitted on the training period only."""
    means: dict[str, float]
    stds: dict[str, float]
    quantiles: list[float]      # sorted composite values
    as_of: int
    n: int

    def z(self, name: str, value: float) -> float:
        std = self.stds.get(name) or 0.0
        if std < 1e-9:
            return 0.0
        return (value - self.means.get(name, 0.0)) / std

    def composite(self, c: Components) -> float:
        return sum(w * self.z(name, getattr(c, name))
                   for name, w in WEIGHTS.items())

    def percentile(self, raw: float) -> int:
        """Where `raw` sits in the training population, 0-100."""
        if not self.quantiles:
            return 50
        import bisect
        i = bisect.bisect_left(self.quantiles, raw)
        return max(0, min(100, round(100.0 * i / len(self.quantiles))))

    def to_json(self) -> str:
        return json.dumps({
            "means": self.means, "stds": self.stds,
            "quantiles": self.quantiles, "as_of": self.as_of, "n": self.n,
        })

    @classmethod
    def load(cls, path: Path | None = None) -> "PerfIndex":
        path = path or INDEX_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"no performance index at {path}; run "
                f"python tools/build_perf_index.py")
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(means=d["means"], stds=d["stds"], quantiles=d["quantiles"],
                   as_of=d["as_of"], n=d["n"])


@dataclass(frozen=True)
class Potential:
    """A player's forward-looking score, plus why."""
    score: int                  # 0-100 percentile
    raw: float                  # composite before the percentile mapping
    components: Components
    reason: str

    @property
    def thin(self) -> bool:
        """Too little history for the number to carry much weight."""
        return self.components.n_games < 5


# Phrasing for whichever component stands out most. Deliberately plain: this
# prints mid-match, where nobody is going to parse a z-score.
_HIGH = {"rating": "consistently strong", "acs": "high combat score",
         "kd": "wins duels", "map_edge": "strong on this map"}
_LOW = {"rating": "below par lately", "acs": "low combat score",
        "kd": "loses duels", "map_edge": "weak on this map"}


def explain(index: PerfIndex, c: Components) -> str:
    """What is most *distinctive* about this player, in words.

    Ranks by raw z-score, deliberately not by weighted contribution. ACS
    carries the largest weight, so weighting made it win almost every time and
    the column read "high combat score" for four players out of five -- true,
    and useless. The unusual thing about a player is what a teammate wants to
    know, even when it is not what moved the score most.
    """
    zs = {name: index.z(name, getattr(c, name)) for name in WEIGHTS}
    name = max(zs, key=lambda k: abs(zs[k]))
    word = (_HIGH if zs[name] > 0 else _LOW)[name]
    if name == "map_edge" and c.n_map_games == 0:
        return "no history on this map"
    if c.n_games < 5:
        return f"{word}, but only {c.n_games} game" + ("s" if c.n_games != 1 else "")
    return word


def evaluate(conn: sqlite3.Connection, puuid: str, as_of: int, map_name: str,
             norms: Norms, index: PerfIndex) -> Potential | None:
    """Score one player for one match, or None if we know nothing about them."""
    c = measure(conn, puuid, as_of, map_name, norms)
    if c is None:
        return None
    raw = index.composite(c)
    return Potential(score=index.percentile(raw), raw=raw, components=c,
                     reason=explain(index, c))
