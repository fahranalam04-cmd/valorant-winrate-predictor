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
# Below this many standard deviations from the population, a component is
# not worth naming -- see explain().
NOTABLE_Z = 0.5

# --- the "playing above their rank" flag ------------------------------
# Two conditions, because either alone is noise. A high band-relative rating
# on a level-400 account with 600 games is a good player, not an anomaly --
# that is `rank_only_smurf`, and it must not fire.
FLAG_TARGET = 0.05          # calibrated to fire on ~1 player in 20
DOMINANT = 0.50             # tops the lobby in half their games or more
MIN_DOMINANCE_GAMES = 3     # below this the rate is noise
DEFAULT_FLAG_CUT = 2.0      # used only if the index predates calibration

# The second condition is match-history dominance, not account level. Raced on
# held-out data against a 29.7% base rate, by lift in the top-third rate:
#
#     account level < 100                     -0.2   <- does nothing at all
#     band-relative z >= 0.90                 +9.3
#     z >= 0.90 AND level < 100               +10.0  <- the first version
#     dominance >= 0.50                       +14.4
#     z >= 0.90 AND dominance >= 0.50         +16.5  <- shipped
#     headshot pct >= 0.30                     +3.5
#     tier climb >= 3                          +1.4
#     performance consistency >= 6             +1.4
#
# Account level on its own is worthless as a smurf signal, which is the
# opposite of the intuition it was built on. What an irregular account
# actually looks like is topping the lobby far more often than one in five --
# and that is visible in the match history rather than on the profile.

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
    # Carried for the above-rank flag, not for the score. The score is
    # deliberately blind to rank: a player is good or not regardless of the
    # badge, and `rating` is already normalised within band.
    tier: int | None = None
    account_level: int | None = None
    # Fraction of prior matches finishing top-2 of the ten-player lobby, and
    # how many matches that is over. Chance is 0.20.
    dominance: float = 0.0
    n_dominance: int = 0


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

    # Newest first, so history[0] is the most recent thing we know about them.
    newest = dict(history[0])
    dom, dom_n = temporal.lobby_dominance(conn, puuid, as_of)
    return Components(rating=rating, acs=acs, kd=kd, map_edge=map_edge,
                      n_games=len(history), n_map_games=n_map,
                      tier=newest.get("tier"),
                      account_level=newest.get("account_level"),
                      dominance=dom, n_dominance=dom_n)


@dataclass(frozen=True)
class PerfIndex:
    """Population reference, fitted on the training period only."""
    means: dict[str, float]
    stds: dict[str, float]
    quantiles: list[float]      # sorted composite values
    as_of: int
    n: int
    # z-score of `rating` above which a dominant player is flagged. Calibrated
    # on the training period by tools/build_perf_index.py to hit FLAG_TARGET,
    # rather than guessed -- the rate is the thing worth controlling.
    flag_cut: float = DEFAULT_FLAG_CUT
    # How often this index picks the best player out of five, measured on
    # held-out data by tools/validate_potential.py --write-index. Carried here
    # so the live view can state it without hardcoding a literal that goes
    # silently wrong at the next retrain. None means "not measured yet", and
    # the renderers then say nothing rather than guessing.
    top1_rate: float | None = None

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
            "flag_cut": self.flag_cut, "top1_rate": self.top1_rate,
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
                   as_of=d["as_of"], n=d["n"],
                   flag_cut=d.get("flag_cut", DEFAULT_FLAG_CUT),
                   top1_rate=d.get("top1_rate"))


@dataclass(frozen=True)
class Potential:
    """A player's forward-looking score, plus why."""
    score: int                  # 0-100 percentile
    raw: float                  # composite before the percentile mapping
    components: Components
    reason: str
    flag: "AboveRank | None" = None     # set by evaluate()

    @property
    def thin(self) -> bool:
        """Too little history for the number to carry much weight."""
        return self.components.n_games < 5


# Phrasing for whichever component stands out most. Deliberately plain: this
# prints mid-match, where nobody is going to parse a z-score.
_HIGH = {"rating": "consistently strong", "acs": "high combat score",
         "kd": "wins duels"}
_LOW = {"rating": "below par lately", "acs": "low combat score",
        "kd": "loses duels"}

# `map_edge` is deliberately absent from both, so `explain()` never narrates it.
# It stays in the *score* -- 0.15 is its measured optimum -- but saying "strong
# on this map" asserts something the data does not support. Measured on 12,000
# held-out player-matches, a player's map-specific history has no relationship
# with how they then perform on that map:
#
#     map games in history   n       spearman
#     0                      4,931    +0.000
#     1                      2,971    -0.036
#     3                      1,000    -0.057
#     5                        349    +0.065
#     6+                       488    +0.017
#     all                   12,000    -0.010
#
# Signs flip at random and the magnitudes are noise, flat even at 6+ games. A
# confident sentence on top of that is exactly the kind of plausible-sounding
# claim this project keeps catching, so the reason column names only the
# components that carry signal.


def explain(index: PerfIndex, c: Components) -> str:
    """What is most *distinctive* about this player, in words.

    Ranks by raw z-score, deliberately not by weighted contribution. ACS
    carries the largest weight, so weighting made it win almost every time and
    the column read "high combat score" for four players out of five -- true,
    and useless. The unusual thing about a player is what a teammate wants to
    know, even when it is not what moved the score most.
    """
    # Only components we can honestly narrate are candidates. Ranging over
    # WEIGHTS instead let `map_edge` win the max and then raise KeyError on the
    # phrase lookup -- a crash in the live view, caught by the test that
    # asserts the map is never named.
    zs = {name: index.z(name, getattr(c, name)) for name in _HIGH}
    name = max(zs, key=lambda k: abs(zs[k]))

    # A player who is unremarkable on every component should be described that
    # way. Without this the largest |z| wins even when it is 0.05, so a
    # perfectly average player was labelled "below par lately" on the strength
    # of noise -- a confident-sounding claim about nothing.
    if abs(zs[name]) < NOTABLE_Z:
        if c.n_games < 5:
            return f"middle of the pack, on only {c.n_games} game" + (
                "s" if c.n_games != 1 else "")
        return "middle of the pack"

    word = (_HIGH if zs[name] > 0 else _LOW)[name]
    if name == "map_edge" and c.n_map_games == 0:
        return "no history on this map"
    if c.n_games < 5:
        return f"{word}, but only {c.n_games} game" + ("s" if c.n_games != 1 else "")
    return word


@dataclass(frozen=True)
class AboveRank:
    """Whether a player looks like they are playing below their real rank."""
    flagged: bool
    z: float                    # band-relative rating, in population sd
    dominant: bool              # tops their lobby far more than one game in five
    note: str

    def __bool__(self) -> bool:
        return self.flagged


def above_rank(index: "PerfIndex", c: Components) -> AboveRank:
    """Is this player performing well above their own rank band?

    `rating` is already z-scored within `band_of(tier)` by `rate_performance`,
    so a high value literally means "better than others at this rank". That is
    the whole signal; this adds the second condition and the calibrated cut.

    The second condition matters. Measured on the collected data, mean ACS is
    almost identical across account levels -- 209.6 under level 40 against
    213.5 at level 300+ -- while mean tier is 6.7 against 19.9. New accounts
    frag like veterans but are ranked far below them. Requiring a young or
    thin account is what separates that pattern from a well-established player
    who is simply good.

    Deliberately NOT called "smurf". This cannot tell a smurf from a returning
    player or someone mid-climb, and the wording should not pretend otherwise.
    """
    z = index.z("rating", c.rating)
    dominant = (c.n_dominance >= MIN_DOMINANCE_GAMES
                and c.dominance >= DOMINANT)
    flagged = bool(z >= index.flag_cut and dominant)

    if not flagged:
        return AboveRank(False, z, dominant, "")
    return AboveRank(
        True, z, dominant,
        f"tops the lobby in {c.dominance:.0%} of their {c.n_dominance} "
        f"tracked games (1 in 5 is normal) -- possible smurf")


def evaluate(conn: sqlite3.Connection, puuid: str, as_of: int, map_name: str,
             norms: Norms, index: PerfIndex) -> Potential | None:
    """Score one player for one match, or None if we know nothing about them."""
    c = measure(conn, puuid, as_of, map_name, norms)
    if c is None:
        return None
    raw = index.composite(c)
    return Potential(score=index.percentile(raw), raw=raw, components=c,
                     reason=explain(index, c), flag=above_rank(index, c))
