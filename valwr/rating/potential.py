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
what the map actually adds.

It is also **gated**: below `MIN_MAP_GAMES` it is exactly zero, so a player
with no history *and* a player with a handful of games are both ranked purely
on the rest. That is the honest answer rather than a guess, and it is stronger
than shrinkage alone -- see the constant for the measurement that forced it.

Gating and scaling are coupled, and getting that wrong cost more than the gate
ever bought: a gated component's z-score must not be scaled on the players it
gated out, or the divisor collapses toward zero. `fit_scales` carries the
measurement and the fix.

Everything is shrunk toward a prior by sample size, the same empirical-Bayes
treatment every rate feature in this project gets. Of the samples we can score
at all, 23.3% have exactly one prior match, and without shrinkage one lucky
game would outrank a hundred consistent ones.

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
# So this is deliberately ACS-led, and `rating` and `kd` earn their place by
# explaining rather than by ranking: they are what `explain()` turns into
# "consistently strong" or "wins duels", which a bare ACS number cannot say.
# The cost of keeping them is inside noise; the gain is a readable reason.
#
# `map_edge` is the exception and is deliberately NOT narrated -- see `_HIGH`.
WEIGHTS = {"acs": 0.45, "rating": 0.25, "kd": 0.15, "map_edge": 0.15}

# Shrinkage strengths, in units of "matches". Map history is thinner than
# overall history, so it is pulled harder toward no-opinion.
PRIOR_N_RATING = 4.0
PRIOR_N_ACS = 4.0
PRIOR_N_KD = 4.0
PRIOR_N_MAP = 6.0

# Below this many games on the map, `map_edge` is exactly zero -- no opinion,
# rather than a shrunk guess. Shrinkage alone was not enough: one 9/17 game on
# Ascent pulled a real account from 68 to 57, and three games on Lotus pushed
# it to 77, a 20-point swing across maps on a component measured to have no
# predictive power at all (Spearman -0.010 over 12,000 held-out
# player-matches). A number that moves that far on one game is worse than one
# that does not move, because the movement reads as insight.
#
# The threshold was swept on 1,500 held-out teams. Accuracy does not choose it
# -- every value sits inside a +/-1.0 point standard error:
#
#     gate   top-1   players it affects   median map swing
#        0   28.9%              100.0%              0.152
#        4   29.5%               14.1%              0.476
#        6   29.7%                5.4%              0.463   <- shipped
#        8   29.3%                2.3%              0.418
#
# Those are simulated: one pass of components measured with the gate off, then
# each threshold applied to the same values. A fresh end-to-end run at gate 6
# -- rebuilt index, refitted percentiles -- came back at 29.2%, not 29.7%. Both
# sit inside the standard error of each other and of gate 4, which is the point:
# accuracy does not choose the threshold, so do not read either figure as one
# beating the other.
#
# What separates them is *exposure*: at 6 the map moves one player in twenty
# rather than one in seven, at the same strength when it does fire. For a
# component with no measured predictive power, less exposure is the right side
# to err on.
MIN_MAP_GAMES = 6

# How many matches count as "recent" for the form figures the scoreboard shows.
# Twenty is roughly a week of steady play, and long enough that one bad night
# does not dominate it.
RECENT_GAMES = 20

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
    # "no opinion" rather than "average player" -- and gated, so a handful of
    # games cannot swing the score at all.
    if n_map >= MIN_MAP_GAMES:
        map_mean = _weighted(map_ratings, map_weights)
        map_edge = _shrink(map_mean, n_map, rating, PRIOR_N_MAP) - rating
    else:
        map_edge = 0.0

    # Newest first, so history[0] is the most recent thing we know about them.
    newest = dict(history[0])
    dom, dom_n = temporal.lobby_dominance(conn, puuid, as_of)
    return Components(rating=rating, acs=acs, kd=kd, map_edge=map_edge,
                      n_games=len(history), n_map_games=n_map,
                      tier=newest.get("tier"),
                      account_level=newest.get("account_level"),
                      dominance=dom, n_dominance=dom_n)


# Below this many gate-clearing samples, a scale fitted on them alone is itself
# noise, and the full sample -- wrong but stable -- is the safer of two bad
# options. The build prints loudly when it falls back.
MIN_SCALE_SAMPLE = 200


def fit_scales(collected: list[Components]) -> tuple[dict, dict]:
    """Component means and standard deviations for a PerfIndex.

    Lives here rather than in `tools/build_perf_index.py` because it is the
    rule that decides what a z-score *means*, and it needs a test.

    `map_edge` is fitted differently from the other three, and the reason is
    the worst measurement bug this scoring code has had. Below `MIN_MAP_GAMES`
    the component is set to exactly 0.0, meaning *no opinion* -- not "measured
    and found average". Those are non-measurements, and 98.7% of a 2,574-sample
    draw were exactly that. Fitting a standard deviation across them measures
    the width of a spike at zero: 0.00500 against 0.04353 over the players who
    actually clear the gate, 8.7x too small.

    Dividing by a scale 8.7x too small turned an ordinary map edge into z =
    -17.8, so a component carrying 15% of the weight -- and no measured
    predictive power on its own, Spearman -0.010 -- outweighed the other three
    combined. One real account read 74 on one map and 2 on another off nothing
    else. Raising MIN_MAP_GAMES from 4 to 6 made it worse, because fewer
    clearers means more zeros means a smaller divisor; the top-1 sweep used to
    choose that gate could not see it, since top-1 ranks players within a team
    where nearly everyone is gated to zero anyway.

    So: the scale comes from the gate-clearing subset, and the mean is pinned
    to exactly 0.0. `map_edge` is a delta against the player's own rating, so
    zero is its true centre by construction -- and pinning it keeps "gated
    implies exactly no contribution" an exact property rather than an accident
    of the sample mean landing near zero.
    """
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in WEIGHTS:
        sample = collected
        if name == "map_edge":
            sample = [c for c in collected if c.n_map_games >= MIN_MAP_GAMES]
            if len(sample) < MIN_SCALE_SAMPLE:
                sample = collected
        vals = [getattr(c, name) for c in sample]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)
        means[name] = 0.0 if name == "map_edge" else mu
        stds[name] = var ** 0.5
    return means, stds


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


# --- the expanded card -------------------------------------------------

COMPONENT_LABELS = {
    "acs": "combat score",
    "rating": "overall rating",
    "kd": "kills per death",
    "map_edge": "map adjustment",
}


def _band(z: float) -> str:
    """A z-score in words. Blunt on purpose: nobody reads sigmas mid-match."""
    a = abs(z)
    if a < 0.35:
        return "about average"
    if a < 1.0:
        return "above average" if z > 0 else "below average"
    if a < 2.0:
        return "well above average" if z > 0 else "well below average"
    return "exceptional" if z > 0 else "very poor"


def _ago(seconds: float) -> str:
    days = seconds / 86400.0
    if days < 1:
        hours = max(1, int(seconds // 3600))
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    if days < 2:
        return "yesterday"
    return f"{int(days)} days ago"


def _stats(c) -> dict:
    """One tally as plain JSON -- the same shape for a career and for one map.

    `acs` is total score over total rounds, which is what ACS means. An earlier
    version of the map block averaged per-match ACS instead, weighting a
    13-round stomp the same as a 29-round grind; the numbers differ slightly
    and this one is the definition.
    """
    return {
        "games": c.games,
        "wins": c.wins,
        "losses": c.decided - c.wins,
        "kills": c.kills, "deaths": c.deaths, "assists": c.assists,
        "acs": round(c.acs, 1) if c.acs is not None else None,
        "kd": round(c.kd, 2) if c.kd is not None else None,
        "headshot_rate": (round(c.headshot_rate, 4)
                          if c.headshot_rate is not None else None),
        "win_rate": round(c.win_rate, 4) if c.win_rate is not None else None,
    }


def detail(conn: sqlite3.Connection, puuid: str, as_of: int, map_name: str,
           norms: Norms, index: PerfIndex, form_games: int = 5) -> dict | None:
    """Everything behind one player's score, as plain data.

    The score on its own is unauditable -- a reader cannot tell whether 68 came
    from consistent play or from three good games and no data. This returns the
    parts, so the number can be checked rather than believed.

    `freshness` is the field that matters most and the one whose absence hid a
    real bug: history for a known player was never refreshed, so a score could
    sit unchanged for twelve days while looking live. If the data is old, the
    card has to say so.
    """
    c = measure(conn, puuid, as_of, map_name, norms)
    if c is None:
        return None

    raw = index.composite(c)
    flag = above_rank(index, c)

    components = []
    for name in WEIGHTS:
        z = index.z(name, getattr(c, name))
        entry = {
            "key": name,
            "label": COMPONENT_LABELS.get(name, name),
            "value": round(getattr(c, name), 3),
            "z": round(z, 2),
            "note": _band(z),
            "weight": WEIGHTS[name],
            "contribution": round(WEIGHTS[name] * z, 3),
        }
        if name == "map_edge" and c.n_map_games < MIN_MAP_GAMES:
            # Say why it is zero rather than letting a reader assume the player
            # is simply average here. Below the gate we have no opinion at all.
            entry["note"] = (f"not counted -- {c.n_map_games} game"
                             f"{'s' if c.n_map_games != 1 else ''} on this map, "
                             f"{MIN_MAP_GAMES} needed")
        components.append(entry)

    # --- this map ------------------------------------------------------
    on_map = [dict(r) for r in
              temporal.player_history_on_map(conn, puuid, as_of, map_name)]
    agents: dict[str, int] = {}
    for r in on_map:
        if r.get("agent"):
            agents[r["agent"]] = agents.get(r["agent"], 0) + 1
    map_acs = [r["score"] / r["rounds_played"] for r in on_map
               if r.get("rounds_played")]
    map_block = {
        "name": map_name,
        "games": len(on_map),
        "counts_toward_score": len(on_map) >= MIN_MAP_GAMES,
        "wins": sum(1 for r in on_map if r.get("won")),
        "losses": sum(1 for r in on_map if r.get("won") is not None
                      and not r["won"]),
        "acs": round(sum(map_acs) / len(map_acs), 1) if map_acs else None,
        # Emitted so the views can name the threshold without hardcoding it;
        # two copies of a constant drift the moment one is tuned.
        "gate": MIN_MAP_GAMES,
        "agents": [{"agent": a, "games": n}
                   for a, n in sorted(agents.items(), key=lambda kv: -kv[1])],
    }
    # K/D/A, headshots and a round-weighted ACS for this map specifically.
    # `on_map` is already in memory, so this is free -- no second query.
    map_block.update(_stats(temporal.Career.from_rows(on_map)))
    map_block["name"] = map_name

    # --- recent form ---------------------------------------------------
    history = temporal.player_history(conn, puuid, as_of, limit=form_games)
    form = []
    for r in history:
        d = dict(r)
        rp = d.get("rounds_played") or 0
        form.append({
            "map": d.get("map"),
            "agent": d.get("agent"),
            "acs": round(d["score"] / rp, 1) if rp else None,
            "kills": d.get("kills"), "deaths": d.get("deaths"),
            "won": bool(d["won"]) if d.get("won") is not None else None,
            "ago": _ago(as_of - (d.get("started_at") or as_of)),
        })

    newest = dict(history[0]).get("started_at") if history else None
    age = (as_of - newest) if newest else None
    return {
        "score": index.percentile(raw),
        "raw": round(raw, 4),
        "reason": explain(index, c),
        "components": components,
        # Raw career totals, deliberately NOT the shrunk component values above.
        # A player with four games shows their real ACS here and a value pulled
        # toward the population mean in `components`; the score must not believe
        # four games, but a scoreboard must not lie about what happened.
        "career": _stats(temporal.career_totals(conn, puuid, as_of)),
        # Form, over the last RECENT_GAMES matches. Usually the same rows as
        # the career block -- the median player here has 8 stored matches -- so
        # the views compare the two and stay quiet when they are identical
        # rather than printing one number under two headings.
        "recent": _stats(temporal.recent_totals(conn, puuid, as_of,
                                                RECENT_GAMES)),
        "recent_window": RECENT_GAMES,
        "map": map_block,
        "form": form,
        "freshness": {
            "games_known": c.n_games,
            "seconds_old": age,
            "label": (f"last match {_ago(age)}" if age is not None
                      else "no matches on record"),
            # Anything past a day is worth flagging: the score cannot reflect
            # games we have not collected.
            "stale": age is not None and age > 86400,
        },
        "dominance": {
            "rate": round(c.dominance, 3),
            "games": c.n_dominance,
            "note": (f"top 2 of the lobby in {c.dominance:.0%} of "
                     f"{c.n_dominance} games (1 in 5 is normal)"
                     if c.n_dominance >= MIN_DOMINANCE_GAMES else
                     "not enough games to judge"),
        },
        "flag": ({"note": flag.note, "z": round(flag.z, 2)}
                 if flag.flagged else None),
    }
