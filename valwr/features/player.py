"""Per-player features, computed strictly from matches before the target.

Every read goes through valwr/store/temporal.py with an explicit `as_of`.
No module in this package may query `match_players` directly -- the audit in
test/test_leakage.py enforces that by inspecting source, because the rule is
only worth anything if it cannot be quietly broken.

Every rate feature is shrunk toward a population prior. A player with three
games at 100% is not a 100% player, and without shrinkage those sparse cells
dominate the model and generalise to nothing (CLAUDE.md rule 4).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from valwr.rating.components import per_round_rates
from valwr.rating.normalize import Norms
from valwr.rating.rating import rate_performance
from valwr.store import temporal

# Prior weights, by how sparse the cell is. A map x agent cell may hold two
# games, so it needs a far heavier pull toward the prior than an overall
# record built from hundreds. Tuned on validation in Phase 5; these are
# starting points, not settled values.
PRIOR_OVERALL = 20.0
PRIOR_MAP = 30.0
PRIOR_AGENT = 30.0
PRIOR_MAP_AGENT = 45.0
PRIOR_RECENT = 10.0

# The rating and the performance averages need shrinking for exactly the same
# reason the win rates do, and this was missed: 62% of players have a single
# prior match, and a rating averaged over one game was being trusted as much
# as one averaged over fifty. The rating is centred on 1.0 by construction, so
# that is the prior to pull toward.
RATING_PRIOR = 1.0
PRIOR_RATING = 4.0
PRIOR_PERF = 4.0

# Recent matches say more about a player than old ones, and this history spans
# over two years. Weight each match by exp(-age / HALFLIFE) so a season-old
# game counts for less without being discarded.
RECENCY_HALFLIFE_DAYS = 30

RECENT_N = 20


@dataclass
class PlayerFeatures:
    puuid: str
    values: dict[str, float] = field(default_factory=dict)
    games: int = 0

    @property
    def has_history(self) -> bool:
        return self.games > 0


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _weighted_mean(xs, ws):
    pairs = [(x, w) for x, w in zip(xs, ws) if x is not None]
    if not pairs:
        return None
    total_w = sum(w for _, w in pairs)
    return sum(x * w for x, w in pairs) / total_w if total_w else None


def _shrink(value, n, prior, weight):
    """Pull a small-sample average toward a prior, by sample size.

    Identical in spirit to the empirical-Bayes shrinkage on the rate features.
    Without it a one-match average carries the same authority as a fifty-match
    one, and most players in this dataset have one match.
    """
    if value is None:
        return prior
    return (value * n + prior * weight) / (n + weight)


def build(conn: sqlite3.Connection, puuid: str, as_of: int, map_name: str,
          agent: str, tier: int | None, account_level: int | None,
          norms: Norms, prior_rate: float,
          roles: dict[str, str | None] | None = None) -> PlayerFeatures:
    """Features for one player entering one match.

    `as_of` is the target match's start time; nothing at or after it is visible.
    """
    history = temporal.player_history(conn, puuid, as_of)
    f: dict[str, float] = {}

    # --- experience -------------------------------------------------
    f["games_played"] = float(len(history))
    f["tier"] = float(tier or 0)
    f["account_level"] = float(account_level or 0)

    # --- shrunk win rates -------------------------------------------
    overall = temporal.record(conn, puuid, as_of)
    f["wr"] = overall.shrunk(prior_rate, PRIOR_OVERALL)

    on_map = temporal.record_on_map(conn, puuid, as_of, map_name)
    f["wr_map"] = on_map.shrunk(prior_rate, PRIOR_MAP)
    f["games_map"] = float(on_map.games)

    on_agent = temporal.record_on_agent(conn, puuid, as_of, agent)
    f["wr_agent"] = on_agent.shrunk(prior_rate, PRIOR_AGENT)
    f["games_agent"] = float(on_agent.games)

    # The interaction from the original brief: "better on Ascent as Jett".
    # Very sparse, so shrunk hardest of all.
    combo = temporal.record_map_agent(conn, puuid, as_of, map_name, agent)
    f["wr_map_agent"] = combo.shrunk(prior_rate, PRIOR_MAP_AGENT)
    f["games_map_agent"] = float(combo.games)

    # --- form -------------------------------------------------------
    recent = temporal.recent_record(conn, puuid, as_of, last_n=RECENT_N)
    f["wr_recent"] = recent.shrunk(prior_rate, PRIOR_RECENT)

    # --- rating and performance averages ----------------------------
    ratings, accs, adrs, kasts, fbs, fds, weights = [], [], [], [], [], [], []
    for row in history:
        d = dict(row)
        r = rate_performance(d, norms)
        ratings.append(r.value if r is not None else None)
        rates = per_round_rates(d)
        accs.append(rates.get("acs"))
        adrs.append(rates.get("adr"))
        kasts.append(rates.get("kast"))
        fbs.append(rates.get("fb_rate"))
        fds.append(rates.get("fd_rate"))
        age_days = max(0.0, (as_of - (d.get("started_at") or as_of)) / 86400.0)
        weights.append(0.5 ** (age_days / RECENCY_HALFLIFE_DAYS))

    n_rated = sum(1 for r in ratings if r is not None)
    f["rating"] = _shrink(_weighted_mean(ratings, weights), n_rated,
                          RATING_PRIOR, PRIOR_RATING)
    f["rating_n"] = float(n_rated)

    # Shrink toward the measured population mean, not a hardcoded guess. The
    # norms already carry exactly these values, fitted on training data at the
    # correct cutoff, so a literal here would be both redundant and something
    # that silently goes stale as the meta shifts.
    n_hist = len(history)
    for name, vals in (("acs", accs), ("adr", adrs), ("kast", kasts),
                       ("fb_rate", fbs), ("fd_rate", fds)):
        cell = norms.glob.get(name)
        prior = cell.mean if cell is not None and cell.n else 0.0
        f[name] = _shrink(_weighted_mean(vals, weights), n_hist,
                          prior, PRIOR_PERF)

    # Trend: are they improving or sliding? Recent half minus older half.
    rated = [r for r in ratings if r is not None]
    if len(rated) >= 4:
        half = len(rated) // 2
        # history is newest-first, so the first half is the recent one.
        f["rating_trend"] = (_mean(rated[:half]) or 0) - (_mean(rated[half:]) or 0)
    else:
        f["rating_trend"] = 0.0

    # --- off-role ----------------------------------------------------
    # Playing outside your usual role is a real signal, and it needs history
    # to detect: it is a comparison against what this player normally picks.
    if roles:
        counts: dict[str, int] = {}
        for row in history:
            r = roles.get(row["agent"])
            if r:
                counts[r] = counts.get(r, 0) + 1
        primary = max(counts, key=counts.get) if counts else None
        current = roles.get(agent)
        f["off_role"] = float(bool(primary and current and primary != current))
    else:
        f["off_role"] = 0.0

    # --- rust / session ---------------------------------------------
    if history:
        f["days_since_last"] = max(0.0, (as_of - history[0]["started_at"]) / 86400.0)
    else:
        f["days_since_last"] = 0.0

    return PlayerFeatures(puuid=puuid, values=f, games=len(history))


# The order every downstream matrix relies on. Kept explicit so a reordering
# cannot silently misalign training and live features.
FEATURE_NAMES = [
    "games_played", "tier", "account_level",
    "wr", "wr_map", "games_map", "wr_agent", "games_agent",
    "wr_map_agent", "games_map_agent", "wr_recent",
    "rating", "rating_n", "acs", "adr", "kast", "fb_rate", "fd_rate",
    "rating_trend", "days_since_last", "off_role",
]


def empty() -> dict[str, float]:
    """A player with no history at all: neutral values, not zeros.

    Zero would say 'never wins', which is a claim the data does not make.
    Neutral says 'we know nothing', which is the truth, and shrinkage already
    encodes that elsewhere.
    """
    return {name: (1.0 if name == "rating" else 0.0) for name in FEATURE_NAMES}
