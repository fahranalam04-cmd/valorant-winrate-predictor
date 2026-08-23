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
    ratings, accs, adrs, kasts, fbs, fds = [], [], [], [], [], []
    for row in history:
        d = dict(row)
        r = rate_performance(d, norms)
        if r is not None:
            ratings.append(r.value)
        rates = per_round_rates(d)
        accs.append(rates.get("acs"))
        adrs.append(rates.get("adr"))
        kasts.append(rates.get("kast"))
        fbs.append(rates.get("fb_rate"))
        fds.append(rates.get("fd_rate"))

    f["rating"] = _mean(ratings) if ratings else 1.0
    f["rating_n"] = float(len(ratings))
    f["acs"] = _mean(accs) or 0.0
    f["adr"] = _mean(adrs) or 0.0
    f["kast"] = _mean(kasts) or 0.0
    f["fb_rate"] = _mean(fbs) or 0.0
    f["fd_rate"] = _mean(fds) or 0.0

    # Trend: are they improving or sliding? Recent half minus older half.
    if len(ratings) >= 4:
        half = len(ratings) // 2
        # history is newest-first, so the first half is the recent one.
        f["rating_trend"] = (_mean(ratings[:half]) or 0) - (_mean(ratings[half:]) or 0)
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
