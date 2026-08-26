"""Canonical player archetypes.

Values are anchored to the population means actually measured on the collected
data rather than invented -- ACS 212, ADR 141, KAST 0.712, first-blood rate
0.099, ~21 rounds per match. `average` sits on those numbers exactly, and the
ladder moves away from them coherently.

Coherence matters. A profile whose ACS is elite while its KAST is terrible
describes a player who does not exist, and predictions about them teach
nothing. `_ladder` derives the correlated statistics from one ability
parameter so the default archetypes cannot drift apart; profiles that are
*deliberately* contradictory (hidden_smurf, rank_only_smurf) override
explicitly and say so.
"""

from __future__ import annotations

from valwr.sandbox.schema import PlayerProfile

# Population anchors, measured. See docs/SANDBOX.md.
POP_ACS = 212.0
POP_ADR = 141.0
POP_KAST = 0.712
POP_FB = 0.099
POP_FD = 0.099


def _ladder(name: str, ability: float, **overrides) -> PlayerProfile:
    """Build a coherent profile from a single ability parameter.

    `ability` is roughly a z-score of overall skill: 0 is the population
    average, +1 is a strong player, -1 a weak one. Every correlated statistic
    is derived from it so the archetypes cannot become internally inconsistent.

    The scaling factors are judgement, not fitted values -- they exist to keep
    the ladder monotone and plausible, not to model VALORANT precisely.
    """
    base = dict(
        name=name,
        acs=POP_ACS + 40.0 * ability,
        adr=POP_ADR + 26.0 * ability,
        kast=min(0.95, max(0.35, POP_KAST + 0.045 * ability)),
        fb_rate=min(0.30, max(0.02, POP_FB + 0.018 * ability)),
        # Better players die first less often: the sign is deliberately inverted.
        fd_rate=min(0.30, max(0.02, POP_FD - 0.012 * ability)),
        win_rate=min(0.75, max(0.25, 0.50 + 0.035 * ability)),
        tier=int(round(15 + 2.2 * ability)),
    )
    base["map_win_rate"] = base["win_rate"]
    base["agent_win_rate"] = base["win_rate"]
    base["map_agent_win_rate"] = base["win_rate"]
    base.update(overrides)
    return PlayerProfile(**base)


# --- the skill ladder -------------------------------------------------
UNKNOWN = PlayerProfile(
    name="unknown", games=0, map_games=0, agent_games=0, map_agent_games=0,
    tier=0, account_level=1,
)
NEW_PLAYER = _ladder("new_player", -0.4, games=3, map_games=1, agent_games=2,
                     map_agent_games=1, account_level=14, tier=0)
WEAK = _ladder("weak", -1.2)
BELOW_AVERAGE = _ladder("below_average", -0.6)
AVERAGE = _ladder("average", 0.0)
ABOVE_AVERAGE = _ladder("above_average", 0.6)
STRONG = _ladder("strong", 1.2)
ELITE = _ladder("elite", 2.0)

# --- contradictory / special --------------------------------------------
# Elite performance on a low-tier account with few games: the pattern a smurf
# leaves behind. Deliberately incoherent between rank and performance, which
# is the entire point of the archetype.
SMURF_LIKE = _ladder("smurf_like", 2.4, games=18, map_games=4, agent_games=8,
                     map_agent_games=3, tier=9, account_level=22)

# High rank, ordinary output. Tests whether rank alone moves the prediction.
RANK_ONLY_SMURF = _ladder("rank_only_smurf", 0.0, tier=24, account_level=300)

HOT = _ladder("hot", 0.0, recent_win_rate=0.75, trend=0.35)
COLD = _ladder("cold", 0.0, recent_win_rate=0.28, trend=-0.35)

def _subset_specialist(name: str, *, games: int, subset_games: int,
                       subset_rate: float, field: str, **extra) -> PlayerProfile:
    """A player who is unusually good (or bad) on one subset of their history.

    Rate features are **compositionally coupled**: map games are a subset of
    all games, so a subset rate and the overall rate cannot both be chosen
    freely -- the complement has to absorb the difference.

    An earlier version of these profiles held the overall rate at 0.50 while
    moving the map rate, which forced the complement to the boundary: the map
    "specialist" ended up 0/15 off-map and the map "weak" player 15/15. They
    were not mirror images, and the resulting scenario looked like a model
    finding when it was arithmetic.

    So the *complement* is pinned at neutral instead, and the overall rate is
    derived from it. A player better on one map genuinely has a slightly
    better overall record, which is also what the real world looks like.
    """
    rest = games - subset_games
    overall = (subset_rate * subset_games + 0.50 * rest) / games
    return _ladder(name, 0.0, games=games, win_rate=overall,
                   **{field: subset_rate}, **extra)


MAP_SPECIALIST = _subset_specialist(
    "map_specialist", games=60, subset_games=45, subset_rate=0.68,
    field="map_win_rate", map_games=45, map_agent_games=20,
    map_agent_win_rate=0.68, agent_win_rate=0.50)
MAP_WEAK = _subset_specialist(
    "map_weak", games=60, subset_games=45, subset_rate=0.32,
    field="map_win_rate", map_games=45, map_agent_games=20,
    map_agent_win_rate=0.32, agent_win_rate=0.50)
# A one-trick player: most of a long history on a single agent. `games` has to
# rise with `agent_games` -- the validator rejects a subset larger than its set,
# which is how the first draft of these profiles was caught.
AGENT_SPECIALIST = _subset_specialist(
    "agent_specialist", games=180, subset_games=90, subset_rate=0.66,
    field="agent_win_rate", map_games=36, agent_games=90,
    map_agent_games=20, map_agent_win_rate=0.66, map_win_rate=0.58)
AGENT_WEAK = _subset_specialist(
    "agent_weak", games=180, subset_games=90, subset_rate=0.34,
    field="agent_win_rate", map_games=36, agent_games=90,
    map_agent_games=20, map_agent_win_rate=0.34, map_win_rate=0.42)

# Ordinary overall, ordinary on the map, ordinary on the agent -- but the
# combination is excellent. Isolates the interaction from its parts.
def _combo_specialist(name: str, combo_rate: float) -> PlayerProfile:
    """Excellent (or poor) specifically in one map-and-agent combination.

    Everything outside the combination is pinned at neutral, and the map,
    agent and overall rates are then *derived*. They necessarily move a little
    -- 25 strong games inside an 80-game agent history cannot leave that
    history untouched -- so this isolates the interaction as far as nested
    rates allow, which is not perfectly.
    """
    games, map_g, agent_g, combo_g = 200, 40, 80, 25
    derive = lambda n, r=combo_rate: (r * combo_g + 0.50 * (n - combo_g)) / n
    return _ladder(name, 0.0, games=games, map_games=map_g,
                   agent_games=agent_g, map_agent_games=combo_g,
                   map_agent_win_rate=combo_rate,
                   map_win_rate=derive(map_g), agent_win_rate=derive(agent_g),
                   win_rate=derive(games))


MAP_AGENT_SPECIALIST = _combo_specialist("map_agent_specialist", 0.78)
MAP_AGENT_WEAK = _combo_specialist("map_agent_weak", 0.22)

OFF_ROLE = _ladder("off_role", 0.0, role="Sentinel")     # picks a Duelist
RUSTY = _ladder("rusty", 0.0, days_since_last=45.0)
ACTIVE = _ladder("active", 0.0, days_since_last=0.0)

EXPERIENCED_AVERAGE = _ladder("experienced_average", 0.0, games=600,
                              map_games=120, agent_games=240,
                              map_agent_games=60, account_level=420)
# Level is decoupled from skill on purpose, so the model cannot be read as
# "account level means good".
HIGH_LEVEL_MEDIOCRE = _ladder("high_level_mediocre", -0.5, account_level=600)
LOW_LEVEL_STRONG = _ladder("low_level_strong", 1.5, account_level=25)

LOW_SAMPLE_HIGH_WR = _ladder(
    "low_sample_high_wr", 0.0, games=3, map_games=1, agent_games=2,
    map_agent_games=1, win_rate=1.0, map_win_rate=1.0, agent_win_rate=1.0,
    map_agent_win_rate=1.0)
LOW_SAMPLE_LOW_WR = _ladder(
    "low_sample_low_wr", 0.0, games=3, map_games=1, agent_games=2,
    map_agent_games=1, win_rate=0.0, map_win_rate=0.0, agent_win_rate=0.0,
    map_agent_win_rate=0.0)
BIG_SAMPLE_GOOD_WR = _ladder(
    "big_sample_good_wr", 0.0, games=500, map_games=100, agent_games=200,
    map_agent_games=50, win_rate=0.55, map_win_rate=0.55,
    agent_win_rate=0.55, map_agent_win_rate=0.55)

# Strong on some measures, weak on others -- an intentionally jagged player,
# used to probe how the team spread statistics respond.
HIGH_VARIANCE = _ladder("high_variance", 1.4, kast=0.60, fd_rate=0.19,
                        win_rate=0.50)

ALL: dict[str, PlayerProfile] = {
    p.name: p for p in (
        UNKNOWN, NEW_PLAYER, WEAK, BELOW_AVERAGE, AVERAGE, ABOVE_AVERAGE,
        STRONG, ELITE, SMURF_LIKE, RANK_ONLY_SMURF, HOT, COLD,
        MAP_SPECIALIST, MAP_WEAK, AGENT_SPECIALIST, AGENT_WEAK,
        MAP_AGENT_SPECIALIST, MAP_AGENT_WEAK, OFF_ROLE, RUSTY, ACTIVE,
        EXPERIENCED_AVERAGE, HIGH_LEVEL_MEDIOCRE, LOW_LEVEL_STRONG,
        LOW_SAMPLE_HIGH_WR, LOW_SAMPLE_LOW_WR, BIG_SAMPLE_GOOD_WR,
        HIGH_VARIANCE,
    )
}


def get(name: str) -> PlayerProfile:
    if name not in ALL:
        raise KeyError(f"unknown profile {name!r}; have {sorted(ALL)}")
    return ALL[name]


def validate_all() -> None:
    for p in ALL.values():
        p.validate()
