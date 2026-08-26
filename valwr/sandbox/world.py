"""Turn profiles into synthetic match history in an in-memory database.

This is what lets the sandbox reuse the production feature path instead of
reimplementing it. Rows are written through `store/normalize.upsert_match` and
`upsert_players` -- the same functions the crawler uses -- into a `:memory:`
SQLite carrying the production schema. `features.build.build_match` then reads
them back through `store/temporal`, exactly as it does for real matches.

Nothing here touches disk, the network, or the real database. The connection is
always `:memory:`, which a test asserts.

Determinism is a requirement, not a convenience: static scenarios are
benchmarks, so history generation uses no randomness at all. Variance is
applied earlier, by perturbing the *profiles*, never by jittering here.
"""

from __future__ import annotations

import sqlite3

from valwr.sandbox.schema import MatchScenario, PlayerProfile
from valwr.store import normalize, schema

# Every synthetic match is this long. Real matches averaged 21.0 rounds, and a
# fixed length keeps per-round rates exactly invertible: score = acs * rounds.
ROUNDS = 21

# Damage taken is not a modelled feature, so it is held at the population
# average rather than varied -- a constant is honest here, noise would not be.
POP_DAMAGE_TAKEN = 141.0 * ROUNDS

# Days between consecutive history matches. A 60-game history therefore spans
# ~4 months, which puts it on a realistic footing against the 90-day recency
# half-life in features/player.py.
SPACING_DAYS = 2.0

# The temporal layer filters on a strict `started_at < as_of`, correctly: a
# match beginning at the prediction instant is not prior history. So
# days_since_last=0 has to mean "very recently", not "exactly now", or the most
# recent game silently vanishes -- which cost one game off every such profile
# before this offset existed.
MIN_AGE_SECONDS = 3600

OTHER_MAP = "Split"
OTHER_AGENT = "Sova"

# Roles for the agents used here, so off-role detection has something to read.
# Taken from the production ref_agents table at runtime where available.
FALLBACK_ROLES = {"Jett": "Duelist", "Sova": "Initiator", "Omen": "Controller",
                  "Killjoy": "Sentinel", "Raze": "Duelist", "Sage": "Sentinel"}


def new_connection() -> sqlite3.Connection:
    """An empty in-memory database with the production schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(schema.SCHEMA)
    conn.executescript(schema.INDEXES)
    return conn


def _buckets(p: PlayerProfile) -> list[tuple[int, bool, bool, float]]:
    """Partition a history into (count, on_map, on_agent, win_rate).

    The four cells are disjoint and sum to `games`, so the map, agent and
    map-x-agent rates can all hold simultaneously rather than fighting.
    """
    both = p.map_agent_games
    map_only = max(0, p.map_games - both)
    agent_only = max(0, p.agent_games - both)
    neither = max(0, p.games - map_only - agent_only - both)

    # The residual cell absorbs whatever is needed for the overall rate to
    # come out right, given the three specified cells.
    specified_wins = (both * p.map_agent_win_rate
                      + map_only * _cell_rate(p.map_win_rate, p.map_agent_win_rate,
                                              p.map_games, both)
                      + agent_only * _cell_rate(p.agent_win_rate,
                                                p.map_agent_win_rate,
                                                p.agent_games, both))
    total_wins = p.games * p.win_rate
    residual_rate = ((total_wins - specified_wins) / neither) if neither else 0.0
    residual_rate = min(1.0, max(0.0, residual_rate))

    return [
        (both, True, True, p.map_agent_win_rate),
        (map_only, True, False,
         _cell_rate(p.map_win_rate, p.map_agent_win_rate, p.map_games, both)),
        (agent_only, False, True,
         _cell_rate(p.agent_win_rate, p.map_agent_win_rate, p.agent_games, both)),
        (neither, False, False, residual_rate),
    ]


def _cell_rate(outer_rate: float, inner_rate: float, outer_n: int,
               inner_n: int) -> float:
    """Rate for the part of `outer` not covered by `inner`.

    The map rate covers all map games including the map-x-agent ones, so the
    map-only cell has to carry whatever is left over for the stated map rate
    to hold across the whole map subset.
    """
    rest = outer_n - inner_n
    if rest <= 0:
        return outer_rate
    wins = outer_rate * outer_n - inner_rate * inner_n
    return min(1.0, max(0.0, wins / rest))


def _win_flags(count: int, wins: int) -> list[int]:
    """Deterministic win/loss flags, spread evenly rather than front-loaded."""
    if count <= 0:
        return []
    wins = min(count, max(0, wins))
    flags = [0] * count
    if wins:
        step = count / wins
        for i in range(wins):
            flags[min(count - 1, int(i * step))] = 1
    return flags


def _allocate_wins(cells: list[tuple[int, bool, bool, float]],
                   games: int, overall_rate: float) -> list[int]:
    """Integer wins per cell, summing to the overall target exactly.

    Rounding each cell independently does not work on small histories: a
    3-game profile at 0.486 rounds every cell to zero and produces a 0.000
    win rate. Largest-remainder allocation keeps each cell near its own rate
    while making the total come out right.

    Where the cell rates and the overall rate genuinely cannot both hold --
    which small integer counts sometimes make impossible -- the overall rate
    wins, because it is the one every scenario depends on.
    """
    ideal = [count * rate for count, _, _, rate in cells]
    floors = [int(x) for x in ideal]
    target = int(round(games * overall_rate))
    deficit = target - sum(floors)

    order = sorted(range(len(cells)),
                   key=lambda i: (ideal[i] - floors[i]), reverse=True)
    out = list(floors)
    step = 1 if deficit > 0 else -1
    i = 0
    while deficit != 0 and order:
        idx = order[i % len(order)]
        capacity = cells[idx][0]
        if 0 <= out[idx] + step <= capacity:
            out[idx] += step
            deficit -= step
        i += 1
        if i > 4 * len(order) + games:      # nothing left to give or take
            break
    return out


def _apply_recency(matches: list[dict], p: PlayerProfile) -> None:
    """Nudge the recent window toward `recent_win_rate`.

    Best effort by design. The per-cell rates take precedence, so this only
    swaps wins and losses of the same cell between the recent window and older
    history -- it never invents wins that would break the overall rate.
    """
    if p.recent_win_rate is None or not matches:
        return
    from valwr.features.player import RECENT_N

    window = min(RECENT_N, len(matches))
    target = int(round(p.recent_win_rate * window))
    recent, older = matches[:window], matches[window:]

    while sum(m["won"] for m in recent) < target and older:
        loss = next((m for m in recent if not m["won"]), None)
        win = next((m for m in older if m["won"]), None)
        if not loss or not win:
            break
        loss["won"], win["won"] = 1, 0
    while sum(m["won"] for m in recent) > target and older:
        win = next((m for m in recent if m["won"]), None)
        loss = next((m for m in older if not m["won"]), None)
        if not win or not loss:
            break
        win["won"], loss["won"] = 0, 1


def _trend_scale(index: int, total: int, trend: float) -> float:
    """Multiplier on performance for a match `index` back from the present.

    A positive trend means the player is improving, so older matches are worse.
    Linear and bounded -- enough to move rating_trend without producing
    implausible statistics at the extremes.
    """
    if not trend or total <= 1:
        return 1.0
    age = index / (total - 1)          # 0 = most recent, 1 = oldest
    return 1.0 + trend * 0.25 * (1.0 - 2.0 * age)


def player_history(p: PlayerProfile, as_of: int, map_name: str,
                   agent: str) -> list[dict]:
    """The synthetic matches this profile implies, newest first."""
    if not p.has_history:
        return []

    # Each cell contributes (position-in-cell, cell) so it can be spread
    # evenly across the timeline. This has to be a true permutation: an
    # earlier version reordered by striding modulo the length, which
    # duplicated some entries and dropped others and silently corrupted every
    # declared rate -- 60 games at 0.500 came back as 0.667.
    cells = _buckets(p)
    wins_per_cell = _allocate_wins(cells, p.games, p.win_rate)

    plan: list[tuple[float, tuple[bool, bool, int]]] = []
    for (count, on_map, on_agent, _), wins in zip(cells, wins_per_cell):
        flags = _win_flags(count, wins)
        for j, won in enumerate(flags):
            spread = (j + 0.5) / len(flags)          # 0..1 within this cell
            plan.append((spread, (on_map, on_agent, won)))

    plan.sort(key=lambda item: item[0])
    ordered = [item for _, item in plan]

    matches = []
    for i, (on_map, on_agent, won) in enumerate(ordered):
        scale = _trend_scale(i, len(ordered), p.trend)
        matches.append({
            "index": i,
            "on_map": on_map,
            "on_agent": on_agent,
            "won": won,
            "map": map_name if on_map else OTHER_MAP,
            "agent": agent if on_agent else OTHER_AGENT,
            "started_at": int(as_of - MIN_AGE_SECONDS
                              - (p.days_since_last + i * SPACING_DAYS) * 86400),
            "acs": p.acs * scale,
            "adr": p.adr * scale,
            "kast": min(1.0, p.kast * scale),
            "fb_rate": min(1.0, p.fb_rate * scale),
            "fd_rate": min(1.0, p.fd_rate),
        })

    _apply_recency(matches, p)
    return matches


def write_player(conn: sqlite3.Connection, puuid: str, p: PlayerProfile,
                 as_of: int, map_name: str, agent: str) -> int:
    """Materialise one profile's history. Returns the number of matches."""
    matches = player_history(p, as_of, map_name, agent)
    for m in matches:
        mid = f"{puuid}#{m['index']}"
        winner = "Blue" if m["won"] else "Red"
        normalize.upsert_match(conn, {
            "match_id": mid, "started_at": m["started_at"], "map": m["map"],
            "mode": "competitive", "queue": "Standard", "region": "na",
            "season": "sandbox", "rounds_red": 13 if not m["won"] else 9,
            "rounds_blue": 13 if m["won"] else 9, "winner": winner,
            "data_quality": None, "ingested_at": as_of,
        })
        normalize.upsert_players(conn, [{
            "match_id": mid, "puuid": puuid, "team": "Blue",
            "agent": m["agent"], "party_id": None, "tier": p.tier,
            "account_level": p.account_level,
            "score": int(round(m["acs"] * ROUNDS)),
            "kills": int(round(0.75 * ROUNDS)),
            "deaths": int(round(0.75 * ROUNDS)),
            "assists": int(round(0.30 * ROUNDS)),
            "headshots": 20, "bodyshots": 60, "legshots": 5,
            "damage_dealt": int(round(m["adr"] * ROUNDS)),
            "damage_taken": int(round(POP_DAMAGE_TAKEN)),
            "started_at": m["started_at"], "map": m["map"], "won": m["won"],
            "rounds_played": ROUNDS,
            "first_bloods": int(round(m["fb_rate"] * ROUNDS)),
            "first_deaths": int(round(m["fd_rate"] * ROUNDS)),
            "multikills": 1, "trade_kills": 2, "traded_deaths": 2,
            "kast_rounds": min(ROUNDS, int(round(m["kast"] * ROUNDS))),
            "clutches": 0, "_name": puuid, "_tag": "SBX",
        }])
    conn.commit()
    return len(matches)


def build_world(scenario: MatchScenario, as_of: int,
                agent_for: dict[str, str] | None = None
                ) -> tuple[sqlite3.Connection, list[dict]]:
    """Materialise a whole scenario. Returns (connection, roster rows).

    The roster is in exactly the shape `features.build.build_match` expects, so
    the caller hands it straight through with no translation layer.
    """
    conn = new_connection()
    roster: list[dict] = []

    for team_label, team in (("Blue", scenario.team_a), ("Red", scenario.team_b)):
        for i, profile in enumerate(team.players):
            puuid = f"{team_label.lower()}{i}"
            agent = team.agent_for(i)
            write_player(conn, puuid, profile, as_of, scenario.map_name, agent)
            roster.append({
                "match_id": "sandbox",
                "puuid": puuid,
                "team": team_label,
                "agent": agent,
                "party_id": profile.party,
                "tier": profile.tier if profile.has_history or profile.tier else None,
                "account_level": profile.account_level,
            })
    return conn, roster
