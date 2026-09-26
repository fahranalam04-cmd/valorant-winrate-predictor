"""A synthetic match, so the dashboard can be looked at without playing one.

    python -m valwr.dash --demo

The page is only visible for the few minutes you are loading into a game, which
makes it awkward to work on and impossible to show anyone. This builds a state
dictionary in exactly the shape `live.state.poll_once` returns and hands it to
the same renderer, so what you see is the real page and not a mock-up of it.

**It invents players.** Nothing here touches the database, the game client or
the network, and no real gamertag or PUUID appears. Agent UUIDs are read from
`ref_agents` when a database is present, purely so the bundled artwork resolves;
without one the page falls back to lettered tiles, which is also worth seeing.
"""

from __future__ import annotations

NAMES = [("Meridian", "na1"), ("halcyon", "0000"), ("nine lives", "eu2"),
         ("Tessellate", "kr1"), ("brief candle", "ap1"), ("Yarrow", "na2"),
         ("Ossuary", "eu1"), ("pale fire", "na3"), ("Quietus", "ap2"),
         ("Winterlight", "eu3")]

AGENTS = ["Jett", "Omen", "Sage", "Sova", "Killjoy",
          "Reyna", "Fade", "Cypher", "Breach", "Neon"]

ROLES = {"Jett": "Duelist", "Reyna": "Duelist", "Neon": "Duelist",
         "Omen": "Controller", "Sage": "Sentinel", "Cypher": "Sentinel",
         "Killjoy": "Sentinel", "Sova": "Initiator", "Fade": "Initiator",
         "Breach": "Initiator"}

# The phrase `explain()` would produce for each profile. Taken from the real
# vocabulary and matched to the numbers beside them -- an earlier version of
# this file gave all eight known players the same string, which made the demo
# look like the model had one opinion about everyone. It does not: measured
# over 2,588 sampled players the real explain() returns 35 distinct phrases and
# the most common covers 8.6% of them.
REASONS = [
    "consistently strong",
    "damages every round",
    None,
    "middle of the pack",
    "below par lately, but only 4 games",
    "wins duels",
    "consistently strong",
    None,
    "middle of the pack",
    "loses duels",
]

# Each player's most recent competitive game: kills, deaths, assists, headshot
# rate, won, rounds for and against, how long ago, the map, and how that game's
# damage compares with their average. The agent-select card prints this on
# every row, and one shared template made five different players look as if
# they had all played the same game -- the problem the reasons above had.
AGO_SECONDS = {"40 minutes ago": 2400, "1 hour ago": 3600, "2 hours ago": 7200,
               "3 hours ago": 10800, "5 hours ago": 18000,
               "yesterday": 86400, "2 days ago": 172800}

LAST_GAMES = [
    (21, 14, 3, 0.27, True, 13, 9, "2 hours ago", "Ascent", 1.15),
    (24, 17, 5, 0.25, True, 13, 11, "40 minutes ago", "Bind", 1.22),
    None,
    (16, 16, 8, 0.22, True, 14, 12, "3 hours ago", "Haven", 1.00),
    (9, 15, 11, 0.19, False, 7, 13, "yesterday", "Lotus", 0.78),
    (29, 12, 4, 0.33, True, 13, 5, "1 hour ago", "Split", 1.35),
    (11, 14, 6, 0.18, False, 10, 13, "5 hours ago", "Sunset", 0.85),
    None,
    (17, 15, 9, 0.24, False, 11, 13, "yesterday", "Icebox", 1.02),
    (13, 16, 4, 0.21, False, 12, 14, "2 days ago", "Pearl", 0.90),
]

# score, career games, acs, k, d, a, kd, hs, winrate, map games
PROFILES = [
    (82, 412, 251.4, 6103, 4980, 1844, 1.23, 0.281, 0.55, 34),
    (67, 188, 224.8, 2610, 2430, 998, 1.07, 0.243, 0.51, 12),
    (None, 0, None, 0, 0, 0, None, None, None, 0),
    (54, 96, 209.1, 1290, 1355, 612, 0.95, 0.219, 0.47, 8),
    (46, 4, 196.3, 52, 61, 30, 0.86, 0.198, 0.42, 1),
    (91, 74, 288.7, 1249, 812, 261, 1.54, 0.334, 0.71, 9),
    (63, 260, 218.2, 3520, 3380, 1502, 1.04, 0.236, 0.50, 21),
    (None, 0, None, 0, 0, 0, None, None, None, 0),
    (58, 143, 213.6, 1902, 1930, 870, 0.99, 0.227, 0.49, 11),
    (56, 64, 219.7, 883, 866, 262, 1.02, 0.244, 0.48, 6),
]

# A plausible spread: mostly Platinum with a Diamond and one unranked.
from valwr.rating import ranks as _r
from valwr.rating import roles as _roles
from valwr.rating.role_score import COMPONENT_LABELS as _LABELS
from valwr.rating.role_score import MIN_MAP_GAMES as _MAP_GATE
from valwr.rating.role_score import standing
RANKS = [_r.describe(t) for t in
         (18, 16, 0, 15, 13, 20, 16, 0, 15, 12)]

# Two duos and a trio, so both the marker and the grouping are visible.
PARTIES = [
    {"members": ["demo-00", "demo-01"], "team": "Blue", "size": 2,
     "label": "duo", "source": "inferred"},
    {"members": ["demo-05", "demo-06", "demo-08"], "team": "Red", "size": 3,
     "label": "trio", "source": "inferred"},
]

MAP = "Ascent"
GATE = _MAP_GATE

# How much a role's standing sits above or below its 0-100, reflecting how
# often that role is the best player on its team: 43.5% for duelists against
# 14.6% for initiators, measured on 3,000 test-period teams.
_ROLE_EDGE = {"Duelist": 0.18, "Sentinel": 0.02, "Controller": 0.0,
              "Initiator": -0.10}

# How far from average each component sits, relative to the player's overall
# standing. Invented, but not uniform: a demo where every component tells the
# same story about a player makes the breakdown look decorative.
_SLANT = {"acs": 1.15, "adr": 1.05, "kd": 0.90, "kda": 0.95, "kast": 0.70,
          "assists": 0.55, "fb": 0.80, "fd": -0.60, "hs": 1.00,
          "abilities": 0.45, "map_edge": 0.35}


def _band(z):
    a = abs(z)
    if a < 0.35:
        return "about average"
    if a < 1.0:
        return "above average" if z > 0 else "below average"
    return "well above average" if z > 0 else "well below average"


def _components(role, agent, score, acs, kd, hs, mg):
    """The breakdown, built from the real weight tables.

    Generated rather than written out, so the published demo cannot drift from
    the weights that actually ship -- an earlier hand-written version outlived
    two changes to the score before anyone noticed it was describing a formula
    the project no longer used.
    """
    weights = _roles.weights_for(role, agent)
    # A percentile back into roughly the z it came from: 50 -> 0, 90 -> +1.3.
    base = (score - 50) / 30.0
    values = {"acs": acs, "adr": round(acs * 0.66, 1), "kd": kd,
              "kda": round(kd * 1.35, 2), "assists": 0.28, "kast": 0.72,
              "fb": 0.11, "fd": 0.10, "hs": hs, "abilities": 2.1,
              "map_edge": round(acs * 0.02, 2)}
    out = []
    for name, w in sorted(weights.items(), key=lambda kv: -abs(kv[1])):
        z = round(base * _SLANT.get(name, 1.0), 2)
        gated = name == "map_edge" and mg < GATE
        out.append({
            "key": name, "label": _LABELS.get(name, name),
            "value": values.get(name), "z": None if gated else z,
            "weight": round(w, 4),
            "contribution": 0.0 if gated else round(w * z, 3),
            "note": (f"not counted -- {mg} game{'s' if mg != 1 else ''} on this "
                     f"map, {GATE} needed") if gated else _band(z),
        })
    return out


def _stats(games, acs, k, d, a, kd, hs, wr):
    wins = round(games * wr) if wr is not None else 0
    return {"games": games, "wins": wins, "losses": games - wins,
            "kills": k, "deaths": d, "assists": a, "acs": acs,
            # Damage per round, which the card shows where combat score used to
            # be. Held at the measured ratio between the two rather than
            # invented separately: 140.3 ADR against 211.9 ACS, dataset-wide.
            "adr": round(acs * 0.662, 1) if acs is not None else None,
            "kd": kd, "headshot_rate": hs, "win_rate": wr}


def _agent_ids(conn=None) -> dict[str, str]:
    """Real UUIDs where we can get them, so the bundled art resolves."""
    if conn is None:
        return {}
    try:
        return {r["name"]: r["uuid"]
                for r in conn.execute("SELECT uuid, name FROM ref_agents")}
    except Exception:                                # noqa: BLE001
        return {}


def demo_state(conn=None, phase: str = "coregame") -> dict:
    """The invented match. `phase="pregame"` is agent select: your team only,
    two players still picking, and one still being looked up."""
    ids = _agent_ids(conn)
    players = []
    for i, (agent, (name, tag), prof) in enumerate(
            zip(AGENTS, NAMES, PROFILES)):
        score, games, acs, k, d, a, kd, hs, wr, mg = prof
        team = "Blue" if i < 5 else "Red"
        known = score is not None
        career = _stats(games, acs, k, d, a, kd, hs, wr) if known else None
        # Form runs a little ahead of the career figure for most of them, which
        # is what a real last-20 usually looks like on an improving account.
        recent = (_stats(min(games, 20), round(acs * 1.04, 1),
                         int(k * min(games, 20) / max(games, 1)),
                         int(d * min(games, 20) / max(games, 1)),
                         int(a * min(games, 20) / max(games, 1)),
                         round(kd * 1.06, 2), round(hs * 1.02, 4),
                         round(min((wr or .5) * 1.08, 1.0), 4))
                  if known else None)
        entry = {
            "puuid": f"demo-{i:02d}", "name": f"{name}#{tag}",
            "known_name": True, "agent": agent, "agent_id": ids.get(agent),
            "role": ROLES.get(agent), "team": team, "is_you": i == 0,
            "score": score, "flag": None, "career": career,
            "rank": RANKS[i],
            "recent": recent,
            "reason": REASONS[i] or "no history",
        }
        if known:
            last = LAST_GAMES[i]
            # The flagged one: dominates lobbies far above their rank.
            if i == 5:
                entry["flag"] = {"note": "Finishes top-2 in 71% of their "
                                         "lobbies, against 20% by chance. "
                                         "Well above what this rank explains."}
            counts = mg >= GATE
            # What the lobby is ordered by. The 0-100 is a percentile inside a
            # role, so it cannot order a mixed team: duelists top a scoreboard
            # far more often than initiators, and scoring each against their
            # own role removes exactly that. Invented like everything else
            # here, but with the same shape -- which is why the demo shows a
            # lower number sitting above a higher one.
            entry["raw"] = round((score - 50) / 30.0 + _ROLE_EDGE.get(
                ROLES.get(agent), 0.0), 3)
            entry["detail"] = {
                "score": score, "raw": entry["raw"], "reason": entry["reason"],
                "components": _components(
                    ROLES.get(agent), agent, score, acs, kd, hs, mg),
                "career": career, "recent": recent, "recent_window": 20,
                "map": dict(_stats(mg, round(acs * 0.96, 1), int(k * mg / max(games, 1)),
                                   int(d * mg / max(games, 1)),
                                   int(a * mg / max(games, 1)),
                                   round(kd * 0.95, 2), round(hs * 0.97, 4),
                                   round((wr or 0.5) * 0.98, 4)),
                            name=MAP, counts_toward_score=counts, gate=GATE,
                            agents=[{"agent": agent, "games": max(mg - 2, 1)},
                                    {"agent": "Omen" if agent != "Omen"
                                     else "Astra", "games": 2}]),
                "form": [
                    {"map": last[8], "agent": agent,
                     "acs": round(acs * last[9], 1),
                     "adr": round(acs * last[9] * 0.662, 1),
                     "kills": last[0], "deaths": last[1], "assists": last[2],
                     "headshot_rate": last[3], "rounds_won": last[5],
                     "rounds_lost": last[6], "won": last[4],
                     "ago": last[7]},
                    {"map": "Lotus", "agent": "Omen", "acs": round(acs * 0.82, 1),
                     "adr": round(acs * 0.82 * 0.662, 1),
                     "kills": 12, "deaths": 18, "assists": 9,
                     "headshot_rate": 0.18, "rounds_won": 8,
                     "rounds_lost": 13, "won": False,
                     "ago": "yesterday"},
                    {"map": "Split", "agent": agent, "acs": round(acs * 1.02, 1),
                     "adr": round(acs * 1.02 * 0.662, 1),
                     "kills": 17, "deaths": 16, "assists": 5,
                     "headshot_rate": 0.24, "rounds_won": 13,
                     "rounds_lost": 11, "won": True, "ago": "2 days ago"},
                ],
                "freshness": {"games_known": games,
                              "seconds_old": AGO_SECONDS[last[7]],
                              "label": f"last match {last[7]}",
                              "stale": AGO_SECONDS[last[7]] > 86400},
            }
        else:
            entry["detail"] = None
        players.append(entry)

    # Ordered the way `live/state._player_rows` orders a real lobby, so the
    # demo cannot show a roster the dashboard would never produce.
    players.sort(key=lambda r: (standing(r) is not None, standing(r) or 0),
                 reverse=True)

    if phase == "pregame":
        return _pregame(players)

    return {
        "match_id": "demo", "phase": "coregame", "is_custom": False,
        "standard_mode": True, "map": MAP, "mode": "BombGameMode",
        "as_of": 0, "own_team": "Blue", "enemy_team": "Red",
        "team_sizes": {"Blue": 5, "Red": 5}, "coverage": 8,
        "confidence": "medium", "fetched": 0, "model": "logistic regression",
        "parties": PARTIES,
        "warnings": ["Demo data. These players are invented and no database, "
                     "game client or network was touched to build this."],
        "players": players,
        "lookup": {"pending": [], "remaining": 0},
        "prediction": {"own_probability": 0.5731, "win_probability": 0.5731,
                       "factors": [{"name": "d_rank", "value": 0.182},
                                   {"name": "d_acs", "value": -0.061},
                                   {"name": "d_kd", "value": 0.044}]},
    }


def _pregame(players: list[dict]) -> dict:
    """Agent select, in the shape poll_once returns there.

    Riot describes only your own team before the match starts, so the enemy
    side is absent rather than unknown. Two players have not picked yet, and
    the one with no stored history is still being looked up -- the state the
    page spends its first seconds in, and the one easiest to render wrongly.
    """
    ours = [dict(p) for p in players if p["team"] == "Blue"]
    for p in ours:
        if p["puuid"] in ("demo-03", "demo-04"):
            p.update(agent=None, agent_id=None, role=None)
    pending = [p["puuid"] for p in ours if p["score"] is None]
    return {
        "match_id": "demo-pregame", "phase": "pregame", "is_custom": False,
        "standard_mode": True, "map": MAP, "mode": "BombGameMode",
        "as_of": 0, "own_team": "Blue", "enemy_team": "Red",
        "team_sizes": {"Blue": len(ours), "Red": 0},
        "coverage": sum(p["score"] is not None for p in ours),
        "confidence": "low", "fetched": 0, "model": "logistic regression",
        "parties": [g for g in PARTIES if g["team"] == "Blue"],
        "warnings": ["Demo data. These players are invented and no database, "
                     "game client or network was touched to build this.",
                     "Enemy team is hidden during agent select. It fills in "
                     "once the match starts."],
        "players": ours,
        "lookup": {"pending": pending, "remaining": 2 * len(pending)},
        "prediction": None,
    }
