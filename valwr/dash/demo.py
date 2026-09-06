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
    "high combat score",
    None,
    "middle of the pack",
    "below par lately, but only 4 games",
    "wins duels",
    "consistently strong",
    None,
    "middle of the pack",
    "loses duels",
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
    (41, 11, 188.4, 134, 166, 80, 0.81, 0.191, 0.39, 2),
]

MAP = "Ascent"
GATE = 6


def _stats(games, acs, k, d, a, kd, hs, wr):
    wins = round(games * wr) if wr is not None else 0
    return {"games": games, "wins": wins, "losses": games - wins,
            "kills": k, "deaths": d, "assists": a, "acs": acs, "kd": kd,
            "headshot_rate": hs, "win_rate": wr}


def _agent_ids(conn=None) -> dict[str, str]:
    """Real UUIDs where we can get them, so the bundled art resolves."""
    if conn is None:
        return {}
    try:
        return {r["name"]: r["uuid"]
                for r in conn.execute("SELECT uuid, name FROM ref_agents")}
    except Exception:                                # noqa: BLE001
        return {}


def demo_state(conn=None) -> dict:
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
            "recent": recent,
            "reason": REASONS[i] or "no history",
        }
        if known:
            # The flagged one: dominates lobbies far above their rank.
            if i == 5:
                entry["flag"] = {"note": "Finishes top-2 in 71% of their "
                                         "lobbies, against 20% by chance. "
                                         "Well above what this rank explains."}
            counts = mg >= GATE
            entry["detail"] = {
                "score": score, "raw": 0.0, "reason": entry["reason"],
                "components": [
                    {"key": "acs", "label": "combat score", "value": acs,
                     "z": 1.1, "note": "well above average", "weight": 0.45,
                     "contribution": 0.5},
                    {"key": "rating", "label": "overall rating", "value": 1.06,
                     "z": 0.4, "note": "above average", "weight": 0.25,
                     "contribution": 0.1},
                    {"key": "kd", "label": "kills per death", "value": kd,
                     "z": 0.6, "note": "above average", "weight": 0.15,
                     "contribution": 0.09},
                    {"key": "map_edge", "label": "map adjustment",
                     "value": 0.01, "z": 0.3,
                     "note": ("above average" if counts else
                              f"not counted -- {mg} game"
                              f"{'s' if mg != 1 else ''} on this map, "
                              f"{GATE} needed"),
                     "weight": 0.15, "contribution": 0.05 if counts else 0.0},
                ],
                "career": career, "recent": recent, "recent_window": 20,
                "map": dict(_stats(mg, round(acs * 0.96, 1), int(k * mg / max(games, 1)),
                                   int(d * mg / max(games, 1)),
                                   int(a * mg / max(games, 1)),
                                   round(kd * 0.95, 2), round(hs * 0.97, 4),
                                   round((wr or 0.5) * 0.98, 4)),
                            name=MAP, counts_toward_score=counts, gate=GATE,
                            agents=[{"agent": agent, "games": max(mg - 2, 1)},
                                    {"agent": "Omen", "games": 2}]),
                "form": [
                    {"map": MAP, "agent": agent, "acs": round(acs * 1.15, 1),
                     "kills": 21, "deaths": 14, "won": True,
                     "ago": "2 hours ago"},
                    {"map": "Lotus", "agent": "Omen", "acs": round(acs * 0.82, 1),
                     "kills": 12, "deaths": 18, "won": False,
                     "ago": "yesterday"},
                    {"map": "Split", "agent": agent, "acs": round(acs * 1.02, 1),
                     "kills": 17, "deaths": 16, "won": True, "ago": "2 days ago"},
                ],
                "freshness": {"games_known": games, "seconds_old": 7200,
                              "label": "last match 2 hours ago", "stale": False},
            }
        else:
            entry["detail"] = None
        players.append(entry)

    return {
        "match_id": "demo", "phase": "coregame", "is_custom": False,
        "standard_mode": True, "map": MAP, "mode": "BombGameMode",
        "as_of": 0, "own_team": "Blue", "enemy_team": "Red",
        "team_sizes": {"Blue": 5, "Red": 5}, "coverage": 8,
        "confidence": "medium", "fetched": 0, "model": "logistic regression",
        "warnings": ["Demo data. These players are invented and no database, "
                     "game client or network was touched to build this."],
        "players": players,
        "prediction": {"own_probability": 0.5731, "win_probability": 0.5731,
                       "factors": [{"name": "d_rank", "value": 0.182},
                                   {"name": "d_acs", "value": -0.061},
                                   {"name": "d_kd", "value": 0.044}]},
    }
