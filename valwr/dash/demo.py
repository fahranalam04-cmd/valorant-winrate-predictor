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

# A plausible spread: mostly Platinum with a Diamond and one unranked.
from valwr.rating import ranks as _r
from valwr.rating import roles as _roles
from valwr.rating.role_score import COMPONENT_LABELS as _LABELS
from valwr.rating.role_score import MIN_MAP_GAMES as _MAP_GATE
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
            "rank": RANKS[i],
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
        "parties": PARTIES,
        "warnings": ["Demo data. These players are invented and no database, "
                     "game client or network was touched to build this."],
        "players": players,
        "prediction": {"own_probability": 0.5731, "win_probability": 0.5731,
                       "factors": [{"name": "d_rank", "value": 0.182},
                                   {"name": "d_acs", "value": -0.061},
                                   {"name": "d_kd", "value": 0.044}]},
    }
