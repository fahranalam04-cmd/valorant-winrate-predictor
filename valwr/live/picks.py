"""Which agents you play best -- the question agent select actually asks you.

Ranked by how well you play each agent, not by how often you win on it. A win
rate is half your teammates', and at the sizes stored here it is mostly noise:
68 games across fourteen agents, 18 on the most played, where a win rate moves
about twelve points either way by chance alone. Match impact is the measure the
rest of the project uses for "played best", and it is yours alone.

Two sections. Agents with three or more games are your established picks,
ranked, each average pulled toward your own in proportion to how few games it
rests on: five virtual games at your average stand beside the real ones, so
three games count for three eighths of themselves. Agents played once or twice
sit underneath with what those games actually showed, unpulled -- pulling one
game five sixths of the way back would call nearly all of them "your usual",
which is disregarding them by another name. Each says how many games it rests
on, so it reads as a game or two rather than a habit.

This map's record sits beside every agent as context, never as the ranking:
one to three games per agent per map is all anyone has.

Read-only, and computed only in agent select, where it is the question.
"""

from __future__ import annotations

import sqlite3
from statistics import mean, pstdev

from valwr.rating import rating
from valwr.store import temporal

# Virtual games at your own average added to every ranked agent. Five keeps
# three games modest while letting eighteen speak for themselves.
SHRINK_GAMES = 5
# Games an agent needs to rank among your established picks. Fewer still
# appear, underneath.
ESTABLISHED_GAMES = 3
SHOWN_ESTABLISHED = 8
SHOWN_FEW = 6
# In units of your game-to-game spread, after shrinking.
ABOVE, WELL_ABOVE = 0.15, 0.35


def label(vs_usual: float) -> str:
    if vs_usual >= WELL_ABOVE:
        return "well above your usual"
    if vs_usual >= ABOVE:
        return "above your usual"
    if vs_usual <= -WELL_ABOVE:
        return "well below your usual"
    if vs_usual <= -ABOVE:
        return "below your usual"
    return "your usual"


def your_picks(conn: sqlite3.Connection, puuid: str, as_of: int,
               map_name: str | None, norms, roles_by_agent: dict[str, str]
               ) -> dict | None:
    """Your best agents, or None with nothing stored to judge them by."""
    rated = []
    for row in temporal.player_history(conn, puuid, as_of):
        r = dict(row)
        impact = rating.match_impact(r, norms)
        if impact is not None and r.get("agent"):
            rated.append((r, impact))
    if not rated:
        return None

    yours = mean(v for _, v in rated)
    spread = pstdev([v for _, v in rated]) or 1.0

    by_agent: dict[str, list[tuple[dict, float]]] = {}
    for r, v in rated:
        by_agent.setdefault(r["agent"], []).append((r, v))

    established, few = [], []
    for agent, games in by_agent.items():
        n = len(games)
        total = sum(v for _, v in games)
        if n >= ESTABLISHED_GAMES:
            vs = ((total + SHRINK_GAMES * yours) / (n + SHRINK_GAMES) - yours) / spread
        else:
            vs = (total / n - yours) / spread      # what the game or two showed
        here = [r for r, _ in games if map_name and r.get("map") == map_name]
        wins = sum(1 for r, _ in games if r.get("won"))
        (established if n >= ESTABLISHED_GAMES else few).append({
            "agent": agent,
            "role": roles_by_agent.get(agent),
            "games": n,
            "wins": wins,
            "losses": sum(1 for r, _ in games if r.get("won") is not None) - wins,
            "vs_usual": round(vs, 2),
            "label": label(vs),
            "here": {"games": len(here),
                     "wins": sum(1 for r in here if r.get("won")),
                     "losses": sum(1 for r in here if r.get("won") is not None
                                   and not r["won"])},
        })
    for group in (established, few):
        group.sort(key=lambda a: (-a["vs_usual"], -a["games"]))
    return {"map": map_name, "games": len(rated),
            "agents": established[:SHOWN_ESTABLISHED], "few": few[:SHOWN_FEW]}
