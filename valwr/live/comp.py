"""Which role a teammate is likely to play, before they have picked.

Agent select leaves about a minute, and the useful thing to know early is
which role your team is about to be missing. Whoever has locked or hovered an
agent has said; for everyone else, the role they have played most over their
last twenty competitive games is the best guess there is, given with its share
so a 19-of-20 main reads differently from an 8-of-20 flex player. Recent
rather than lifetime, because people change roles and twenty games is how far
back the rest of the card looks.

Ties go to the role played most recently. The page turns these per-player
facts into the team's strip and its open roles, so the dashboard, the replay
and the published demo all arrange them the same way.
"""

from __future__ import annotations

import sqlite3
from collections import Counter

from valwr.store import temporal

RECENT_GAMES = 20                     # keep in step with potential.RECENT_GAMES


def likely_role(conn: sqlite3.Connection, puuid: str, as_of: int,
                roles_by_agent: dict[str, str]) -> dict | None:
    """{"role", "games", "of"} over their last 20 games, or None unknown."""
    roles = Counter()
    for row in temporal.player_history(conn, puuid, as_of, limit=RECENT_GAMES):
        role = roles_by_agent.get(dict(row).get("agent"))
        if role:
            roles[role] += 1        # newest first, so a tie keeps the latest
    if not roles:
        return None
    role, games = roles.most_common(1)[0]
    return {"role": role, "games": games, "of": sum(roles.values())}
