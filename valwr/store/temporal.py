"""The temporal query layer -- the leakage firewall.

This is the ONLY sanctioned way for feature code to read player history.
Feature modules must never query `match_players` directly; the audit in
test/test_leakage.py enforces that.

Every function here takes `as_of` as a required positional argument and filters
`started_at < as_of`, strictly. Not `<=`: a match must never inform a
prediction about itself.

The design goal is not convenience, it is that leakage should be hard to write
by accident. A single chokepoint that always demands `as_of` is far more
reliable than remembering to add a time filter in twenty separate places. See
"Why this design and not a feature store" in docs/DATA.md before optimising.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

# started_at is denormalised onto match_players (see docs/DATA.md), so history
# is one index range scan rather than a join fanning out per match.
_SELECT = """
SELECT mp.match_id, mp.started_at, mp.puuid, mp.team, mp.agent, mp.tier,
       mp.party_id, mp.score, mp.kills, mp.deaths, mp.assists,
       mp.headshots, mp.bodyshots, mp.legshots,
       mp.damage_dealt, mp.damage_taken,
       mp.map, mp.won
FROM match_players mp
WHERE mp.puuid = ? AND mp.started_at < ?
"""


def _query(conn, puuid: str, as_of: int, extra: str = "", params: tuple = (),
           limit: int | None = None) -> list[sqlite3.Row]:
    sql = _SELECT + extra + " ORDER BY mp.started_at DESC"
    args: list = [puuid, as_of, *params]
    if limit is not None:
        sql += " LIMIT ?"
        args.append(limit)
    return conn.execute(sql, args).fetchall()


def player_history(conn, puuid: str, as_of: int,
                   limit: int | None = None) -> list[sqlite3.Row]:
    """Matches for `puuid` that had already started before `as_of`.

    `as_of` is the start time of the match being predicted.
    """
    return _query(conn, puuid, as_of, limit=limit)


def player_history_on_map(conn, puuid: str, as_of: int, map_name: str,
                          limit: int | None = None) -> list[sqlite3.Row]:
    return _query(conn, puuid, as_of, " AND mp.map = ?", (map_name,), limit)


def player_history_on_agent(conn, puuid: str, as_of: int, agent: str,
                            limit: int | None = None) -> list[sqlite3.Row]:
    return _query(conn, puuid, as_of, " AND mp.agent = ?", (agent,), limit)


def player_history_map_agent(conn, puuid: str, as_of: int, map_name: str,
                             agent: str, limit: int | None = None) -> list[sqlite3.Row]:
    """The sparse cell -- 'better on Ascent specifically as Jett'.

    Expect very few rows. Whatever consumes this must shrink hard toward a
    prior; see rule 4 in CLAUDE.md.
    """
    return _query(conn, puuid, as_of, " AND mp.map = ? AND mp.agent = ?",
                  (map_name, agent), limit)


@dataclass(frozen=True)
class Record:
    """Win/loss tally. `games` counts only matches with a decided winner."""
    games: int
    wins: int

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.games if self.games else None

    def shrunk(self, prior_rate: float, prior_weight: float) -> float:
        """Empirical-Bayes shrinkage toward a prior.

        Three games at 100% is not a 100% player. Every rate feature in this
        project goes through something like this -- see docs/MODELING.md.
        """
        return (self.wins + prior_weight * prior_rate) / (self.games + prior_weight)


def _record(conn, puuid: str, as_of: int, extra: str = "",
            params: tuple = (), last_n: int | None = None) -> Record:
    inner = ("SELECT mp.won FROM match_players mp "
             "WHERE mp.puuid = ? AND mp.started_at < ? AND mp.won IS NOT NULL" + extra)
    args: list = [puuid, as_of, *params]
    if last_n is not None:
        inner += " ORDER BY mp.started_at DESC LIMIT ?"
        args.append(last_n)
    row = conn.execute(
        f"SELECT COUNT(*) games, COALESCE(SUM(won), 0) wins FROM ({inner})", args
    ).fetchone()
    return Record(games=row["games"], wins=row["wins"])


def record(conn, puuid: str, as_of: int) -> Record:
    return _record(conn, puuid, as_of)


def record_on_map(conn, puuid: str, as_of: int, map_name: str) -> Record:
    return _record(conn, puuid, as_of, " AND mp.map = ?", (map_name,))


def record_on_agent(conn, puuid: str, as_of: int, agent: str) -> Record:
    return _record(conn, puuid, as_of, " AND mp.agent = ?", (agent,))


def record_map_agent(conn, puuid: str, as_of: int, map_name: str, agent: str) -> Record:
    return _record(conn, puuid, as_of, " AND mp.map = ? AND mp.agent = ?",
                   (map_name, agent))


def recent_record(conn, puuid: str, as_of: int, last_n: int = 20) -> Record:
    """Form. Note the LIMIT applies to matches before `as_of`, never around it."""
    return _record(conn, puuid, as_of, last_n=last_n)


def population_win_rate(conn, as_of: int) -> float:
    """Prior for shrinkage. ~0.5 by construction, but measure rather than assume."""
    row = conn.execute(
        "SELECT COUNT(*) games, COALESCE(SUM(won), 0) wins FROM match_players "
        "WHERE started_at < ? AND won IS NOT NULL", (as_of,)
    ).fetchone()
    return (row["wins"] / row["games"]) if row["games"] else 0.5
