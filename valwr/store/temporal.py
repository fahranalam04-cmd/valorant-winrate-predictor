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
# Every figure this project computes is about *competitive* play, because that
# is the only thing the model was trained on and the only thing it predicts.
# Swiftplay is shorter, customs are not matchmade, and pooling them would move
# a player's K/D without saying so.
#
# The database happens to hold nothing else today -- all 68,847 matches are
# competitive -- so this filter changes no current result. It is here because
# "correct as long as nobody ingests another queue" is not a guarantee, and
# the failure would be silent: every number would still look plausible.
COMPETITIVE = "competitive"

# Written as a correlated EXISTS against the matches primary key, and the form
# matters enormously. The first version was `mp.match_id IN (SELECT match_id
# FROM matches WHERE mode = ?)`, which rebuilds a list of every match in the
# database on each execution: 199 ms per history lookup against 0.37 ms for
# this one, measured over 200 players with identical rows returned. Every
# feature row runs several of these, so the retrain after that change managed
# one match per second -- about twenty hours for the full set -- and the live
# view paid the same cost on every lookup. EXISTS checks only the rows the
# (puuid, started_at) index already found, one primary-key probe each.
_ONLY_COMPETITIVE = (
    " AND EXISTS (SELECT 1 FROM matches m"
    " WHERE m.match_id = mp.match_id AND m.mode = ?)")


_SELECT = """
SELECT mp.match_id, mp.started_at, mp.puuid, mp.team, mp.agent, mp.tier,
       -- Pre-match identity, same class as tier: known before the first round
       -- and carrying no outcome information. Read by the above-rank flag in
       -- rating/potential.py.
       mp.account_level,
       mp.party_id, mp.score, mp.kills, mp.deaths, mp.assists,
       mp.headshots, mp.bodyshots, mp.legshots,
       mp.damage_dealt, mp.damage_taken,
       mp.map, mp.won,
       -- Round- and kill-derived components (Phase 3). Omitting these is not
       -- a visible error: per_round_rates() simply divides by a missing
       -- rounds_played, returns None for everything, and every performance
       -- feature silently collapses to its default. Keep this list in step
       -- with valwr/rating/components.py.
       mp.rounds_played, mp.first_bloods, mp.first_deaths, mp.multikills,
       mp.trade_kills, mp.traded_deaths, mp.kast_rounds, mp.clutches
FROM match_players mp
WHERE mp.puuid = ? AND mp.started_at < ?
"""


def _query(conn, puuid: str, as_of: int, extra: str = "", params: tuple = (),
           limit: int | None = None) -> list[sqlite3.Row]:
    sql = _SELECT + _ONLY_COMPETITIVE + extra + " ORDER BY mp.started_at DESC"
    args: list = [puuid, as_of, COMPETITIVE, *params]
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
             "WHERE mp.puuid = ? AND mp.started_at < ? AND mp.won IS NOT NULL"
             + _ONLY_COMPETITIVE + extra)
    args: list = [puuid, as_of, COMPETITIVE, *params]
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


@dataclass(frozen=True)
class Career:
    """Raw career totals, for display rather than for scoring.

    Nothing here is shrunk. The scoring path in rating/potential.py deliberately
    pulls its components toward the population prior, because four good games
    are not evidence of a good player. These are the actual totals, which is
    what a scoreboard should show. The two therefore disagree for anyone with a
    short history -- a player with four games can show a career ACS of 260 next
    to a component value near the population mean -- and that disagreement is
    correct rather than a bug.
    """
    games: int
    kills: int
    deaths: int
    assists: int
    headshots: int
    bodyshots: int
    legshots: int
    score: int
    rounds: int
    wins: int
    decided: int          # matches with a winner; `games` includes draws/aborts

    @property
    def acs(self) -> float | None:
        """Average combat score per round, the number the scoreboard shows."""
        return self.score / self.rounds if self.rounds else None

    @property
    def kd(self) -> float | None:
        if not self.games:
            return None
        # A player who has never died reports their kill count rather than
        # dividing by zero. Vanishingly rare over a career, but it is one row.
        return self.kills / self.deaths if self.deaths else float(self.kills)

    @property
    def headshot_rate(self) -> float | None:
        """Headshots over every shot that landed, not over kills."""
        shots = self.headshots + self.bodyshots + self.legshots
        return self.headshots / shots if shots else None

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.decided if self.decided else None


    @classmethod
    def from_rows(cls, rows) -> "Career":
        """The same tally from history rows already in memory.

        Used for per-map stats, where the rows have been fetched anyway and a
        second aggregate query would be pure waste. Takes anything indexable by
        column name -- sqlite3.Row or dict.
        """
        def total(key):
            return sum((r[key] or 0) for r in rows)

        return cls(games=len(rows), kills=total("kills"),
                   deaths=total("deaths"), assists=total("assists"),
                   headshots=total("headshots"), bodyshots=total("bodyshots"),
                   legshots=total("legshots"), score=total("score"),
                   rounds=total("rounds_played"),
                   wins=sum(1 for r in rows if r["won"]),
                   decided=sum(1 for r in rows if r["won"] is not None))


def _totals(conn, puuid: str, as_of: int, last_n: int | None = None) -> Career:
    inner = ("SELECT kills, deaths, assists, headshots, bodyshots, legshots, "
             "score, rounds_played, won FROM match_players mp "
             "WHERE puuid = ? AND started_at < ?" + _ONLY_COMPETITIVE)
    args: list = [puuid, as_of, COMPETITIVE]
    if last_n is not None:
        # Newest first, then cut. The LIMIT applies to matches before `as_of`,
        # never around it -- ordering after the time filter, not instead of it.
        inner += " ORDER BY started_at DESC LIMIT ?"
        args.append(last_n)
    row = conn.execute(
        "SELECT COUNT(*) games, "
        "COALESCE(SUM(kills), 0) kills, COALESCE(SUM(deaths), 0) deaths, "
        "COALESCE(SUM(assists), 0) assists, "
        "COALESCE(SUM(headshots), 0) headshots, "
        "COALESCE(SUM(bodyshots), 0) bodyshots, "
        "COALESCE(SUM(legshots), 0) legshots, "
        "COALESCE(SUM(score), 0) score, "
        "COALESCE(SUM(rounds_played), 0) rounds, "
        "COALESCE(SUM(won), 0) wins, COUNT(won) decided "
        f"FROM ({inner})", args).fetchone()
    return Career(games=row["games"], kills=row["kills"], deaths=row["deaths"],
                  assists=row["assists"], headshots=row["headshots"],
                  bodyshots=row["bodyshots"], legshots=row["legshots"],
                  score=row["score"], rounds=row["rounds"], wins=row["wins"],
                  decided=row["decided"])


def recent_totals(conn: sqlite3.Connection, puuid: str, as_of: int,
                  last_n: int = 20) -> Career:
    """The same tally over the most recent `last_n` matches only.

    Form rather than record. Worth knowing how thin this is in practice: the
    median player in this database has 8 stored matches and only 2.1% have more
    than 20, so for most people this returns their entire history and equals
    the career figure. The views say so rather than printing the same number
    twice under two headings.
    """
    return _totals(conn, puuid, as_of, last_n)


def career_totals(conn: sqlite3.Connection, puuid: str, as_of: int) -> Career:
    """Every counting stat for one player, in a single aggregate.

    One indexed range scan rather than materialising the whole history: a lobby
    of ten costs ten cheap sums, and the live view repeats them every five
    seconds.

    Time-gated like everything else in this module, strictly before `as_of`.
    """
    return _totals(conn, puuid, as_of)


def current_tier(conn: sqlite3.Connection, puuid: str, as_of: int) -> int | None:
    """Their competitive tier as of the most recent match we hold.

    Rank at the loading screen is not something the stored data carries
    directly -- what it has is the tier each player *was* in each match they
    played. The newest of those is the best available answer, and it is the
    same column the model already uses as a feature.

    It can lag: a player who climbed since their last stored game shows the
    older rank. That is why the views print it next to how fresh the data is.
    """
    row = conn.execute(
        "SELECT mp.tier FROM match_players mp "
        "WHERE mp.puuid = ? AND mp.started_at < ? AND mp.tier IS NOT NULL"
        + _ONLY_COMPETITIVE +
        " ORDER BY mp.started_at DESC LIMIT 1",
        (puuid, as_of, COMPETITIVE)).fetchone()
    return row["tier"] if row else None


def times_partied(conn: sqlite3.Connection, a: str, b: str, as_of: int) -> int:
    """How many earlier matches these two entered in the same party.

    The client does not tell you who the *enemy* queued with -- that would be
    a competitive advantage -- so a live lobby can only be inferred from
    history. Measured over 8,000 same-team pairs in 400 recent matches, "they
    have shared a party before" identifies a real party with 99.8% precision
    and 53.2% recall: when it fires it is almost never wrong, and it misses
    about half. So a view may say "these two queue together" on the strength
    of it, and must not say "solo" from its silence.
    """
    return conn.execute(
        "SELECT COUNT(*) FROM match_players x "
        "JOIN match_players y ON x.match_id = y.match_id "
        "                    AND x.party_id = y.party_id "
        "WHERE x.puuid = ? AND y.puuid = ? AND x.started_at < ? "
        "  AND x.party_id IS NOT NULL",
        (a, b, as_of)).fetchone()[0]


def match_parties(conn: sqlite3.Connection, match_id: str) -> dict[str, str]:
    """puuid -> party id, for a match already stored.

    Exact, unlike the inference above, because a finished match records who
    queued with whom. Only usable for a replay; a live lobby has no row yet.
    """
    return {r["puuid"]: r["party_id"] for r in conn.execute(
        "SELECT puuid, party_id FROM match_players WHERE match_id = ?",
        (match_id,)) if r["party_id"]}


# Columns knowable at the loading screen, before a single round is played.# Columns knowable at the loading screen, before a single round is played.
# Deliberately excludes score, kills, deaths, damage, won and the derived
# components -- those are the outcome of the match being predicted.
ROSTER_COLUMNS = ("match_id", "puuid", "team", "agent", "party_id",
                  "tier", "account_level")


def match_roster(conn, match_id: str) -> list[sqlite3.Row]:
    """Who is in this match, and nothing about how it went.

    The target match's roster is legitimate input -- at the loading screen you
    can see all ten players, their agents and their ranks. Its statistics are
    not. `SELECT *` would hand back kills, damage and `won` alongside, which is
    the shortest path to a model that predicts the past, so this returns only
    the pre-match columns and there is no variant that returns more.
    """
    cols = ", ".join(ROSTER_COLUMNS)
    return conn.execute(
        f"SELECT {cols} FROM match_players WHERE match_id = ?", (match_id,)
    ).fetchall()


def population_win_rate(conn, as_of: int) -> float:
    """Prior for shrinkage. ~0.5 by construction, but measure rather than assume."""
    row = conn.execute(
        "SELECT COUNT(*) games, COALESCE(SUM(won), 0) wins FROM match_players mp "
        "WHERE mp.started_at < ? AND mp.won IS NOT NULL" + _ONLY_COMPETITIVE,
        (as_of, COMPETITIVE)).fetchone()
    return (row["wins"] / row["games"]) if row["games"] else 0.5

# A match needs at least this many players present before finishing near
# the top of it means anything.
MIN_LOBBY = 6


def lobby_dominance(conn: sqlite3.Connection, puuid: str, as_of: int,
                    top_n: int = 2) -> tuple[float, int]:
    """How often this player finished near the top of their own lobby.

    Returns (fraction in the top `top_n` by ACS, matches counted). Chance is
    `top_n / 10`, so 0.20 by default -- measured population mean is 0.20 on the
    nose, which is the sanity check that this is computed correctly.

    This is the "visible irregularity" an account looks like it has: a player
    who tops the lobby in most of their games is doing something the rank
    around them is not. Measured on held-out data it lifts the top-third rate
    by 14.4 points, against 9.3 for band-relative performance alone and -0.2
    for account level, which does nothing whatsoever on its own.

    Time-gated like everything else in this module: strictly before `as_of`.
    The comparison is against the other players in each historical match, so
    it needs the lobby, not just the player's own row.
    """
    row = conn.execute(
        """
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN better < ? THEN 1 ELSE 0 END) AS top
        FROM (
            SELECT (SELECT COUNT(*) FROM match_players o
                    WHERE o.match_id = mp.match_id AND o.rounds_played > 0
                      AND 1.0 * o.score / o.rounds_played
                          > 1.0 * mp.score / mp.rounds_played) AS better
            FROM match_players mp
            WHERE mp.puuid = ? AND mp.started_at < ? AND mp.rounds_played > 0
              AND EXISTS (SELECT 1 FROM matches m
                          WHERE m.match_id = mp.match_id AND m.mode = ?)
              -- Only matches where we actually hold a lobby to compare
              -- against. Without this a partially-scraped match counts as a
              -- win by default, and a synthetic solo match scores a perfect
              -- 1.00 -- you cannot top a lobby of one. Costs nothing on real
              -- data: 100% of collected matches have all ten players.
              AND (SELECT COUNT(*) FROM match_players q
                   WHERE q.match_id = mp.match_id AND q.rounds_played > 0) >= ?
        )
        """, (top_n, puuid, as_of, COMPETITIVE, MIN_LOBBY)).fetchone()
    n = row["n"] or 0
    return ((row["top"] or 0) / n, n) if n else (0.0, 0)

