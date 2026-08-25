"""Turning a live roster into players we know something about.

This is the hard constraint in the whole live path. Agent select lasts roughly
30 seconds. The HenrikDev basic tier sustains about 3 requests per minute, so
one uncached player costs ~20 seconds. Ten uncached players would take four
minutes, and no amount of client-side cleverness changes that -- three
different limiter designs already established the ceiling is the API's.

So the strategy is not "fetch faster", it is:

1. **Cache first.** History from hours ago is fine. Measured against real
   lobbies, about 7 of 10 players are already in the database, because the
   crawl was seeded from this account and preferentially collected the people
   it queues against.
2. **Own team first.** A partial answer about your own side is worth more than
   a uniformly incomplete one.
3. **Degrade, never block.** Return what is known with an explicit count, and
   let the caller widen its confidence rather than wait.

A fetch that does not finish in time is not an error. It is the normal case.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

from valwr.collect.client import HenrikError, RateLimited, TransientError
from valwr.live.roster import LiveMatch
from valwr.store import temporal


@dataclass
class Resolution:
    """Which players we can build features for, and which we cannot."""
    known: set[str] = field(default_factory=set)
    unknown: set[str] = field(default_factory=set)
    fetched: int = 0
    seconds: float = 0.0

    @property
    def coverage(self) -> int:
        return len(self.known)

    @property
    def confidence(self) -> str:
        """How much to trust a prediction built on this.

        Thresholds come from the measured coverage strata: a lobby where fewer
        than half the players are known predicts barely better than rank alone.
        """
        n = self.coverage
        if n >= 9:
            return "high"
        if n >= 7:
            return "moderate"
        if n >= 5:
            return "low"
        return "very low"

    def summary(self) -> str:
        return (f"{self.coverage}/10 players known "
                f"({self.fetched} fetched live, {self.seconds:.0f}s) "
                f"-- confidence {self.confidence}")


def has_history(conn: sqlite3.Connection, puuid: str, as_of: int) -> bool:
    """Do we already hold anything about this player from before `as_of`?"""
    return bool(temporal.player_history(conn, puuid, as_of, limit=1))


def order_for_fetching(match: LiveMatch, own_puuid: str) -> list[str]:
    """Own team first, then everyone else.

    Under a deadline the ordering decides what you end up knowing, so it is a
    deliberate choice rather than whatever the roster happened to list.
    """
    own_team = match.team_of(own_puuid)
    mine = [p.puuid for p in match.players if p.team == own_team]
    theirs = [p.puuid for p in match.players if p.team != own_team]
    return mine + theirs


def resolve(conn: sqlite3.Connection, match: LiveMatch, own_puuid: str,
            as_of: int, client=None, deadline_seconds: float = 25.0,
            region: str = "na", platform: str = "pc",
            on_progress=None) -> Resolution:
    """Resolve as many players as the deadline allows.

    `client` may be None, in which case this is cache-only -- useful for a
    dry run, and for the dashboard's first paint before any fetching starts.
    """
    out = Resolution()
    started = time.monotonic()

    ordered = order_for_fetching(match, own_puuid)
    for puuid in ordered:
        if has_history(conn, puuid, as_of):
            out.known.add(puuid)
        else:
            out.unknown.add(puuid)

    if on_progress:
        on_progress(out)

    if client is None or not out.unknown:
        out.seconds = time.monotonic() - started
        return out

    # Fetch the unknowns in priority order until the deadline.
    for puuid in [p for p in ordered if p in out.unknown]:
        if time.monotonic() - started >= deadline_seconds:
            break
        try:
            client.matches(region, platform, puuid, size=10, mode="competitive")
            out.fetched += 1
            # The crawler normalises inline, so a fetched player is queryable
            # immediately -- no separate parse step to wait on.
            if has_history(conn, puuid, as_of):
                out.known.add(puuid)
                out.unknown.discard(puuid)
            if on_progress:
                on_progress(out)
        except (RateLimited, TransientError):
            # Out of quota or off the network. Neither is worth waiting on
            # inside agent select; the cached answer is what ships.
            break
        except HenrikError:
            continue        # this player is unfetchable; the rest are not

    out.seconds = time.monotonic() - started
    return out
