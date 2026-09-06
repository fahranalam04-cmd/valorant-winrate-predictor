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
from valwr.store import normalize, temporal


# "Known" has to mean known *recently*. Treating any stored row as sufficient
# froze the local player's score for twelve days: `has_history` found an
# August match, marked the account known, and nothing ever refetched it. The
# score was recomputed every game from identical history and never moved.
#
# Two hours, measured rather than guessed. Over 1,484 player-appearances with
# prior history, 19.3% of players queue again within an hour and 22.4% within
# three -- so a twelve-hour rule called a player "current" whose stored history
# was already missing the games they had just played. That matters most for the
# last-20 K/D on every row, which is computed from stored history and looks
# entirely plausible while being several games behind.
#
# The budget allows it. The API grants 30 requests a minute, so a whole ten-man
# lobby costs about 20 seconds against a 25-second deadline; being aggressive
# here spends capacity that was otherwise idle.
STALE_AFTER_SECONDS = 2 * 3600

# Below this many stored matches the score is mostly prior anyway, so it is
# worth a fetch even if the newest row is recent.
THIN_HISTORY = 5

# One matchlist call returns ten matches. The form figure on every row is the
# last twenty, so one page cannot fill it: a player we have never crawled
# deeply shows a "last 20" padded out with whatever older games we happen to
# hold, which is a different sample from their actual last twenty.
#
# Measured against tracker.gg on one account. Our twenty reached back to Aug 23
# and read 0.914; their twenty stopped at Aug 26 and read 0.952. We were
# missing six of their last twenty games -- not because the API withholds them
# but because we only ever asked for the first page. Fetching the second
# reproduced their figure exactly: 318/334 = 0.9521 against 0.9521.
PAGE_SIZE = 10
FORM_WINDOW = 20                       # keep in step with potential.RECENT_GAMES
HISTORY_PAGES = -(-FORM_WINDOW // PAGE_SIZE)


def stored_depth(conn: sqlite3.Connection, puuid: str, as_of: int) -> int:
    """How many competitive matches we hold for them, up to the form window."""
    return len(temporal.player_history(conn, puuid, as_of, limit=FORM_WINDOW))


@dataclass
class Resolution:
    """Which players we can build features for, and which we cannot."""
    known: set[str] = field(default_factory=set)
    unknown: set[str] = field(default_factory=set)
    stale: set[str] = field(default_factory=set)
    fetched: int = 0
    refreshed: int = 0
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
        got = f"{self.coverage}/10 players known"
        how = [f"{self.fetched} fetched"] if self.fetched else []
        if self.refreshed:
            how.append(f"{self.refreshed} refreshed")
        if self.stale:
            how.append(f"{len(self.stale)} still stale")
        detail = f" ({', '.join(how)}, {self.seconds:.0f}s)" if how else ""
        return f"{got}{detail} -- confidence {self.confidence}"


def has_history(conn: sqlite3.Connection, puuid: str, as_of: int) -> bool:
    """Do we already hold anything about this player from before `as_of`?"""
    return bool(temporal.player_history(conn, puuid, as_of, limit=1))


def newest_match(conn: sqlite3.Connection, puuid: str, as_of: int) -> int | None:
    """Start time of the most recent stored match before `as_of`, if any."""
    row = conn.execute(
        "SELECT MAX(started_at) FROM match_players "
        "WHERE puuid = ? AND started_at < ?", (puuid, as_of)).fetchone()
    return row[0] if row and row[0] else None


def is_stale(conn: sqlite3.Connection, puuid: str, as_of: int) -> bool:
    """Is what we hold too old, or too little, to describe this player now?

    A player with no history at all is *unknown*, not stale -- the caller
    distinguishes them because they cost the same fetch but mean different
    things on screen.
    """
    newest = newest_match(conn, puuid, as_of)
    if newest is None:
        return False
    if as_of - newest > STALE_AFTER_SECONDS:
        return True
    return len(temporal.player_history(conn, puuid, as_of,
                                       limit=THIN_HISTORY)) < THIN_HISTORY


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
            if is_stale(conn, puuid, as_of):
                out.stale.add(puuid)
        else:
            out.unknown.add(puuid)

    if on_progress:
        on_progress(out)

    # The local account is always worth one call, so it alone is not enough
    # reason to stop here. This return used to fire whenever the lobby looked
    # current, which skipped the unconditional refresh below entirely -- the
    # comment said "unconditionally" and the code never reached it.
    if client is None or not (out.unknown or out.stale or own_puuid):
        out.seconds = time.monotonic() - started
        return out

    def fetch(puuid: str, start: int = 0) -> bool:
        """One matchlist page. True if it landed, False to stop fetching."""
        try:
            payload = client.matches(region, platform, puuid, size=PAGE_SIZE,
                                     mode="competitive", start=start)
            # The client only fetches and caches the raw body -- it does not
            # normalise. Without this the fetch stored a blob nothing read and
            # the player stayed exactly as unknown as before, which is why the
            # live path appeared to work while never actually learning
            # anything. Storing here is what makes the next line true.
            normalize.ingest(conn, payload)
            if has_history(conn, puuid, as_of):
                out.known.add(puuid)
                out.unknown.discard(puuid)
            out.stale.discard(puuid)
            if on_progress:
                on_progress(out)
            return True
        except (RateLimited, TransientError):
            # Out of quota or off the network. Neither is worth waiting on
            # inside agent select; the cached answer is what ships.
            return False
        except HenrikError:
            out.stale.discard(puuid)    # unfetchable; do not keep retrying it
            return True                 # the rest of the lobby still can be

    # The local account first and *genuinely* unconditionally, before any
    # deadline accounting. This comment claimed "unconditionally" while the
    # code gated it on the staleness rule above, so finishing a game and
    # requeueing five minutes later left your own last-20 missing the game you
    # had just played -- the one row you actually read.
    #
    # It costs one call, it is the account nothing else keeps current (the
    # crawl follows the players it discovers, not the person running this), and
    # skipping it is what left the reported score frozen for twelve days.
    keep_going = True
    was_unknown = own_puuid in out.unknown
    if own_puuid:
        keep_going = fetch(own_puuid)
        if keep_going:
            out.fetched += was_unknown
            out.refreshed += not was_unknown

    # Everything else, as (player, page) work in priority order. Under a
    # deadline the ordering decides what you end up knowing, so it is explicit:
    #
    #   1. the rest of your own history, because your own row is the one read
    #      every game and it is the only account nothing else keeps current
    #   2. page one for players we know nothing about -- a missing player
    #      costs the prediction more than a shallow one
    #   3. page one for players whose data is merely old
    #   4. page two for anyone still short of the form window, which is what
    #      makes their "last 20" actually their last 20
    others = [p for p in ordered if p != own_puuid]
    work: list[tuple[str, int]] = []
    if own_puuid:
        work += [(own_puuid, page * PAGE_SIZE)
                 for page in range(1, HISTORY_PAGES)]
    work += [(p, 0) for p in others if p in out.unknown]
    work += [(p, 0) for p in others if p in out.stale]
    work += [(p, page * PAGE_SIZE) for p in others
             for page in range(1, HISTORY_PAGES)
             if stored_depth(conn, p, as_of) < FORM_WINDOW]

    for puuid, start in work:
        if not keep_going or time.monotonic() - started >= deadline_seconds:
            break
        was_unknown = puuid in out.unknown
        keep_going = fetch(puuid, start)
        if keep_going and start == 0:
            out.fetched += was_unknown
            out.refreshed += not was_unknown

    out.seconds = time.monotonic() - started
    return out
