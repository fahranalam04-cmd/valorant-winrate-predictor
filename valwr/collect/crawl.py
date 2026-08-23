"""The crawl loop.

Claim a PUUID, fetch its recent competitive matches, store the response
verbatim, harvest the other nine players into the frontier, repeat.

The efficiency lever against the rate limit: one matchlist request returns up
to `size` full matches, each carrying all ten players. A single request can
therefore yield ~10 matches and discover up to ~90 new PUUIDs.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

import sqlite3 as _sqlite3

from valwr.collect import frontier
from valwr.collect.client import (HenrikClient, HenrikError, RateLimited,
                                  TransientError)
from valwr.collect.limiter import TokenBucket

# Cap on the retry backoff for network failures. Long enough to ride out a
# suspend/resume cycle or a router reboot, short enough that a run recovers
# promptly once connectivity returns.
MAX_TRANSIENT_BACKOFF = 300.0


@dataclass
class CrawlStats:
    requests: int = 0
    matches_new: int = 0
    matches_seen_again: int = 0
    puuids_discovered: int = 0
    players_fetched: int = 0
    failures: int = 0
    rate_limit_hits: int = 0
    transient_errors: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at


def harvest(doc: dict) -> list[tuple[str, list[tuple[str, int | None]]]]:
    """Pull (match_id, [(puuid, tier), ...]) out of a matchlist response.

    Deliberately shallow -- the frontier needs puuids and tiers, nothing else.
    Full parsing is Phase 2's job and reads from the stored raw body.
    """
    out = []
    for m in doc.get("data") or []:
        match_id = (m.get("metadata") or {}).get("match_id")
        if not match_id:
            continue
        players = [
            (p["puuid"], (p.get("tier") or {}).get("id"))
            for p in (m.get("players") or [])
            if p.get("puuid")
        ]
        out.append((match_id, players))
    return out


class Crawler:
    def __init__(
        self,
        conn: sqlite3.Connection,
        client: HenrikClient,
        limiter: TokenBucket,
        region: str,
        platform: str,
        size: int = 10,
    ):
        self.conn = conn
        self.client = client
        self.limiter = limiter
        self.region = region
        self.platform = platform
        self.size = size
        self.stats = CrawlStats()
        self._transient_streak = 0

    def _note_matches(self, harvested) -> None:
        now = int(time.time())
        for match_id, players in harvested:
            tiers = [t for _, t in players if t]
            band = frontier.band_of(int(sum(tiers) / len(tiers)) if tiers else None)
            cur = self.conn.execute(
                "INSERT INTO crawl_seen_match (match_id, first_seen_at, tier_band) "
                "VALUES (?,?,?) ON CONFLICT(match_id) DO NOTHING",
                (match_id, now, band),
            )
            if cur.rowcount:
                self.stats.matches_new += 1
            else:
                self.stats.matches_seen_again += 1

            self.stats.puuids_discovered += frontier.enqueue_many(self.conn, players)
        self.conn.commit()

    def _fetch_one(self, puuid: str) -> bool:
        """Fetch one player's matchlist. Returns False if it should be retried."""
        try:
            # No acquire() here: the limiter is wired into HenrikClient, so it
            # cannot be bypassed by a caller that forgets. See CLAUDE.md.
            doc = self.client.matches(
                self.region, self.platform, puuid, size=self.size, mode="competitive"
            )
            self.stats.requests += 1
        except RateLimited as e:
            # Our accounting disagreed with the server's. Back off and retry
            # this same puuid -- it is not the puuid's fault, so no attempt
            # is charged against it.
            self.stats.rate_limit_hits += 1
            wait = e.retry_after or 60.0
            print(f"  [429] rate limited, backing off {wait:.0f}s")
            self.limiter.penalise(wait)
            time.sleep(wait)
            return False
        except TransientError as e:
            # Network-level, not this puuid's fault. Back off and retry the
            # same player rather than charging it an attempt.
            self.stats.transient_errors += 1
            self._transient_streak += 1
            wait = min(60.0 * 2 ** (self._transient_streak - 1), MAX_TRANSIENT_BACKOFF)
            print(f"  [net] {e} -- retrying in {wait:.0f}s "
                  f"(streak {self._transient_streak})")
            time.sleep(wait)
            return False
        except HenrikError as e:
            frontier.fail(self.conn, puuid, str(e))
            self.stats.failures += 1
            return True

        self._transient_streak = 0
        self._note_matches(harvest(doc))
        frontier.complete(self.conn, puuid)
        self.stats.players_fetched += 1
        return True

    def run(self, minutes: float, verbose: bool = True) -> CrawlStats:
        try:
            recovered = frontier.recover_stale(self.conn)
        except _sqlite3.OperationalError as e:
            # An analysis pass holds the write lock. Wait rather than crash --
            # the supervisor would only restart into the same contention.
            print(f"  [db] {e} -- waiting for the writer to finish")
            time.sleep(60)
            recovered = 0
        if recovered:
            print(f"  recovered {recovered} stale claim(s) from a previous run")

        deadline = time.monotonic() + minutes * 60
        last_reported = 0
        while time.monotonic() < deadline:
            row = frontier.claim(self.conn)
            if row is None:
                print("  frontier empty -- nothing left to crawl")
                break

            try:
                while not self._fetch_one(row["puuid"]):
                    if time.monotonic() >= deadline:
                        # Out of time mid-backoff. Release rather than fail --
                        # our deadline is not this puuid's fault, and charging
                        # an attempt would eventually blacklist it.
                        frontier.release(self.conn, row["puuid"])
                        break
            except KeyboardInterrupt:
                frontier.release(self.conn, row["puuid"])
                raise

            done = self.stats.players_fetched
            if verbose and done and done != last_reported and done % 10 == 0:
                self._progress()
                last_reported = done

        return self.stats

    def _progress(self) -> None:
        s = self.stats
        pend = frontier.counts_by_state(self.conn).get("pending", 0)
        rate = s.requests / max(s.elapsed / 60, 1e-9)
        quota = self.limiter.server_remaining
        quota_s = f" | quota {quota}/{self.limiter.server_limit}" if quota is not None else ""
        print(
            f"  {s.players_fetched:>5} players | {s.matches_new:>6} new matches | "
            f"{pend:>6} pending | {rate:.1f} req/min{quota_s}"
        )
