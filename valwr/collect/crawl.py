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
from pathlib import Path
from typing import Callable

import sqlite3 as _sqlite3

from valwr.collect import frontier
from valwr.collect.client import (HenrikClient, HenrikError, RateLimited,
                                  TransientError)
from valwr.collect.limiter import TokenBucket
from valwr.store import normalize

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


# While VALORANT is running the crawler stands aside. The API key's quota is
# one fixed window a minute shared by every process using it, and the crawler
# is built to spend each window down -- so a lobby looked up just after it had
# done so waited up to a minute for quota, inside a minute-long agent select.
# Looking up one lobby can cost two thirds of a window on its own.
PAUSE_POLL_SECONDS = 5.0


def pause_marker_path() -> Path:
    """Refreshed while the crawler is paused, so the watchdog can tell a
    crawler idling on purpose from one that has hung."""
    from valwr import config
    return config.load(require_key=False).database_path.parent / "crawl-paused"


class Crawler:
    def __init__(
        self,
        conn: sqlite3.Connection,
        client: HenrikClient,
        limiter: TokenBucket,
        region: str,
        platform: str,
        size: int = 10,
        game_running: Callable[[], bool] | None = None,
        pause_marker: Path | None = None,
    ):
        self.conn = conn
        self.client = client
        self.limiter = limiter
        self.region = region
        self.platform = platform
        self.size = size
        self.stats = CrawlStats()
        self._transient_streak = 0
        # Passed in by whoever starts a real crawl rather than defaulted here,
        # so a test never pauses because VALORANT happens to be open on the
        # machine running it.
        self.game_running = game_running
        self.pause_marker = pause_marker

    def _normalise(self, doc: dict) -> int:
        """Normalise this response inline, in the crawler process.

        SQLite allows exactly one writer. Running the normaliser as a separate
        process against a live crawl deadlocks whichever one loses the race --
        both directions were observed: 59 crawler runs died with "database is
        locked", and later the normaliser did. Doing it here makes the crawler
        the only writer that ever exists, so analysis processes are pure
        readers and WAL handles them concurrently without contention.

        It also keeps the normalised tables continuously current, instead of
        drifting behind raw until someone remembers to catch them up.
        """
        done = 0
        for m in doc.get("data") or []:
            try:
                row, players, _ = normalize.parse_match(m)
            except normalize.ParseError:
                continue
            normalize.upsert_match(self.conn, row)
            if players:
                normalize.upsert_players(self.conn, players)
            done += 1
        return done

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
            # cannot be bypassed by a caller that forgets. See docs/API-NOTES.md.
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
        self._normalise(doc)
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
            # Checked before claiming, so a pause never holds a player's claim.
            if not self._stand_aside_for_the_game(deadline):
                break
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

    def _stand_aside_for_the_game(self, deadline: float) -> bool:
        """Wait while VALORANT is running. False if the run's time ran out."""
        if self.game_running is None or not self.game_running():
            return True
        print("  VALORANT is running -- paused so the dashboard has the whole "
              "quota; crawling resumes when the game closes", flush=True)
        while time.monotonic() < deadline:
            self._mark_paused()
            time.sleep(PAUSE_POLL_SECONDS)
            if not self.game_running():
                break
        else:
            # Out of time with the game still open. The marker stays: the next
            # run pauses again at once, and a watchdog looking in between must
            # not read the gap as a hang.
            return False
        self._clear_paused()
        print("  VALORANT closed -- crawling again", flush=True)
        return True

    def _mark_paused(self) -> None:
        if self.pause_marker is None:
            return
        try:
            self.pause_marker.write_text(str(int(time.time())), encoding="utf-8")
        except OSError:
            pass                     # the pause matters; the marker is a courtesy

    def _clear_paused(self) -> None:
        if self.pause_marker is not None:
            try:
                self.pause_marker.unlink(missing_ok=True)
            except OSError:
                pass

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
