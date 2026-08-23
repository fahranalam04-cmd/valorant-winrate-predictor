"""Supervised crawling: python -m valwr.collect.supervise

An overnight run died at 03:52 with nothing in the Windows event log -- no
sleep, no crash, no network event. Absent evidence, the useful response is not
a better guess; it is to make the cause irrelevant and to leave a record next
time.

This wrapper restarts the crawler whenever it exits for any reason, and writes
a timestamped log to data/crawl.log so the next unexplained stop explains
itself. Progress lives in SQLite, so a restart resumes rather than repeats.

It cannot survive its own console window being closed. For that, see the
Scheduled Task in docs/ROADMAP.md.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime

from valwr import config
from valwr.collect import frontier, seed
from valwr.collect.client import HenrikClient
from valwr.collect.crawl import Crawler
from valwr.collect.keepawake import KeepAwake
from valwr.collect.limiter import TokenBucket
from valwr.store import schema

RESTART_DELAY = 30.0


def log(path, msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="valwr.collect.supervise")
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--allow-sleep", action="store_true")
    args = ap.parse_args(argv)

    s = config.load()
    conn = schema.connect(s.database_path)
    schema.create_all(conn)
    logfile = s.database_path.parent / "crawl.log"

    deadline = time.time() + args.hours * 3600
    attempt = 0

    log(logfile, f"=== supervisor start: {args.hours}h budget, pid {__import__('os').getpid()} ===")

    with KeepAwake(enabled=not args.allow_sleep) as awake:
        log(logfile, f"power: {awake.status}")
        while time.time() < deadline:
            attempt += 1
            remaining_min = (deadline - time.time()) / 60
            try:
                limiter = TokenBucket(s.requests_per_minute)
                with HenrikClient(s.henrik_api_key, conn=conn, limiter=limiter) as client:
                    crawler = Crawler(conn, client, limiter, s.region, s.platform,
                                      size=args.size)
                    log(logfile, f"run #{attempt} starting ({remaining_min:.0f} min left)")
                    stats = crawler.run(remaining_min, verbose=True)
                    log(logfile,
                        f"run #{attempt} returned: {stats.players_fetched} players, "
                        f"{stats.matches_new} new matches, {stats.rate_limit_hits} 429s, "
                        f"{stats.transient_errors} net errors")
            except KeyboardInterrupt:
                log(logfile, "interrupted by user -- progress saved")
                return 0
            except Exception:
                # Anything unexpected is logged in full and then survived.
                # The whole point is that an unattended run does not end
                # because of one bad hour.
                log(logfile, f"run #{attempt} CRASHED:\n{traceback.format_exc()}")

            if time.time() >= deadline:
                break
            pending = frontier.counts_by_state(conn).get("pending", 0)
            if not pending:
                log(logfile, "frontier empty -- nothing left to crawl")
                break
            log(logfile, f"restarting in {RESTART_DELAY:.0f}s ({pending:,} pending)")
            time.sleep(RESTART_DELAY)

    log(logfile, "=== supervisor finished ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
