"""python -m valwr.collect [--minutes N] [--seed self|leaderboard|both|none]

Runs the crawler for a bounded time, then reports. Safe to interrupt: all
progress lives in the database, so a restart picks up where this left off.
"""

from __future__ import annotations

import argparse
import sys

from valwr import config
from valwr.collect import frontier, seed
from valwr.collect.client import HenrikClient
from valwr.collect.crawl import Crawler
from valwr.collect.limiter import TokenBucket
from valwr.store import raw, schema

BAND_NAMES = {
    0: "Unranked", 1: "Iron", 2: "Bronze", 3: "Silver", 4: "Gold",
    5: "Platinum", 6: "Diamond", 7: "Ascendant", 8: "Immortal", 9: "Radiant",
}


def report(conn, stats) -> None:
    print("\n" + "=" * 58)
    print("crawl summary")
    print("=" * 58)
    mins = stats.elapsed / 60
    print(f"  ran for            {mins:.1f} min")
    print(f"  requests           {stats.requests}  ({stats.requests / max(mins, 1e-9):.1f}/min)")
    print(f"  players fetched    {stats.players_fetched}")
    print(f"  matches new        {stats.matches_new}")
    print(f"  matches duplicate  {stats.matches_seen_again}")
    print(f"  puuids discovered  {stats.puuids_discovered}")
    print(f"  failures           {stats.failures}")
    print(f"  429s               {stats.rate_limit_hits}"
          f"{'   <-- limiter needs tuning' if stats.rate_limit_hits else ''}")

    states = frontier.counts_by_state(conn)
    print(f"\n  frontier: " + "  ".join(f"{k}={v}" for k, v in sorted(states.items())))

    store = raw.storage_summary(conn)
    print(f"  storage:  {store['responses']} responses, "
          f"{store['compressed_bytes'] / 1e6:.1f} MB compressed")

    # The health check. A distribution piled into one band means the sampling
    # is broken, and every downstream result inherits the problem.
    print("\n  rank distribution of crawled players (done):")
    bands = frontier.counts_by_band(conn)
    total = sum(b.get("done", 0) for b in bands.values()) or 1
    for band in sorted(bands):
        done = bands[band].get("done", 0)
        if not done:
            continue
        pct = done / total * 100
        bar = "#" * int(pct / 2)
        print(f"    {BAND_NAMES.get(band, f'band{band}'):<10} {done:>6}  {pct:>5.1f}%  {bar}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="valwr.collect")
    ap.add_argument("--minutes", type=float, default=30, help="how long to crawl")
    ap.add_argument("--seed", choices=["self", "leaderboard", "both", "none"],
                    default="self",
                    help="self: snowball from your account (reaches your band "
                         "+/- a few tiers). leaderboard: top of ladder, a "
                         "disconnected high-elo cluster. See seed.py.")
    ap.add_argument("--size", type=int, default=10, help="matches per request")
    ap.add_argument("--report-only", action="store_true", help="print state and exit")
    args = ap.parse_args(argv)

    s = config.load()
    conn = schema.connect(s.database_path)
    schema.create_all(conn)

    limiter = TokenBucket(s.requests_per_minute)
    print(f"limiter: {s.henrik_tier} tier, {limiter.stated_per_minute} req/min stated "
          f"-> {limiter.effective_per_minute} effective")

    with HenrikClient(s.henrik_api_key, conn=conn, limiter=limiter) as client:
        if args.report_only:
            from valwr.collect.crawl import CrawlStats
            report(conn, CrawlStats())
            return 0

        if args.seed in ("self", "both"):
            if not (s.riot_name and s.riot_tag):
                print("RIOT_NAME/RIOT_TAG not set -- cannot seed from self", file=sys.stderr)
                return 1
            puuid = seed.seed_self(conn, client, s.riot_name, s.riot_tag)
            print(f"seeded self: {s.riot_id} -> {puuid[:8]}...")
        if args.seed in ("leaderboard", "both"):
            n = seed.seed_leaderboard(conn, client, s.region, s.platform)
            print(f"seeded leaderboard: {n} new puuids")

        crawler = Crawler(conn, client, limiter, s.region, s.platform, size=args.size)
        print(f"crawling for {args.minutes} min ... (ctrl-c to stop; progress is saved)\n")
        try:
            crawler.run(args.minutes)
        except KeyboardInterrupt:
            print("\n  interrupted -- progress is saved")

    report(conn, crawler.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
