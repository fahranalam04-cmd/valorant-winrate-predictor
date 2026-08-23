"""Live crawl monitor: python -m valwr.collect.watch

Reads the database rather than the crawler's stdout, so it works no matter how
that process buffers -- and it can be started, stopped and restarted freely
without touching the crawl itself.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

from valwr import config
from valwr.collect import frontier
from valwr.store import raw, schema

BANDS = {0: "Unranked", 1: "Iron", 2: "Bronze", 3: "Silver", 4: "Gold",
         5: "Platinum", 6: "Diamond", 7: "Ascendant", 8: "Immortal", 9: "Radiant"}

CLEAR = "\033[2J\033[H"
BOLD, DIM, GREEN, YELLOW, RED, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m")


def snapshot(conn) -> dict:
    one = lambda sql: conn.execute(sql).fetchone()[0]
    return {
        "raw": one("SELECT COUNT(*) FROM raw_response"),
        "matches": one("SELECT COUNT(*) FROM crawl_seen_match"),
        "normalised": one("SELECT COUNT(*) FROM matches"),
        "players": one("SELECT COUNT(*) FROM frontier"),
        "last_fetch": one("SELECT COALESCE(MAX(fetched_at), 0) FROM raw_response"),
        "bytes": raw.storage_summary(conn)["compressed_bytes"],
        "states": frontier.counts_by_state(conn),
        "bands": frontier.counts_by_band(conn),
    }


def render(now: dict, start: dict, started_at: float) -> str:
    age = int(time.time() - now["last_fetch"]) if now["last_fetch"] else 999
    if age < 30:
        health, colour = "RUNNING", GREEN
    elif age < 120:
        health, colour = "SLOW / BACKING OFF", YELLOW
    else:
        health, colour = "STALLED?", RED

    mins = max((time.time() - started_at) / 60, 1e-9)
    d_match = now["matches"] - start["matches"]
    d_req = now["raw"] - start["raw"]
    st = now["states"]

    out = [
        f"{BOLD}valwr crawl monitor{RESET}   {datetime.now():%H:%M:%S}",
        "=" * 60,
        f"  status         {colour}{BOLD}{health}{RESET}  "
        f"{DIM}(last fetch {age}s ago){RESET}",
        "",
        f"  matches seen   {BOLD}{now['matches']:>8,}{RESET}   "
        f"{GREEN}+{d_match:,}{RESET} this session",
        f"  requests       {now['raw']:>8,}   {GREEN}+{d_req:,}{RESET}",
        f"  normalised     {now['normalised']:>8,}   {DIM}(run store.normalize){RESET}",
        f"  players known  {now['players']:>8,}",
        f"  storage        {now['bytes'] / 1e6:>8.1f} MB",
        "",
        f"  rate           {d_match / mins:>8.1f} matches/min   "
        f"{d_req / mins:.1f} req/min",
        "",
        "  frontier       " + "   ".join(
            f"{k}={v:,}" for k, v in sorted(st.items())),
        "",
        f"  {DIM}rank coverage of crawled players{RESET}",
    ]

    total = sum(b.get("done", 0) for b in now["bands"].values()) or 1
    for band in sorted(now["bands"]):
        done = now["bands"][band].get("done", 0)
        if not done:
            continue
        pct = done / total * 100
        out.append(f"    {BANDS.get(band, band):<10} {done:>6}  {pct:>5.1f}%  "
                   f"{'#' * int(pct / 2)}")

    out += ["", f"  {DIM}ctrl-c to close this monitor "
                f"(the crawl keeps running){RESET}"]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="valwr.collect.watch")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args(argv)

    os.system("")  # enables ANSI escapes on Windows terminals
    # Windows consoles default to cp1252; force UTF-8 so the output cannot
    # crash the monitor on an unencodable character.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)

    start = snapshot(conn)
    started_at = time.time()
    try:
        while True:
            print(CLEAR + render(snapshot(conn), start, started_at), flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nmonitor closed. the crawl is unaffected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
