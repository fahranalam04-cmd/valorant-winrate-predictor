"""Restart the crawler if it has stopped: python -m valwr.collect.watchdog

The supervisor inside the crawler handles crashes, but it cannot survive its
own process being killed -- and that has now happened twice, both times with
no traceback, no clean exit and no Windows event. The last log line each time
was `run #N starting`, which is what abrupt termination looks like from the
inside: nothing.

So this runs *outside* the crawler, from Task Scheduler, and asks the only
question that matters: has anything been fetched recently? A hung process that
holds a PID while fetching nothing is just as dead as a missing one, so
liveness is measured from the database rather than the process table.

Safe to run on a timer. It starts a crawler only when one is genuinely needed,
because two crawlers share one quota and would spend it twice.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from valwr import config
from valwr.store import schema

# No fetch in this long means stopped. Generous: at the Basic tier's ~2.5
# req/min a legitimate quota stall can approach a minute, and restarting a
# healthy crawler is worse than waiting one more cycle.
STALE_SECONDS = 420

# A process exists but nothing has been fetched for this long -- hung rather
# than merely throttled. Kill it before starting a replacement.
HUNG_SECONDS = 1200

REPO = Path(__file__).resolve().parent.parent.parent


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  [watchdog] {msg}"
    print(line, flush=True)
    try:
        with open(REPO / "data" / "crawl.log", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def seconds_since_last_fetch() -> float | None:
    s = config.load(require_key=False)
    if not s.database_path.exists():
        return None
    conn = schema.connect(s.database_path)
    row = conn.execute("SELECT MAX(fetched_at) t FROM raw_response").fetchone()
    conn.close()
    if row is None or row["t"] is None:
        return None
    return time.time() - row["t"]


def crawler_pids() -> list[int]:
    """PIDs of running crawler processes, via WMIC command-line matching."""
    try:
        out = subprocess.run(
            ["wmic", "process", "where",
             "name='pythonw.exe' or name='python.exe'",
             "get", "ProcessId,CommandLine", "/format:csv"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for line in out.splitlines():
        if "valwr.collect.supervise" in line:
            for part in reversed(line.strip().split(",")):
                if part.isdigit():
                    pids.append(int(part))
                    break
    return pids


def start_crawler(hours: float) -> None:
    pythonw = REPO / ".venv" / "Scripts" / "pythonw.exe"
    exe = pythonw if pythonw.exists() else Path(sys.executable)
    subprocess.Popen(
        [str(exe), "-u", "-m", "valwr.collect.supervise", "--hours", str(hours)],
        cwd=str(REPO),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def kill(pid: int) -> None:
    subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                   capture_output=True, timeout=30)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="valwr.collect.watchdog")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    age = seconds_since_last_fetch()
    pids = crawler_pids()

    if age is None:
        log("no fetches recorded yet; leaving it alone")
        return 0

    if age < STALE_SECONDS:
        return 0                      # healthy: stay quiet, this runs on a timer

    if pids and age < HUNG_SECONDS:
        log(f"stale {age:.0f}s but {len(pids)} process(es) alive -- likely a "
            f"quota stall, waiting")
        return 0

    if pids:
        log(f"hung: {age:.0f}s since last fetch with {len(pids)} process(es). "
            f"killing {pids}")
        if not args.dry_run:
            for pid in pids:
                kill(pid)
            time.sleep(5)

    log(f"restarting crawler ({age:.0f}s since last fetch)")
    if not args.dry_run:
        start_crawler(args.hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
