"""Check everything the live view needs, in plain language.

Run by live.bat before it starts watching, so a missing piece produces a
sentence telling you what to do rather than a traceback forty lines later.

Exit codes are what the batch file branches on:

    0  ready -- the game is running and every dependency is present
    2  everything present, but VALORANT is not running yet
    1  something is missing that needs fixing first
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")

OK = "  [ok]  "
NO = "  [--]  "
BAD = "  [!!]  "


def check_all() -> tuple[list[str], list[str], bool]:
    """Returns (lines to print, problems to fix, whether the game is running)."""
    lines: list[str] = []
    problems: list[str] = []

    from valwr import config
    from valwr.live import lockfile
    from valwr.rating import potential as pot
    from valwr.store import schema

    s = config.load(require_key=False)

    if s.database_path.exists():
        conn = schema.connect(s.database_path)
        n = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        agents = conn.execute("SELECT COUNT(*) FROM ref_agents").fetchone()[0]
        lines.append(f"{OK}match database        {n:,} matches")
        if agents:
            lines.append(f"{OK}agent names           {agents} agents")
        else:
            lines.append(f"{BAD}agent names           empty -- agents will show as '?'")
            problems.append("Agent reference table is empty. Run:  "
                            ".venv\\Scripts\\python -m valwr.collect.seed")
        conn.close()
    else:
        lines.append(f"{BAD}match database        missing")
        problems.append(f"No database at {s.database_path}. Run the crawler "
                        f"first:  crawl.bat")

    model = s.database_path.parent.parent / "models" / "model.joblib"
    if model.exists():
        import joblib
        b = joblib.load(model)
        lines.append(f"{OK}win model             {b['best']}, "
                     f"{len(b['columns'])} features")
    else:
        lines.append(f"{BAD}win model             missing")
        problems.append("No trained model. Run:  "
                        ".venv\\Scripts\\python -m valwr.model.train")

    try:
        idx = pot.PerfIndex.load()
        lines.append(f"{OK}player score          fitted on {idx.n:,} samples")
    except FileNotFoundError:
        lines.append(f"{NO}player score          not built -- table will be skipped")
        problems.append("Optional. For the per-player 0-100 table, run:  "
                        ".venv\\Scripts\\python tools\\build_perf_index.py")

    try:
        if config.load().henrik_api_key:
            lines.append(f"{OK}API key               present")
        else:
            raise ValueError("empty")
    except Exception:
        lines.append(f"{NO}API key               missing -- cache-only mode")

    running = lockfile.game_is_running()
    lines.append((OK if running else NO) + "VALORANT              " +
                 ("running" if running else "not running yet"))
    return lines, problems, running


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="preflight")
    ap.add_argument("--wait", type=int, default=0,
                    help="seconds to wait for VALORANT to start (0 = do not)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    print()
    print("  Checking everything the live view needs")
    print("  " + "-" * 52)
    lines, problems, running = check_all()
    print("\n".join(lines))
    print()

    blocking = [p for p in problems if not p.startswith("Optional")]
    if blocking:
        print("  Fix these first:")
        for p in blocking:
            print(f"    - {p}")
        print()
        return 1

    for p in problems:
        print(f"  Note: {p}")
    if problems:
        print()

    if running:
        return 0

    if args.wait <= 0:
        return 2

    from valwr.live import lockfile
    print(f"  Waiting up to {args.wait // 60} minutes for VALORANT to start.")
    print("  Launch the game now -- this will pick it up automatically.")
    print("  Press Ctrl+C to give up.")
    deadline = time.time() + args.wait
    try:
        while time.time() < deadline:
            if lockfile.game_is_running():
                print("\n  VALORANT is up. Starting the live view.\n")
                return 0
            print("  .", end="", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        return 2
    print("\n  Gave up waiting.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
