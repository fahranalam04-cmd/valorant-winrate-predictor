"""Watch for a live match and predict it: python -m valwr.live

Read-only throughout. This never writes to the client API, never selects or
locks an agent, and never touches process memory. See docs/ETHICS-AND-TOS.md
-- that boundary is what separates a tolerated overlay from a ban.

The poll itself lives in `live/state.py` and the printing in `live/render.py`,
so the browser dashboard renders exactly the same data. Two views assembling
their own state is the failure mode this split exists to prevent.
"""

from __future__ import annotations

import argparse
import sys
import time

from valwr.live import lockfile, render
from valwr.live import state as st

POLL_SECONDS = 5.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="valwr.live")
    ap.add_argument("--once", action="store_true",
                    help="check a single time and exit")
    ap.add_argument("--no-fetch", action="store_true",
                    help="cache only; never spend API quota")
    ap.add_argument("--deadline", type=float, default=st.DEFAULT_DEADLINE,
                    help="seconds to spend resolving unknown players")
    args = ap.parse_args(argv)

    # Real gamertags in this dataset include Japanese characters, and Windows'
    # console defaults to cp1252 -- printing one raised UnicodeEncodeError in
    # testing. A teammate's name must not be able to kill the live view.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    print(f"client: {lockfile.describe()}")
    try:
        ctx = st.open_context(no_fetch=args.no_fetch, deadline=args.deadline)
    except st.NotReady as e:
        print(e)
        return 1

    print(f"model : {ctx.model_name}  ({len(ctx.bundle['columns'])} features)")
    if ctx.index is None:
        print("note  : no player-score index; run "
              "python tools/build_perf_index.py")
    print(f"account: {ctx.session.puuid[:8]}...  shard={ctx.session.shard}\n")
    top1 = ctx.index.top1_rate if ctx.index is not None else None

    seen: str | None = None
    try:
        while True:
            state = st.poll_once(ctx)
            if state is None:
                if args.once:
                    print("not in a match (lobby).")
                    return 0
                time.sleep(POLL_SECONDS)
                continue

            if state["match_id"] != seen:
                seen = state["match_id"]
                render.show(state, top1_rate=top1)

            if args.once:
                return 0
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0
    finally:
        ctx.close()


if __name__ == "__main__":
    raise SystemExit(main())
