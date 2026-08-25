"""Watch for a live match and predict it: python -m valwr.live

Read-only throughout. This never writes to the client API, never selects or
locks an agent, and never touches process memory. See docs/ETHICS-AND-TOS.md
-- that boundary is what separates a tolerated overlay from a ban.
"""

from __future__ import annotations

import argparse
import time

from valwr import config
from valwr.collect.client import HenrikClient
from valwr.collect.limiter import TokenBucket
from valwr.live import lockfile, predict as P, resolve as R, roster
from valwr.live import session as S
from valwr.store import schema

POLL_SECONDS = 5.0


def agents_by_id(conn) -> dict[str, str]:
    return {r["uuid"].lower(): r["name"]
            for r in conn.execute("SELECT uuid, name FROM ref_agents")}


def load_bundle(path):
    import joblib
    if not path.exists():
        raise SystemExit(f"no model at {path}; run python -m valwr.model.train")
    return joblib.load(path)


def show(match, resolution, prediction) -> None:
    print("\n" + "=" * 58)
    print(f"  {match.phase.upper()}  ·  {match.map_name or 'unknown map'}"
          f"  ·  {len(match.players)} players")
    print("=" * 58)
    if prediction is None:
        print("  not enough of the roster to predict yet")
    else:
        p = prediction
        bar = int(round(p.own_probability * 30))
        print(f"\n  YOUR TEAM ({p.own_team})   {p.own_probability * 100:5.1f}%")
        print(f"  {'#' * bar}{'-' * (30 - bar)}")
        print(f"  ENEMY                {(1 - p.own_probability) * 100:5.1f}%")
        print(f"\n  {resolution.summary()}")
        print(f"  model: {p.model}")
        if p.factors:
            print("\n  strongest factors:")
            for name, contribution in p.factors:
                arrow = "+" if contribution > 0 else "-"
                print(f"    {arrow} {name.replace('d_', ''):<26} "
                      f"{abs(contribution):.3f}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="valwr.live")
    ap.add_argument("--once", action="store_true",
                    help="check a single time and exit")
    ap.add_argument("--no-fetch", action="store_true",
                    help="cache only; never spend API quota")
    ap.add_argument("--deadline", type=float, default=25.0,
                    help="seconds to spend resolving unknown players")
    args = ap.parse_args(argv)

    print(f"client: {lockfile.describe()}")
    if not lockfile.game_is_running():
        print("VALORANT is not running -- start the game and try again.")
        return 1

    settings = config.load(require_key=False)
    conn = schema.connect(settings.database_path)
    bundle = load_bundle(settings.database_path.parent.parent / "models" /
                         "model.joblib")
    print(f"model : {bundle['best']}  ({len(bundle['columns'])} features)")

    session = S.build()
    print(f"account: {session.puuid[:8]}...  shard={session.shard}\n")

    client = None
    if not args.no_fetch:
        key = config.load().henrik_api_key
        limiter = TokenBucket(config.load().requests_per_minute)
        client = HenrikClient(key, conn=conn, limiter=limiter)

    seen: str | None = None
    try:
        while True:
            match = roster.current(session, agents_by_id(conn))
            if match is None:
                if args.once:
                    print("not in a match (lobby).")
                    return 0
                time.sleep(POLL_SECONDS)
                continue

            if match.match_id != seen:
                seen = match.match_id
                as_of = int(time.time())
                resolution = R.resolve(
                    conn, match, session.puuid, as_of, client=client,
                    deadline_seconds=args.deadline,
                    region=settings.region, platform=settings.platform)
                prediction = P.predict(conn, match, bundle, resolution,
                                       session.puuid, as_of=as_of)
                show(match, resolution, prediction)

            if args.once:
                return 0
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
