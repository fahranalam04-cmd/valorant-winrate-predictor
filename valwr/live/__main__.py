"""Watch for a live match and predict it: python -m valwr.live

Read-only throughout. This never writes to the client API, never selects or
locks an agent, and never touches process memory. See docs/ETHICS-AND-TOS.md
-- that boundary is what separates a tolerated overlay from a ban.
"""

from __future__ import annotations

import argparse
import sys
import time

from valwr import config
from valwr.collect.client import HenrikClient
from valwr.collect.limiter import TokenBucket
from valwr.live import lockfile, predict as P, resolve as R, roster
from valwr.live import session as S
from valwr.rating import potential as pot
from valwr.store import schema

POLL_SECONDS = 5.0
NAME_WIDTH = 20


def _display_width(s: str) -> int:
    """Terminal columns a string occupies, not its character count.

    Korean and Japanese gamertags are common in this data and render two
    columns per glyph, so `f"{name:<20}"` under-pads them and the table's
    columns walk out of line.
    """
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
               for c in s)


def _fit(s: str, width: int) -> str:
    """Truncate to `width` display columns, then pad to exactly that."""
    if _display_width(s) > width:
        out = ""
        for ch in s:
            if _display_width(out + ch) > width - 1:
                break
            out += ch
        s = out + "…"
    return s + " " * max(0, width - _display_width(s))


def gamertags(conn, puuids: list[str]) -> dict[str, str]:
    """puuid -> "name#tag" for whoever we already hold.

    Straight from the `players` table -- every player the crawler has ever
    seen has one. The client exposes names through a PUT on its name-service
    endpoint, which docs/ETHICS-AND-TOS.md rules out without exception, so an
    unrecognised teammate keeps a short PUUID rather than being looked up.
    """
    if not puuids:
        return {}
    q = ",".join("?" * len(puuids))
    out = {}
    for r in conn.execute(
            f"SELECT puuid, name, tag FROM players WHERE puuid IN ({q})",
            puuids):
        if r["name"]:
            out[r["puuid"]] = f"{r['name']}#{r['tag']}" if r["tag"] else r["name"]
    return out


def team_table(conn, match, bundle, own_puuid: str, as_of: int, index) -> None:
    """Your team, ranked by who is likely to play best."""
    # Spectating or coaching a custom: there is no "your team", but the table
    # is still the useful part of the output, so fall back to Blue rather than
    # printing nothing.
    own_team = match.team_of(own_puuid) or "Blue"
    mine = [p for p in match.players if p.team == own_team]
    if not mine:
        return

    names = gamertags(conn, [p.puuid for p in mine])
    rows = []
    for p in mine:
        who = names.get(p.puuid, p.puuid[:8])
        pot_p = pot.evaluate(conn, p.puuid, as_of, match.map_name or "?",
                             bundle["norms"], index)
        rows.append((pot_p, who, p.agent, p.puuid == own_puuid))

    # Unscored players sort last rather than being dropped: they are on the
    # team whether or not we know anything about them, and saying so is the
    # point.
    rows.sort(key=lambda r: (r[0] is not None, r[0].score if r[0] else 0),
              reverse=True)

    print(f"  YOUR TEAM ({own_team})".ljust(NAME_WIDTH + 22) + "potential")
    print("  " + "-" * (NAME_WIDTH + 30))
    for i, (pot_p, who, agent, is_me) in enumerate(rows, 1):
        mark = "*" if is_me else " "
        label = _fit(who, NAME_WIDTH)
        if pot_p is None:
            print(f"  {i}{mark} {label} {agent:<10} {'--':>3}   no history")
        else:
            print(f"  {i}{mark} {label} {agent:<10} "
                  f"{pot_p.score:>3}   {pot_p.reason}")
    print(f"\n  * you. Score is a percentile: 70 means likely to outperform "
          f"70% of players.\n  Ranks the top player correctly 30.5% of the "
          f"time against 20% chance -- a\n  real edge, not a reliable one. "
          f"See docs/MODEL-CHOICE.md.")


def agents_by_id(conn) -> dict[str, str]:
    return {r["uuid"].lower(): r["name"]
            for r in conn.execute("SELECT uuid, name FROM ref_agents")}


def load_bundle(path):
    import joblib
    if not path.exists():
        raise SystemExit(f"no model at {path}; run python -m valwr.model.train")
    return joblib.load(path)


def show(match, resolution, prediction, conn=None, bundle=None,
         own_puuid=None, as_of=None, index=None) -> None:
    print("\n" + "=" * 58)
    kind = "CUSTOM" if match.is_custom else match.phase.upper()
    print(f"  {kind}  ·  {match.map_name or 'unknown map'}"
          f"  ·  {match.mode or 'unknown mode'}"
          f"  ·  {match.team_size('Blue')}v{match.team_size('Red')}")
    print("=" * 58)

    # Custom lobbies are where the model's training distribution stops being a
    # safe assumption, so say so rather than printing a confident number.
    if not match.is_standard_mode:
        print(f"  {match.mode} is not bomb defusal. The model only ever saw")
        print("  standard 5v5, so a win probability here means nothing.")
        print()
    elif match.is_custom and not match.is_even_5v5:
        print(f"  Uneven teams ({match.team_size('Blue')}v"
              f"{match.team_size('Red')}). Team features are averages, so this")
        print("  still computes -- but the model was only trained on 5v5.")
        print()
    if own_puuid and match.team_of(own_puuid) is None and match.players:
        print("  You are not on either team (spectating or coaching);")
        print("  percentages below are from Team Blue's side.")
        print()

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
    if index is not None and conn is not None:
        print()
        try:
            team_table(conn, match, bundle, own_puuid, as_of, index)
        except Exception as e:          # never let the table kill the view
            print(f"  (team table unavailable: {type(e).__name__}: {e})")
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

    # Real gamertags in this dataset include Japanese characters, and Windows'
    # console defaults to cp1252 -- printing one raised UnicodeEncodeError in
    # testing. A teammate's name must not be able to kill the live view.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    print(f"client: {lockfile.describe()}")
    if not lockfile.game_is_running():
        print("VALORANT is not running -- start the game and try again.")
        return 1

    settings = config.load(require_key=False)
    conn = schema.connect(settings.database_path)
    bundle = load_bundle(settings.database_path.parent.parent / "models" /
                         "model.joblib")
    print(f"model : {bundle['best']}  ({len(bundle['columns'])} features)")

    # Optional: the live view still works without it, minus the team table.
    try:
        index = pot.PerfIndex.load()
    except FileNotFoundError as e:
        index = None
        print(f"note  : {e}")

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
                show(match, resolution, prediction, conn=conn,
                     bundle=bundle, own_puuid=session.puuid,
                     as_of=as_of, index=index)

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
