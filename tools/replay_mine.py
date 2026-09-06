"""Replay your own recent matches and compare the model against what happened.

    python tools/replay_mine.py --fetch          # refresh, then replay 10
    python tools/replay_mine.py --games 20       # cache only

`--fetch` spends API quota: one matchlist call for your account, which is also
the call the live path now makes at the start of a match. Without it this
replays whatever is already stored, which may be weeks old -- the exact problem
that made the live score look frozen.

**Ten games cannot validate anything.** The model is right about 53% of the
time, so over ten matches anywhere from 3 to 8 correct is unremarkable: the
standard error on ten is about 16 percentage points. This exists to show you
what the model said and what happened, side by side, not to score it. The
output repeats that where the numbers are, because a 7/10 looks like skill.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

sys.path.insert(0, ".")

import joblib

from valwr import config
from valwr.live import predict as LP
from valwr.live.resolve import Resolution
from valwr.live.roster import LiveMatch, LivePlayer
from valwr.rating import potential as P
from valwr.rating.normalize import build_norms
from valwr.rating.rating import rate_performance
from valwr.store import schema


def local_puuid(conn, name: str, tag: str | None) -> str | None:
    """Find the local account, tolerating a rename.

    A Riot ID is not a stable key -- this account was `OldName#tag` when the
    crawl started and is `NewName#tag` now, so a name lookup against a
    configured name silently finds nothing. The PUUID is the stable identifier;
    the tag survives most renames, so it is the better fallback.
    """
    row = conn.execute("SELECT puuid FROM players WHERE lower(name) = lower(?)",
                       (name,)).fetchone()
    if row:
        return row["puuid"]
    if tag:
        rows = conn.execute(
            "SELECT p.puuid, p.name, MAX(mp.started_at) AS seen "
            "FROM players p JOIN match_players mp ON mp.puuid = p.puuid "
            "WHERE lower(p.tag) = lower(?) GROUP BY p.puuid "
            "ORDER BY seen DESC LIMIT 1", (tag,)).fetchall()
        if rows:
            print(f"note: no account named '{name}'; using #{tag} -> "
                  f"'{rows[0]['name']}' (renamed since the crawl began)")
            return rows[0]["puuid"]
    return None


def refresh(conn, settings, puuid: str) -> int:
    """One matchlist call for the local account. Returns matches gained."""
    from valwr.collect.client import HenrikClient
    from valwr.collect.limiter import TokenBucket

    before = conn.execute(
        "SELECT COUNT(*) FROM match_players WHERE puuid = ?", (puuid,)
    ).fetchone()[0]
    full = config.load()
    client = HenrikClient(full.henrik_api_key, conn=conn,
                          limiter=TokenBucket(full.requests_per_minute))
    try:
        client.matches(settings.region, settings.platform, puuid, size=10,
                       mode="competitive")
    finally:
        client.close()
    after = conn.execute(
        "SELECT COUNT(*) FROM match_players WHERE puuid = ?", (puuid,)
    ).fetchone()[0]
    return after - before


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="replay_mine")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--fetch", action="store_true",
                    help="refresh your account first (one API call)")
    ap.add_argument("--puuid", help="use this account directly, bypassing the "
                                    "name lookup (survives a rename)")
    args = ap.parse_args(argv)

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    puuid = args.puuid or local_puuid(conn, s.riot_name,
                                      getattr(s, "riot_tag", None))
    if not puuid:
        print(f"'{s.riot_name}' is not in the database yet; run with --fetch")
        return 1

    if args.fetch:
        gained = refresh(conn, s, puuid)
        print(f"refreshed your account: {gained:+d} new player-rows\n")

    bundle = joblib.load(s.database_path.parent.parent / "models" / "model.joblib")
    norms = build_norms(conn, bundle["norms_as_of"])
    index = P.PerfIndex.load()

    mine = conn.execute(
        "SELECT match_id, started_at, map, team, won FROM match_players "
        "WHERE puuid = ? ORDER BY started_at DESC LIMIT ?",
        (puuid, args.games)).fetchall()
    if not mine:
        print("no matches stored for your account")
        return 1

    print(f"model: {bundle['best']}   index top-1 rate: "
          f"{index.top1_rate if index.top1_rate is not None else 'unmeasured'}")
    print(f"replaying your last {len(mine)} matches\n")
    print(f"  {'when':<12}{'map':<9}{'pred':>6}{'result':>8}  "
          f"{'call':<6}{'best player predicted':<24}{'actually best':<22}")
    print("  " + "-" * 92)

    right = calls = best_hits = best_n = 0
    for m in mine:
        as_of = m["started_at"]
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM match_players WHERE match_id = ?",
            (m["match_id"],)).fetchall()]
        if len(rows) < 10:
            continue

        live = LiveMatch(
            match_id=m["match_id"], phase="coregame", map_name=m["map"],
            mode="BombGameMode", flow="Matchmaking",
            players=[LivePlayer(puuid=r["puuid"], team=r["team"], agent_id="x",
                                agent=r["agent"] or "?") for r in rows])
        res = Resolution(known={r["puuid"] for r in rows})
        pred = LP.predict(conn, live, bundle, res, puuid, as_of=as_of)

        # Who did the score expect to top *your* team, and who actually did?
        mine_rows = [r for r in rows if r["team"] == m["team"]]
        scored, actual = {}, {}
        for r in mine_rows:
            d = P.detail(conn, r["puuid"], as_of, m["map"], norms, index)
            if d:
                scored[r["puuid"]] = d["score"]
            a = rate_performance(r, norms)
            if a:
                actual[r["puuid"]] = a.value

        def who(pu):
            row = conn.execute("SELECT name FROM players WHERE puuid = ?",
                               (pu,)).fetchone()
            return (row["name"] if row and row["name"] else pu[:8])[:20]

        picked = max(scored, key=scored.get) if scored else None
        truly = max(actual, key=actual.get) if actual else None
        if picked and truly:
            best_n += 1
            best_hits += picked == truly

        won = bool(m["won"])
        if pred is not None:
            calls += 1
            said_win = pred.own_probability >= 0.5
            ok = said_win == won
            right += ok
            call = "hit" if ok else "miss"
            shown = f"{pred.own_probability * 100:5.1f}%"
        else:
            call, shown = "-", "  --  "

        print(f"  {datetime.fromtimestamp(as_of):%m-%d %H:%M} "
              f"{m['map']:<9}{shown:>6}{'WIN' if won else 'LOSS':>8}  "
              f"{call:<6}"
              f"{(who(picked) + f' ({scored[picked]})') if picked else '-':<24}"
              f"{who(truly) if truly else '-':<22}")

    print()
    if calls:
        print(f"  win prediction : {right}/{calls} correct")
    if best_n:
        print(f"  top of your team: {best_hits}/{best_n} correct "
              f"(1 in 5 by chance)")
    print()
    print("  Neither number means much at this sample size. The model is right")
    print("  about 53% of the time, so 3 to 8 correct out of 10 is the ordinary")
    print("  range -- the standard error on ten games is roughly 16 points.")
    print("  Held-out measurements over thousands of matches are in")
    print("  docs/MODEL-CHOICE.md; these ten are for looking at, not scoring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
