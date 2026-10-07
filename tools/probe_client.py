"""Does the game client hand out the real Performance Score?

    python tools/probe_client.py            # the most recent match
    python tools/probe_client.py <match_id>

VALORANT must be running. Patch 13.06 replaced ACS with Performance Score, and
no public API carries it -- checked on the day: live match responses hold only
`assists, bodyshots, damage, deaths, headshots, kills, legshots, score`, and
HenrikDev's newest match endpoint is still v4. But the End of Game screen shows
the real number, so the client can fetch it from somewhere.

This asks. `pd.{shard}.a.pvp.net/match-details/v1/matches/{id}` is what that
screen is built from, and `live/session.py` already knows how to reach it --
the `pd` host has simply never been called.

Read-only, like everything under `live/`: one GET, nothing written, nothing
sent to the game. Field NAMES and numeric ranges are printed, never a gamertag
or a PUUID, so the output can be pasted anywhere.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from valwr import config
from valwr.live import lockfile, session as S
from valwr.store import schema

# What a Performance Score field would plausibly be called, and the range it
# would sit in. Riot documents 0-500 and nothing else.
WORDS = ("perf", "score", "impact", "master", "grade", "rating", "contribution")
PS_RANGE = (0.0, 500.0)


def walk(node, path="", depth=0, out=None):
    """Every leaf in the response, as path -> type, without any values."""
    out = {} if out is None else out
    if depth > 6:
        return out
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, f"{path}.{k}" if path else k, depth + 1, out)
    elif isinstance(node, list):
        if node:
            walk(node[0], f"{path}[]", depth + 1, out)
    else:
        out[path] = type(node).__name__
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="probe_client")
    ap.add_argument("match_id", nargs="?")
    args = ap.parse_args(argv)

    try:
        sess = S.build()
    except lockfile.ClientNotRunning as e:
        print(f"  VALORANT is not running -- {e}")
        print("  Start the game (the menus are enough) and run this again.")
        return 1

    match_id = args.match_id
    if not match_id:
        s = config.load(require_key=False)
        conn = schema.connect(s.database_path)
        row = conn.execute("SELECT match_id FROM live_predictions "
                           "ORDER BY made_at DESC LIMIT 1").fetchone()
        conn.close()
        if row is None:
            print("  no recorded match to ask about; pass a match id")
            return 1
        match_id = row["match_id"]
    print(f"  asking the client about {match_id[:8]}...\n")

    import httpx
    url = f"{sess.pd}/match-details/v1/matches/{match_id}"
    try:
        # Riot's own server, with a real certificate: verified, unlike the
        # game's local API. Turning it off sent the session token to a host
        # nobody had checked.
        r = httpx.get(url, headers=sess.headers, timeout=20)
    except httpx.HTTPError as e:
        print(f"  the request failed: {type(e).__name__}: {e}")
        return 1
    print(f"  {r.status_code} from match-details/v1")
    if r.status_code != 200:
        print(f"  body starts: {r.text[:160]}")
        print("\n  A 404 here usually means the shard is wrong or the match is "
              "too old for the client to hold.")
        return 1

    data = r.json()
    players = data.get("players") or []
    print(f"  {len(players)} players in the response\n")
    if not players:
        print("  no player list; the shape is not what this expected")
        return 1

    shape = walk(players[0])
    print("  per-player fields (names and types only):")
    for path, kind in sorted(shape.items()):
        print(f"    {path:<52}{kind}")

    # Anything that looks like it could be the number.
    named = [p for p in shape if any(w in p.lower() for w in WORDS)]
    print("\n  fields whose NAME suggests a performance score:")
    print("   ", ", ".join(named) if named else "none")

    def numbers(node, path=""):
        found = []
        if isinstance(node, dict):
            for k, v in node.items():
                found += numbers(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for v in node:
                found += numbers(v, f"{path}[]")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            found.append((path, float(node)))
        return found

    in_range = {}
    for p in players:
        for path, value in numbers(p):
            if PS_RANGE[0] <= value <= PS_RANGE[1]:
                in_range.setdefault(path, []).append(value)
    # A per-match score varies between the ten players and is not a count of
    # something small, so the spread is what separates it from kills or rounds.
    candidates = {k: v for k, v in in_range.items()
                  if len(set(v)) >= 5 and max(v) >= 50}
    print("\n  numeric fields inside 0-500 that differ across the ten players:")
    for path, values in sorted(candidates.items()):
        print(f"    {path:<52}{min(values):.0f} to {max(values):.0f}")
    if not candidates:
        print("    none -- the real Performance Score is not in this response")

    print("\n  Next: if one of those is the number, record it per match and fit "
          "\n  the estimate against it. If not, the 0-100 stays as it is and no "
          "\n  0-500 figure gets invented. See docs/SCORE-SPEC.md, phase 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
