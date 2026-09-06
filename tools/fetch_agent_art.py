"""Download agent artwork for the dashboard, once.

    python tools/fetch_agent_art.py

Saves two files per agent into valwr/dash/static/agents/:

    <uuid>-icon.png       the square portrait, shown on every scoreboard row
    <uuid>-portrait.png   the full-body art, shown in the detail panel

The UUIDs are Riot's own, the same ones already in `ref_agents` and the same
ones the live client reports as `CharacterID`, so the page addresses a file
directly from the roster with no lookup table.

**Why bundle rather than hotlink.** The dashboard runs while you are in a game.
An external image request per agent means the page depends on a CDN being up
and reachable at exactly the moment you are loading into a match, and it leaks
one request per agent to a third party. Downloaded once, the page never touches
the network again.

Re-run it when Riot adds an agent. Nothing else calls this, and it writes only
into the static directory.

This talks to valorant-api.com, which is public game-asset data -- no player
information is sent or received, and it is unrelated to the HenrikDev API the
crawler uses. It needs no key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

import httpx

API = "https://valorant-api.com/v1/agents?isPlayableCharacter=true"
DEST = Path("valwr/dash/static/agents")
TIMEOUT = 30.0

# What to save from each agent entry: (api field, filename suffix).
WANTED = (("displayIcon", "icon"), ("fullPortrait", "portrait"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fetch_agent_art")
    ap.add_argument("--force", action="store_true",
                    help="re-download files that already exist")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    DEST.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        try:
            listing = client.get(API).json()
        except Exception as e:                       # noqa: BLE001
            print(f"could not reach valorant-api.com ({type(e).__name__}: {e})")
            print("The dashboard still works; agents fall back to a lettered "
                  "tile until this succeeds.")
            return 1

        agents = listing.get("data") or []
        if not agents:
            print("the agent listing came back empty; nothing written")
            return 1

        got = skipped = failed = 0
        for a in agents:
            uuid, name = a.get("uuid"), a.get("displayName", "?")
            if not uuid:
                continue
            for field, suffix in WANTED:
                url = a.get(field)
                path = DEST / f"{uuid}-{suffix}.png"
                if not url:
                    print(f"  {name}: no {field}")
                    continue
                if path.exists() and not args.force:
                    skipped += 1
                    continue
                try:
                    r = client.get(url)
                    r.raise_for_status()
                    path.write_bytes(r.content)
                    got += 1
                except Exception as e:               # noqa: BLE001
                    failed += 1
                    print(f"  {name} {suffix}: {type(e).__name__}")

        print(f"\n{len(agents)} agents: {got} downloaded, {skipped} already "
              f"present, {failed} failed")
        print(f"into {DEST}")
        size = sum(f.stat().st_size for f in DEST.glob('*.png'))
        print(f"{len(list(DEST.glob('*.png')))} files, {size / 1e6:.1f} MB")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
