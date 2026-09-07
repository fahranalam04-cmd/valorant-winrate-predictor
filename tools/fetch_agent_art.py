"""Download the dashboard's artwork, once.

    python tools/fetch_agent_art.py            # agents and maps
    python tools/fetch_agent_art.py --reprocess  # resize what is on disk

Three files per agent, into valwr/dash/static/agents/:

    <uuid>-row.png        480px wide, the art bleeding across a scoreboard row
    <uuid>-portrait.png   560px tall, the detail-panel hero
    <uuid>-icon.png       96px square, the compact agent tile

and one per map, into valwr/dash/static/maps/:

    <name>-splash.jpg     1280px wide, the background some themes use

The agent UUIDs are Riot's own, the same ones already in `ref_agents` and the
same ones the live client reports as `CharacterID`; map files are keyed by the
lowercased map name, which is exactly what the live state carries. Either way
the page addresses a file straight from the match with no lookup table.

Everything is resized on the way in. Riot ships these at print resolution --
the agent portraits are ~705 KB each and the scoreboard shows ten at once --
so resizing once here saves the browser megabytes of decode on every paint.

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
MAPS_API = "https://valorant-api.com/v1/maps"
DEST = Path("valwr/dash/static/agents")
MAPS = Path("valwr/dash/static/maps")
TIMEOUT = 30.0

# What to save from each agent entry: (api field, filename suffix).
WANTED = (("displayIcon", "icon"), ("fullPortrait", "portrait"))

# Riot ships these at print resolution -- the portraits are ~705 KB each, and
# the scoreboard puts ten of them on screen at once. Nothing is displayed
# anywhere near that large, so each file is resized once, here, rather than
# making the browser decode 11 MB on every first paint.
#
#   row       the art bleeding across a scoreboard row, ~230 CSS px wide
#   portrait  the detail panel hero, ~240 CSS px tall
#   icon      square agent tile, 30 CSS px
#
# Each is sized for a 2x display and no larger.
SIZES = {"row": ("width", 480), "portrait": ("height", 560), "icon": ("box", 96),
         # Map splashes are only ever shown blurred and darkened behind the
         # scoreboard, so they need far less resolution than they ship at.
         "splash": ("width", 1280)}


def _shrink(path: Path, kind: str) -> None:
    """Resize one file in place. A failure here is not fatal: the oversized
    original still renders, it just costs more to decode."""
    how, n = SIZES[kind]
    try:
        from PIL import Image
    except ImportError:
        return
    try:
        with Image.open(path) as im:
            im = im.convert("RGBA")
            w, h = im.size
            if how == "width":
                if w <= n:
                    return
                size = (n, max(1, round(h * n / w)))
            elif how == "height":
                if h <= n:
                    return
                size = (max(1, round(w * n / h)), n)
            else:
                if max(w, h) <= n:
                    return
                size = (n, n)
            im.resize(size, Image.LANCZOS).save(path, "PNG", optimize=True)
    except Exception:                                # noqa: BLE001
        pass


def _save_jpeg(path: Path, blob: bytes) -> None:
    """Map splashes are photographs shown blurred behind the scoreboard.

    PNG is the wrong container for them -- 20 of these came to 27.6 MB as PNG
    and about a tenth of that as JPEG, with no visible difference through a
    blur and a dark scrim. Alpha is dropped deliberately: a background has
    nothing to composite against.
    """
    try:
        from PIL import Image
    except ImportError:
        path.with_suffix(".png").write_bytes(blob)
        return
    import io
    with Image.open(io.BytesIO(blob)) as im:
        im = im.convert("RGB")
        w, h = im.size
        target = SIZES["splash"][1]
        if w > target:
            im = im.resize((target, max(1, round(h * target / w))), Image.LANCZOS)
        im.save(path, "JPEG", quality=82, optimize=True, progressive=True)


def slug(name: str) -> str:
    """`Pearl` -> `pearl`. The live state names maps exactly as this API does,
    so the page can address a file straight from the match without a lookup."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def fetch_maps(client) -> int:
    """Map splash art, used as a background by the themes that want one."""
    MAPS.mkdir(parents=True, exist_ok=True)
    try:
        data = client.get(MAPS_API).json().get("data") or []
    except Exception as e:                           # noqa: BLE001
        print(f"  maps: {type(e).__name__}; skipping")
        return 0
    got = 0
    for m in data:
        name, url = m.get("displayName"), m.get("splash")
        # Every map with art, not just the competitive pool. The live view
        # follows you into swiftplay, team deathmatch and customs, and a
        # skirmish arena needs a background as much as Ascent does -- filtering
        # these out left those modes on a bare gradient.
        if not name or not url:
            continue
        path = MAPS / f"{slug(name)}-splash.jpg"
        if path.exists():
            continue
        try:
            r = client.get(url)
            r.raise_for_status()
            _save_jpeg(path, r.content)
            got += 1
        except Exception as e:                       # noqa: BLE001
            print(f"  map {name}: {type(e).__name__}")
    size = sum(f.stat().st_size for f in MAPS.glob("*.jpg"))
    print(f"maps: {got} downloaded, {len(list(MAPS.glob('*.jpg')))} on disk, "
          f"{size / 1e6:.1f} MB")
    return got


def reprocess() -> int:
    """Resize whatever is already on disk, without re-downloading.

    Used when the sizes above change, and by anyone who downloaded the art
    before this step existed.
    """
    n = 0
    for path in sorted(DEST.glob("*-portrait.png")):
        row = DEST / path.name.replace("-portrait", "-row")
        if not row.exists():
            row.write_bytes(path.read_bytes())
        _shrink(row, "row")
        _shrink(path, "portrait")
        n += 1
    for path in sorted(DEST.glob("*-icon.png")):
        _shrink(path, "icon")
    for path in sorted(MAPS.glob("*-splash.png")):
        _save_jpeg(path.with_suffix(".jpg"), path.read_bytes())
        path.unlink()
    size = sum(f.stat().st_size for f in DEST.glob("*.png"))
    print(f"reprocessed {n} agents; {len(list(DEST.glob('*.png')))} files, "
          f"{size / 1e6:.1f} MB")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fetch_agent_art")
    ap.add_argument("--force", action="store_true",
                    help="re-download files that already exist")
    ap.add_argument("--reprocess", action="store_true",
                    help="resize what is already downloaded; no network")
    args = ap.parse_args(argv)

    if args.reprocess:
        return reprocess()

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
                    _shrink(path, suffix)
                    if suffix == "portrait":
                        # The row art is the same source at a different size.
                        row = DEST / f"{uuid}-row.png"
                        row.write_bytes(r.content)
                        _shrink(row, "row")
                    got += 1
                except Exception as e:               # noqa: BLE001
                    failed += 1
                    print(f"  {name} {suffix}: {type(e).__name__}")

        fetch_maps(client)
        print(f"\n{len(agents)} agents: {got} downloaded, {skipped} already "
              f"present, {failed} failed")
        print(f"into {DEST}")
        size = sum(f.stat().st_size for f in DEST.glob('*.png'))
        print(f"{len(list(DEST.glob('*.png')))} files, {size / 1e6:.1f} MB")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
