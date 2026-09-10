"""The README's dashboard images, captured from the demo build.

    pip install playwright               # once; drives your installed Chrome or Edge
    python tools/build_pages_demo.py
    python tools/capture_dashboard.py

Writes docs/images/:

    dashboard.jpg              the scoreboard with one player's card open
    dashboard-walkthrough.webp an animated tour: select, switch map and side,
                               a thin lobby, closing the card
    dashboard-phone.jpg        the phone layout, list and card side by side

Everything is captured from the demo -- the real page fed invented players --
so no real gamertag can end up in a committed image.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import io
import socketserver
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
OUT = ROOT / "docs" / "images"
FONT = Path("C:/Windows/Fonts/bahnschrift.ttf")

# The steps of the walkthrough: (caption, action, hold in ms). An action is
# ("click", puuid), ("set", key, value), ("key", name) or None.
TOUR = [
    ("All ten players, by team. Brighter rows have match history.", None, 2600),
    ("Click a player for their card: score, record, form, this map.",
     ("click", "demo-00"), 3200),
    ("Another player. The diamond flags one playing far above their rank.",
     ("click", "demo-05"), 3200),
    ("The background and map record follow the map being played.",
     ("set", "map", "Lotus"), 2800),
    ("Switch sides and the odds, 'you' and the verdict follow.",
     ("set", "side", "Red"), 2800),
    ("A lobby of strangers: fewer known players, lower confidence.",
     ("set", "known", "thin"), 3000),
    ("Esc closes the card.", ("key", "Escape"), 2200),
]


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(directory: Path):
    handler = functools.partial(_Quiet, directory=str(directory))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/"


# Every page is opened with the content policy bypassed. The demo's policy
# forbids evaluating strings as script, which is how Playwright polls a wait
# condition; the published page keeps its policy -- only this harness skips it.
CSP = {"bypass_csp": True}


def settle(page) -> None:
    """Wait for the render, every image, and the background to paint."""
    page.wait_for_selector("button.row")
    page.wait_for_function(
        "[...document.images].every(i => i.complete)", timeout=15000)
    page.wait_for_timeout(700)


# A name column narrower than this shows three or four characters. The roster
# once collapsed it to zero at 1600px, and screenshots are where that shows.
MIN_NAME_PX = 90
CHECK_WIDTHS = (1280, 1440, 1600, 1920)


def check_layout(browser, url: str) -> list[str]:
    """Every player's name must have room, at every common desktop width."""
    problems = []
    for width in CHECK_WIDTHS:
        page = browser.new_page(viewport={"width": width, "height": 1000}, **CSP)
        page.goto(url + "?clean")
        settle(page)
        narrowest = page.evaluate(
            "Math.min(...[...document.querySelectorAll('button.row .nm')]"
            ".map(e => e.clientWidth))")
        if narrowest < MIN_NAME_PX:
            problems.append(f"{width}px: a name column is {narrowest}px wide")
        page.close()
    return problems


def launch(p):
    for channel in ("chrome", "msedge"):
        try:
            return p.chromium.launch(channel=channel)
        except Exception:                            # noqa: BLE001
            continue
    return p.chromium.launch()


def caption(img, text: str):
    """A caption strip under a frame, in the dashboard's own face."""
    from PIL import Image, ImageDraw, ImageFont
    strip = 46
    out = Image.new("RGB", (img.width, img.height + strip), (7, 10, 14))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, img.height, 4, img.height + strip], fill=(255, 70, 85))
    try:
        font = ImageFont.truetype(str(FONT), 19)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, img.height + 12), text, fill=(236, 232, 225), font=font)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="capture_dashboard")
    ap.add_argument("--site", default=str(SITE))
    args = ap.parse_args(argv)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("needs playwright: pip install playwright")
        return 1
    from PIL import Image

    site = Path(args.site)
    if not (site / "index.html").exists():
        print(f"no demo build at {site}; run tools/build_pages_demo.py first")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    httpd, url = serve(site)

    with sync_playwright() as p:
        browser = launch(p)
        problems = check_layout(browser, url)
        if problems:
            print("refusing to capture; names do not fit:")
            print("\n".join(f"  {m}" for m in problems))
            browser.close()
            httpd.shutdown()
            return 1
        print(f"layout: every name has at least {MIN_NAME_PX}px at "
              f"{', '.join(map(str, CHECK_WIDTHS))}px")

        # --- the hero: the scoreboard with a card open --------------------
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, **CSP)
        page.goto(url + "?clean")
        settle(page)
        page.click('button.row[data-puuid="demo-05"]')
        settle(page)
        page.screenshot(path=str(OUT / "dashboard.jpg"), type="jpeg", quality=86)
        page.close()

        # --- the walkthrough ------------------------------------------------
        page = browser.new_page(viewport={"width": 1440, "height": 900}, **CSP)
        page.goto(url + "?clean")
        settle(page)
        frames, holds = [], []
        for text, action, hold in TOUR:
            if action and action[0] == "click":
                page.click(f'button.row[data-puuid="{action[1]}"]')
            elif action and action[0] == "set":
                page.evaluate("([k, v]) => window.__demo.set(k, v)",
                              [action[1], action[2]])
            elif action and action[0] == "key":
                page.keyboard.press(action[1])
            settle(page)
            shot = Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB")
            shot = shot.resize((1080, round(shot.height * 1080 / shot.width)),
                               Image.LANCZOS)
            frames.append(caption(shot, text))
            holds.append(hold)
        frames[0].save(OUT / "dashboard-walkthrough.webp", save_all=True,
                       append_images=frames[1:], duration=holds, loop=0,
                       quality=74, method=6)
        page.close()

        # --- the phone ------------------------------------------------------
        ctx = browser.new_context(viewport={"width": 390, "height": 844},
                                  device_scale_factor=2, is_mobile=True,
                                  has_touch=True, **CSP)
        page = ctx.new_page()
        page.goto(url + "?clean")
        settle(page)
        listing = Image.open(io.BytesIO(page.screenshot(type="png")))
        page.tap('button.row[data-puuid="demo-00"]')
        settle(page)
        page.locator("aside.panel").scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        card = Image.open(io.BytesIO(page.screenshot(type="png")))
        gap = 36
        both = Image.new("RGB", (listing.width * 2 + gap, listing.height), (7, 10, 14))
        both.paste(listing, (0, 0))
        both.paste(card, (listing.width + gap, 0))
        both = both.resize((both.width // 2, both.height // 2), Image.LANCZOS)
        both.save(OUT / "dashboard-phone.jpg", quality=86)
        ctx.close()
        browser.close()
    httpd.shutdown()

    for f in sorted(OUT.iterdir()):
        print(f"  {f.relative_to(ROOT)}  {f.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
