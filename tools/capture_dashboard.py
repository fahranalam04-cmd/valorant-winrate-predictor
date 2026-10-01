"""The README's dashboard images, captured from the demo build.

    pip install playwright               # once; drives your installed Chrome or Edge
    python tools/build_pages_demo.py
    python tools/capture_dashboard.py

Writes docs/images/:

    dashboard.jpg              the scoreboard with one player's card open
    dashboard-walkthrough.webp an animated tour: select, switch map and side,
                               a thin lobby, closing the card
    dashboard-phone.jpg        the phone layout, list and card side by side
    dashboard-agent-select.jpg agent select: your team as cards, some still
                               picking, one still being looked up

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


def settle(page, rendered: str = "button.row") -> None:
    """Wait for the render, every image, and the background to paint.

    `rendered` is what the page draws once it has a match: scoreboard rows,
    or in agent select the team's cards.
    """
    page.wait_for_selector(rendered)
    page.wait_for_function(
        "[...document.images].every(i => i.complete)", timeout=15000)
    page.wait_for_timeout(700)


# A name column narrower than this shows three or four characters. The roster
# once collapsed it to zero at 1600px, and screenshots are where that shows.
MIN_NAME_PX = 90
# Wide displays included on purpose. The roster split into two columns above a
# container width of 900px and clipped every gamertag and reason from 1440
# upward -- on a 2560 screen it was unreadable -- while this guard reported all
# clear, because 144px of name passes a 90px minimum. A width that is never
# checked is a width that breaks.
CHECK_WIDTHS = (1280, 1440, 1600, 1920, 2030, 2560, 3440)


def check_layout(browser, url: str) -> list[str]:
    """Nothing a reader needs may be clipped, at any common desktop width.

    Measured as actual truncation -- `scrollWidth` past `clientWidth` -- rather
    than as a pixel minimum, because the pixel minimum is what let a roster
    ship with every gamertag ellipsised.
    """
    problems = []
    for width in CHECK_WIDTHS:
        page = browser.new_page(viewport={"width": width, "height": 1000}, **CSP)
        page.goto(url + "?clean")
        settle(page)
        got = page.evaluate("""() => {
          const clipped = e => e.scrollWidth > e.clientWidth + 1;
          const names = [...document.querySelectorAll('button.row .nm')];
          const rest = [...document.querySelectorAll(
              'button.row .why, button.row .rsn, button.row .sub')];
          return {narrowest: Math.min(...names.map(e => e.clientWidth)),
                  names: names.filter(clipped).length,
                  rest: rest.filter(clipped).length};
        }""")
        if got["narrowest"] < MIN_NAME_PX:
            problems.append(f"{width}px: a name column is "
                            f"{got['narrowest']}px wide")
        if got["names"]:
            problems.append(f"{width}px: {got['names']} gamertag(s) clipped")
        if got["rest"]:
            problems.append(f"{width}px: {got['rest']} reason line(s) clipped")
        page.close()
    return problems


# Agent select allows about a minute to lock in, which is no time to scroll
# for the fifth player. A maximised browser on a 1920x1080 monitor has about
# 950px of height; 800 leaves room for a bookmarks bar, a window that is not
# quite maximised, or a little zoom.
FIT_VIEWPORTS = ((1920, 800), (2560, 1300))


def check_agent_select(browser, url: str) -> list[str]:
    """All five teammates inside the window, and nothing on a card clipped.

    The first agent-select layout stacked every line of a card and ran 152px
    each, which put the fifth player 56px below a maximised 1080p window.
    """
    problems = []
    for width, height in FIT_VIEWPORTS:
        page = browser.new_page(viewport={"width": width, "height": height},
                                **CSP)
        page.goto(url + "?clean&phase=pregame")
        settle(page, "button.pcard")
        got = page.evaluate("""() => {
          const cards = [...document.querySelectorAll('button.pcard')];
          return {n: cards.length, bottom: Math.max(
            ...cards.map(e => e.getBoundingClientRect().bottom))};
        }""")
        if got["n"] != 5:
            problems.append(f"{width}x{height}: {got['n']} agent-select cards, "
                            f"expected the whole team of 5")
        elif got["bottom"] > height:
            problems.append(f"{width}x{height}: agent select runs "
                            f"{got['bottom'] - height:.0f}px past the window")
        page.close()
    for width in CHECK_WIDTHS:
        page = browser.new_page(viewport={"width": width, "height": 950}, **CSP)
        page.goto(url + "?clean&phase=pregame")
        settle(page, "button.pcard")
        clipped = page.evaluate("""() => [...document.querySelectorAll(
            '.pcard .nm, .pcard .rs, .pcard .cline, .pcard .st b, .pcard .st span'
          )].filter(e => e.scrollWidth > e.clientWidth + 1).length""")
        if clipped:
            problems.append(f"{width}px: {clipped} agent-select line(s) clipped")
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
        problems = check_layout(browser, url) + check_agent_select(browser, url)
        if problems:
            print("refusing to capture:")
            print("\n".join(f"  {m}" for m in problems))
            browser.close()
            httpd.shutdown()
            return 1
        print("agent select: all five fit "
              + " and ".join(f"{w}x{h}" for w, h in FIT_VIEWPORTS))
        print(f"layout: nothing clipped, and every name has at least "
              f"{MIN_NAME_PX}px, at "
              f"{', '.join(map(str, CHECK_WIDTHS))}px")

        # --- the hero: the scoreboard with a card open --------------------
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, **CSP)
        page.goto(url + "?clean")
        settle(page)
        page.click('button.row[data-puuid="demo-05"]')
        settle(page)
        page.screenshot(path=str(OUT / "dashboard.jpg"), type="jpeg", quality=86)

        # ...and the same match once it has been played, which is what a
        # pinned tab settles into.
        page.evaluate("() => window.__demo.set('status', 'finished')")
        settle(page)
        # The card is closed and the comparison scrolled to, because it is the
        # point of this shot and it sits below the rosters. Its own top, not
        # scroll_into_view_if_needed's -- the section is taller than the
        # viewport now, so "if needed" lands halfway down it.
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        page.evaluate(
            "() => window.scrollTo(0, document.querySelector('.compare')"
            ".getBoundingClientRect().top + window.scrollY - 12)")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "dashboard-result.jpg"), type="jpeg",
                        quality=86)
        page.close()

        # --- agent select: your team as cards ------------------------------
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, **CSP)
        page.goto(url + "?clean&phase=pregame")
        settle(page, "button.pcard")
        page.screenshot(path=str(OUT / "dashboard-agent-select.jpg"),
                        type="jpeg", quality=86)
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
