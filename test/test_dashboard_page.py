"""The dashboard page, executed rather than grepped.

`valwr/dash/static/index.html` is the one part of this project a Python test
cannot reach, and it is exactly where a rename in `poll_once` shows up as a
blank column instead of an exception -- the same silent-failure class that has
cost this project the most.

So the page's own script is run under Node against a real state dictionary, and
the HTML it produces is asserted on. Skipped when Node is absent; the rest of
the suite does not depend on it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "valwr" / "dash" / "static" / "index.html"
CHECK = ROOT / "test" / "render_check.mjs"

node = shutil.which("node")
needs_node = pytest.mark.skipif(node is None, reason="node is not installed")


@needs_node
def test_the_page_renders_a_full_scoreboard(tmp_path):
    from valwr.dash.demo import demo_state

    s = demo_state()
    # demo_state() reads agent UUIDs from ref_agents, which a test must not
    # depend on. Inject them so the artwork path is exercised either way; the
    # no-art path has its own test below.
    for i, p_ in enumerate(s["players"]):
        p_["agent_id"] = f"agent-{i:02d}"
    state = tmp_path / "state.json"
    state.write_text(json.dumps(s), encoding="utf-8")
    r = subprocess.run([node, str(CHECK), str(PAGE), str(state)],
                       capture_output=True, text=True, timeout=120)
    sys.stdout.write(r.stdout)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_page_survives_a_state_with_no_agent_art(tmp_path):
    """A fresh clone has not run tools/fetch_agent_art.py yet.

    Every <img> must be optional -- a missing file removes the element and
    reveals a lettered tile, rather than leaving a broken-image icon on every
    row of the scoreboard.
    """
    page = PAGE.read_text(encoding="utf-8")
    assert page.count("onerror=\"this.remove()\"") >= 2, (
        "both the row icon and the panel portrait need a removal fallback")


def test_the_page_escapes_everything_it_prints():
    """Gamertags are attacker-controlled text from other players' accounts.

    Nothing may reach innerHTML unescaped. This checks the helper exists and
    that the risky fields go through it, since the page builds HTML by
    concatenation.
    """
    page = PAGE.read_text(encoding="utf-8")
    assert "const esc = t =>" in page
    # Prefix match: several of these are escaped as esc(f.map || "?").
    for field in ("p.name", "p.agent", "m.name", "f.map", "f.agent",
                  "a.agent", "p.reason", "f.ago", "p.team",
                  # Numbers too: they come from stored API responses, and
                  # SQLite keeps whatever it was given.
                  "p.score", "m.gate", "a.games", "s.coverage",
                  "recent.games", "career.games", "win"):
        assert f"esc({field}" in page, f"{field} is printed without esc()"
    assert 'const i0 = v => v == null ? "—" : esc(v);' in page


# --- the map background ------------------------------------------------

# Grounds, inks and the two team accents. The page has one palette now, so
# these have to live on :root or components fall back to browser defaults.
CRITICAL = ("--void", "--steel", "--sunk", "--line",
            "--bone", "--bone-dim", "--slate", "--ghost", "--atk", "--def")

MAPS = ROOT / "valwr" / "dash" / "static" / "maps"


def test_the_palette_lives_on_root():
    page = PAGE.read_text(encoding="utf-8")
    root = re.search(r":root\s*\{([^}]*)\}", page).group(1)
    for token in CRITICAL:
        assert f"{token}:" in root, f":root must define {token}"


def test_there_is_exactly_one_palette():
    """The picker and its four alternates are gone. A stray data-theme rule
    would be dead CSS that never applies, since nothing sets the attribute."""
    page = PAGE.read_text(encoding="utf-8")
    assert "data-theme" not in page
    assert "localStorage" not in page


def test_the_background_is_named_from_the_map():
    """The file is addressed by the lowercased map name with punctuation
    stripped, which is exactly what the live state carries -- no lookup."""
    page = PAGE.read_text(encoding="utf-8")
    fn = page[page.index("function setMapArt"):]
    fn = fn[:fn.index("function verdict")]
    assert 'toLowerCase().replace(/[^a-z0-9]/g, "")' in fn
    # Addressed from the directory the page is served from. A bare relative
    # path 404s on a match's own page at /m/<id>; a rooted one breaks the demo
    # under the GitHub Pages sub-path.
    assert 'url("${base()}maps/${name}-splash.jpg")' in fn
    assert '"none"' in fn, "a map with no art must resolve to none, not a 404"


@pytest.mark.skipif(not MAPS.is_dir(), reason="art not downloaded")
def test_every_map_the_database_has_seen_has_a_background():
    """The point of the theme: Pearl looks like Pearl and Bind like Bind.

    A missing file is not an exception anywhere -- the background simply does
    not paint -- so nothing would report it but this.
    """
    import sqlite3

    from dotenv import dotenv_values

    from valwr import config

    # About this machine's data, so it asks for the database that is really
    # configured -- the suite otherwise never reads .env -- and opens it
    # read-only: no test may write to real data.
    where = dotenv_values(config.REPO_ROOT / ".env").get("DATABASE_PATH")
    path = Path(where or "data/valwr.db")
    if not path.is_absolute():
        path = config.REPO_ROOT / path
    if not path.exists():
        pytest.skip("no database")
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    played = [r[0] for r in conn.execute(
        "SELECT DISTINCT map FROM matches WHERE map IS NOT NULL")]
    conn.close()
    assert played, "no maps in the database to check"

    missing = []
    for name in played:
        slug = "".join(ch for ch in name.lower() if ch.isalnum())
        if not (MAPS / f"{slug}-splash.jpg").exists():
            missing.append(name)
    assert not missing, (
        f"no splash art for {missing}; run tools/fetch_agent_art.py")


# --- the stylesheet ----------------------------------------------------

# Every class the renderer emits. The page builds HTML by string
# concatenation, so a rule that disappears takes no JavaScript with it: the
# markup is identical, the render harness passes, and the page renders as
# unstyled serif text on white boxes. That happened -- a slice meant to delete
# the theme picker ran to </style> and removed the entire component sheet, and
# all 275 tests still passed.
EMITTED = (
    "shell", "top", "brand", "mapwrap", "map", "mapsub", "odds", "oddsbar",
    "verdict", "warn", "grid", "rosters", "team", "tbar", "tname", "tside",
    "trule", "tcount", "row", "art", "score", "who", "nm", "tag", "rs",
    "alert", "st", "panel", "phero", "veil", "stripe", "pid", "pbody", "pcol",
    "bigscore", "pwhy", "sect", "cmp", "chips", "formrow", "flag", "hint",
    "fx", "dir", "track", "mag", "idle", "foot", "num", "disp", "pulse",
    "rank", "party", "partynote", "result", "compare",
)


def test_the_stylesheet_defines_every_class_the_page_emits():
    page = PAGE.read_text(encoding="utf-8")
    css = page[page.index("<style>"):page.index("</style>")]
    missing = [c for c in EMITTED if f".{c}" not in css]
    assert not missing, f"no rule for {missing} -- the page renders unstyled"


def test_the_stylesheet_is_not_truncated():
    """A blunter version of the same guard: the sheet has a known floor and
    must still close. Catches a slice that ate the tail of it."""
    page = PAGE.read_text(encoding="utf-8")
    assert page.count("<style>") == 1 and page.count("</style>") == 1
    css = page[page.index("<style>"):page.index("</style>")]
    assert len(css) > 12000, f"stylesheet is only {len(css)} chars; truncated?"
    assert css.count("{") == css.count("}"), "unbalanced braces in the sheet"


def test_nothing_paints_over_the_map():
    """The map is a `background-image` on `body`, and so is the weave texture.

    Both were plain `body{...}` rules of equal specificity, so source order
    decided which won -- and the weave came later, resetting the map to none.
    The page looked correct in every other respect and the render harness,
    which executes JavaScript, could not see it.

    So: whichever body rule sets the map must be the last one that touches
    body's background at all.
    """
    page = PAGE.read_text(encoding="utf-8")
    css = page[page.index("<style>"):page.index("</style>")]
    # Comments first: a `}` inside one truncates the block match, and these
    # comments quote CSS. That silently made this check pass on nothing.
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    # Every `body { ... }` block -- not body::before, not body[data-x].
    blocks = [(m.start(), m.group(1))
              for m in re.finditer(r"(?:^|\n)(?:html,)?body\s*\{([^}]*)\}", css)]
    assert blocks, "no body rule at all"

    paints = [(pos, body) for pos, body in blocks
              if re.search(r"(?<!-)\bbackground(?:-image)?\s*:", body)]
    assert paints, "nothing paints body's background"

    last_pos, last_body = paints[-1]
    assert "--mapart" in last_body, (
        "the last body rule to touch the background does not set the map; "
        "something paints over it")

    # And the shorthand must not appear after it either, since `background:`
    # resets background-image wholesale.
    for pos, body in blocks:
        if pos > last_pos:
            assert not re.search(r"(?<!-)\bbackground\s*:", body), (
                "a later body rule uses the `background` shorthand, which "
                "resets the map image")


@needs_node
def test_agent_select_shows_every_number_on_each_card(tmp_path):
    """About a minute to lock in: the rating, last-20 K/D/A, headshots, their
    agents on this map and their last game, readable without a click -- and a
    player still being looked up never shown as having no history."""
    from valwr.dash.demo import demo_state

    s = demo_state(phase="pregame")
    for i, p_ in enumerate(s["players"]):
        if p_["agent"]:
            p_["agent_id"] = f"agent-{i:02d}"
    state = tmp_path / "state.json"
    state.write_text(json.dumps(s), encoding="utf-8")
    # The same match once it loads: Riot keeps one match id from agent select
    # into the game, and this tab follows it there.
    loaded = demo_state(phase="coregame")
    loaded["match_id"] = s["match_id"]
    match = tmp_path / "match.json"
    match.write_text(json.dumps(loaded), encoding="utf-8")
    r = subprocess.run([node, str(ROOT / "test" / "pregame_check.mjs"),
                        str(PAGE), str(state), str(match)],
                       capture_output=True, text=True, timeout=120)
    sys.stdout.write(r.stdout)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_pregame_demo_is_shaped_like_a_real_poll():
    """The page is tested against this; if it drifted from what poll_once
    sends in agent select, the test would pass on a screen nobody sees."""
    from valwr.dash.demo import demo_state

    s = demo_state(phase="pregame")
    assert s["phase"] == "pregame"
    assert {p["team"] for p in s["players"]} == {s["own_team"]}, (
        "Riot describes only your own team in agent select")
    assert s["prediction"] is None
    assert s["lookup"]["pending"], "the looking-up state must be exercised"
    assert any(p["agent"] is None for p in s["players"])
