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
                  "a.agent", "p.reason", "f.ago", "p.team"):
        assert f"esc({field}" in page, f"{field} is printed without esc()"


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
    assert "/maps/${name}-splash.jpg" in fn
    assert '"none"' in fn, "a map with no art must resolve to none, not a 404"


@pytest.mark.skipif(not MAPS.is_dir(), reason="art not downloaded")
def test_every_map_the_database_has_seen_has_a_background():
    """The point of the theme: Pearl looks like Pearl and Bind like Bind.

    A missing file is not an exception anywhere -- the background simply does
    not paint -- so nothing would report it but this.
    """
    from valwr import config
    from valwr.store import schema

    s = config.load(require_key=False)
    if not s.database_path.exists():
        pytest.skip("no database")
    conn = schema.connect(s.database_path)
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
