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
