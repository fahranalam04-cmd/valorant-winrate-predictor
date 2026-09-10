"""The GitHub Pages demo is the real page, fed invented players.

A demo that re-implemented the dashboard would drift from it, and would show
visitors something the tool does not do. So these check that the page script is
copied verbatim, that the stand-in socket really drives it, and that nothing
real can reach a public build.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "valwr" / "dash" / "static" / "index.html"
node = shutil.which("node")


def _tool():
    spec = importlib.util.spec_from_file_location(
        "build_pages_demo", ROOT / "tools" / "build_pages_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def site(tmp_path):
    tool = _tool()
    info = tool.build(tmp_path / "site", conn=None)
    return tool, info, (tmp_path / "site" / "index.html").read_text(encoding="utf-8")


def test_the_page_script_is_copied_verbatim(site):
    _, _, html = site
    page = PAGE.read_text(encoding="utf-8")
    own = page[page.index("<script>"):page.index("</script>") + len("</script>")]
    assert own in html, "the demo must run the dashboard's own script, unedited"
    assert html.index("window.WebSocket = class") < html.index(own), (
        "the stand-in has to be defined before the page opens its socket")


def test_the_demo_carries_only_invented_players(site):
    _, _, html = site
    from valwr.dash.demo import NAMES
    state = json.loads(re.search(r"const BASE = (\{.*?\});\n", html).group(1))
    invented = {f"{n}#{t}" for n, t in NAMES}
    assert {p["name"] for p in state["players"]} <= invented
    assert all(p["puuid"].startswith("demo-") for p in state["players"])
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                         html), "no database, so no agent UUIDs either"


def test_the_demo_declares_a_policy_and_requests_nothing_external(site):
    tool, _, html = site
    assert f'content="{tool.POLICY}"' in html
    assert "connect-src" not in tool.POLICY, "the demo opens no connection"
    external = set(re.findall(r'(?:src|href)="(https?://[^"]+)"', html))
    assert external <= {tool.REPO_URL}, f"unexpected external links: {external}"


def test_a_build_refuses_to_replace_a_directory_it_did_not_create(tmp_path):
    tool = _tool()
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(SystemExit):
        tool.build(precious, conn=None)
    assert (precious / "notes.txt").exists()
    # Its own output is replaced freely.
    tool.build(tmp_path / "site", conn=None)
    tool.build(tmp_path / "site", conn=None)


@pytest.mark.skipif(node is None, reason="node is not installed")
def test_the_stand_in_socket_drives_the_real_page(site, tmp_path):
    r = subprocess.run([node, str(ROOT / "test" / "pages_check.mjs"),
                        str(tmp_path / "site" / "index.html")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
