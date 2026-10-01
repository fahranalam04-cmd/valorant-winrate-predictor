"""Agent select has to fit on screen, all five players, without scrolling.

Measured in a real browser on the real page, because a layout can only be
checked by laying it out: the first agent-select version passed every
content check there was and still put the fifth player below the fold of a
maximised 1080p window. Needs playwright and an installed Chrome or Edge, so
it skips in CI and runs wherever the dashboard is developed.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agent_select_fits_the_window_and_clips_nothing(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    capture = _tool("capture_dashboard")
    _tool("build_pages_demo").build(tmp_path / "site", conn=None)

    with sync_api.sync_playwright() as p:
        try:
            browser = capture.launch(p)
        except Exception as e:                          # noqa: BLE001
            pytest.skip(f"no browser to lay the page out in: {e}")
        httpd, url = capture.serve(tmp_path / "site")
        try:
            problems = capture.check_agent_select(browser, url)
        finally:
            browser.close()
            httpd.shutdown()
    assert not problems, "\n".join(problems)
