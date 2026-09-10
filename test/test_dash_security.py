"""The dashboard answers only the page it serves, on the machine it runs on.

Binding 127.0.0.1 keeps the network out but not the browser. Any site open in
another tab can reach localhost, and websockets are exempt from the
same-origin policy, so without these checks that site could open /ws and read
the live match: every player's gamertag and stored statistics. DNS rebinding
gets further still, pointing an attacker's own domain at 127.0.0.1 so the
browser treats this server as same-origin with them.

Websocket URLs here are absolute on purpose. Starlette's TestClient ignores
`base_url` for websockets and sends `Host: testserver`, which the server
refuses as a domain name -- so a relative "/ws" made every refusal test pass
for the wrong reason, and never reached the origin check at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from valwr.dash import server as S

LOCAL = "http://127.0.0.1:8787"
PHONE = "http://192.168.1.20:8787"


def ws_url(base: str) -> str:
    return base.replace("http://", "ws://", 1) + "/ws"


@pytest.fixture
def make_app(monkeypatch):
    """A fresh app per client: each TestClient shuts its app's poll pool down.

    No game client and no database -- a handshake is what is under test, so
    the context reports VALORANT as closed and the socket sends that error.
    """
    def closed(**_):
        raise S.st.NotReady("VALORANT is not running")

    monkeypatch.setattr(S.st, "open_context", closed)
    return lambda: S.build_app(no_fetch=True)


@pytest.mark.parametrize("host", [
    "127.0.0.1:8787", "127.0.0.1", "localhost:8787", "[::1]:8787",
    "192.168.1.20:8787", "10.0.0.5",
])
def test_loopback_and_ip_literal_hosts_are_allowed(host):
    assert S.host_allowed(host)


@pytest.mark.parametrize("host", [
    None, "", "testserver", "attacker.example", "attacker.example:8787",
    "127.0.0.1.nip.io:8787", "localhost.attacker.example", "[::1",
])
def test_any_domain_name_is_refused(host):
    assert not S.host_allowed(host)


def test_a_rebound_domain_can_neither_load_the_page_nor_open_the_socket(make_app):
    rebound = "http://attacker.example:8787"
    with TestClient(make_app(), base_url=rebound) as client:
        assert client.get("/").status_code == 403
        # Same-origin with itself, as a rebinding page would be: only the
        # host check can stop this one.
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(ws_url(rebound),
                                          headers={"origin": rebound}):
                pass
        assert refused.value.code == 1008


@pytest.mark.parametrize("origin", [
    "http://attacker.example",
    "null",                          # sandboxed iframes and file:// pages
    "http://localhost:8787",         # a different origin, even if it is us
])
def test_another_origin_cannot_open_the_socket(make_app, origin):
    with TestClient(make_app(), base_url=LOCAL) as client:
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(ws_url(LOCAL),
                                          headers={"origin": origin}):
                pass
        assert refused.value.code == 1008


@pytest.mark.parametrize("base", [LOCAL, PHONE])
def test_the_page_itself_still_connects_locally_and_from_a_phone(make_app, base):
    with TestClient(make_app(), base_url=base) as client:
        assert client.get("/").status_code == 200
        with client.websocket_connect(ws_url(base),
                                      headers={"origin": base}) as ws:
            assert ws.receive_json()["status"] == "error"


def test_a_client_that_is_not_a_browser_page_needs_no_origin(make_app):
    """A page cannot drop the Origin header; only a local program can, and a
    local program could reach the port anyway."""
    with TestClient(make_app(), base_url=LOCAL) as client:
        with client.websocket_connect(ws_url(LOCAL)) as ws:
            assert ws.receive_json()["status"] == "error"


def test_the_page_is_served_with_a_policy_that_allows_nothing_external(make_app):
    with TestClient(make_app(), base_url=LOCAL) as client:
        r = client.get("/")
    policy = r.headers["content-security-policy"]
    assert "default-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert "connect-src 'self' ws://127.0.0.1:8787" in policy
    assert "http" not in policy, "no external origin may be allowed"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"


def test_downloaded_art_cannot_be_written_outside_its_directory():
    """tools/fetch_agent_art.py names files after ids from a third-party API."""
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "tools" / "fetch_agent_art.py"
    spec = importlib.util.spec_from_file_location("fetch_agent_art", path)
    art = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(art)

    # Built rather than written out, so no UUID-shaped literal sits in the repo.
    good = "-".join(c * n for c, n in zip("abcde", (8, 4, 4, 4, 12)))
    assert art.UUID.fullmatch(good)
    for bad in ("../../evil", good + "/../../x", "..\\..\\evil", good.upper(), ""):
        assert not art.UUID.fullmatch(bad), bad
    assert art.slug("../Pearl/..") == "pearl"


def test_the_page_requests_nothing_from_outside():
    """The policy forbids it; this keeps the page from depending on it."""
    page = (Path(S.STATIC) / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script[^>]+src=", page)
    assert not re.search(r"<link\b", page)
    assert "@import" not in page
    assert not re.search(r"https?://", page)
