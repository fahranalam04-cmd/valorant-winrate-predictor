"""The game client's own event channel, waking the dashboard's poll early."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from valwr.live import events as E


def frame(uri, op=8, event=E.SUBSCRIPTION):
    return json.dumps([op, event, {"data": {}, "eventType": "Update",
                                   "uri": uri}])


PREGAME = "/riot-messaging-service/v1/message/ares-pregame/pregame/v1/matches/abc"
COREGAME = ("/riot-messaging-service/v1/message/ares-core-game/core-game/v1/"
            "matches/abc")


@pytest.mark.parametrize("message", [frame(PREGAME), frame(COREGAME),
                                     frame(PREGAME).encode()])
def test_agent_select_and_the_match_starting_are_announcements(message):
    assert E.is_match_event(message)


@pytest.mark.parametrize("message", [
    frame("/riot-messaging-service/v1/message/ares-parties/parties/v1/parties/x"),
    frame("/chat/v4/presences"),
    frame(PREGAME, op=5),                       # a subscribe, not an event
    frame(PREGAME, event="OnJsonApiEvent"),     # the firehose, never asked for
    json.dumps([8, E.SUBSCRIPTION, "not an object"]),
    json.dumps([8, E.SUBSCRIPTION]),
    "",
    "not json",
    "null",
])
def test_everything_else_is_not(message):
    assert not E.is_match_event(message)


class FakeSocket:
    """A websocket that hands out queued frames, then times out."""

    def __init__(self, frames, drop_after=False):
        self.frames = list(frames)
        self.drop_after = drop_after
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def send(self, message):
        self.sent.append(message)

    def recv(self, timeout=None):
        if self.frames:
            return self.frames.pop(0)
        if self.drop_after:
            raise ConnectionResetError("the game closed")
        time.sleep(min(timeout or 0.01, 0.01))
        raise TimeoutError


def run_until(listener, done, seconds=3.0):
    listener.start()
    end = time.monotonic() + seconds
    while time.monotonic() < end and not done():
        time.sleep(0.005)
    listener.stop()
    listener._thread.join(timeout=3)
    assert not listener._thread.is_alive(), "stop() must end the thread"


def test_it_wakes_once_per_announcement_and_sends_only_the_subscription():
    """docs/ETHICS-AND-TOS.md: read-only. The one message ever sent is the
    request to hear the messaging service -- nothing that acts on the game."""
    sock = FakeSocket([frame("/chat/v4/presences"), frame(PREGAME),
                       frame(PREGAME), frame(COREGAME)])
    woke = []
    listener = E.MatchEvents(lambda: woke.append(1), connect=lambda: sock)
    run_until(listener, lambda: len(woke) == 3 and not sock.frames)
    assert len(woke) == 3 and listener.events == 3
    assert sock.sent == [json.dumps([5, E.SUBSCRIPTION])]


def test_a_refused_or_dropped_channel_is_retried_with_a_fresh_connection():
    """The game restarting moves the port, so each attempt connects anew --
    and nothing about the poll depends on this succeeding."""
    attempts = []
    woke = []

    def connect():
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionRefusedError("game not running yet")
        if len(attempts) == 2:
            return FakeSocket([frame(PREGAME)], drop_after=True)
        return FakeSocket([frame(COREGAME)])

    listener = E.MatchEvents(lambda: woke.append(1), connect=connect,
                             retry_seconds=0.01)
    run_until(listener, lambda: len(woke) == 2)
    assert len(attempts) >= 3, "it gave up after a failure"
    assert len(woke) == 2


def test_a_failing_wake_does_not_end_the_listener():
    calls = []

    def wake():
        calls.append(1)
        raise RuntimeError("event loop closed")

    sockets = iter([FakeSocket([frame(PREGAME)], drop_after=True),
                    FakeSocket([frame(PREGAME)])])
    listener = E.MatchEvents(wake, connect=lambda: next(sockets),
                             retry_seconds=0.01)
    run_until(listener, lambda: len(calls) == 2)
    assert len(calls) == 2


def test_a_failure_that_is_not_the_game_being_closed_is_said_once(capsys):
    """Swallowed silently, a missing library or a changed API cost the
    instant detection with nothing to say why."""
    from valwr.live.lockfile import ClientNotRunning
    failures = iter([ClientNotRunning("closed"), ConnectionRefusedError(),
                     TypeError("unexpected keyword 'ssl'"),
                     TypeError("unexpected keyword 'ssl'")])
    seen = []

    def connect():
        seen.append(1)
        raise next(failures)
    listener = E.MatchEvents(lambda: None, connect=connect, retry_seconds=0.01)
    run_until(listener, lambda: len(seen) >= 4)
    out = capsys.readouterr().out
    assert out.count("match announcements unavailable") == 1
    assert "TypeError" in out and "closed" not in out


def test_the_library_it_needs_is_new_enough():
    """`ssl=` arrived in websockets 13; on 12 every connect failed."""
    import websockets
    major = int(websockets.__version__.split(".")[0])
    assert major >= 13


# --- the poll's wait --------------------------------------------------

def _timed(coro_fn):
    async def go():
        start = asyncio.get_running_loop().time()
        await coro_fn()
        return asyncio.get_running_loop().time() - start
    return asyncio.run(go())


def test_with_no_announcement_the_wait_is_the_full_poll_interval():
    from valwr.dash import server as DS

    async def wait():
        await DS.nap(asyncio.Event(), 0.15, asyncio.get_running_loop().time())
    assert _timed(wait) >= 0.14


def test_an_announcement_cuts_the_wait_short(monkeypatch):
    from valwr.dash import server as DS
    monkeypatch.setattr(DS, "MIN_WAKE_GAP_SECONDS", 0.0)

    async def wait():
        wake = asyncio.Event()
        asyncio.get_running_loop().call_later(0.03, wake.set)
        await DS.nap(wake, 5.0, asyncio.get_running_loop().time())
        assert not wake.is_set(), "taken, so the next wait is a full one"
    assert _timed(wait) < 1.0


def test_one_that_came_during_a_slow_lookup_is_not_lost(monkeypatch):
    """A lookup can run for seconds; a lock-in announced meanwhile must still
    bring the next poll forward."""
    from valwr.dash import server as DS
    monkeypatch.setattr(DS, "MIN_WAKE_GAP_SECONDS", 1.0)

    async def wait():
        wake = asyncio.Event()
        wake.set()                       # arrived while the poll was busy
        began = asyncio.get_running_loop().time() - 4.0   # a 4 s lookup
        await DS.nap(wake, 5.0, began)
    assert _timed(wait) < 0.5


def test_a_burst_of_announcements_does_not_poll_faster_than_the_gap(monkeypatch):
    """A lobby locking in announces every hover. Each still waits until the
    gap since the last poll has passed."""
    from valwr.dash import server as DS
    monkeypatch.setattr(DS, "MIN_WAKE_GAP_SECONDS", 0.2)

    async def wait():
        wake = asyncio.Event()
        wake.set()
        await DS.nap(wake, 5.0, asyncio.get_running_loop().time())
    assert 0.18 <= _timed(wait) < 1.0


def test_the_gap_is_well_under_the_poll_interval():
    """Otherwise the wake would do nothing."""
    from valwr.dash import server as DS
    assert DS.MIN_WAKE_GAP_SECONDS <= DS.POLL_SECONDS / 2


# --- wired into the dashboard ----------------------------------------

class _Listener:
    made: list = []

    def __init__(self, on_match):
        self.on_match = on_match
        self.started = self.stopped = False
        _Listener.made.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


@pytest.fixture
def listener(monkeypatch):
    _Listener.made = []
    monkeypatch.setattr(E, "MatchEvents", _Listener)
    return _Listener


def test_an_announcement_reaches_the_page_before_the_poll_interval(
        monkeypatch, listener):
    """End to end: the client's event, through the listener's thread, to the
    next state on the page -- with the poll interval long enough that only the
    wake could explain it."""
    from fastapi.testclient import TestClient

    from valwr.dash import server as DS

    class Ctx:
        index = role_index = conn = client = None
        settings = type("S", (), {"region": "na"})()

        def close(self):
            pass

    polls = []

    def poll(ctx, on_progress=None):
        polls.append(time.monotonic())
        return None                               # in the menus

    monkeypatch.setattr(DS.st, "open_context", lambda **kw: Ctx())
    monkeypatch.setattr(DS, "poll_and_record", poll)
    monkeypatch.setattr(DS, "recent_rows", lambda: [])
    monkeypatch.setattr(DS, "POLL_SECONDS", 4.0)
    monkeypatch.setattr(DS, "MIN_WAKE_GAP_SECONDS", 0.0)

    with TestClient(DS.build_app(no_fetch=True, listen=True),
                    base_url="http://127.0.0.1:8787") as client:
        assert len(listener.made) == 1 and listener.made[0].started
        with client.websocket_connect("ws://127.0.0.1:8787/ws") as ws:
            assert ws.receive_json()["status"] == "working"
            ws.receive_json()                      # the first poll
            woken_at = time.monotonic()
            # As the listener does it: from its own thread.
            threading.Thread(target=listener.made[0].on_match).start()
            ws.receive_json()                      # the next one
            assert time.monotonic() - woken_at < 2.0, "the wake did nothing"
    assert listener.made[0].stopped, "the listener outlived the server"


@pytest.mark.parametrize("kw", [{"demo": True}, {"demo": "pregame"},
                                {"match": "m1"}, {}])
def test_only_the_live_dashboard_listens(kw, listener):
    """The demo and a replay have no game behind them; and only main() asks
    for a listener, so tests building an app get none by default."""
    from fastapi.testclient import TestClient

    from valwr.dash import server as DS
    with TestClient(DS.build_app(no_fetch=True, **kw)):
        pass
    if kw:
        with TestClient(DS.build_app(no_fetch=True, listen=True, **kw)):
            pass
    assert listener.made == []


def test_the_launcher_asks_for_the_listener(tmp_path, monkeypatch):
    import uvicorn

    from valwr import config
    from valwr.dash import server as S
    # Never the real port file beside the user's database.
    monkeypatch.setattr(S, "port_file", lambda: tmp_path / "dashboard-port")
    monkeypatch.setattr(S, "port_free", lambda host, port: True)

    def no_database(**kw):
        raise RuntimeError("tests must not open the real database")
    monkeypatch.setattr(config, "load", no_database)
    asked = {}
    real = S.build_app

    def build_app(**kw):
        asked.update(kw)
        return real(**kw)
    monkeypatch.setattr(S, "build_app", build_app)

    class _Server:
        def __init__(self, cfg):
            self.started = True

        def run(self):
            pass
    monkeypatch.setattr(uvicorn, "Server", _Server)

    assert S.main(["--port", "8791", "--no-browser", "--no-tabs"]) == 0
    assert asked.get("listen") is True
