"""Wake the dashboard the moment a match starts, from the client's own events.

The dashboard asks the game whether you are in a match every three seconds,
so agent select was noticed 1.5 seconds late on average -- from a minute in
which to pick. The client also pushes an event the instant agent select or a
match begins, over the websocket on the same port as its local API, and
docs/ETHICS-AND-TOS.md allows listening to it.

This listens and nothing else: the only message it ever sends is the
subscription. It does not replace the poll -- it cuts the poll's wait short.
If the websocket is refused, drops, or the game restarts on a new port, the
poll carries on exactly as before while this reconnects in the background.
"""

from __future__ import annotations

import json
import ssl
import threading
from typing import Callable

# Only the messaging service: the full event stream is mostly chat presence.
SUBSCRIPTION = "OnJsonApiEvent_riot-messaging-service_v1_message"
# Agent select starting or changing, and the match itself starting.
MATCH_URIS = ("ares-pregame/pregame/v1/matches/",
              "ares-core-game/core-game/v1/matches/")
RETRY_SECONDS = 5.0


def is_match_event(message: str | bytes) -> bool:
    """Does this websocket frame announce agent select or a match?"""
    try:
        op, event, payload = json.loads(message)
    except (ValueError, TypeError):
        return False
    if op != 8 or event != SUBSCRIPTION or not isinstance(payload, dict):
        return False
    uri = payload.get("uri") or ""
    return any(part in uri for part in MATCH_URIS)


def _local_tls() -> ssl.SSLContext:
    # The game's own API on 127.0.0.1, with a self-signed certificate -- the
    # same exception lockfile.py makes for its REST calls, and only for them.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _connect():
    from websockets.sync.client import connect

    from valwr.live import lockfile
    lock = lockfile.read()                 # re-read: a restart moves the port
    return connect(f"wss://127.0.0.1:{lock.port}", ssl=_local_tls(),
                   additional_headers=lock.auth_header, open_timeout=5)


class MatchEvents:
    """A background thread calling `on_match` when agent select or a match
    starts. Start it, and stop it on the way out."""

    def __init__(self, on_match: Callable[[], None], connect=_connect,
                 retry_seconds: float = RETRY_SECONDS):
        self.on_match = on_match
        self._connect = connect
        self.retry_seconds = retry_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False
        self.events = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="valwr-match-events")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with self._connect() as ws:
                    ws.send(json.dumps([5, SUBSCRIPTION]))
                    self.connected = True
                    while not self._stop.is_set():
                        try:
                            message = ws.recv(timeout=1.0)
                        except TimeoutError:
                            continue
                        if is_match_event(message):
                            self.events += 1
                            self.on_match()
            except Exception:                       # noqa: BLE001
                # Game closed, refused, or restarted. The poll is unaffected;
                # try again shortly, re-reading the lockfile for the new port.
                pass
            self.connected = False
            self._stop.wait(self.retry_seconds)
