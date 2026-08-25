"""Local Riot client authentication.

The lockfile is written by the Riot Client and holds the port and password for
its local HTTP API. Read-only use only -- see docs/ETHICS-AND-TOS.md, which is
a ban-safety document rather than a formality.

One trap worth stating up front, because it cost a wrong assumption during
Phase 6: **the lockfile existing does not mean VALORANT is running.** It is
written whenever the Riot Client launcher is up, which is most of the time on a
machine that has the game installed. With the launcher alone, `/help` exposes
seven functions -- Exit, Help, Subscribe and friends -- and every endpoint this
project needs returns 404. The game's own API only appears once VALORANT
itself launches, so liveness is checked by asking what the API can actually do.
"""

from __future__ import annotations

import base64
import os
import pathlib
from dataclasses import dataclass

import httpx

LOCKFILE = pathlib.Path(
    os.environ.get("LOCALAPPDATA", pathlib.Path.home() / "AppData/Local")
) / "Riot Games" / "Riot Client" / "Config" / "lockfile"

# Functions the launcher exposes on its own. Seeing only these means the game
# is not running, however present the lockfile is.
LAUNCHER_ONLY = {
    "Exit", "Help", "Subscribe", "Unsubscribe", "WebSocketFormat",
    "GetRiotclientappV1IsXbgpRunning", "PostRiotclientappV1NewArgs",
}

TIMEOUT = 10.0


class ClientNotRunning(RuntimeError):
    """The Riot Client is not up, or is up without the game."""


@dataclass(frozen=True)
class Lock:
    name: str
    pid: int
    port: int
    password: str
    protocol: str

    @property
    def base(self) -> str:
        return f"{self.protocol}://127.0.0.1:{self.port}"

    @property
    def auth_header(self) -> dict[str, str]:
        # Username is the literal string "riot".
        token = base64.b64encode(f"riot:{self.password}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    def client(self) -> httpx.Client:
        """An HTTP client for the local API.

        TLS verification is disabled because the certificate is self-signed --
        and *only* here. It must never be disabled for the HenrikDev client,
        which talks to the public internet.
        """
        return httpx.Client(base_url=self.base, headers=self.auth_header,
                            verify=False, timeout=TIMEOUT)


def read(path: pathlib.Path | None = None) -> Lock:
    """Parse the lockfile. Raises ClientNotRunning if it is absent."""
    path = path or LOCKFILE
    if not path.exists():
        raise ClientNotRunning(
            f"no lockfile at {path} -- the Riot Client is not running")
    parts = path.read_text(encoding="utf-8").strip().split(":")
    if len(parts) != 5:
        raise ClientNotRunning(
            f"lockfile has {len(parts)} fields, expected 5 "
            f"(name:pid:port:password:protocol)")
    name, pid, port, password, protocol = parts
    return Lock(name=name, pid=int(pid), port=int(port),
                password=password, protocol=protocol)


def functions(lock: Lock) -> set[str]:
    """Everything the local API currently exposes."""
    with lock.client() as c:
        r = c.get("/help")
        r.raise_for_status()
        return set(r.json().get("functions", {}))


def game_is_running(lock: Lock | None = None) -> bool:
    """Is VALORANT itself up, as opposed to just the launcher?

    Asks the API what it can do rather than trusting the lockfile's existence
    or scanning the process table -- the API's own answer is what determines
    whether the endpoints this project needs will work.
    """
    try:
        lock = lock or read()
        return bool(functions(lock) - LAUNCHER_ONLY)
    except (ClientNotRunning, httpx.HTTPError, ValueError):
        return False


def describe() -> str:
    """A one-line status suitable for a CLI or the dashboard."""
    try:
        lock = read()
    except ClientNotRunning as e:
        return f"not running ({e})"
    try:
        extra = functions(lock) - LAUNCHER_ONLY
    except httpx.HTTPError as e:
        return f"lockfile found on port {lock.port}, but the API refused: {e}"
    if not extra:
        return (f"Riot Client up on port {lock.port}, but VALORANT is not "
                f"running -- only launcher functions are exposed")
    return f"VALORANT running on port {lock.port} ({len(extra)} game functions)"
