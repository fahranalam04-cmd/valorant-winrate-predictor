"""Authentication context for the live client.

Everything the remote game endpoints need, gathered from the local API. All of
it was verified against a running client rather than taken from documentation,
because three details differ from what the docs imply:

1. **`X-Riot-ClientVersion` is required.** Without it the glz endpoints return
   an opaque `400`, which reads like a malformed URL rather than a missing
   header. It comes from `/product-session/v1/external-sessions`.

2. **Use `/riotclient/region-locale` for the shard, not `/chat/v1/session`.**
   The chat session reports its own server -- `la1` on this account -- which is
   an XMPP region, not the game shard. Building a glz URL from it silently
   produces a host that does not exist.

3. **The region does not need scraping off the process command line.**
   docs/API-NOTES.md suggests parsing `-ares-deployment` from the running
   process; the local API reports it directly, which is simpler and does not
   break when the launcher changes its arguments.

Read-only throughout. See docs/ETHICS-AND-TOS.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from valwr.live.lockfile import ClientNotRunning, Lock, read

# Base64 of the platform descriptor the client sends. Static; it identifies the
# platform, not the user.
CLIENT_PLATFORM = (
    "ew0KCSJwbGF0Zm9ybVR5cGUiOiAiUEMiLA0KCSJwbGF0Zm9ybU9TIjogIldpbmRvd3MiLA0K"
    "CSJwbGF0Zm9ybU9TVmVyc2lvbiI6ICIxMC4wLjE5MDQyLjEuMjU2LjY0Yml0IiwNCgkicGxh"
    "dGZvcm1DaGlwc2V0IjogIlVua25vd24iDQp9"
)

VERSION_FALLBACK = "https://valorant-api.com/v1/version"
TIMEOUT = 20.0


@dataclass(frozen=True)
class Session:
    puuid: str
    shard: str
    access_token: str
    entitlements_token: str
    client_version: str

    @property
    def glz(self) -> str:
        return f"https://glz-{self.shard}-1.{self.shard}.a.pvp.net"

    @property
    def pd(self) -> str:
        return f"https://pd.{self.shard}.a.pvp.net"

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "X-Riot-Entitlements-JWT": self.entitlements_token,
            "X-Riot-ClientVersion": self.client_version,
            "X-Riot-ClientPlatform": CLIENT_PLATFORM,
        }

    def __repr__(self) -> str:
        # Never let tokens reach a log or a traceback.
        return (f"Session(puuid={self.puuid[:8]}..., shard={self.shard!r}, "
                f"version={self.client_version!r}, tokens=<redacted>)")


# The external-sessions map is keyed by opaque session id, but it also carries
# a "host_app" entry whose version is the literal string "0" -- a placeholder
# for the launcher itself. Taking the first dict with a version key returns
# that placeholder, and a bogus X-Riot-ClientVersion earns an opaque 400 from
# the glz endpoints rather than anything that names the real problem.
PLACEHOLDER_SESSIONS = {"host_app"}
MIN_VERSION_LENGTH = 4


def _client_version(local: httpx.Client) -> str:
    """Read the running game's client version, falling back to valorant-api.com."""
    try:
        data = local.get("/product-session/v1/external-sessions").json()
        for key, value in data.items():
            if key in PLACEHOLDER_SESSIONS or not isinstance(value, dict):
                continue
            version = value.get("version")
            if version and len(str(version)) >= MIN_VERSION_LENGTH:
                return str(version)
    except (httpx.HTTPError, ValueError, AttributeError):
        pass
    return httpx.get(VERSION_FALLBACK, timeout=TIMEOUT).json()["data"][
        "riotClientVersion"]


def build(lock: Lock | None = None) -> Session:
    """Gather the auth context. Raises ClientNotRunning if the game is not up."""
    lock = lock or read()
    with lock.client() as local:
        try:
            ent = local.get("/entitlements/v1/token")
            ent.raise_for_status()
            ent = ent.json()
        except httpx.HTTPError as e:
            raise ClientNotRunning(
                "the entitlements endpoint is unavailable -- the Riot Client "
                f"is up but VALORANT probably is not ({e})") from e

        if not ent.get("subject"):
            raise ClientNotRunning("no puuid in the entitlements response")

        # region-locale, deliberately: /chat/v1/session reports its XMPP
        # server region and would give a shard that does not exist.
        try:
            shard = (local.get("/riotclient/region-locale").json()
                     .get("region") or "na").lower()
        except (httpx.HTTPError, ValueError):
            shard = "na"

        return Session(
            puuid=ent["subject"],
            shard=shard,
            access_token=ent["accessToken"],
            entitlements_token=ent["token"],
            client_version=_client_version(local),
        )
