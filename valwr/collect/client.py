"""HenrikDev API client.

Every call to api.henrikdev.xyz in this project goes through here. Phase 1 adds
the token-bucket limiter behind this same interface, so nothing else needs to
know about rate limiting.

Endpoint reference: docs/API-NOTES.md. Do not invent paths.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import httpx

from valwr.store import raw

BASE = "https://api.henrikdev.xyz"
TIMEOUT = 30.0


class HenrikError(RuntimeError):
    pass


class TransientError(HenrikError):
    """A network-level failure that says nothing about the request itself.

    Connection resets, timeouts, DNS blips -- and critically, every socket
    dying when the machine suspends. These must never end a crawl: an
    unattended overnight run has to survive the laptop going to sleep.
    """


class RateLimited(HenrikError):
    def __init__(self, retry_after: float | None):
        self.retry_after = retry_after
        super().__init__(
            f"rate limited by HenrikDev"
            + (f", retry after {retry_after}s" if retry_after else "")
        )


class HenrikClient:
    """Thin wrapper. Stores every response verbatim when given a connection.

    The raw store matters more than it looks: API calls are the rate-limited
    resource and parsing is free, so a parsing bug should cost a re-parse, not
    a re-crawl.
    """

    def __init__(
        self,
        api_key: str,
        conn: sqlite3.Connection | None = None,
        limiter: Any = None,
    ):
        self._conn = conn
        self._limiter = limiter
        self._api_key = api_key
        self._client = httpx.Client(
            base_url=BASE,
            timeout=TIMEOUT,
            # The raw key, no Bearer prefix -- see docs/API-NOTES.md.
            headers={"Authorization": api_key, "Accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def _reconnect(self) -> None:
        """Rebuild the connection pool after a network-level failure."""
        try:
            self._client.close()
        except Exception:
            pass
        self._client = httpx.Client(
            base_url=BASE, timeout=TIMEOUT,
            headers={"Authorization": self._api_key, "Accept": "application/json"},
        )

    def __enter__(self) -> "HenrikClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get(self, path: str, **params) -> dict:
        if self._limiter is not None:
            self._limiter.acquire()

        try:
            r = self._client.get(path, params=params or None)
        except httpx.HTTPError as e:
            # Sleep/resume kills every pooled connection. Drop the pool so the
            # next attempt dials fresh rather than reusing a dead socket.
            self._reconnect()
            raise TransientError(f"{type(e).__name__}: {e}") from e

        # Reconcile before anything can raise -- a 429 carries quota headers
        # too, and that is exactly when we most need them.
        if self._limiter is not None:
            self._limiter.observe(r.headers)

        if self._conn is not None:
            self._record(path, params, r)

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            raise RateLimited(float(retry_after) if retry_after else None)
        if r.status_code >= 400:
            raise HenrikError(f"{r.status_code} on {path}: {r.text[:200]}")

        return r.json()

    def _record(self, path: str, params: dict, r: httpx.Response) -> None:
        raw.record(self._conn, path, params, r.status_code, r.text)

    # --- endpoints (docs/API-NOTES.md) ---------------------------------

    def account(self, name: str, tag: str) -> dict:
        return self.get(f"/valorant/v2/account/{name}/{tag}")

    def account_by_puuid(self, puuid: str) -> dict:
        return self.get(f"/valorant/v2/by-puuid/account/{puuid}")

    def matches(
        self, region: str, platform: str, puuid: str, size: int = 10, **filters
    ) -> dict:
        """Matchlist by PUUID.

        One call returns up to `size` full matches, each with all 10 players --
        the efficiency lever against the rate limit.
        """
        return self.get(
            f"/valorant/v4/by-puuid/matches/{region}/{platform}/{puuid}",
            size=size,
            **filters,
        )

    def leaderboard(self, region: str, platform: str, **params) -> dict:
        return self.get(f"/valorant/v3/leaderboard/{region}/{platform}", **params)

    def mmr_history(self, region: str, platform: str, puuid: str) -> dict:
        return self.get(
            f"/valorant/v2/by-puuid/stored-mmr-history/{region}/{platform}/{puuid}"
        )
