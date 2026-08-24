"""Token-bucket rate limiter.

Every call to api.henrikdev.xyz goes through one of these. The limit is the
binding constraint on this whole project, and the maintainer runs the service
for free -- see docs/ETHICS-AND-TOS.md. Do not add a bypass.
"""

from __future__ import annotations

import threading
import time

# Fraction of the stated ceiling we actually target. Bursting at exactly the
# published limit trips it: the server's window and ours never line up
# perfectly, so the last few percent buys 429s, not throughput.
HEADROOM = 0.9


class TokenBucket:
    def __init__(self, per_minute: int, headroom: float = HEADROOM):
        self.stated_per_minute = per_minute
        self.effective_per_minute = max(1, int(per_minute * headroom))
        self.rate = self.effective_per_minute / 60.0
        self.capacity = float(self.effective_per_minute)
        # Start empty, not full. A full bucket lets the first N requests fire
        # with zero spacing, and the server's rolling window may already hold
        # requests from a previous run -- which is exactly how the first live
        # crawl earned 429s while averaging under 3 req/min.
        self._tokens = 0.0
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self.acquired = 0
        self.waited_seconds = 0.0
        self.server_remaining: int | None = None
        self.server_limit: int = per_minute
        # Learned cost of one request in quota units. A size=10 matchlist fans
        # out to Riot and bills roughly 2, but this is measured rather than
        # assumed because it varies by endpoint and by cache hit.
        self.cost_per_request: float = 1.0
        self._prev_remaining: int | None = None

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    def acquire(self, tokens: int = 1) -> float:
        """Block until `tokens` are available. Returns seconds spent waiting."""
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self.acquired += tokens
                    self.waited_seconds += waited
                    return waited
                deficit = tokens - self._tokens
                sleep_for = deficit / self.rate
            time.sleep(sleep_for)
            waited += sleep_for

    def observe(self, headers) -> None:
        """Reconcile against the server's own quota accounting.

        HenrikDev reports x-ratelimit-limit / -remaining / -reset on every
        response. Modelling the limit client-side is guesswork; the server
        knows. We only ever revise *down* -- never grant ourselves more budget
        than it says we have.
        """
        try:
            remaining = int(headers.get("x-ratelimit-remaining", -1))
            reset = float(headers.get("x-ratelimit-reset", 0) or 0)
            limit = int(headers.get("x-ratelimit-limit", 0) or 0)
        except (TypeError, ValueError):
            return
        if remaining < 0:
            return

        with self._lock:
            # Learn what a request actually costs, from consecutive readings.
            if (self._prev_remaining is not None
                    and 0 < self._prev_remaining - remaining <= 20):
                observed = float(self._prev_remaining - remaining)
                self.cost_per_request = 0.7 * self.cost_per_request + 0.3 * observed
            self._prev_remaining = remaining

            self.server_remaining = remaining
            self.server_limit = limit or self.server_limit

            # Pace to the sustainable request rate rather than the raw unit
            # ceiling: 30 units/min at ~2 units per request is ~15 req/min.
            sustainable = self.server_limit / max(self.cost_per_request, 1.0)
            self.rate = max(sustainable * HEADROOM / 60.0, 1 / 120.0)

            # Keep a reserve and never sprint to zero. Hitting zero triggers a
            # fixed backoff whose length we cannot infer -- the reset header
            # reports the window size, not the time left in it -- and stalling
            # blind for a full window is what cost 82% of an hour's wall clock.
            reserve = max(2.0, 0.15 * self.server_limit)
            usable = max(0.0, remaining - reserve)
            self._refill()
            self._tokens = min(self._tokens, usable)

        if remaining == 0 and reset > 0:
            self.penalise(reset)

    def penalise(self, seconds: float) -> None:
        """Drain the bucket after a 429.

        A 429 means our accounting disagrees with the server's. Emptying the
        bucket and refusing to spend for `seconds` is the conservative
        response -- back off rather than probe the boundary.
        """
        with self._lock:
            self._refill()
            self._tokens = 0.0
            self._last = time.monotonic() + max(0.0, seconds)
