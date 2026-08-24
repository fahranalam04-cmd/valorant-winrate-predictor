"""Fixed-window rate limiter.

Measured, not assumed. Probing the API every 10 seconds showed:

    t=1   remaining 29  reset 60
    t=13  remaining 28  reset 48
    t=59  remaining 24  reset  2
    t=71  remaining 29  reset 60     <- jumped back to full

So `reset` counts down to the window boundary, and the window is FIXED: the
allowance refills all at once rather than trickling. Two consequences, both
the opposite of what a token bucket assumes.

Unused quota **expires** at the boundary, so holding a reserve is pure waste --
an earlier version kept 15% back and it simply evaporated every minute.
And when the allowance runs out, `reset` is the exact time to wait, not a
guess.

The right shape is therefore: spend the window down, sleep precisely until it
rolls, repeat.


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
        self._reset_at: float | None = None

    def _affordable(self) -> float:
        """How many further requests the current window can pay for."""
        if self.server_remaining is None:
            # Before the first response, allow one probe so the real numbers
            # can be observed.
            return 1.0
        if self._reset_at is not None and time.monotonic() >= self._reset_at:
            # The window boundary has passed. Assume it refilled and allow a
            # probe: quota is only observable by spending some, so waiting for
            # a fresh reading that can never arrive is a deadlock. This is what
            # hung the test suite -- with remaining at 0 and no new response to
            # correct it, acquire() slept forever.
            self.server_remaining = self.server_limit
            self._reset_at = None
        return self.server_remaining / max(self.cost_per_request, 1.0)

    def _seconds_to_reset(self) -> float:
        if self._reset_at is None:
            return 5.0
        return max(0.0, self._reset_at - time.monotonic())

    def acquire(self, tokens: int = 1) -> float:
        """Block until the window can pay for a request. Returns seconds waited.

        No smooth pacing: in a fixed window, quota not spent before the
        boundary is lost, so spending it as it becomes available is strictly
        better than trickling.
        """
        waited = 0.0
        while True:
            with self._lock:
                if self._affordable() >= tokens:
                    # Debit optimistically; observe() overwrites this with the
                    # authoritative count when the response returns.
                    if self.server_remaining is not None:
                        self.server_remaining -= self.cost_per_request
                    self.acquired += tokens
                    self.waited_seconds += waited
                    return waited
                sleep_for = max(1.0, self._seconds_to_reset())
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

            # The server's count is authoritative; ours was only a placeholder
            # between responses.
            self.server_remaining = remaining
            self.server_limit = limit or self.server_limit
            if reset > 0:
                self._reset_at = time.monotonic() + reset

    def penalise(self, seconds: float) -> None:
        """Back off after a 429: treat the window as spent until it rolls."""
        with self._lock:
            self.server_remaining = 0
            self._reset_at = time.monotonic() + max(1.0, seconds)
