"""Phase 1 tests: limiter, frontier state machine, stratification, crash safety."""

from __future__ import annotations

import time

import pytest

from valwr.collect import frontier
from valwr.collect.crawl import Crawler, harvest
from valwr.collect.limiter import TokenBucket
from valwr.store import schema


@pytest.fixture
def conn(tmp_path):
    c = schema.connect(tmp_path / "t.db")
    schema.create_all(c)
    return c


# --- limiter ---------------------------------------------------------

def test_limiter_leaves_headroom_below_stated_limit():
    """Bursting at exactly the published limit buys 429s, not throughput."""
    b = TokenBucket(30)
    assert b.stated_per_minute == 30
    assert b.effective_per_minute < 30


def test_limiter_starts_empty_so_it_paces_from_the_first_request():
    """A full bucket lets the first N requests fire unspaced.

    That is how the first live crawl earned 429s while averaging under
    3 req/min -- the server's rolling window still held earlier requests.
    """
    b = TokenBucket(6000)  # 90/s effective -- keeps the test fast
    assert b.acquire() > 0.0


def test_limiter_never_grants_more_than_the_server_reports():
    b = TokenBucket(6000)
    b.observe({"x-ratelimit-limit": "30", "x-ratelimit-remaining": "2",
               "x-ratelimit-reset": "60"})
    assert b.server_remaining == 2
    assert b._tokens <= 2.0


def test_exhausted_quota_triggers_a_backoff():
    b = TokenBucket(6000)
    b.observe({"x-ratelimit-limit": "30", "x-ratelimit-remaining": "0",
               "x-ratelimit-reset": "0.05"})
    assert b.acquire() > 0.0


def test_observe_ignores_responses_without_quota_headers():
    b = TokenBucket(6000)
    b.observe({})
    b.observe({"x-ratelimit-remaining": "not-a-number"})
    assert b.server_remaining is None


def test_penalise_drains_the_bucket():
    b = TokenBucket(6000)
    b.penalise(0.05)
    assert b.acquire() > 0.0


# --- frontier --------------------------------------------------------

def test_enqueue_is_deduplicated(conn):
    assert frontier.enqueue(conn, "p1", 13) is True
    assert frontier.enqueue(conn, "p1", 13) is False
    assert frontier.counts_by_state(conn) == {"pending": 1}


def test_band_of_maps_tiers_to_divisions():
    assert frontier.band_of(None) == 0      # unranked
    assert frontier.band_of(13) == 4        # Gold 2
    assert frontier.band_of(9) == 3         # Silver 1
    assert frontier.band_of(27) == 9        # Radiant


def test_claim_marks_fetching_then_complete(conn):
    frontier.enqueue_many(conn, [("p1", 13)])
    row = frontier.claim(conn)
    assert row["puuid"] == "p1"
    assert frontier.counts_by_state(conn) == {"fetching": 1}
    frontier.complete(conn, "p1")
    assert frontier.counts_by_state(conn) == {"done": 1}


def test_fail_retries_then_gives_up(conn):
    frontier.enqueue_many(conn, [("p1", 13)])
    for _ in range(frontier.MAX_ATTEMPTS - 1):
        frontier.claim(conn)
        frontier.fail(conn, "p1", "boom")
        assert frontier.counts_by_state(conn) == {"pending": 1}
    frontier.claim(conn)
    frontier.fail(conn, "p1", "boom")
    assert frontier.counts_by_state(conn) == {"failed": 1}


def test_release_does_not_charge_an_attempt(conn):
    """Our own deadline must not blacklist a blameless puuid."""
    frontier.enqueue_many(conn, [("p1", 13)])
    frontier.claim(conn)
    frontier.release(conn, "p1")
    assert frontier.counts_by_state(conn) == {"pending": 1}
    assert conn.execute("SELECT attempts FROM frontier").fetchone()["attempts"] == 0


# --- crash safety (acceptance criterion #2) --------------------------

def test_stale_claims_are_recovered_after_a_crash(conn):
    """Simulates kill -9 between claim and complete."""
    frontier.enqueue_many(conn, [("p1", 13), ("p2", 13)])
    frontier.claim(conn)
    assert frontier.counts_by_state(conn)["fetching"] == 1

    # Backdate the claim to look like a worker that died.
    conn.execute("UPDATE frontier SET claimed_at = ? WHERE state='fetching'",
                 (int(time.time()) - frontier.STALE_SECONDS - 1,))
    conn.commit()

    assert frontier.recover_stale(conn) == 1
    assert frontier.counts_by_state(conn) == {"pending": 2}
    assert "fetching" not in frontier.counts_by_state(conn)


def test_fresh_claims_are_not_stolen(conn):
    """recover_stale must not reclaim a worker that is merely slow."""
    frontier.enqueue_many(conn, [("p1", 13)])
    frontier.claim(conn)
    assert frontier.recover_stale(conn) == 0
    assert frontier.counts_by_state(conn) == {"fetching": 1}


# --- stratification (acceptance criterion #3) ------------------------

def test_claim_prefers_the_least_crawled_band(conn):
    """Inverse-frequency selection: a band that pulls ahead stops being chosen."""
    frontier.enqueue_many(conn, [("gold1", 13), ("gold2", 13), ("plat1", 16)])
    # Gold is already ahead.
    frontier.claim(conn)
    frontier.complete(conn, "gold1")

    row = frontier.claim(conn)
    assert row["tier_band"] == frontier.band_of(16), "should switch to the behind band"


def test_stratification_equalises_across_bands(conn):
    frontier.enqueue_many(conn, [(f"g{i}", 13) for i in range(10)]
                                + [(f"p{i}", 16) for i in range(10)])
    for _ in range(10):
        row = frontier.claim(conn)
        frontier.complete(conn, row["puuid"])

    bands = frontier.counts_by_band(conn)
    gold = bands[frontier.band_of(13)].get("done", 0)
    plat = bands[frontier.band_of(16)].get("done", 0)
    assert abs(gold - plat) <= 1, f"uneven: gold={gold} plat={plat}"


# --- harvest ---------------------------------------------------------

def test_harvest_extracts_match_and_players():
    doc = {"data": [{
        "metadata": {"match_id": "m1"},
        "players": [{"puuid": "a", "tier": {"id": 13}}, {"puuid": "b", "tier": None}],
    }]}
    assert harvest(doc) == [("m1", [("a", 13), ("b", None)])]


def test_harvest_skips_matches_without_an_id():
    assert harvest({"data": [{"metadata": {}, "players": []}]}) == []


def test_harvest_tolerates_empty_response():
    assert harvest({}) == []
    assert harvest({"data": None}) == []


# --- crawler integration ---------------------------------------------

class FakeClient:
    """Returns two matches sharing a player, so dedup is exercised."""

    def __init__(self):
        self.calls = 0

    def matches(self, region, platform, puuid, size=10, **kw):
        self.calls += 1
        return {"data": [
            {"metadata": {"match_id": f"m{self.calls}"},
             "players": [{"puuid": "shared", "tier": {"id": 13}},
                         {"puuid": f"new{self.calls}", "tier": {"id": 16}}]},
            {"metadata": {"match_id": "m_always"},
             "players": [{"puuid": "shared", "tier": {"id": 13}}]},
        ]}


def test_crawler_counts_new_versus_duplicate_matches(conn):
    frontier.enqueue_many(conn, [("seed", 13)])
    c = Crawler(conn, FakeClient(), TokenBucket(6000), "na", "pc")
    c.run(minutes=0.02, verbose=False)

    # 'm_always' recurs in every response, so it is new once and duplicate after.
    assert c.stats.matches_new >= 2
    assert c.stats.matches_seen_again >= 1
    assert c.stats.puuids_discovered >= 2
    assert c.stats.rate_limit_hits == 0
