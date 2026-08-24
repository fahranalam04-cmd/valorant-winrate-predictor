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


# --- resilience (the overnight-run failure) ---------------------------

class FlakyClient:
    """Fails with a network error N times, then succeeds.

    Models what actually happened overnight: the laptop slept, every pooled
    socket died, and an unhandled httpx error ended a 10-hour run three
    minutes in.
    """

    def __init__(self, failures: int):
        self.remaining = failures
        self.calls = 0

    def matches(self, region, platform, puuid, size=10, **kw):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            from valwr.collect.client import TransientError
            raise TransientError("ConnectError: connection reset")
        return {"data": [{"metadata": {"match_id": f"m{self.calls}"},
                          "players": [{"puuid": "x", "tier": {"id": 13}}]}]}


def test_network_failure_does_not_end_the_crawl(conn, monkeypatch):
    monkeypatch.setattr("valwr.collect.crawl.time.sleep", lambda s: None)
    frontier.enqueue_many(conn, [("seed", 13)])
    c = Crawler(conn, FlakyClient(failures=3), TokenBucket(6000), "na", "pc")
    c.run(minutes=0.05, verbose=False)

    assert c.stats.transient_errors == 3
    assert c.stats.players_fetched >= 1, "should have recovered and fetched"


def test_network_failure_does_not_charge_the_puuid_an_attempt(conn, monkeypatch):
    """A dead socket is not the player's fault; blacklisting them would be
    the wrong repair and would silently shrink the frontier."""
    monkeypatch.setattr("valwr.collect.crawl.time.sleep", lambda s: None)
    frontier.enqueue_many(conn, [("seed", 13)])
    c = Crawler(conn, FlakyClient(failures=2), TokenBucket(6000), "na", "pc")
    c.run(minutes=0.05, verbose=False)

    assert conn.execute("SELECT attempts FROM frontier WHERE puuid='seed'"
                        ).fetchone()["attempts"] == 0
    assert c.stats.failures == 0


def test_transient_backoff_is_capped(conn, monkeypatch):
    """Exponential backoff must not run away to hours on a long outage."""
    from valwr.collect.crawl import MAX_TRANSIENT_BACKOFF
    waits = []
    monkeypatch.setattr("valwr.collect.crawl.time.sleep", lambda s: waits.append(s))
    frontier.enqueue_many(conn, [("seed", 13)])
    c = Crawler(conn, FlakyClient(failures=12), TokenBucket(6000), "na", "pc")
    c.run(minutes=0.05, verbose=False)
    assert waits, "should have backed off"
    assert max(waits) <= MAX_TRANSIENT_BACKOFF


# --- coverage leverage (the breadth-vs-depth fix) ---------------------

def test_claim_prefers_players_appearing_in_more_collected_matches(conn):
    """Fetching a player who appears in five of our matches adds history to
    five at once. The first version ignored this and optimised breadth, which
    left 86% of players on a single match and only 1.3% of matches trainable.
    """
    from valwr.store import normalize
    frontier.enqueue_many(conn, [("hub", 13), ("leaf", 13)])

    # 'hub' appears in three collected matches, 'leaf' in one.
    for i, roster in enumerate([["hub", "leaf"], ["hub", "x"], ["hub", "y"]]):
        normalize.upsert_match(conn, {
            "match_id": f"m{i}", "started_at": 1000 + i, "map": "Sunset",
            "mode": "competitive", "queue": None, "region": "na", "season": None,
            "rounds_red": 13, "rounds_blue": 5, "winner": "Red",
            "data_quality": None, "ingested_at": 0})
        normalize.upsert_players(conn, [
            {"match_id": f"m{i}", "puuid": p, "team": "Red", "agent": "Jett",
             "party_id": None, "tier": 13, "account_level": 1, "score": 1,
             "kills": 1, "deaths": 1, "assists": 1, "headshots": 1,
             "bodyshots": 1, "legshots": 1, "damage_dealt": 1, "damage_taken": 1,
             "started_at": 1000 + i, "map": "Sunset", "won": 1,
             "rounds_played": 20, "first_bloods": 0, "first_deaths": 0,
             "multikills": 0, "trade_kills": 0, "traded_deaths": 0,
             "kast_rounds": 10, "clutches": 0, "_name": p, "_tag": "NA1"}
            for p in roster])
    conn.commit()

    assert frontier.claim(conn)["puuid"] == "hub"


def test_coverage_summary_counts_trainable_matches(conn):
    """Match count alone is misleading -- a match whose players have no prior
    history carries no features."""
    s = frontier.coverage_summary(conn)
    assert set(s) == {"matches", "full_10", "usable_8", "partial_5"}
    assert s["matches"] == 0


def test_crawler_normalises_inline_so_there_is_one_writer(conn):
    """SQLite allows exactly one writer, and a separate normaliser process
    deadlocked against a live crawl in both directions. The crawler now writes
    both raw and normalised tables itself."""
    class RealShapeClient:
        def matches(self, region, platform, puuid, size=10, **kw):
            return {"data": [{
                "metadata": {"match_id": "m1", "started_at": "2026-08-01T00:00:00Z",
                             "map": {"name": "Sunset"}, "region": "na",
                             "queue": {"id": "competitive", "mode_type": "Standard"},
                             "season": {"short": "e11a5"}, "is_completed": True},
                "teams": [{"team_id": "Red", "won": False, "rounds": {"won": 12}},
                          {"team_id": "Blue", "won": True, "rounds": {"won": 14}}],
                "rounds": [{"id": i, "ceremony": "CeremonyDefault",
                            "winning_team": "Blue", "stats": []} for i in range(20)],
                "kills": [],
                "players": [{"puuid": f"p{i}", "name": f"n{i}", "tag": "NA1",
                             "team_id": "Blue" if i < 5 else "Red",
                             "agent": {"name": "Jett"}, "tier": {"id": 13},
                             "party_id": None, "account_level": 50,
                             "stats": {"score": 4000, "kills": 15, "deaths": 14,
                                       "assists": 5, "headshots": 10,
                                       "bodyshots": 20, "legshots": 2,
                                       "damage": {"dealt": 2800, "received": 2900}}}
                            for i in range(10)]}]}

    frontier.enqueue_many(conn, [("seed", 13)])
    c = Crawler(conn, RealShapeClient(), TokenBucket(6000), "na", "pc")
    c.run(minutes=0.02, verbose=False)

    assert conn.execute("SELECT COUNT(*) n FROM matches").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) n FROM match_players").fetchone()["n"] == 10


# --- adaptive pacing --------------------------------------------------

def test_limiter_learns_the_real_cost_of_a_request():
    """A size=10 matchlist bills ~2 units, not 1. Pacing to the raw unit
    ceiling overshoots by 2x and ends in a stall."""
    b = TokenBucket(30)
    b.observe({"x-ratelimit-limit": "30", "x-ratelimit-remaining": "28",
               "x-ratelimit-reset": "60"})
    b.observe({"x-ratelimit-limit": "30", "x-ratelimit-remaining": "26",
               "x-ratelimit-reset": "60"})
    b.observe({"x-ratelimit-limit": "30", "x-ratelimit-remaining": "24",
               "x-ratelimit-reset": "60"})
    assert 1.5 < b.cost_per_request < 2.5, b.cost_per_request


def test_limiter_keeps_a_reserve_instead_of_sprinting_to_zero():
    """Hitting zero triggers a blind full-window backoff. 53 of those cost
    82% of one hour's wall clock."""
    b = TokenBucket(30)
    b.observe({"x-ratelimit-limit": "30", "x-ratelimit-remaining": "3",
               "x-ratelimit-reset": "60"})
    # reserve is max(2, 15% of 30) = 4.5, so 3 remaining leaves nothing usable
    assert b._tokens == 0.0
    assert b.acquire() > 0.0


def test_pacing_adapts_downward_when_requests_cost_more():
    b = TokenBucket(30)
    fast = b.rate
    for r in (25, 20, 15, 10):
        b.observe({"x-ratelimit-limit": "30", "x-ratelimit-remaining": str(r),
                   "x-ratelimit-reset": "60"})
    assert b.cost_per_request > 3.0
    assert b.rate < fast, "expensive requests must slow the pace"
