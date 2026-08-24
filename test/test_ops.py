"""Tests for the operational modules: seeding and the watchdog.

Both had real logic and zero coverage. `seed_leaderboard` in particular had
never been executed at all -- every crawl so far used `--seed self` -- so its
response parsing was entirely unverified against the shape the API returns.

The watchdog is what keeps an unattended crawl alive after two silent deaths,
so its decision logic is worth pinning down: restarting a healthy crawler is
worse than waiting, and starting a second one is worse than both.
"""

from __future__ import annotations

import time

import pytest

from valwr.collect import frontier, seed, watchdog
from valwr.store import schema


@pytest.fixture
def conn(tmp_path):
    c = schema.connect(tmp_path / "t.db")
    schema.create_all(c)
    return c


class FakeLeaderboardClient:
    """Mirrors the real v3 leaderboard shape: data -> {players: [...]}."""

    def __init__(self, players):
        self._players = players
        self.calls = 0

    def leaderboard(self, region, platform, **kw):
        self.calls += 1
        return {"status": 200,
                "data": {"updated_at": "now", "thresholds": [],
                         "players": self._players}}

    def account(self, name, tag):
        return {"data": {"puuid": "me-puuid", "name": name, "tag": tag}}


def player(puuid, tier=27, anon=False, banned=False):
    return {"puuid": puuid, "name": "n", "tag": "t", "tier": tier,
            "rr": 100, "wins": 10, "leaderboard_rank": 1,
            "is_anonymized": anon, "is_banned": banned}


# --- seeding ----------------------------------------------------------

def test_leaderboard_seeding_parses_the_real_response_shape(conn):
    """Verified against the live endpoint: data is a dict containing
    `players`, each with puuid / tier / is_anonymized / is_banned."""
    client = FakeLeaderboardClient([player(f"p{i}") for i in range(5)])
    n = seed.seed_leaderboard(conn, client, "na", "pc")
    assert n == 5
    assert frontier.counts_by_state(conn)["pending"] == 5


def test_anonymised_and_banned_players_are_skipped(conn):
    """Anonymised entries carry no usable puuid and would otherwise fill the
    frontier with rows that can never be fetched."""
    client = FakeLeaderboardClient([
        player("good1"), player("anon", anon=True),
        player("banned", banned=True), player("good2"),
    ])
    assert seed.seed_leaderboard(conn, client, "na", "pc") == 2
    kept = {r["puuid"] for r in conn.execute("SELECT puuid FROM frontier")}
    assert kept == {"good1", "good2"}


def test_seeding_is_idempotent(conn):
    client = FakeLeaderboardClient([player(f"p{i}") for i in range(4)])
    assert seed.seed_leaderboard(conn, client, "na", "pc") == 4
    assert seed.seed_leaderboard(conn, client, "na", "pc") == 0, "re-seeding adds nothing"


def test_seed_limit_is_respected(conn):
    client = FakeLeaderboardClient([player(f"p{i}") for i in range(50)])
    assert seed.seed_leaderboard(conn, client, "na", "pc", limit=10) == 10


def test_leaderboard_tier_maps_to_a_rank_band(conn):
    """Leaderboard `tier` is a bare int; match data nests it under {id,name}.
    Mixing the two up would silently band everyone as Unranked."""
    client = FakeLeaderboardClient([player("radiant", tier=27)])
    seed.seed_leaderboard(conn, client, "na", "pc")
    band = conn.execute("SELECT tier_band FROM frontier").fetchone()["tier_band"]
    assert band == frontier.band_of(27) == 9


def test_seed_self_enqueues_and_returns_the_puuid(conn):
    client = FakeLeaderboardClient([])
    puuid = seed.seed_self(conn, client, "Name", "TAG")
    assert puuid == "me-puuid"
    assert frontier.counts_by_state(conn)["pending"] == 1


def test_seed_self_fails_loudly_on_an_unresolvable_id(conn):
    class NoAccount(FakeLeaderboardClient):
        def account(self, name, tag):
            return {"data": {}}
    with pytest.raises(RuntimeError, match="could not resolve"):
        seed.seed_self(conn, NoAccount([]), "Nobody", "0000")


# --- watchdog ---------------------------------------------------------

def test_watchdog_stays_quiet_when_the_crawl_is_healthy(monkeypatch, capsys):
    """It runs on a timer. A watchdog that logs every ten minutes is a log
    nobody reads."""
    monkeypatch.setattr(watchdog, "seconds_since_last_fetch", lambda: 30.0)
    monkeypatch.setattr(watchdog, "crawler_pids", lambda: [123])
    started = []
    monkeypatch.setattr(watchdog, "start_crawler", lambda h: started.append(h))

    watchdog.main(["--dry-run"])
    assert capsys.readouterr().out == ""
    assert not started


def test_watchdog_waits_rather_than_restarting_a_slow_crawler(monkeypatch):
    """At ~2.5 req/min a legitimate quota stall approaches a minute.
    Restarting a healthy crawler is worse than waiting one more cycle."""
    monkeypatch.setattr(watchdog, "seconds_since_last_fetch",
                        lambda: watchdog.STALE_SECONDS + 60)
    monkeypatch.setattr(watchdog, "crawler_pids", lambda: [123])
    started, killed = [], []
    monkeypatch.setattr(watchdog, "start_crawler", lambda h: started.append(h))
    monkeypatch.setattr(watchdog, "kill", lambda p: killed.append(p))

    watchdog.main([])
    assert not started and not killed


def test_watchdog_restarts_when_nothing_is_running(monkeypatch):
    monkeypatch.setattr(watchdog, "seconds_since_last_fetch",
                        lambda: watchdog.STALE_SECONDS + 60)
    monkeypatch.setattr(watchdog, "crawler_pids", lambda: [])
    started = []
    monkeypatch.setattr(watchdog, "start_crawler", lambda h: started.append(h))

    watchdog.main(["--hours", "12"])
    assert started == [12.0]


def test_watchdog_kills_a_hung_process_before_replacing_it(monkeypatch):
    """A process holding a PID while fetching nothing is as dead as a missing
    one -- and leaving it running would mean two crawlers sharing one quota."""
    monkeypatch.setattr(watchdog, "seconds_since_last_fetch",
                        lambda: watchdog.HUNG_SECONDS + 60)
    monkeypatch.setattr(watchdog, "crawler_pids", lambda: [111, 222])
    started, killed = [], []
    monkeypatch.setattr(watchdog, "start_crawler", lambda h: started.append(h))
    monkeypatch.setattr(watchdog, "kill", lambda p: killed.append(p))
    monkeypatch.setattr(time, "sleep", lambda s: None)

    watchdog.main([])
    assert killed == [111, 222], "every survivor must go before a restart"
    assert len(started) == 1


def test_watchdog_does_nothing_before_the_first_fetch(monkeypatch):
    """An empty database is not a stalled crawl."""
    monkeypatch.setattr(watchdog, "seconds_since_last_fetch", lambda: None)
    monkeypatch.setattr(watchdog, "crawler_pids", lambda: [])
    started = []
    monkeypatch.setattr(watchdog, "start_crawler", lambda h: started.append(h))

    watchdog.main([])
    assert not started


def test_watchdog_thresholds_are_ordered_sensibly():
    assert watchdog.STALE_SECONDS < watchdog.HUNG_SECONDS
    assert watchdog.STALE_SECONDS > 300, "must outlast a normal quota stall"
