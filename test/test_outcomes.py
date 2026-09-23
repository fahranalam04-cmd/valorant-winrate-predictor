"""Recording what the live view predicted, and scoring it against the result.

The held-out test set measures the model on matches collected alongside its
training data. This measures it on the matches you played, which is the only
figure a user can check -- so the recording has to be honest about *when* the
prediction was made, and the scoring has to be honest about how little a
handful of matches can say.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

from valwr.live import outcomes, review
from valwr.store import schema

ROOT = pathlib.Path(__file__).resolve().parent


def _leakage():
    """The match-payload builder the leakage tests already own."""
    spec = importlib.util.spec_from_file_location(
        "test_leakage", ROOT / "test_leakage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ingest(conn, *matches):
    """Store finished matches the way the crawler does."""
    _leakage().ingest(conn, *matches)


@pytest.fixture
def conn(tmp_path):
    c = schema.connect(tmp_path / "t.db")
    schema.create_all(c)
    return c


def state(match_id="m1", phase="coregame", own="Blue", p=0.62, mode="Bomb",
          scores=(90, 70, 50, 30, 10), career=None):
    """A dashboard state in the shape poll_once returns."""
    players = []
    for i, puuid in enumerate([f"b{i}" for i in range(5)] + [f"r{i}" for i in range(5)]):
        players.append({
            "puuid": puuid, "name": f"{puuid}#NA1", "team": "Blue" if i < 5 else "Red",
            "agent": "Jett", "is_you": puuid == "b0",
            "score": scores[i] if i < 5 else None,
            "reason": "wins duels", "recent": {"kd": 1.1},
            "career": dict(career) if career else None,
        })
    return {
        "match_id": match_id, "phase": phase, "is_custom": False,
        "standard_mode": mode == "Bomb", "map": "Sunset", "mode": mode,
        "as_of": 1000, "own_team": own, "enemy_team": "Red",
        "coverage": 8, "confidence": "moderate", "model": "logistic regression",
        "warnings": [], "parties": [], "players": players,
        "prediction": {"own_probability": p, "win_probability": p,
                       "factors": []},
    }


# --- recording ---------------------------------------------------------

def test_the_first_sight_of_a_match_is_what_gets_scored(conn):
    """A prediction made at the loading screen is the claim being tested.

    Later polls know more -- players resolve, coverage rises -- so letting them
    overwrite would score a prediction nobody saw.
    """
    assert outcomes.record(conn, state(p=0.62), now=100) is True
    assert outcomes.record(conn, state(p=0.71), now=200) is False
    row = conn.execute("SELECT * FROM live_predictions").fetchone()
    assert row["own_probability"] == pytest.approx(0.62)
    assert row["made_at"] == 100


def test_agent_select_is_upgraded_once_the_match_starts(conn):
    """In agent select the enemy team is hidden, so that prediction is made on
    half a lobby. The first complete one replaces it."""
    outcomes.record(conn, state(phase="pregame", p=0.55), now=100)
    assert outcomes.record(conn, state(phase="coregame", p=0.62), now=120) is True
    row = conn.execute("SELECT * FROM live_predictions").fetchone()
    assert row["phase"] == "coregame"
    assert row["own_probability"] == pytest.approx(0.62)
    assert row["made_at"] == 100, "the clock still starts at the first sight"


def test_a_state_with_no_prediction_is_not_recorded(conn):
    thin = state()
    thin["prediction"] = None
    assert outcomes.record(conn, thin) is False
    assert conn.execute("SELECT COUNT(*) FROM live_predictions").fetchone()[0] == 0


def test_a_match_still_being_played_is_not_asked_about(conn):
    """The clock starts when the match LOADS, not when it ends.

    This is the bug that made the whole feature look broken: the first version
    asked three minutes in and gave up after six tries a minute apart, so every
    attempt landed while the match was still being played. Every recorded match
    failed to settle, no comparison ever appeared, and the log filled with 404s
    that read like a dead endpoint.
    """
    outcomes.record(conn, state(), now=1000)
    assert outcomes.pending(conn, now=1000) == []
    # Ten minutes in: still playing. A competitive match runs 25-45 minutes.
    assert outcomes.pending(conn, now=1000 + 10 * 60) == []
    assert outcomes.pending(conn, now=1000 + 21 * 60) == ["m1"]


def test_the_retries_spread_out_instead_of_burning_down(conn):
    """Each attempt waits longer, so a slow match is still chased hours later
    rather than having its budget spent in the first ten minutes."""
    outcomes.record(conn, state(), now=1000)
    conn.execute("UPDATE live_predictions SET attempts = 5")
    assert outcomes.pending(conn, now=1000 + 60 * 60) == [], "too soon for a 6th"
    assert outcomes.pending(conn, now=1000 + 6 * 3600) == ["m1"]
    # The last attempt reaches beyond a day, so an overnight publish lands.
    assert outcomes.due_after(outcomes.MAX_ATTEMPTS - 1) > 24 * 3600

    conn.execute("UPDATE live_predictions SET attempts = ?",
                 (outcomes.MAX_ATTEMPTS,))
    assert outcomes.pending(conn, now=10 ** 9) == [],         "a match that never lands stops being asked about"


# --- settling ----------------------------------------------------------

class FakeAPI:
    """Stands in for HenrikDev: hands back one finished match, or nothing."""

    def __init__(self, payload=None):
        self.payload = payload
        self.calls = 0

    def match(self, region, match_id):
        self.calls += 1
        if self.payload is None:
            raise RuntimeError("404 on /valorant/v4/match")
        return {"data": self.payload}


def finished(winner="Blue", best="b3"):
    """A completed match where `best` had the highest combat score."""
    payload = _leakage().make_match("m1", "2026-08-01T00:00:00Z", winner=winner)
    for p in payload["players"]:
        p["stats"]["score"] = 6000 if p["puuid"] == best else 3000
    return payload


def test_a_won_match_is_scored_against_what_was_predicted(conn):
    outcomes.record(conn, state(own="Blue", p=0.62), now=1000)
    api = FakeAPI(finished(winner="Blue"))
    assert outcomes.settle(conn, api, "na", "m1", now=2000) == "settled"

    row = conn.execute("SELECT * FROM live_predictions").fetchone()
    assert row["winner"] == "Blue" and row["own_won"] == 1
    assert row["correct"] == 1
    assert row["brier"] == pytest.approx((0.62 - 1) ** 2)
    assert row["settled_at"] == 2000


def test_a_lost_match_marks_the_prediction_wrong(conn):
    outcomes.record(conn, state(own="Blue", p=0.62), now=1000)
    outcomes.settle(conn, FakeAPI(finished(winner="Red")), "na", "m1", now=2000)
    row = conn.execute("SELECT * FROM live_predictions").fetchone()
    assert row["own_won"] == 0 and row["correct"] == 0
    assert row["brier"] == pytest.approx(0.62 ** 2)


def test_the_top_pick_is_checked_against_who_actually_played_best(conn):
    # b0 is the highest-scored teammate; b3 actually had the best game.
    outcomes.record(conn, state(scores=(90, 70, 50, 30, 10)), now=1000)
    outcomes.settle(conn, FakeAPI(finished(best="b3")), "na", "m1", now=2000)
    assert conn.execute("SELECT top_pick_hit FROM live_predictions").fetchone()[0] == 0

    conn.execute("DELETE FROM live_predictions")
    outcomes.record(conn, state(scores=(10, 20, 30, 99, 5)), now=1000)
    outcomes.settle(conn, FakeAPI(finished(best="b3")), "na", "m1", now=2000)
    assert conn.execute("SELECT top_pick_hit FROM live_predictions").fetchone()[0] == 1


def test_a_match_the_api_does_not_have_yet_is_left_open(conn):
    outcomes.record(conn, state(), now=1000)
    assert outcomes.settle(conn, FakeAPI(None), "na", "m1", now=2000) == "error"
    row = conn.execute("SELECT * FROM live_predictions").fetchone()
    assert row["settled_at"] is None
    assert row["attempts"] == 1
    assert "RuntimeError" in row["last_error"]


def test_settle_pending_walks_everything_due(conn):
    outcomes.record(conn, state(match_id="m1"), now=1000)
    api = FakeAPI(finished())
    out = outcomes.settle_pending(conn, api, "na", now=1000 + 10_000)
    assert out["settled"] == 1 and api.calls == 1
    # Nothing left to do, so no further calls are spent.
    assert outcomes.settle_pending(conn, api, "na", now=1000 + 20_000)["settled"] == 0
    assert api.calls == 1


def test_no_client_means_no_calls_and_no_crash(conn):
    """Without an API client there is nothing to fetch, but nothing breaks --
    the match is simply still waiting."""
    outcomes.record(conn, state(), now=1000)
    assert outcomes.settle_pending(conn, None, "na", now=99_999) == {
        "settled": 0, "waiting": 1, "error": 0}


def test_a_match_the_crawler_already_holds_settles_without_an_api_call(conn):
    """The crawler collects matches through other players' histories.

    When it has already stored this one, asking the API again spends a
    rate-limited call to learn what is in front of us -- and the settle has to
    work even when no client exists at all.
    """
    outcomes.record(conn, state(), now=1000)
    _ingest(conn, finished(winner="Blue"))
    assert outcomes.settle_pending(conn, None, "na", now=99_999) == {
        "settled": 1, "waiting": 0, "error": 0}
    row = conn.execute("SELECT * FROM live_predictions").fetchone()
    assert row["settled_at"] is not None
    assert row["own_won"] == 1 and row["correct"] == 1


def test_the_api_is_not_called_when_the_match_is_already_stored(conn):
    outcomes.record(conn, state(), now=1000)
    _ingest(conn, finished(winner="Blue"))
    api = FakeAPI(finished())
    assert outcomes.settle(conn, api, "na", "m1", now=99_999) == "settled"
    assert api.calls == 0, "it already had the match"


# --- reading it back ---------------------------------------------------

def test_the_comparison_pairs_prediction_with_performance(conn):
    outcomes.record(conn, state(scores=(90, 70, 50, 30, 10)), now=1000)
    outcomes.settle(conn, FakeAPI(finished(winner="Blue", best="b3")), "na",
                    "m1", now=2000)
    got = review.compare(conn, "m1")

    assert got["settled"] is True
    assert got["actual"]["winner"] == "Blue"
    assert got["actual"]["score"] == "14-12"
    assert got["correct"] == 1
    assert got["top_pick"]["hit"] == 0
    assert got["top_pick"]["picked"] == "b0#NA1"
    assert got["top_pick"]["actually_best"] == "b3#NA1"

    me = next(p for p in got["players"] if p["puuid"] == "b0")
    assert me["is_you"] and me["predicted_rank"] == 1
    assert me["acs"] is not None and me["kd"] is not None
    best = next(p for p in got["players"] if p["puuid"] == "b3")
    assert best["actual_rank"] == 1, "highest combat score in the match"


def test_every_player_is_compared_against_their_own_history(conn):
    """The per-player block reads a match against the player, not the lobby.

    The 0-100 already ranks players against each other. What the scoreboard can
    add is whether someone played like themselves, so each figure the page held
    beforehand has to survive into the comparison beside what they did.
    """
    outcomes.record(conn, state(career={"acs": 200.0, "kd": 1.0,
                                        "headshot_rate": 0.20, "games": 40}),
                    now=1000)
    outcomes.settle(conn, FakeAPI(finished(best="b3")), "na", "m1", now=2000)
    got = review.compare(conn, "m1")

    top = next(p for p in got["players"] if p["puuid"] == "b3")
    assert top["career_acs"] == 200.0 and top["career_games"] == 40
    assert top["career_kd"] == 1.0 and top["career_hs"] == 0.20
    assert top["recent_kd"] == 1.1
    # 6000 combat score over the fixture's rounds, against a career 200.
    assert top["acs"] is not None
    assert top["acs_delta"] == pytest.approx(top["acs"] - 200.0, abs=0.05)
    assert top["versus_usual"] == "well above their usual"
    assert top["place_delta"] == top["predicted_rank"] - top["actual_rank"]

    # Everyone else scored 3000, which is well under their career line.
    other = next(p for p in got["players"] if p["puuid"] == "b1")
    assert other["acs_delta"] < 0
    assert other["versus_usual"] == "well below their usual"


def test_a_player_with_no_history_is_compared_against_nothing(conn):
    """Half the lobby is usually unknown. The block must render for them and
    say nothing rather than invent a baseline of zero."""
    outcomes.record(conn, state(career=None), now=1000)
    outcomes.settle(conn, FakeAPI(finished()), "na", "m1", now=2000)
    got = review.compare(conn, "m1")
    for player in got["players"]:
        assert player["career_acs"] is None
        assert player["acs_delta"] is None and player["kd_delta"] is None
        assert player["hs_delta"] is None
        assert player["versus_usual"] is None


def test_how_a_match_reads_against_a_career_average():
    """The bands are wide on purpose -- ACS swings hugely match to match."""
    assert review.against_usual(240, 200) == "well above their usual"
    assert review.against_usual(212, 200) == "above their usual"
    assert review.against_usual(200, 200) == "about their usual"
    assert review.against_usual(188, 200) == "below their usual"
    assert review.against_usual(150, 200) == "well below their usual"
    # Nothing to compare against is not the same as average.
    assert review.against_usual(240, None) is None
    assert review.against_usual(None, 200) is None
    assert review.against_usual(240, 0) is None


def test_an_unsettled_match_says_it_is_waiting(conn):
    outcomes.record(conn, state(), now=1000)
    got = review.compare(conn, "m1")
    assert got["settled"] is False and got["actual"] is None
    assert "published" in got["waiting"]
    assert review.compare(conn, "never-seen") is None


def test_the_scorecard_counts_only_matches_the_model_was_built_for(conn):
    outcomes.record(conn, state(match_id="m1", mode="Bomb"), now=1000)
    outcomes.settle(conn, FakeAPI(finished()), "na", "m1", now=2000)
    outcomes.record(conn, state(match_id="m2", mode="Deathmatch"), now=1500)

    card = review.scorecard(conn)
    assert card["recorded"] == 2
    assert card["competitive"]["n"] == 1, "deathmatch is recorded, not scored"
    assert card["pending"] == 1
    assert [m["match_id"] for m in card["matches"]] == ["m2", "m1"], "newest first"


def test_the_scorecard_refuses_to_read_anything_into_two_matches(conn):
    for i, winner in enumerate(("Blue", "Blue")):
        outcomes.record(conn, state(match_id=f"m{i}"), now=1000 + i)
        payload = finished(winner=winner)
        payload["metadata"]["match_id"] = f"m{i}"
        outcomes.settle(conn, FakeAPI(payload), "na", f"m{i}", now=2000)

    card = review.scorecard(conn)
    assert card["competitive"]["n"] == 2
    joined = " ".join(card["insights"]).lower()
    assert "30 matches" in joined, "it must say how thin the evidence is"
    assert card["competitive"]["se"] > 0


def test_the_verdict_wording_matches_the_page():
    assert review.verdict(0.5) == "too close to call"
    assert review.verdict(0.56) == "your side favoured"
    assert review.verdict(0.30) == "the enemy clearly favoured"
    assert review.verdict(None) == "no prediction"


def test_a_recorded_state_is_json_and_replays_the_card(conn):
    """The whole state is stored so a pinned tab can rebuild the page exactly
    as it looked, months later, without the model or the client."""
    outcomes.record(conn, state(), now=1000)
    stored = json.loads(
        conn.execute("SELECT state_json FROM live_predictions").fetchone()[0])
    assert stored["players"][0]["name"] == "b0#NA1"
    assert stored["prediction"]["own_probability"] == pytest.approx(0.62)


# --- how much it got right, and what that suggests ---------------------

def _improve():
    spec = importlib.util.spec_from_file_location(
        "improve", ROOT.parent / "tools" / "improve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_match_summary_counts_what_was_right(conn):
    outcomes.record(conn, state(scores=(90, 70, 50, 30, 10)), now=1000)
    outcomes.settle(conn, FakeAPI(finished(winner="Blue", best="b3")), "na",
                    "m1", now=2000)
    got = review.compare(conn, "m1")["summary"]

    assert got["winner_called"] == 1
    assert got["top_pick_hit"] == 0
    assert got["rated"] == 5, "only the five with a predicted score"
    assert 0 <= got["within_one"] <= got["rated"]
    assert got["order"] is not None, "a rank correlation over those five"


def test_the_ordering_is_pooled_across_matches(conn):
    """One lobby is five players and a lot of luck; the pool is the measure."""
    for i in range(3):
        payload = finished(winner="Blue", best="b3")
        payload["metadata"]["match_id"] = f"m{i}"
        outcomes.record(conn, state(match_id=f"m{i}"), now=1000 + i)
        outcomes.settle(conn, FakeAPI(payload), "na", f"m{i}", now=2000)

    order = review.scorecard(conn)["order"]
    assert order["players"] == 15, "five rated players in each of three matches"
    assert 0 <= order["within_one_rate"] <= 1
    assert order["order"] is not None


def test_the_report_says_which_matches_can_still_judge_the_model(conn):
    """A match inside the training window cannot measure the model any more."""
    improve = _improve()
    outcomes.record(conn, state(), now=1000)
    outcomes.settle(conn, FakeAPI(finished()), "na", "m1", now=2000)
    rows = list(conn.execute("SELECT * FROM live_predictions"))

    # One match is too little to draw a split from, so it says nothing rather
    # than inventing a boundary.
    assert improve.fairness(conn, rows) == {}

    # With a spread of matches in the store it places each one.
    leak = _leakage()
    for i in range(40):
        leak.ingest(conn, leak.make_match(
            f"bulk{i}", f"2026-07-{i % 28 + 1:02d}T00:00:00Z"))
    fair = improve.fairness(conn, rows)
    assert sum(fair.values()) == 1
    assert set(fair) == {"train", "val", "test"}


def test_the_report_runs_and_writes_its_findings(conn, capsys):
    improve = _improve()
    outcomes.record(conn, state(), now=1000)
    outcomes.settle(conn, FakeAPI(finished()), "na", "m1", now=2000)

    card = improve.report(conn)
    printed = capsys.readouterr().out
    assert "WHAT YOUR OWN MATCHES SAY" in printed
    assert "1 recorded" in printed
    assert card["competitive"]["n"] == 1
    assert "fairness" in card


def test_the_suggestions_lead_with_the_thinness_of_the_evidence():
    improve = _improve()
    thin = {"competitive": {"n": 4, "se": 0.25, "actual": 0.5, "predicted": 0.5},
            "by_coverage": [], "order": {}}
    lines = improve.suggestions(thin)
    assert "30 matches" in lines[0]
    assert any("--retrain" in line for line in lines)


def test_a_thin_lobby_points_at_the_crawler():
    improve = _improve()
    card = {"competitive": {"n": 40, "se": 0.08, "actual": 0.5, "predicted": 0.5},
            "by_coverage": [{"label": "5/10 known", "n": 20, "accuracy": 0.45,
                             "predicted": 0.5, "actual": 0.45, "se": 0.1},
                            {"label": "9/10 known", "n": 20, "accuracy": 0.65,
                             "predicted": 0.5, "actual": 0.65, "se": 0.1}],
            "order": {}}
    assert any("crawler" in line for line in improve.suggestions(card))


def test_the_recorded_pick_is_the_one_shown_at_the_top(conn):
    """What gets scored afterwards has to be what the player saw first.

    The scoreboard orders by the cross-role figure while displaying the
    within-role percentile, so a pick chosen by the percentile would sometimes
    be the second name on the list.
    """
    s = state(scores=(60, 90, 50, 30, 10))
    for p, raw in zip(s["players"], [0.9, 0.2, 0.1, 0.0, -0.1]):
        p["raw"] = raw
    assert outcomes.top_pick(s, "Blue") == "b0", "highest raw, not highest 0-100"

    # With no raw at all -- a match recorded before the change -- the 0-100 is
    # what it was ordered by, and stays.
    for p in s["players"]:
        p.pop("raw", None)
    assert outcomes.top_pick(s, "Blue") == "b1"


def test_the_post_match_ranking_matches_the_order_that_was_shown(conn):
    s = state(scores=(60, 90, 50, 30, 10))
    for p, raw in zip(s["players"], [0.9, 0.2, 0.1, 0.0, -0.1]):
        p["raw"] = raw
    outcomes.record(conn, s, now=1000)
    outcomes.settle(conn, FakeAPI(finished()), "na", "m1", now=2000)
    got = review.compare(conn, "m1")
    ranked = {p["puuid"]: p["predicted_rank"] for p in got["players"]}
    assert ranked["b0"] == 1 and ranked["b1"] == 2


# --- getting back to a match after it ends ------------------------------

def test_recorded_matches_are_listed_newest_first(conn):
    """The live page follows the current match, so when one ends the game just
    played leaves the screen. This is the list that makes it reachable."""
    outcomes.record(conn, state(match_id="old"), now=1000)
    outcomes.record(conn, state(match_id="m1"), now=5000)
    _ingest(conn, finished(winner="Blue"))
    outcomes.settle(conn, FakeAPI(finished()), "na", "m1", now=9000)

    got = review.recent(conn)
    assert [r["match_id"] for r in got] == ["m1", "old"]
    assert got[0]["settled"] is True and got[0]["own_won"] == 1
    assert got[0]["score"] == "14-12", "from the player's own side"
    assert got[1]["settled"] is False, "still waiting is not an error"
    assert got[1]["score"] is None


def test_the_recent_list_is_capped(conn):
    for i in range(9):
        outcomes.record(conn, state(match_id=f"m{i}"), now=1000 + i)
    assert len(review.recent(conn, limit=4)) == 4
    assert len(review.recent(conn)) == 6


def test_an_empty_record_lists_nothing_rather_than_failing(conn):
    assert review.recent(conn) == []


def test_predictions_stranded_by_the_old_schedule_are_recoverable(conn):
    """Anyone who ran the version that gave up after ten minutes has rows with
    a spent budget and no result. The matches are long finished; the attempts
    were wasted on asking too early."""
    outcomes.record(conn, state(match_id="stranded"), now=1000)
    conn.execute("UPDATE live_predictions SET attempts = 6, last_error = '404'")
    conn.commit()
    assert outcomes.pending(conn, now=1000 + 3600) == [], "still backing off"

    freed = outcomes.unstick(conn, now=1000 + 3600)
    conn.commit()
    assert freed == 1
    assert outcomes.pending(conn, now=1000 + 3600) == ["stranded"]


def test_unstick_leaves_a_match_that_is_still_being_played_alone(conn):
    outcomes.record(conn, state(match_id="live-now"), now=1000)
    conn.execute("UPDATE live_predictions SET attempts = 1")
    conn.commit()
    assert outcomes.unstick(conn, now=1000 + 60) == 0


def test_unstick_does_not_touch_settled_matches(conn):
    outcomes.record(conn, state(), now=1000)
    _ingest(conn, finished())
    outcomes.settle(conn, FakeAPI(finished()), "na", "m1", now=9000)
    assert outcomes.unstick(conn, now=10 ** 9) == 0


def test_the_client_leaving_a_match_asks_for_the_result_at_once(conn):
    """The backoff is anchored on the loading screen, so waiting for it means
    sitting through the rest of the game before the first attempt."""
    outcomes.record(conn, state(), now=1000)
    conn.execute("UPDATE live_predictions SET attempts = 3")
    conn.commit()
    _ingest(conn, finished(winner="Blue"))

    assert outcomes.match_ended(conn, None, "na", "m1", now=3000) == "settled"
    row = conn.execute("SELECT * FROM live_predictions").fetchone()
    assert row["settled_at"] == 3000 and row["own_won"] == 1


def test_a_match_that_ended_but_is_not_published_keeps_its_place(conn):
    """One immediate attempt, then the ordinary schedule -- from now rather
    than from the loading screen."""
    outcomes.record(conn, state(), now=1000)
    conn.execute("UPDATE live_predictions SET attempts = 9")
    conn.commit()
    assert outcomes.match_ended(conn, FakeAPI(None), "na", "m1", now=3000) == "error"
    row = conn.execute("SELECT * FROM live_predictions").fetchone()
    assert row["attempts"] == 1, "the spent budget is given back, then one used"
    assert row["settled_at"] is None


# --- what "played best" means, after 13.06 -----------------------------

def test_the_best_game_is_match_impact_not_combat_score(conn):
    """Patch 13.06 removed combat score from the game, and ranking by it also
    disagreed with how the rest of the project measures the same question."""
    outcomes.record(conn, state(), now=1000)
    # b3 is given the highest combat score, b1 the better all-round match.
    payload = finished(best="b3")
    for p in payload["players"]:
        if p["puuid"] == "b1":
            p["stats"].update(kills=30, deaths=2, assists=10,
                              damage={"dealt": 6000, "received": 1200})
    _ingest(conn, payload)
    best = outcomes.actual_best(conn, "m1", "Blue")
    assert best == "b1", "the better match wins, not the bigger combat score"


def test_rescoring_replaces_a_verdict_recorded_under_the_old_rule(conn):
    """Stored rows were judged by combat score. Left alone, the scorecard
    would average two different definitions of the same column."""
    outcomes.record(conn, state(scores=(90, 70, 50, 30, 10)), now=1000)
    outcomes.settle(conn, FakeAPI(finished(best="b3")), "na", "m1", now=2000)
    conn.execute("UPDATE live_predictions SET top_pick_hit = 1")
    conn.commit()

    assert outcomes.rescore(conn) == 1
    got = conn.execute("SELECT top_pick_hit FROM live_predictions").fetchone()[0]
    assert got == 0, "the pick did not have the best match after all"
    # Idempotent: a second pass changes nothing, so it can run every launch.
    assert outcomes.rescore(conn) == 0
