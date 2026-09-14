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


@pytest.fixture
def conn(tmp_path):
    c = schema.connect(tmp_path / "t.db")
    schema.create_all(c)
    return c


def state(match_id="m1", phase="coregame", own="Blue", p=0.62, mode="Bomb",
          scores=(90, 70, 50, 30, 10)):
    """A dashboard state in the shape poll_once returns."""
    players = []
    for i, puuid in enumerate([f"b{i}" for i in range(5)] + [f"r{i}" for i in range(5)]):
        players.append({
            "puuid": puuid, "name": f"{puuid}#NA1", "team": "Blue" if i < 5 else "Red",
            "agent": "Jett", "is_you": puuid == "b0",
            "score": scores[i] if i < 5 else None,
            "reason": "wins duels", "career": None, "recent": None,
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


def test_only_matches_old_enough_are_chased_for_a_result(conn):
    outcomes.record(conn, state(), now=1000)
    assert outcomes.pending(conn, now=1000) == []
    assert outcomes.pending(conn, now=1000 + outcomes.SETTLE_AFTER_SECONDS + 1) == ["m1"]
    conn.execute("UPDATE live_predictions SET attempts = ?",
                 (outcomes.MAX_ATTEMPTS,))
    assert outcomes.pending(conn, now=10_000) == [], "a match that never lands stops being asked about"


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
    outcomes.record(conn, state(), now=1000)
    assert outcomes.settle_pending(conn, None, "na", now=99_999) == {
        "settled": 0, "waiting": 0, "error": 0}


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
