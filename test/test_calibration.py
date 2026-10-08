"""What the model's numbers have meant, passed on only for the model they
describe."""

from __future__ import annotations

import json

import pytest

from valwr.model import calibration as C

RESULTS = {
    "shipped": "logistic regression", "n_test": 10_304,
    "reliability": [[0.3969, 0.4393, 1031], [0.5071, 0.5233, 1030],
                    [0.6040, 0.6106, 1030]],
    "coverage_strata": [{"name": "5-6", "n": 1469, "accuracy": 0.5228},
                        {"name": "9-10", "n": 5623, "accuracy": 0.5469}],
}
BUNDLE = {"best": "logistic regression",
          "metrics": {"logistic regression": {"n": 10_304}}}


@pytest.fixture
def results(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(RESULTS), encoding="utf-8")
    return path


def test_the_record_is_the_reliability_table_and_the_strata(results):
    got = C.track_record(BUNDLE, results)
    assert got["n_test"] == 10_304
    assert got["bins"][-1] == [0.604, 0.6106, 1030]
    assert all(isinstance(b[2], int) for b in got["bins"])
    assert got["strata"] == [{"known": "5-6", "n": 1469, "accuracy": 0.5228},
                             {"known": "9-10", "n": 5623, "accuracy": 0.5469}]


@pytest.mark.parametrize("bundle", [
    {"best": "lightgbm", "metrics": {"lightgbm": {"n": 10_304}}},
    {"best": "logistic regression",
     "metrics": {"logistic regression": {"n": 8_301}}},
    {},
])
def test_another_models_record_is_not_passed_on(results, bundle):
    """Retrained since results.json was written, or a different model: its
    record would describe numbers this model never produced."""
    assert C.track_record(bundle, results) is None


def test_no_results_or_unreadable_results_mean_no_record(tmp_path):
    assert C.track_record(BUNDLE, tmp_path / "absent.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert C.track_record(BUNDLE, bad) is None
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({**RESULTS, "reliability": []}), encoding="utf-8")
    assert C.track_record(BUNDLE, empty) is None


def test_the_committed_results_describe_the_committed_manifest():
    """The demo shows the shipped record; it must be the manifest's model."""
    from valwr.model import manifest
    rec = C.published_record()
    model = manifest.load()["artifacts"]["model.joblib"]
    assert rec is not None and rec["n_test"] == model["test"]["n"]
    assert len(rec["bins"]) == 10 and rec["strata"]


def test_the_live_payload_carries_the_contexts_record():
    from valwr.dash import server as S

    class Ctx:
        index = role_index = None
        record = {"bins": [[0.5, 0.5, 10]], "strata": []}
    got = S._match_payload(Ctx(), {"match_id": "m1"}, None)
    assert got["record"] is Ctx.record


def test_the_demo_payload_carries_the_shipped_record():
    from valwr.dash import server as S
    got = S._demo_payload()
    assert got["record"] == C.published_record()


def test_a_context_is_opened_with_its_models_record(monkeypatch, tmp_path):
    """open_context reads the record for the bundle it loaded."""
    import joblib

    from valwr.live import state as st
    models = tmp_path / "models"
    models.mkdir()
    joblib.dump(BUNDLE, models / "model.joblib")
    monkeypatch.setenv("MODELS_PATH", str(models))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "v.db"))
    monkeypatch.setattr(st.lockfile, "game_is_running", lambda: True)
    monkeypatch.setattr(st.S, "build", lambda: object())
    seen = []
    monkeypatch.setattr(C, "track_record",
                        lambda bundle, path=None: seen.append(bundle) or "REC")
    ctx = st.open_context(no_fetch=True)
    assert ctx.record == "REC" and seen[0]["best"] == "logistic regression"
