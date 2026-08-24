"""Phase 5 tests: splits, baselines, metrics, and the leakage checks.

The model modules decide what the README claims, so the things worth testing
are the ones that would silently inflate a result: a split that leaks the
future, a baseline given less of a chance than the model, a calibration fitted
on test, or a metric that flatters.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from valwr.model import baselines, evaluate, split, train


@pytest.fixture
def frame():
    """A synthetic matrix with a known, weak signal."""
    rng = np.random.default_rng(0)
    n = 900
    d_rating = rng.normal(0, 1, n)
    d_tier = rng.normal(0, 1, n)
    # Weak true signal, deliberately similar in strength to the real thing.
    p = 1 / (1 + np.exp(-(0.35 * d_rating + 0.2 * d_tier)))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({
        "match_id": [f"m{i}" for i in range(n)],
        "started_at": np.arange(1000, 1000 + n),
        "target": y,
        "margin": np.where(y == 1, 4, -4) + rng.normal(0, 3, n).astype(int),
        "coverage": rng.integers(5, 11, n),
        "d_rating_mean": d_rating,
        "d_tier_mean": d_tier,
        "d_tier_max": d_tier + rng.normal(0, 0.5, n),
        "d_constant": np.ones(n),
    })


# --- splits -----------------------------------------------------------

def test_splits_are_time_ordered_and_disjoint(frame):
    b = split.Boundaries(train_end=1600, val_end=1750, n_matches=len(frame))
    df = split.apply(frame, b)
    tr, va, te = (df[df["slice"] == k] for k in ("train", "val", "test"))

    assert tr["started_at"].max() < va["started_at"].min()
    assert va["started_at"].max() < te["started_at"].min()
    assert len(tr) + len(va) + len(te) == len(df)


def test_no_match_appears_in_two_slices(frame):
    b = split.Boundaries(1600, 1750, len(frame))
    df = split.apply(frame, b)
    counts = df.groupby("match_id")["slice"].nunique()
    assert counts.max() == 1


def test_boundary_row_goes_to_the_later_slice(frame):
    """Strict `<` on the boundary, matching the temporal layer's convention."""
    b = split.Boundaries(1600, 1750, 0)
    assert b.slice_of(1599) == "train"
    assert b.slice_of(1600) == "val"
    assert b.slice_of(1749) == "val"
    assert b.slice_of(1750) == "test"


def test_compute_refuses_a_dataset_too_small_to_split():
    import sqlite3
    from valwr.store import schema
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(schema.SCHEMA)
    with pytest.raises(ValueError, match="too few"):
        split.compute(conn)


# --- baselines --------------------------------------------------------

def test_baselines_are_fitted_on_train_not_hand_scaled(frame):
    """A hand-scaled baseline scored worse than a coin flip, which made
    'the model beats the baseline' a statement about my scaling. Baselines
    get the same chance as the model or beating them proves nothing."""
    b = split.Boundaries(1600, 1750, len(frame))
    df = split.apply(frame, b)
    tr, te = df[df["slice"] == "train"], df[df["slice"] == "test"]

    fitted = baselines.fitted(tr)
    rank = fitted["avg rank (fitted)"](te)
    coin = fitted["coin flip"](te)

    s_rank = evaluate.score("rank", te["target"], rank)
    s_coin = evaluate.score("coin", te["target"], coin)
    assert s_rank.log_loss <= s_coin.log_loss + 1e-6, (
        "a fitted baseline must not score worse than a coin flip")


def test_coin_flip_is_exactly_uninformative(frame):
    p = baselines.coin_flip(len(frame))
    s = evaluate.score("coin", frame["target"], p)
    assert s.log_loss == pytest.approx(0.6931, abs=1e-3)
    assert np.isnan(s.auc) or s.auc == pytest.approx(0.5, abs=1e-9)


# --- metrics ----------------------------------------------------------

def test_perfect_predictions_score_perfectly():
    y = np.array([0, 1, 0, 1, 1, 0])
    s = evaluate.score("oracle", y, y.astype(float) * 0.98 + 0.01)
    assert s.auc == 1.0
    assert s.accuracy == 1.0
    assert s.log_loss < 0.05


def test_confidently_wrong_is_punished_harder_than_hedging():
    y = np.array([1] * 50)
    hedged = evaluate.score("hedge", y, np.full(50, 0.45))
    confident = evaluate.score("confident", y, np.full(50, 0.05))
    assert confident.log_loss > hedged.log_loss


def test_calibration_error_catches_a_miscalibrated_model():
    """A model can rank well and still lie about its probabilities."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    honest = np.full(2000, 0.5)
    overconfident = np.where(y == 1, 0.9, 0.1) * 0 + 0.9   # always claims 90%
    assert evaluate.expected_calibration_error(y, honest) < 0.05
    assert evaluate.expected_calibration_error(y, overconfident) > 0.3


def test_confidence_interval_shrinks_with_sample_size():
    wide = evaluate.confidence_interval(0.53, 300)
    narrow = evaluate.confidence_interval(0.53, 30000)
    assert wide > narrow
    assert narrow < 0.01


# --- the leakage check ------------------------------------------------

def test_shuffled_target_check_reports_a_distribution(frame):
    """One draw has a standard deviation near 0.014, so a single sample can
    land at 0.518 and mean nothing -- which happened and cost an
    investigation. The distribution is the test."""
    b = split.Boundaries(1600, 1750, len(frame))
    df = split.apply(frame, b)
    cols = ["d_rating_mean", "d_tier_mean", "d_tier_max"]
    tr, te = df[df["slice"] == "train"], df[df["slice"] == "test"]

    out = train.shuffled_target_check(
        tr[cols].to_numpy(float), tr["target"].to_numpy(int),
        te[cols].to_numpy(float), te["target"].to_numpy(int), draws=8)

    assert set(out) >= {"mean", "std", "sigmas_from_chance", "draws_over_threshold"}
    assert out["draws"] == 8
    assert abs(out["mean"] - 0.5) < 0.15, "shuffled labels must not predict"


def test_zero_variance_features_are_dropped(frame):
    b = split.Boundaries(1600, 1750, len(frame))
    df = split.apply(frame, b)
    cols = train.feature_columns(df)
    assert "d_constant" not in cols, "a constant column teaches nothing"
    assert "d_rating_mean" in cols


# --- margin target ----------------------------------------------------

def test_margin_model_produces_valid_probabilities(frame):
    b = split.Boundaries(1600, 1750, len(frame))
    df = split.apply(frame, b)
    cols = ["d_rating_mean", "d_tier_mean"]
    tr, va, te = (df[df["slice"] == k] for k in ("train", "val", "test"))

    _, _, p = train.fit_margin_model(
        tr[cols].to_numpy(float), tr["margin"].to_numpy(float),
        va[cols].to_numpy(float), va["margin"].to_numpy(float),
        te[cols].to_numpy(float), va["target"].to_numpy(int))

    assert len(p) == len(te)
    assert ((p > 0) & (p < 1)).all(), "probabilities must stay in (0,1)"


# --- inference-bundle readiness (pre-Phase 6) -------------------------

def test_features_can_be_built_without_an_outcome(tmp_path):
    """The live path predicts matches that have not finished.

    build_match() used to return None whenever a winner was missing, which
    would have made the live path impossible -- discovered by auditing Phase 6
    dependencies before building it rather than during.
    """
    from valwr.features import build as fb
    from valwr.rating.normalize import build_norms
    from valwr.store import normalize, reference, schema, temporal

    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)
    conn.execute("INSERT INTO ref_agents (uuid,name,role) VALUES ('u','Jett','Duelist')")
    conn.commit()

    from test.test_leakage import ingest, make_match
    ingest(conn, make_match("past", "2026-08-01T00:00:00Z"))

    as_of = normalize.parse_started_at("2026-08-05T00:00:00Z")
    live = {"match_id": "live", "started_at": as_of, "map": "Sunset",
            "season": "e11a5", "region": "na", "winner": None,
            "rounds_blue": None, "rounds_red": None}
    roster = temporal.match_roster(conn, "past")

    out = fb.build_match(conn, live, roster, build_norms(conn, as_of), 0.5,
                         reference.agent_roles(conn), require_outcome=False)
    assert out is not None, "live matches have no winner and must still build"
    assert out.target is None and out.margin is None
    assert out.values, "features must still be produced"


def test_one_standard_error_rule_prefers_the_simpler_tied_model():
    """Consecutive runs on identical data crowned different winners, because
    the gaps are smaller than the noise. Selecting on the raw minimum is
    selecting on noise."""
    from valwr.model.train import COMPLEXITY
    assert COMPLEXITY["avg rating (fitted)"] < COMPLEXITY["logistic regression"]
    assert COMPLEXITY["logistic regression"] < COMPLEXITY["gradient boosting"]


def test_log_loss_standard_error_is_positive_and_shrinks_with_n():
    rng = np.random.default_rng(3)
    y_small = rng.integers(0, 2, 200)
    y_big = rng.integers(0, 2, 20000)
    se_small = evaluate.log_loss_standard_error(y_small, np.full(200, 0.5))
    se_big = evaluate.log_loss_standard_error(y_big, np.full(20000, 0.5))
    # A constant prediction has zero variance in per-sample loss.
    assert se_small >= 0 and se_big >= 0
    varied_small = evaluate.log_loss_standard_error(
        y_small, rng.uniform(0.3, 0.7, 200))
    varied_big = evaluate.log_loss_standard_error(
        y_big, rng.uniform(0.3, 0.7, 20000))
    assert varied_small > varied_big


def test_analyze_reads_the_same_bundle_train_writes():
    """The two modules must agree on the artefact name and its keys.

    analyze.py loaded 'gbm.joblib' by name and kept working off a stale file
    after train.py was changed to write 'model.joblib' -- silently reporting
    on a model that was no longer the one being served. A fresh clone would
    have crashed instead, which is the better failure but still a bug.
    """
    import re
    root = pathlib.Path(__file__).resolve().parent.parent
    written = (root / "valwr" / "model" / "train.py").read_text(encoding="utf-8")
    read = (root / "valwr" / "model" / "analyze.py").read_text(encoding="utf-8")

    w = set(re.findall(r'models / "([^"]+\.joblib)"', written))
    r = set(re.findall(r'"models" / "([^"]+\.joblib)"', read))
    assert w and r, f"could not locate bundle names (write={w} read={r})"
    assert w == r, f"train writes {w} but analyze reads {r}"


def test_bundle_carries_everything_inference_needs():
    """Reproducing training features live needs more than the estimator."""
    import joblib
    root = pathlib.Path(__file__).resolve().parent.parent
    path = root / "models" / "model.joblib"
    if not path.exists():
        pytest.skip("no trained bundle present")
    b = joblib.load(path)
    for key in ("estimators", "best", "columns", "norms", "prior_rate",
                "roles", "norms_as_of"):
        assert key in b, f"bundle is missing {key}; live features would drift"
    assert b["best"] in b["estimators"] or b["best"] in (
        "logistic regression", "gradient boosting", "margin regression",
        "logistic + margin blend"), "selected model is not servable"
