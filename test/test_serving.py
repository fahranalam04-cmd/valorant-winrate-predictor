"""The shipped model is the one that gets served.

`bundle["best"]` names a result ("gradient boosting") while the estimators are
keyed by component ("gbm"). Every consumer used to bridge that with
`estimators.get(best) or estimators["logistic"]`, which misses for three of
the four shippable models -- so selecting any of them quietly served logistic
regression under the other model's name. These stubs return a different
probability per component, so a wrong route cannot pass by coincidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from valwr.model import serving
from valwr.sandbox import predictor as pred

COLS = ["d_a", "d_b"]


class Fixed:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


class Margin:
    def predict(self, X):
        return np.full(len(X), 3.0)


class Link:
    def predict_proba(self, m):
        assert m.shape[1] == 1, "the link is fitted on a single margin column"
        return np.column_stack([np.full(len(m), 0.8), np.full(len(m), 0.2)])


def bundle(best: str, **overrides) -> dict:
    est = {"logistic": Fixed(0.6), "gbm": Fixed(0.7),
           "margin_reg": Margin(), "margin_link": Link()}
    est.update(overrides)
    return {"best": best, "columns": COLS, "estimators": est}


FRAME = pd.DataFrame([{"d_a": 1.0, "d_b": -1.0}])


@pytest.mark.parametrize("best, expected", [
    ("logistic regression", 0.6),
    ("gradient boosting", 0.7),
    ("margin regression", 0.2),
    ("logistic + margin blend", 0.4),
])
def test_each_shippable_model_routes_to_its_own_components(best, expected):
    got = serving.probabilities(bundle(best), FRAME)
    assert got[0] == pytest.approx(expected)


def test_a_baseline_is_served_as_a_callable_over_the_frame():
    seen = {}

    def baseline(frame):
        seen["columns"] = list(frame.columns)
        return np.full(len(frame), 0.55)

    b = bundle("avg rank (fitted)", **{"avg rank (fitted)": baseline})
    assert serving.probabilities(b, FRAME)[0] == pytest.approx(0.55)
    assert seen["columns"] == COLS


def test_an_unservable_selection_raises_instead_of_falling_back():
    # The message is the contract, not just the exception type: a bare
    # KeyError('margin_link') from deep in the routing says nothing about
    # which selection could not be served, or what the bundle held instead.
    with pytest.raises(KeyError, match="ships 'logistic \\+ isotonic' but"):
        serving.probabilities(bundle("logistic + isotonic"), FRAME)
    b = bundle("margin regression")
    del b["estimators"]["margin_link"]
    with pytest.raises(KeyError, match="does not hold what it needs"):
        serving.probabilities(b, FRAME)
    assert not serving.servable("margin regression", b["estimators"])


def test_missing_columns_read_neutral_and_extra_ones_are_ignored():
    class Echo:
        def predict_proba(self, X):
            assert X.shape[1] == len(COLS)
            p = 0.5 + 0.1 * X[:, 0] + 0.01 * X[:, 1]
            return np.column_stack([1 - p, p])

    b = bundle("logistic regression", logistic=Echo())
    frame = pd.DataFrame([{"d_b": 2.0, "not_a_feature": 99.0}])
    assert serving.probabilities(b, frame)[0] == pytest.approx(0.52)


def test_a_non_finite_probability_raises():
    b = bundle("gradient boosting", gbm=Fixed(float("nan")))
    with pytest.raises(ValueError):
        serving.probabilities(b, FRAME)


def test_the_sandbox_serves_the_selected_model_not_logistic():
    model = pred.shipped(bundle("gradient boosting"))
    assert model.name == "gradient boosting"
    assert model.predict_proba({"d_a": 1.0, "d_b": -1.0}) == pytest.approx(0.7)


def test_consumers_do_not_resolve_the_model_themselves():
    """The fallback idiom was copied twice before; keep it from coming back.

    Reading the "logistic" component by its literal key stays allowed: the
    factor explainers do that on purpose, because they explain that linear
    model specifically and return nothing for any other.
    """
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parent.parent / "valwr"
    lookup = re.compile(r'estimators"\]\.get\((?!"logistic"\))')
    fallback = re.compile(r'\bor\s+bundle\["estimators"\]')
    offenders = [str(p.relative_to(root)) for p in root.rglob("*.py")
                 if p.name != "serving.py"
                 and (lookup.search(text := p.read_text(encoding="utf-8"))
                      or fallback.search(text))]
    assert offenders == [], f"resolve the shipped model via serving: {offenders}"
