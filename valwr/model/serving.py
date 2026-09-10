"""Serve the model the bundle says it ships.

`bundle["best"]` is a *result* name ("logistic regression"), while
`bundle["estimators"]` is keyed by *component* ("logistic", "gbm",
"margin_reg"). Every consumer used to bridge the two itself, as
`estimators.get(best) or estimators["logistic"]` -- which matches nothing for
three of the four models train.py is allowed to ship, and so silently served
the logistic model while labelling the prediction with the other one's name.
It only looked right because logistic regression is what currently ships.

One mapping, used by the live path, the sandbox and the analysis, and an
unknown name raises instead of falling back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Result name -> the bundle components it needs.
COMPONENTS = {
    "logistic regression": ("logistic",),
    "gradient boosting": ("gbm",),
    "margin regression": ("margin_reg", "margin_link"),
    "logistic + margin blend": ("logistic", "margin_reg", "margin_link"),
}


def servable(name: str, estimators: dict) -> bool:
    """Whether a bundle holding `estimators` can serve the model `name`."""
    if name in COMPONENTS:
        return all(k in estimators for k in COMPONENTS[name])
    return callable(estimators.get(name))


def probabilities(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """P(TEAM_A wins) for each row of `frame`, from the shipped model.

    Columns the model does not use are ignored and missing ones read as 0.0,
    which is the neutral value for a difference feature.
    """
    name = bundle["best"]
    est = bundle["estimators"]
    if not servable(name, est):
        raise KeyError(f"bundle ships {name!r} but does not hold what it needs "
                       f"to serve it (has {sorted(est)})")

    frame = frame.reindex(columns=bundle["columns"], fill_value=0.0)
    X = frame.to_numpy(float)

    def margin() -> np.ndarray:
        m = est["margin_reg"].predict(X).reshape(-1, 1)
        return est["margin_link"].predict_proba(m)[:, 1]

    if name == "logistic regression":
        p = est["logistic"].predict_proba(X)[:, 1]
    elif name == "gradient boosting":
        p = est["gbm"].predict_proba(X)[:, 1]
    elif name == "margin regression":
        p = margin()
    elif name == "logistic + margin blend":
        # The same equal weighting train.py scored.
        p = 0.5 * est["logistic"].predict_proba(X)[:, 1] + 0.5 * margin()
    else:
        # A fitted single-feature baseline: a callable over a frame.
        p = np.asarray(est[name](frame), dtype=float)

    p = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} returned a non-finite probability")
    return np.clip(p, 0.0, 1.0)
