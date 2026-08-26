"""Adapter over the trained model bundle.

The sandbox probes every estimator in `models/model.joblib`, not just the
shipped one, and that is deliberate. The shipped model is a linear logistic
regression on antisymmetric difference features, so single-factor sweeps are
monotonic *by construction*, pairwise interactions are exactly zero, and
mirroring holds automatically. Those checks prove almost nothing about it.

The gradient booster has none of those guarantees. Kinks, real interactions and
genuine mirror asymmetry can all appear there, so comparing the two is where
the sandbox earns its keep.

`Predictor` is a narrow protocol -- `predict_proba(features) -> float` -- so
nothing here depends on the estimator's type, and a future model slots in
without touching a scenario.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

REPO = Path(__file__).resolve().parent.parent.parent
BUNDLE = REPO / "models" / "model.joblib"


class MissingModel(RuntimeError):
    pass


class Predictor(Protocol):
    name: str

    def predict_proba(self, features: dict[str, float]) -> float:
        """P(team A wins) for one difference-feature vector."""


class BundlePredictor:
    """One estimator out of the bundle, wrapped in the protocol."""

    def __init__(self, name: str, estimator, columns: list[str]):
        self.name = name
        self._estimator = estimator
        self._columns = columns

    def _vector(self, features: dict[str, float]) -> list[float]:
        return [features.get(c, 0.0) for c in self._columns]

    def predict_proba(self, features: dict[str, float]) -> float:
        row = self._vector(features)
        if hasattr(self._estimator, "predict_proba"):
            p = float(self._estimator.predict_proba([row])[0][1])
        else:
            # A fitted single-feature baseline is a callable over a frame.
            import pandas as pd
            frame = pd.DataFrame({c: [features.get(c, 0.0)]
                                  for c in self._columns})
            p = float(self._estimator(frame)[0])
        if not math.isfinite(p):
            raise ValueError(f"{self.name} returned a non-finite probability")
        return min(1.0, max(0.0, p))


class MarginPredictor:
    """Margin regression plus its probability link, treated as one model."""

    name = "margin"

    def __init__(self, reg, link, columns: list[str]):
        self._reg, self._link, self._columns = reg, link, columns

    def predict_proba(self, features: dict[str, float]) -> float:
        row = [[features.get(c, 0.0) for c in self._columns]]
        margin = self._reg.predict(row).reshape(-1, 1)
        p = float(self._link.predict_proba(margin)[0][1])
        if not math.isfinite(p):
            raise ValueError("margin model returned a non-finite probability")
        return min(1.0, max(0.0, p))


def load_bundle(path: Path | None = None) -> dict:
    import joblib
    path = path or BUNDLE
    if not path.exists():
        raise MissingModel(
            f"no model at {path}. Run `python -m valwr.model.train` first; "
            f"the sandbox needs the bundle's norms, prior and roles so its "
            f"features are comparable to production.")
    return joblib.load(path)


def predictors(bundle: dict, which: str = "all") -> list[Predictor]:
    """Build the requested predictors from a bundle.

    `which` is 'all', or a comma-separated selection of logistic / gbm /
    margin / baselines.
    """
    cols = bundle["columns"]
    est = bundle["estimators"]
    out: list[Predictor] = []
    wanted = {w.strip() for w in which.split(",")} if which != "all" else None

    def include(key: str) -> bool:
        return wanted is None or key in wanted

    if include("logistic") and "logistic" in est:
        out.append(BundlePredictor("logistic", est["logistic"], cols))
    if include("gbm") and "gbm" in est:
        out.append(BundlePredictor("gbm", est["gbm"], cols))
    if include("margin") and {"margin_reg", "margin_link"} <= set(est):
        out.append(MarginPredictor(est["margin_reg"], est["margin_link"], cols))
    if include("baselines"):
        for key, estimator in est.items():
            if key.endswith("(fitted)") or key == "best player rank":
                out.append(BundlePredictor(key, estimator, cols))

    if not out:
        raise MissingModel(f"no predictors matched {which!r}")
    return out


def shipped(bundle: dict) -> Predictor:
    """The estimator the project actually serves."""
    name = bundle.get("best", "logistic")
    est = bundle["estimators"]
    if name in est:
        return BundlePredictor(name, est[name], bundle["columns"])
    return BundlePredictor("logistic", est["logistic"], bundle["columns"])


def linear_contributions(bundle: dict, features: dict[str, float],
                         n: int = 6) -> list[tuple[str, float]]:
    """Signed per-feature contributions from the linear model.

    Exact rather than approximated -- weight times standardised value is the
    whole story for a linear model, which is a real benefit of shipping the
    simple one. Returns [] for any estimator this does not apply to.
    """
    est = bundle["estimators"].get("logistic")
    if est is None or not hasattr(est, "named_steps"):
        return []
    try:
        scaler = est.named_steps["standardscaler"]
        clf = est.named_steps["logisticregression"]
    except (AttributeError, KeyError):
        return []

    out = []
    for i, col in enumerate(bundle["columns"]):
        raw = features.get(col, 0.0)
        scaled = (raw - scaler.mean_[i]) / (scaler.scale_[i] or 1.0)
        out.append((col, float(clf.coef_[0][i] * scaled)))
    out.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return out[:n]
