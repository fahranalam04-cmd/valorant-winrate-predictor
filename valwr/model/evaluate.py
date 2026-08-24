"""Metrics.

Log loss is primary. Accuracy is the least informative number here and the one
people ask about, so it is reported last.

Calibration matters more than ranking for this product: it displays "58%", and
that has to mean 58%. A model can have decent AUC and useless calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-12


@dataclass(frozen=True)
class Scores:
    name: str
    n: int
    log_loss: float
    brier: float
    auc: float
    accuracy: float
    ece: float

    def row(self) -> str:
        return (f"  {self.name:<22} {self.log_loss:>8.4f} {self.brier:>8.4f} "
                f"{self.auc:>7.3f} {self.accuracy*100:>7.1f}% {self.ece:>7.3f}")


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Mean gap between predicted probability and observed frequency.

    Binned by prediction. A model claiming 58% on a bucket that actually wins
    52% of the time is miscalibrated by 6 points there, however good its
    ranking is.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not m.any():
            continue
        total += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return total


def score(name: str, y, p) -> Scores:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    # AUC is undefined on a single-class slice; report 0.5 rather than crash.
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    return Scores(
        name=name,
        n=len(y),
        log_loss=log_loss(y, p, labels=[0, 1]),
        brier=brier_score_loss(y, p),
        auc=auc,
        accuracy=float(((p >= 0.5).astype(int) == y).mean()),
        ece=expected_calibration_error(y, p),
    )


def header() -> str:
    return (f"  {'model':<22} {'logloss':>8} {'brier':>8} {'auc':>7} "
            f"{'acc':>8} {'ece':>7}\n  " + "-" * 63)


def confidence_interval(accuracy: float, n: int) -> float:
    """95% half-width on an accuracy estimate.

    Reported alongside every result because at these sample sizes the interval
    is often wider than the effect being claimed.
    """
    return 1.96 * math.sqrt(max(accuracy * (1 - accuracy), 1e-9) / n)


def log_loss_standard_error(y, p) -> float:
    """Standard error of the mean per-sample log loss.

    Model differences here are in the third decimal place. Without knowing the
    noise floor there is no way to say whether 0.6896 genuinely beats 0.6924
    or whether they are the same model wearing different hats.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    per_sample = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(per_sample.std(ddof=1) / np.sqrt(len(per_sample)))


def reliability_table(y, p, bins: int = 10) -> list[tuple[float, float, int]]:
    """(mean predicted, observed frequency, count) per bin, for the diagram."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(p.min(), p.max(), bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < edges[-1] else p <= hi)
        if m.sum() >= 5:
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out
