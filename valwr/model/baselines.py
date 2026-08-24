"""Baselines. Permanent fixtures, reported in every evaluation.

A model that does not beat these has demonstrated nothing, and the second one
is stronger than it looks -- rank is most of what there is to know about a
player, and matchmaking has already equalised it.

Measured on the built matrix, "higher average rank wins" scores about 52%.
That is not a weak baseline, it is a hard problem: matchmaking works, so the
room between 52% and the ceiling is all the room there is.
"""

from __future__ import annotations

import numpy as np


def coin_flip(n: int) -> np.ndarray:
    """The floor. Anything below this is inverted somewhere."""
    return np.full(n, 0.5)


class FittedBaseline:
    """A single-feature logistic model, fitted on train like any other.

    An earlier version squashed the rank difference through a hand-scaled
    sigmoid. That handicapped it: its log loss came out worse than a coin
    flip, so "the model beats the rank baseline" was measuring my arbitrary
    scaling rather than the baseline. A baseline has to be given the same
    chance as the model or beating it proves nothing.
    """

    def __init__(self, column: str):
        self.column = column
        self._model = None

    def fit(self, train_df):
        from sklearn.linear_model import LogisticRegression
        X = train_df[[self.column]].to_numpy(float)
        self._model = LogisticRegression(max_iter=1000).fit(
            X, train_df["target"].to_numpy(int))
        return self

    def predict(self, df) -> np.ndarray:
        X = df[[self.column]].to_numpy(float)
        return self._model.predict_proba(X)[:, 1]


def fitted(train_df) -> dict:
    """Baselines fitted on the training slice, ready to score on test."""
    return {
        "coin flip": lambda df: coin_flip(len(df)),
        "avg rank (fitted)": FittedBaseline("d_tier_mean").fit(train_df).predict,
        "best player rank": FittedBaseline("d_tier_max").fit(train_df).predict,
        "avg rating (fitted)": FittedBaseline("d_rating_mean").fit(train_df).predict,
    }
