"""The significance test behind the rating-versus-ACS verdict.

A 0.003 gap between two correlations on 50,000 players once printed "ACS wins
-- rating earned nothing". The predictors correlate around 0.9 with each other,
so whether a gap that size is real needs a test for dependent correlations.
"""

from __future__ import annotations

import numpy as np
import pytest

from valwr.rating.validate import dependent_corr_z


def test_equal_correlations_give_zero():
    assert dependent_corr_z(0.2, 0.2, 0.9, 1000) == pytest.approx(0.0)


def test_swapping_the_predictors_flips_the_sign():
    a = dependent_corr_z(0.25, 0.20, 0.9, 1000)
    assert a > 0
    assert dependent_corr_z(0.20, 0.25, 0.9, 1000) == pytest.approx(-a)


def test_more_players_make_the_same_gap_more_significant():
    assert (abs(dependent_corr_z(0.25, 0.20, 0.9, 5000))
            > abs(dependent_corr_z(0.25, 0.20, 0.9, 500)))


def test_correlated_predictors_make_a_gap_more_certain():
    """Near-duplicates share their noise, so their difference is sharper."""
    assert (abs(dependent_corr_z(0.25, 0.20, 0.95, 1000))
            > abs(dependent_corr_z(0.25, 0.20, 0.10, 1000)))


def test_under_the_null_it_is_standard_normal():
    rng = np.random.default_rng(0)
    cov = np.array([[1, .2, .2], [.2, 1, .9], [.2, .9, 1]])
    chol = np.linalg.cholesky(cov)
    zs = []
    for _ in range(600):
        r = np.corrcoef((rng.standard_normal((400, 3)) @ chol.T).T)
        zs.append(dependent_corr_z(r[0, 1], r[0, 2], r[1, 2], 400))
    zs = np.array(zs)
    assert abs(zs.mean()) < 0.15
    assert 0.85 < zs.std() < 1.15
    assert 0.02 < np.mean(np.abs(zs) > 1.96) < 0.09
