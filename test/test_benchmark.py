"""Tests for sandbox/benchmark.py -- the drift detector itself.

A drift detector that quietly reports nothing is worse than none at all: it
converts "I did not check" into "I checked and it was fine". So these tests
push scenarios across the thresholds in both directions and assert the
comparison actually says so.

`NOISE_FLOOR` and the flip guard are the two judgement calls in the module.
Both are asserted from the outside, by moving a probability just over and just
under each threshold, rather than by reading the constants back.
"""

from __future__ import annotations

import pytest

from valwr.sandbox import benchmark as B
from valwr.sandbox.schema import ScenarioResult

BUNDLE = {"best": "logistic regression", "norms_as_of": 1_700_000_000,
          "columns": ["d_rank", "d_acs", "d_kd"]}


def _result(name="baseline", probability=0.5, features=None, category="core",
            mirror=None, expect=None):
    return ScenarioResult(
        scenario=name, category=category, model="logistic regression",
        probability=probability,
        features=features or {"d_rank": 0.25, "d_acs": -0.5},
        mirror_probability=mirror, expect=expect,
        factors=(("d_rank", 0.125),))


def _saved(tmp_path, results):
    path = tmp_path / "bench.json"
    B.save(results, BUNDLE, path)
    return B.load(path)


# --- persistence -------------------------------------------------------

def test_a_benchmark_round_trips_through_disk(tmp_path):
    old = _saved(tmp_path, [_result(probability=0.612345,
                                    mirror=0.387655, expect="A")])
    assert old["model"] == "logistic regression"
    assert old["norms_as_of"] == 1_700_000_000
    assert old["n_features"] == 3
    assert old["schema_version"] == B.SCHEMA_VERSION
    entry = old["scenarios"]["baseline"]
    assert entry["probability"] == 0.612345
    assert entry["mirror_probability"] == 0.387655
    assert entry["expect"] == "A"
    assert entry["category"] == "core"
    assert entry["factors"] == [["d_rank", 0.125]]
    assert entry["features"] == {"d_rank": 0.25, "d_acs": -0.5}


def test_a_scenario_without_a_mirror_stays_none_rather_than_zero(tmp_path):
    """Rounding None would turn "not measured" into a mirror error of 0.5 --
    a perfect-looking symmetry score for a scenario that was never mirrored.
    """
    old = _saved(tmp_path, [_result(mirror=None)])
    assert old["scenarios"]["baseline"]["mirror_probability"] is None


def test_loading_a_missing_benchmark_names_the_command_to_create_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="sandbox benchmark"):
        B.load(tmp_path / "absent.json")


def test_saving_creates_the_directory_it_needs(tmp_path):
    path = tmp_path / "deep" / "nested" / "bench.json"
    assert B.save([_result()], BUNDLE, path).exists()


# --- the comparison ----------------------------------------------------

def test_a_scenario_that_moved_is_reported_with_both_numbers(tmp_path):
    """The failure this module exists to prevent: a real change passing
    quietly. 0.500 to 0.560 must appear, by name and by number."""
    old = _saved(tmp_path, [_result(probability=0.50)])
    text = B.compare(old, [_result(probability=0.56)])
    assert "moved by more than" in text
    assert "baseline" in text
    assert "50.0%" in text and "56.0%" in text
    assert "+6.0" in text


def test_a_move_smaller_than_the_noise_floor_is_not_news(tmp_path):
    old = _saved(tmp_path, [_result(probability=0.50)])
    text = B.compare(old, [_result(probability=0.5 + B.NOISE_FLOOR / 2)])
    assert f"more than {B.NOISE_FLOOR}: 0" in text


def test_a_move_just_over_the_noise_floor_is(tmp_path):
    old = _saved(tmp_path, [_result(probability=0.50)])
    text = B.compare(old, [_result(probability=0.5 + B.NOISE_FLOOR * 2)])
    assert f"more than {B.NOISE_FLOOR}: 1" in text


def test_a_favourite_changing_side_is_called_out(tmp_path):
    old = _saved(tmp_path, [_result(probability=0.45)])
    text = B.compare(old, [_result(probability=0.55)])
    assert "favourite flipped side: 1" in text
    assert "45.0% -> 55.0%" in text


def test_a_coin_flip_wobbling_across_50_is_not_a_flip(tmp_path):
    """0.4995 to 0.5005 crosses the line but means nothing. Reporting it would
    fill the output with noise on every retrain and train the reader to skim
    past the flips that matter."""
    old = _saved(tmp_path, [_result(probability=0.4995)])
    text = B.compare(old, [_result(probability=0.5005)])
    assert "favourite flipped side: 0" in text


def test_added_and_removed_scenarios_are_both_named(tmp_path):
    old = _saved(tmp_path, [_result(name="gone"), _result(name="kept")])
    text = B.compare(old, [_result(name="kept"), _result(name="fresh")])
    assert "new scenarios     : 1" in text and "fresh" in text
    assert "removed scenarios : 1" in text and "gone" in text


def test_a_new_scenario_is_not_counted_as_having_moved(tmp_path):
    """It has no previous value to move from. Treating absent as zero would
    report every addition as a full-scale change."""
    old = _saved(tmp_path, [_result(name="kept", probability=0.5)])
    text = B.compare(old, [_result(name="kept", probability=0.5),
                           _result(name="fresh", probability=0.99)])
    assert f"more than {B.NOISE_FLOOR}: 0" in text


def test_a_feature_that_changed_is_surfaced_with_its_old_and_new_value(tmp_path):
    old = _saved(tmp_path, [_result(features={"d_rank": 0.25})])
    text = B.compare(old, [_result(features={"d_rank": 0.75})])
    assert "largest feature-vector changes" in text
    assert "0.2500 -> 0.7500" in text


def test_identical_runs_report_no_movement_at_all(tmp_path):
    old = _saved(tmp_path, [_result()])
    text = B.compare(old, [_result()])
    assert f"more than {B.NOISE_FLOOR}: 0" in text
    assert "favourite flipped side: 0" in text
    assert "largest feature-vector changes" not in text
