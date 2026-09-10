"""Precision, recall and PR-AUC, and the tool that reports them per model."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from valwr.model import evaluate

ROOT = Path(__file__).resolve().parent.parent


def _tool():
    spec = importlib.util.spec_from_file_location(
        "model_metrics", ROOT / "tools" / "model_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precision_recall_and_pr_auc_by_hand():
    # Called Blue: rows 0 and 2. Right on row 0, wrong on row 2; row 1 missed.
    y = [1, 1, 0, 0]
    p = [0.9, 0.4, 0.6, 0.1]
    sc = evaluate.score("hand", y, p)
    assert sc.precision == pytest.approx(0.5)
    assert sc.recall == pytest.approx(0.5)
    assert sc.f1 == pytest.approx(0.5)
    # Ranked .9(+) .6(-) .4(+) .1(-): precision 1/1 at the first positive and
    # 2/3 at the second, averaged.
    assert sc.pr_auc == pytest.approx((1 + 2 / 3) / 2)


def test_a_model_that_always_says_blue_shows_why_calls_blue_is_reported():
    y = np.array([1, 0, 1, 0, 0])
    sc = evaluate.score("always blue", y, np.full(5, 0.5))
    assert sc.recall == 1.0
    assert sc.precision == pytest.approx(y.mean())


def test_a_slice_with_no_blue_wins_does_not_crash():
    sc = evaluate.score("all red", [0, 0, 0], [0.2, 0.7, 0.4])
    assert math.isnan(sc.pr_auc)
    assert sc.precision == 0.0 and sc.recall == 0.0


def test_verification_refuses_predictions_that_do_not_reproduce():
    tool = _tool()
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.3, 0.6, 0.4])
    recorded = [{"name": "m", "log_loss": evaluate.score("m", y, p).log_loss}]
    assert tool.verify(recorded, {"m": p}, y) == []
    assert tool.verify(recorded, {"m": p * 0.9}, y), "a changed model passed"
    assert tool.verify(recorded, {}, y) == ["m: not reproduced"]


def test_the_table_marks_the_shipped_model_and_blanks_undefined_numbers():
    tool = _tool()
    y = np.array([1, 0, 1, 1, 0])
    metrics = {
        "n_test": 5, "blue_win_rate": 0.6, "shipped": "b",
        "models": [
            tool.row("a", y, np.full(5, 0.5), shipped="b", candidate=True, n_features=3),
            tool.row("b", y, np.array([.7, .4, .6, .8, .3]), shipped="b",
                     candidate=True, n_features=3),
        ],
    }
    metrics["models"][0]["pr_auc"] = float("nan")
    text = tool.render(metrics)
    assert "| **b (shipped)** |" in text
    assert "| —" in text
    # Sorted by log loss, so the shipped, better model is listed first.
    assert text.index("b (shipped)") < text.index("| a |")


def test_doc_regeneration_only_touches_the_marked_block(tmp_path):
    tool = _tool()
    doc = tmp_path / "doc.md"
    doc.write_text(f"before\n{tool.START}\nold\n{tool.END}\nafter\n", encoding="utf-8")
    assert tool.write_doc("new", doc)
    assert doc.read_text(encoding="utf-8") == f"before\n{tool.START}\nnew\n{tool.END}\nafter\n"
    bare = tmp_path / "bare.md"
    bare.write_text("no markers", encoding="utf-8")
    assert not tool.write_doc("new", bare)
