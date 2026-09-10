"""Post-training analysis and the charts the README embeds."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import struct
import zlib

import numpy as np
import pandas as pd
import pytest

from valwr.model import analyze, evaluate

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _audit():
    spec = importlib.util.spec_from_file_location("audit", ROOT / "tools" / "audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- labels -----------------------------------------------------------

def test_labels_strip_only_the_prefix():
    """replace("d_", "") printed "games_playemean" on the shipped chart."""
    assert analyze.label("d_games_played_mean") == "games_played_mean"
    assert analyze.label("d_rating_trend_mean") == "rating_trend_mean"
    assert analyze.label("role_balance") == "role_balance"


def test_families_group_the_aggregates_of_one_stat():
    assert {analyze.family(f) for f in
            ("d_acs_mean", "d_acs_max", "d_acs_min", "d_acs_std")} == {"acs"}
    assert analyze.family("d_wr_recent_mean") == "wr_recent"
    assert analyze.family("d_role_balance") == "role_balance"
    # A bare suffix is a name, not an aggregate of nothing.
    assert analyze.family("d_max") == "max"


# --- importance ---------------------------------------------------------

def test_family_importance_finds_the_family_that_carries_the_signal():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n = 4000
    signal = rng.standard_normal(n)
    frame = pd.DataFrame({
        "d_a_mean": signal + 0.1 * rng.standard_normal(n),
        "d_a_max": signal + 0.1 * rng.standard_normal(n),
        "d_noise_mean": rng.standard_normal(n),
    })
    y = (signal + rng.standard_normal(n) > 0).astype(int)
    cols = list(frame.columns)
    # Fitted on an array, as train.py fits, since serving predicts on one.
    model = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(fit_intercept=False)
                          ).fit(frame.to_numpy(float), y)
    bundle = {"best": "logistic regression", "columns": cols,
              "estimators": {"logistic": model}}

    imp = analyze.family_importance(bundle, frame, y, repeats=3)
    assert list(imp["family"]) == ["a", "noise"]
    assert imp.loc[0, "features"] == 2
    assert imp.loc[0, "rise"] > 0.05
    assert abs(imp.loc[1, "rise"]) < 0.01


# --- reliability bins ------------------------------------------------------

def test_reliability_bins_hold_equal_counts():
    """Equal-width bins once plotted points built from 6 and 25 matches."""
    rng = np.random.default_rng(1)
    p = np.concatenate([[0.01, 0.02, 0.99], rng.uniform(0.45, 0.55, 997)])
    y = rng.integers(0, 2, len(p))
    table = evaluate.reliability_table(y, p)
    counts = [n for *_, n in table]
    assert counts == [100] * 10
    assert [m for m, *_ in table] == sorted(m for m, *_ in table)


def test_reliability_bins_never_exceed_the_rows():
    table = evaluate.reliability_table([0, 1, 1, 0], [0.2, 0.4, 0.6, 0.8])
    assert [n for *_, n in table] == [1, 1, 1, 1]


# --- chart provenance ---------------------------------------------------------

def _chunk(kind: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(kind + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)


def _png(*chunks: bytes) -> bytes:
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + b"".join(chunks)
            + _chunk(b"IEND", b""))


def test_png_text_reads_plain_and_international_chunks(tmp_path):
    audit = _audit()
    stamp = json.dumps({"model": "logistic regression", "norms_as_of": 7})
    path = tmp_path / "chart.png"
    path.write_bytes(_png(
        _chunk(b"tEXt", b"Source\0" + stamp.encode("latin-1")),
        _chunk(b"iTXt", b"Plain\0\0\0\0\0" + "é".encode("utf-8")),
        _chunk(b"iTXt", b"Packed\0\1\0\0\0" + zlib.compress(b"deflated")),
    ))
    text = audit.png_text(path)
    assert json.loads(text["Source"])["norms_as_of"] == 7
    assert text["Plain"] == "é"
    assert text["Packed"] == "deflated"
    (tmp_path / "junk.png").write_bytes(b"not a png")
    assert audit.png_text(tmp_path / "junk.png") == {}


def test_matplotlib_writes_a_stamp_the_audit_can_read(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundle = {"best": "logistic regression", "norms_as_of": 1_788_233_215}
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = tmp_path / "reliability.png"
    fig.savefig(path, metadata={analyze.STAMP_KEY: analyze.stamp(bundle, 10)})
    plt.close(fig)
    drawn = json.loads(_audit().png_text(path)[analyze.STAMP_KEY])
    assert drawn == {"model": "logistic regression",
                     "norms_as_of": 1_788_233_215, "n_test": 10}


def test_the_readme_embeds_both_charts():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for chart in ("reports/reliability.png", "reports/importance.png"):
        assert f"]({chart})" in readme, f"README no longer shows {chart}"
        assert (ROOT / chart).exists()
