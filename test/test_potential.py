"""The 0-100 potential score, and the table that prints it.

Everything here runs against synthetic players in an in-memory database, so
these tests need neither the real database nor a fitted index.
"""

from __future__ import annotations

import json

import pytest

from valwr.live.__main__ import _display_width, _fit
from valwr.rating import potential as P
from valwr.sandbox import profiles, world


def an_index() -> P.PerfIndex:
    """A hand-built index, so the tests do not depend on a fitted one."""
    return P.PerfIndex(
        means={"acs": 212.0, "rating": 1.0, "kd": 1.05, "map_edge": 0.0},
        stds={"acs": 20.0, "rating": 0.06, "kd": 0.13, "map_edge": 0.02},
        quantiles=[i / 100.0 - 1.0 for i in range(201)],   # -1.0 .. 1.0
        as_of=0, n=201)


# --- the score ---------------------------------------------------------

def test_percentile_is_bounded_and_monotone():
    idx = an_index()
    assert idx.percentile(-99.0) == 0
    assert idx.percentile(99.0) == 100
    assert 0 <= idx.percentile(0.0) <= 100
    assert idx.percentile(-0.5) < idx.percentile(0.0) < idx.percentile(0.5)


def test_a_player_with_no_history_scores_nothing():
    """None, never a number.

    Inventing a score for someone we know nothing about is the failure this
    project keeps guarding against; the caller prints "--" instead.
    """
    conn = world.new_connection()
    norms = _norms(conn)
    assert P.measure(conn, "nobody", 1_800_000_000, "Ascent", norms) is None


def test_map_edge_is_zero_without_map_history():
    """The map component is a delta, so "no history here" means "no opinion".

    An absolute map rating would restate overall skill, which `rating` already
    carries; the delta has to vanish when there is nothing to compare.
    """
    conn, as_of, norms = _one_player(map_played="Ascent")
    c = P.measure(conn, "p", as_of, "Icebox", norms)     # never played Icebox
    assert c is not None
    assert c.n_map_games == 0
    assert c.map_edge == pytest.approx(0.0, abs=1e-12)


def test_components_come_back_populated():
    conn, as_of, norms = _one_player(map_played="Ascent")
    c = P.measure(conn, "p", as_of, "Ascent", norms)
    assert c.n_games > 0 and c.n_map_games > 0
    for name in P.WEIGHTS:
        assert isinstance(getattr(c, name), float)


def test_weights_are_a_convex_combination():
    """They should sum to 1, or the composite silently changes scale."""
    assert sum(P.WEIGHTS.values()) == pytest.approx(1.0)


def test_explain_picks_the_most_unusual_component():
    """Ranked by raw z, not weighted contribution.

    Weighting let ACS -- the heaviest weight -- win nearly every row, so the
    column read "high combat score" for four players out of five.
    """
    idx = an_index()
    c = P.Components(rating=1.0, acs=212.0, kd=1.05, map_edge=0.10,
                     n_games=50, n_map_games=20)      # map_edge is +5 sd
    assert "map" in P.explain(idx, c)


def test_thin_history_is_called_out():
    idx = an_index()
    c = P.Components(rating=1.2, acs=280.0, kd=1.6, map_edge=0.0,
                     n_games=2, n_map_games=0)
    assert "2 game" in P.explain(idx, c)


def test_index_round_trips_through_json():
    idx = an_index()
    back = json.loads(idx.to_json())
    assert back["means"]["acs"] == idx.means["acs"]
    assert len(back["quantiles"]) == len(idx.quantiles)


# --- the table ---------------------------------------------------------

def test_display_width_counts_wide_glyphs_as_two():
    assert _display_width("abc") == 3
    assert _display_width("테스트샘플") == 10         # 5 glyphs, 10 columns
    assert _display_width("테스트#abcd") == 11


def test_fit_pads_to_exact_display_width():
    """A wide-glyph name must occupy the same columns as an ASCII one.

    Character-count padding under-pads wide glyphs and the table's columns
    walk out of line, which is what the first version did.
    """
    for name in ("short", "테스트샘플#abcd", "a" * 40, "샘플" * 20, ""):
        assert _display_width(_fit(name, 20)) == 20


# --- helpers -----------------------------------------------------------

def _norms(conn):
    from valwr.rating.normalize import build_norms
    return build_norms(conn, 2_000_000_000)


def _one_player(map_played: str):
    as_of = 1_800_000_000
    conn = world.new_connection()
    world.write_player(conn, "p", profiles.get("average"), as_of,
                       map_played, "Jett")
    return conn, as_of, _norms(conn)
