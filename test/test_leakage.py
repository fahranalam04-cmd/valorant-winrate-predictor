"""Phase 4: the leakage audit.

Leakage makes results look BETTER, which is why it survives -- nobody
investigates a good number. These checks are mechanical rather than a matter of
discipline, because discipline is exactly what fails at 2am.

The strongest one here is the truncation test: build a match's features from
the full database, then delete every row at or after that match's start time
and build them again. If any feature drew on the future, the two disagree.
"""

from __future__ import annotations

import inspect
import pathlib
import re

import pytest

from valwr.features import build as fb
from valwr.features import player as pf
from valwr.rating.normalize import build_norms
from valwr.store import normalize, reference, schema, temporal

FEATURES_DIR = pathlib.Path(__file__).resolve().parent.parent / "valwr" / "features"


def make_match(mid, started, winner="Blue", map_name="Sunset", agent="Jett"):
    puuids = [f"b{i}" for i in range(5)] + [f"r{i}" for i in range(5)]
    return {
        "metadata": {"match_id": mid, "started_at": started,
                     "map": {"name": map_name}, "region": "na",
                     "queue": {"id": "competitive", "mode_type": "Standard"},
                     "season": {"short": "e11a5"}, "is_completed": True},
        "teams": [{"team_id": "Red", "won": winner == "Red", "rounds": {"won": 12}},
                  {"team_id": "Blue", "won": winner == "Blue", "rounds": {"won": 14}}],
        "rounds": [{"id": i, "ceremony": "CeremonyDefault", "winning_team": winner,
                    "stats": []} for i in range(20)],
        "kills": [],
        "players": [{"puuid": p, "name": p, "tag": "NA1",
                     "team_id": "Blue" if i < 5 else "Red",
                     "agent": {"name": agent}, "tier": {"id": 13 + i},
                     "party_id": None, "account_level": 100,
                     "stats": {"score": 4000 + i * 100, "kills": 15, "deaths": 14,
                               "assists": 5, "headshots": 10, "bodyshots": 20,
                               "legshots": 2,
                               "damage": {"dealt": 2800, "received": 2900}}}
                    for i, p in enumerate(puuids)],
    }


def ingest(conn, *matches):
    for m in matches:
        row, players, _ = normalize.parse_match(m)
        normalize.upsert_match(conn, row)
        normalize.upsert_players(conn, players)
    conn.commit()


@pytest.fixture
def conn(tmp_path):
    c = schema.connect(tmp_path / "t.db")
    schema.create_all(c)
    c.execute("INSERT INTO ref_agents (uuid,name,role) VALUES ('u1','Jett','Duelist')")
    c.execute("INSERT INTO ref_agents (uuid,name,role) VALUES ('u2','Omen','Controller')")
    c.commit()
    return c


# --- 1. the truncation audit -----------------------------------------

def test_feature_matrix_requires_an_explicit_training_cutoff(conn):
    """The orchestration layer must not silently fit norms on held-out rows.

    Per-match history queries can all be correctly time-gated while population
    normalisation still reaches into validation and test. Requiring the split
    boundary makes that failure mode explicit instead of defaulting to the
    latest match in the database.
    """
    ingest(conn,
           make_match("train", "2026-08-01T00:00:00Z"),
           make_match("test", "2026-08-05T00:00:00Z"))

    with pytest.raises(TypeError, match="norms_as_of"):
        fb.build_all(conn, verbose=False)


def test_feature_matrix_fits_population_stats_at_training_cutoff(conn, monkeypatch):
    ingest(conn,
           make_match("train", "2026-08-01T00:00:00Z"),
           make_match("validation", "2026-08-05T00:00:00Z"),
           make_match("test", "2026-08-09T00:00:00Z"))
    cutoff = normalize.parse_started_at("2026-08-05T00:00:00Z")
    seen: dict[str, int] = {}
    real_build_norms = fb.build_norms
    real_population_win_rate = temporal.population_win_rate

    def tracked_build_norms(c, as_of):
        seen["norms"] = as_of
        return real_build_norms(c, as_of)

    def tracked_population_win_rate(c, as_of):
        seen["prior"] = as_of
        return real_population_win_rate(c, as_of)

    monkeypatch.setattr(fb, "build_norms", tracked_build_norms)
    monkeypatch.setattr(temporal, "population_win_rate", tracked_population_win_rate)

    rows = fb.build_all(conn, norms_as_of=cutoff, verbose=False)

    assert rows
    assert seen == {"norms": cutoff, "prior": cutoff}

def test_features_are_identical_when_the_future_is_deleted(conn):
    """The core audit.

    Build a match's features from the full database, then delete everything at
    or after its start time and build again. Any feature reaching forward in
    time would change value.
    """
    days = [f"2026-08-{d:02d}T00:00:00Z" for d in range(1, 15)]
    ingest(conn, *[make_match(f"m{i}", d) for i, d in enumerate(days)])

    target = conn.execute(
        "SELECT * FROM matches ORDER BY started_at LIMIT 1 OFFSET 7").fetchone()
    as_of = target["started_at"]
    rows = conn.execute("SELECT * FROM match_players WHERE match_id=?",
                        (target["match_id"],)).fetchall()

    norms = build_norms(conn, as_of)
    prior = temporal.population_win_rate(conn, as_of)
    roles = reference.agent_roles(conn)
    full = fb.build_match(conn, dict(target), rows, norms, prior, roles)

    conn.execute("DELETE FROM match_players WHERE started_at >= ?", (as_of,))
    conn.execute("DELETE FROM matches WHERE started_at >= ?", (as_of,))
    conn.commit()
    truncated = fb.build_match(conn, dict(target), rows, norms, prior, roles)

    assert full is not None and truncated is not None
    for k, v in full.values.items():
        assert truncated.values[k] == pytest.approx(v), (
            f"feature {k} changed when the future was deleted: "
            f"{v} -> {truncated.values[k]} -- it was reading ahead")


def test_player_features_ignore_that_players_own_later_matches(conn):
    ingest(conn,
           make_match("past", "2026-08-01T00:00:00Z"),
           make_match("target", "2026-08-05T00:00:00Z"),
           make_match("future", "2026-08-09T00:00:00Z"))

    as_of = normalize.parse_started_at("2026-08-05T00:00:00Z")
    norms = build_norms(conn, as_of)
    p = pf.build(conn, "b0", as_of, "Sunset", "Jett", 13, 100, norms, 0.5)
    assert p.games == 1, "only the single prior match should be visible"


def test_history_at_exactly_as_of_is_excluded(conn):
    """Strict `<`, not `<=`. A match cannot inform a prediction about itself."""
    ingest(conn, make_match("m", "2026-08-01T00:00:00Z"))
    at = normalize.parse_started_at("2026-08-01T00:00:00Z")
    assert temporal.player_history(conn, "b0", at) == []
    assert len(temporal.player_history(conn, "b0", at + 1)) == 1


# --- 2. the chokepoint is respected ----------------------------------

def test_no_feature_module_queries_match_players_directly():
    """All history reads must go through store/temporal.py.

    One chokepoint that always demands `as_of` is far more reliable than
    remembering a time filter in twenty call sites -- but only if nothing
    bypasses it, which is what this checks.
    """
    offenders = []
    for path in FEATURES_DIR.glob("*.py"):
        code = path.read_text(encoding="utf-8")
        code = re.sub(r'"{3}.*?"{3}', "", code, flags=re.S)   # drop docstrings
        code = re.sub(r"#.*", "", code)                        # drop comments
        if re.search(r"\bFROM\s+match_players\b", code, re.I):
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders} query match_players directly; go through store/temporal.py")


def test_every_temporal_function_requires_as_of():
    """No default for `as_of` anywhere -- omitting it must be impossible."""
    for name, fn in vars(temporal).items():
        if not inspect.isfunction(fn) or name.startswith("_"):
            continue
        params = inspect.signature(fn).parameters
        if "as_of" in params:
            assert params["as_of"].default is inspect.Parameter.empty, (
                f"temporal.{name} gives as_of a default; it must stay required")


def test_match_roster_cannot_return_outcome_columns(conn):
    """The target match's roster is legitimate input; its statistics are not.

    This test exists because the direct-query audit caught build.py doing
    `SELECT *` on the target match, which hands back kills, damage and `won`
    alongside the roster.
    """
    ingest(conn, make_match("m", "2026-08-01T00:00:00Z"))
    rows = temporal.match_roster(conn, "m")
    assert rows, "roster should not be empty"
    returned = set(rows[0].keys())
    forbidden = {"score", "kills", "deaths", "assists", "damage_dealt",
                 "damage_taken", "won", "started_at", "kast_rounds",
                 "first_bloods", "clutches"}
    assert not (returned & forbidden), (
        f"roster leaked outcome columns: {sorted(returned & forbidden)}")


# --- 3. antisymmetry --------------------------------------------------

def test_swapping_teams_negates_every_feature(conn):
    """Otherwise the model learns 'the first team wins more often', which is an
    artefact of the order rows were written in, not a fact about the game."""
    ingest(conn, make_match("a", "2026-08-01T00:00:00Z"),
           make_match("b", "2026-08-05T00:00:00Z"))
    target = conn.execute("SELECT * FROM matches WHERE match_id='b'").fetchone()
    rows = conn.execute("SELECT * FROM match_players WHERE match_id='b'").fetchall()
    norms = build_norms(conn, target["started_at"])

    mf = fb.build_match(conn, dict(target), rows, norms, 0.5,
                        reference.agent_roles(conn))
    mirrored = fb.mirror(mf)

    for k, v in mf.values.items():
        assert mirrored.values[k] == pytest.approx(-v)
    assert mirrored.target == 1 - mf.target


def test_identical_teams_produce_an_all_zero_vector(conn):
    """Proof the representation carries no side-specific bias of its own."""
    ingest(conn, make_match("m", "2026-08-01T00:00:00Z"))
    conn.execute("UPDATE match_players SET tier = 13")
    conn.commit()
    target = conn.execute("SELECT * FROM matches").fetchone()
    rows = conn.execute("SELECT * FROM match_players").fetchall()
    norms = build_norms(conn, target["started_at"])

    mf = fb.build_match(conn, dict(target), rows, norms, 0.5,
                        reference.agent_roles(conn))
    for k, v in mf.values.items():
        assert v == pytest.approx(0.0), f"{k} is {v} for identical teams"


# --- 4. shrinkage is actually applied --------------------------------

def test_a_three_game_winner_is_not_treated_as_a_certainty(conn):
    ingest(conn, *[make_match(f"w{i}", f"2026-08-0{i + 1}T00:00:00Z", winner="Blue")
                   for i in range(3)])
    as_of = normalize.parse_started_at("2026-08-10T00:00:00Z")
    norms = build_norms(conn, as_of)
    p = pf.build(conn, "b0", as_of, "Sunset", "Jett", 13, 100, norms, 0.5)
    assert p.values["wr"] < 0.65, (
        f"a 3-0 record shrank to {p.values['wr']:.3f}; unshrunk it would be 1.0")


def test_a_player_with_no_history_gets_neutral_not_zero(conn):
    """Zero would assert 'never wins', a claim the data does not make."""
    empty = pf.empty()
    assert empty["rating"] == 1.0
    assert empty["wr"] == 0.0


def test_history_rows_carry_everything_the_rating_needs(conn):
    """Regression: the history SELECT predated the Phase 3 component columns,
    so rounds_played came back missing. per_round_rates() then divided by a
    missing denominator, returned None for every rate, and rate_performance()
    fell through to its default -- leaving rating, ACS, ADR, KAST and
    first-blood rate identically zero for all 2,168 rows of the first built
    matrix. No exception, no warning; the signal was just gone.
    """
    from valwr.rating.components import per_round_rates
    from valwr.rating.rating import rate_performance

    ingest(conn, make_match("past", "2026-08-01T00:00:00Z"),
           make_match("later", "2026-08-05T00:00:00Z"))
    as_of = normalize.parse_started_at("2026-08-05T00:00:00Z")
    rows = temporal.player_history(conn, "b0", as_of)
    assert rows, "expected prior history"

    d = dict(rows[0])
    for col in ("rounds_played", "kast_rounds", "first_bloods", "score",
                "damage_dealt"):
        assert col in d, f"history is missing {col}; the rating cannot be computed"

    assert d["rounds_played"], "rounds_played must be a usable denominator"
    assert per_round_rates(d)["acs"] is not None
    assert rate_performance(d, build_norms(conn, as_of)) is not None


def test_built_features_are_not_all_zero(conn):
    """A whole feature family collapsing to zero must fail loudly, not pass."""
    ingest(conn, *[make_match(f"m{i}", f"2026-08-{i + 1:02d}T00:00:00Z",
                              winner="Blue" if i % 2 else "Red")
                   for i in range(10)])
    target = conn.execute(
        "SELECT * FROM matches ORDER BY started_at DESC LIMIT 1").fetchone()
    rows = temporal.match_roster(conn, target["match_id"])
    norms = build_norms(conn, target["started_at"])
    mf = fb.build_match(conn, dict(target), rows, norms, 0.5,
                        reference.agent_roles(conn))

    perf = [k for k in mf.values if any(t in k for t in ("rating", "acs", "adr", "kast"))]
    assert perf, "expected performance features"
    assert any(mf.values[k] != 0.0 for k in perf), (
        "every performance feature is zero -- the rating pipeline is dead")
