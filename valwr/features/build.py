"""Assemble the feature matrix.

Two structural decisions worth stating, because both are easy to get wrong in
ways that inflate results.

**Antisymmetry.** The model sees `blue - red` differences, never the two teams
side by side. Swapping the teams therefore negates the vector exactly, and the
model cannot learn "the first team wins more often" -- an artefact of the order
rows were written in, not a fact about VALORANT. Asserted in
test/test_leakage.py rather than assumed.

**Norms.** The rating's z-scores need population statistics, and those are
derived from data. They are fitted once, using only matches before the
train/validation boundary, and then applied everywhere -- the standard
fit-on-train pattern. Within the training period this means a match may be
normalised using statistics that include slightly later training matches. That
is accepted practice for population-level scalers and carries no outcome
information, but it is a real subtlety and it is recorded here rather than
buried.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from valwr.features import context as ctx
from valwr.features import player as pf
from valwr.features import team as tf
from valwr.rating.normalize import Norms, build_norms
from valwr.store import reference, temporal

TEAM_A, TEAM_B = "Blue", "Red"


@dataclass
class MatchFeatures:
    match_id: str
    started_at: int
    target: int                       # 1 if TEAM_A won
    values: dict[str, float]
    context: dict[str, str]
    coverage: int                     # players with history, 0-10


def feature_names() -> list[str]:
    return [f"d_{n}" for n in tf.feature_names()]


def _team_rows(rows: list[sqlite3.Row], team: str):
    return [r for r in rows if r["team"] == team]


def build_match(conn, match: dict, rows: list[sqlite3.Row], norms: Norms,
                prior_rate: float, roles: dict[str, str | None]) -> MatchFeatures | None:
    """Features for one match, from data strictly before it started."""
    as_of = match["started_at"]
    if match.get("winner") not in (TEAM_A, TEAM_B):
        return None                    # draws and unresolved matches carry no target

    sides = {}
    coverage = 0
    for team in (TEAM_A, TEAM_B):
        members = _team_rows(rows, team)
        if not members:
            return None
        feats, agents, parties = [], [], []
        for r in members:
            p = pf.build(conn, r["puuid"], as_of, match["map"], r["agent"],
                         r["tier"], r["account_level"], norms, prior_rate,
                         roles=roles)
            feats.append(p)
            agents.append(r["agent"])
            parties.append(r["party_id"])
            coverage += int(p.has_history)
        sides[team] = tf.build(feats, agents, parties, roles)

    a, b = sides[TEAM_A], sides[TEAM_B]
    values = {f"d_{k}": a[k] - b.get(k, 0.0) for k in a}

    return MatchFeatures(
        match_id=match["match_id"],
        started_at=as_of,
        target=int(match["winner"] == TEAM_A),
        values=values,
        context=ctx.build(match),
        coverage=coverage,
    )


def mirror(mf: MatchFeatures) -> MatchFeatures:
    """Swap the teams. Every difference negates and the target flips.

    Used by the symmetry test, and available for mirror-augmenting training
    data if the difference representation is ever replaced.
    """
    return MatchFeatures(
        match_id=mf.match_id, started_at=mf.started_at,
        target=1 - mf.target,
        values={k: -v for k, v in mf.values.items()},
        context=mf.context, coverage=mf.coverage,
    )


def build_all(conn: sqlite3.Connection, norms_as_of: int | None = None,
              min_coverage: int = 0, limit: int | None = None,
              verbose: bool = True) -> list[MatchFeatures]:
    """Build features for every resolved match.

    `min_coverage` filters on how many of the ten players have any prior
    history. A match where nobody is known contributes only population means
    and teaches the model nothing.
    """
    matches = conn.execute(
        "SELECT * FROM matches WHERE winner IN (?,?) ORDER BY started_at"
        + (" LIMIT ?" if limit else ""),
        (TEAM_A, TEAM_B, limit) if limit else (TEAM_A, TEAM_B),
    ).fetchall()
    if not matches:
        return []

    cutoff = norms_as_of or (matches[-1]["started_at"] + 1)
    norms = build_norms(conn, cutoff)
    prior_rate = temporal.population_win_rate(conn, cutoff)
    roles = reference.agent_roles(conn)
    if verbose:
        print(f"  norms from {norms.rows_used:,} rows, "
              f"population win rate {prior_rate:.3f}")

    out, started, skipped = [], time.time(), 0
    for i, m in enumerate(matches, 1):
        rows = temporal.match_roster(conn, m["match_id"])
        mf = build_match(conn, dict(m), rows, norms, prior_rate, roles)
        if mf is None or mf.coverage < min_coverage:
            skipped += 1
            continue
        out.append(mf)
        if verbose and i % 500 == 0:
            rate = i / max(time.time() - started, 1e-9)
            print(f"  {i:,}/{len(matches):,} matches  ({rate:.0f}/s, {len(out):,} kept)")

    if verbose:
        print(f"  built {len(out):,}, skipped {skipped:,} "
              f"(coverage < {min_coverage} or unresolved)")
    return out


def to_frame(rows: list[MatchFeatures]):
    import pandas as pd
    recs = []
    for r in rows:
        d = {"match_id": r.match_id, "started_at": r.started_at,
             "target": r.target, "coverage": r.coverage}
        d.update(r.context)
        d.update(r.values)
        recs.append(d)
    return pd.DataFrame.from_records(recs)


def main(argv=None) -> int:
    import argparse
    from valwr import config
    from valwr.store import schema

    ap = argparse.ArgumentParser(prog="valwr.features.build")
    ap.add_argument("--min-coverage", type=int, default=5,
                    help="require at least N of 10 players to have prior history")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="data/features.parquet")
    args = ap.parse_args(argv)

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    rows = build_all(conn, min_coverage=args.min_coverage, limit=args.limit)
    if not rows:
        print("no matches met the criteria")
        return 1

    df = to_frame(rows)
    out = s.database_path.parent / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}  {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"target balance: {df['target'].mean():.3f}  (0.5 = balanced)")
    miss = df.isna().mean().sort_values(ascending=False)
    bad = miss[miss > 0]
    print(f"columns with missing values: {len(bad)}")
    for k, v in bad.head(5).items():
        print(f"  {k}: {v:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
