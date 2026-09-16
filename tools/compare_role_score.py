"""The old score against the per-role one, on the same teams.

    python tools/compare_role_score.py [--teams 1500] [--index PATH]

Both scores are measured on identical test-period teams, against an identical
definition of "played best" -- the ten-part match rating from rating/rating.py.
Anything else would be comparing two measurements rather than two scores.

The success criteria are fixed in docs/SCORE-SPEC.md, written down before the
per-role score existed so they could not be adjusted afterwards:

  1. overall top-1 must not fall by more than one standard error
  2. Initiator and Controller rho must rise from +0.129 and +0.172
  3. per-role numbers are reported whether or not they are flattering

This tool prints the verdict against those criteria and does not soften it.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

from valwr import config
from valwr.model import split
from valwr.rating import potential as P
from valwr.rating import role_score as R
from valwr.rating import roleindex, roles
from valwr.rating.normalize import build_norms
from valwr.rating.rating import rate_performance
from valwr.store import schema

ROOT = Path(__file__).resolve().parent.parent
TEAM_SIZE = 5

# The baseline this has to beat, from the run recorded in the spec.
BASELINE_RHO = {"Duelist": 0.191, "Sentinel": 0.171,
                "Controller": 0.172, "Initiator": 0.129}


def _harness():
    """`top1`, `spearman` and `role_breakdown` from the validation tool.

    Imported by path rather than duplicated: two copies of a metric drift, and
    the whole point here is that both scores are measured the same way.
    """
    spec = importlib.util.spec_from_file_location(
        "validate_potential", ROOT / "tools" / "validate_potential.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="compare_role_score")
    ap.add_argument("--teams", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--index", default=str(roleindex.INDEX_PATH),
                    help="which role index to measure (default: the shipped one)")
    args = ap.parse_args(argv)

    h = _harness()
    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    b = split.compute(conn)
    norms = build_norms(conn, b.train_end)
    old = P.PerfIndex.load()
    index_path = Path(args.index)
    if not index_path.exists():
        print(f"no {index_path}; run tools/build_role_index.py first")
        return 1
    new = roleindex.RoleIndex.load(index_path)
    roles_by_agent = R.roles_by_agent(conn)
    print(f"old index: {old.n:,} samples   new index: {new.n:,} samples")
    print(f"evaluating on the TEST period (after {b.val_end})\n")

    rows = conn.execute(
        "SELECT * FROM match_players WHERE started_at >= ? AND rounds_played > 0",
        (b.val_end,)).fetchall()
    by_team = defaultdict(list)
    for r in rows:
        by_team[(r["match_id"], r["team"])].append(dict(r))
    full = [v for v in by_team.values() if len(v) == TEAM_SIZE]
    rng = random.Random(args.seed)
    rng.shuffle(full)

    started = time.time()
    teams = []
    for squad in full:
        as_of = squad[0]["started_at"]
        scored = []
        for row in squad:
            role = roles_by_agent.get(row["agent"])
            comp_old = P.measure(conn, row["puuid"], as_of, row["map"], norms)
            comp_new = R.measure(conn, row["puuid"], as_of, row["map"], role,
                                 roles_by_agent,
                                 role_means=new.role_means.get(role or "?"),
                                 agent=row["agent"])
            actual = rate_performance(row, norms)
            if comp_old is None or comp_new is None or actual is None:
                break
            raw_new = new.composite(comp_new, roles.weights_for(role, row["agent"]))
            if raw_new is None:
                break
            scored.append({"raw": old.composite(comp_old), "new": raw_new,
                           "acs": comp_old.acs, "actual": actual.value,
                           "role": role or "?"})
        if len(scored) == TEAM_SIZE:
            teams.append(scored)
        if len(teams) >= args.teams:
            break
    if len(teams) < 100:
        print(f"only {len(teams)} fully-known teams; not enough to measure")
        return 1

    flat = [p for t in teams for p in t]
    n = len(teams)
    se = (0.2 * 0.8 / n) ** 0.5 * 100
    print(f"{n:,} teams / {len(flat):,} players ({time.time() - started:.0f}s)\n")

    print("=" * 68)
    print("PICKING THE BEST PLAYER OUT OF FIVE   (chance = 20.0%)")
    print("=" * 68)
    rng2 = random.Random(args.seed + 1)
    results = {}
    for label, key in (("the old score", lambda p: p["raw"]),
                       ("the per-role score", lambda p: p["new"]),
                       ("career ACS alone", lambda p: p["acs"]),
                       ("shuffled (control)", lambda p: rng2.random())):
        acc, _ = h.top1(teams, key)
        results[label] = acc
        print(f"  {label:<24}{acc * 100:>7.1f}%{(acc - 0.2) * 100:>+9.1f} vs chance")
    print(f"\n  standard error on {n:,} teams: +/-{se:.1f} points")

    print("\n" + "=" * 68)
    print("BY ROLE")
    print("=" * 68)
    print(f"  {'role':<12}{'rho old':>9}{'rho new':>9}{'change':>9}"
          f"{'named old':>11}{'named new':>11}")
    print("  " + "-" * 62)
    old_rows = {r["role"]: r for r in h.role_breakdown(teams, flat)}
    new_rows = {r["role"]: r
                for r in h.role_breakdown(teams, flat, key=lambda p: p["new"])}

    verdicts = []
    for role in ("Duelist", "Sentinel", "Controller", "Initiator"):
        a, c = old_rows.get(role), new_rows.get(role)
        if not a or not c:
            continue
        print(f"  {role:<12}{a['rho']:>+9.3f}{c['rho']:>+9.3f}"
              f"{c['rho'] - a['rho']:>+9.3f}"
              f"{a['recall'] * 100:>10.1f}%{c['recall'] * 100:>10.1f}%")
        verdicts.append((role, a["rho"], c["rho"]))

    print("\n" + "=" * 68)
    print("AGAINST THE CRITERIA WRITTEN DOWN BEFOREHAND")
    print("=" * 68)
    drop = (results["the old score"] - results["the per-role score"]) * 100
    ok_overall = drop <= se
    print(f"  1. overall top-1 within one standard error: "
          f"{'PASS' if ok_overall else 'FAIL'}"
          f"  ({results['the old score'] * 100:.1f}% -> "
          f"{results['the per-role score'] * 100:.1f}%, "
          f"tolerance {se:.1f})")
    for role in ("Initiator", "Controller"):
        row = next(((a, c) for r, a, c in verdicts if r == role), None)
        if row is None:
            continue
        was, got = row
        # Against this run's own old-score figure, not the one recorded in the
        # spec. The recorded number came from a different sample of teams --
        # this tool keeps only teams where BOTH scores can be computed -- so
        # comparing across the two would read sampling noise as progress.
        print(f"  2. {role} rho improves and clears +0.200: "
              f"{'PASS' if got > 0.2 and got > was else 'FAIL'}  "
              f"(old {was:+.3f} -> new {got:+.3f} on these teams; "
              f"{BASELINE_RHO[role]:+.3f} recorded on the spec's sample)")
    print("  3. per-role numbers reported above, flattering or not.")

    worse = [r for r, a, c in verdicts if c < a - 0.01]
    if worse:
        print(f"\n  Worse than the old score for: {', '.join(worse)}. That is a"
              "\n  result, not a rounding error -- read it before shipping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
