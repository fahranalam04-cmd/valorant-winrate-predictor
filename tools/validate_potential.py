"""Does the potential score actually predict who plays best?

Measured on the **test** period only -- matches after the validation boundary,
never seen while the index was fitted. Touches no model bundle and writes
nothing.

The headline is deliberately the question the live chart claims to answer:
inside a real five-player team, how often is the top-ranked player the one who
actually had the best game? Chance is exactly 20%. Spearman correlation is
reported too, but it is the softer number -- ranking a whole population is an
easier task than picking the best of five.

Two controls keep this honest:

  * a **shuffled** run, which must land at 20%. If it does not, the harness is
    measuring itself rather than the score.
  * two **baselines** -- rank by career ACS, and rank by the existing rating
    alone. The composite has to beat both, or its extra components are
    decoration and should be dropped rather than shipped.

    python tools/validate_potential.py [--teams 1500]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")

from valwr import config
from valwr.model import split
from valwr.rating import potential as P
from valwr.rating.normalize import build_norms
from valwr.rating.rating import rate_performance
from valwr.store import schema

TEAM_SIZE = 5


def spearman(xs, ys) -> float:
    """Rank correlation, without pulling in scipy for one number."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = shared
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def top1(teams, key) -> tuple[float, int]:
    """How often the highest `key` is also the best actual performance."""
    hits = 0
    for team in teams:
        best_pred = max(team, key=lambda p: key(p))
        best_real = max(p["actual"] for p in team)
        if abs(best_pred["actual"] - best_real) < 1e-12:
            hits += 1
    return hits / len(teams), hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="validate_potential")
    ap.add_argument("--teams", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--period", choices=("val", "test"), default="test",
                    help="tune on val; touch test once, at the end")
    ap.add_argument("--sweep", action="store_true",
                    help="compare candidate weightings (validation only)")
    args = ap.parse_args(argv)

    if args.sweep and args.period == "test":
        print("refusing to sweep weights on the test set -- that is selecting "
              "on the measurement.\nRun with --period val.")
        return 2

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    b = split.compute(conn)
    index = P.PerfIndex.load()
    norms = build_norms(conn, b.train_end)
    print(f"index fitted on {index.n:,} training samples")

    if args.period == "val":
        window = "started_at >= ? AND started_at < ?"
        params = (b.train_end, b.val_end)
        print(f"evaluating on the VALIDATION period "
              f"({b.train_end}..{b.val_end})\n")
    else:
        window = "started_at >= ? AND ? = ?"
        params = (b.val_end, 1, 1)
        print(f"evaluating on the TEST period (after {b.val_end})\n")

    rows = conn.execute(
        f"SELECT * FROM match_players WHERE {window} AND rounds_played > 0",
        params).fetchall()
    by_team = defaultdict(list)
    for r in rows:
        by_team[(r["match_id"], r["team"])].append(dict(r))
    full = [v for v in by_team.values() if len(v) == TEAM_SIZE]
    print(f"{len(rows):,} test player-rows -> {len(full):,} complete teams")

    rng = random.Random(args.seed)
    rng.shuffle(full)

    started = time.time()
    teams, scanned = [], 0
    for squad in full:
        scanned += 1
        as_of = squad[0]["started_at"]
        scored = []
        for row in squad:
            c = P.measure(conn, row["puuid"], as_of, row["map"], norms)
            if c is None:
                break
            actual = rate_performance(row, norms)
            if actual is None:
                break
            scored.append({"c": c, "raw": index.composite(c),
                           "actual": actual.value})
        if len(scored) == TEAM_SIZE:
            teams.append(scored)
        if len(teams) >= args.teams:
            break
        if scanned % 2000 == 0:
            print(f"  scanned {scanned:,}, kept {len(teams):,} "
                  f"({time.time() - started:.0f}s)")

    if len(teams) < 100:
        print(f"only {len(teams)} fully-known teams; not enough to measure")
        return 1

    flat = [p for t in teams for p in t]
    print(f"\n{len(teams):,} teams where all five had prior history "
          f"({len(flat):,} players, {time.time() - started:.0f}s)\n")

    n = len(teams)
    se = (0.2 * 0.8 / n) ** 0.5 * 100

    if args.sweep:
        # Candidate weightings, evaluated on validation only. Measuring is
        # cheap here -- the components are already computed, so a weighting is
        # just a dot product.
        def raw_with(c, w):
            return sum(wt * index.z(k, getattr(c, k)) for k, wt in w.items())

        candidates = {
            "acs only": {"acs": 1.0},
            "rating only": {"rating": 1.0},
            "kd only": {"kd": 1.0},
            "acs + rating": {"acs": 0.5, "rating": 0.5},
            "acs + kd": {"acs": 0.5, "kd": 0.5},
            "acs-led + map": {"acs": 0.45, "rating": 0.25, "kd": 0.15,
                              "map_edge": 0.15},
            "acs-led, no map": {"acs": 0.5, "rating": 0.3, "kd": 0.2},
            "even quarters": {k: 0.25 for k in P.WEIGHTS},
            "shipped weights": dict(P.WEIGHTS),
        }
        print("=" * 68)
        print(f"WEIGHT SWEEP on VALIDATION   ({n:,} teams, chance 20.0%, "
              f"SE +/-{se:.1f})")
        print("=" * 68)
        scored = []
        for label, w in candidates.items():
            a, _ = top1(teams, lambda p, w=w: raw_with(p["c"], w))
            r = spearman([raw_with(p["c"], w) for p in flat],
                         [p["actual"] for p in flat])
            scored.append((a, r, label))
        for a, r, label in sorted(scored, reverse=True):
            print(f"  {label:<20}{a * 100:>8.1f}%   spearman {r:+.3f}")
        print("\n  Pick the best here, set it in valwr/rating/potential.py,")
        print("  then run once with --period test.")
        return 0

    acc, hits = top1(teams, lambda p: p["raw"])
    acs_acc, _ = top1(teams, lambda p: p["c"].acs)
    rat_acc, _ = top1(teams, lambda p: p["c"].rating)
    rng2 = random.Random(args.seed + 1)
    shuf_acc, _ = top1(teams, lambda p: rng2.random())

    print("=" * 68)
    print("PICKING THE BEST PLAYER OUT OF FIVE   (chance = 20.0%)")
    print("=" * 68)
    print(f"  {'ranked by':<28}{'top-1':>9}{'vs chance':>12}")
    print("  " + "-" * 64)
    for label, a in (("potential score", acc), ("existing rating alone", rat_acc),
                     ("career ACS alone", acs_acc), ("shuffled (control)", shuf_acc)):
        print(f"  {label:<28}{a * 100:>8.1f}%{(a - 0.2) * 100:>+11.1f}")
    print(f"\n  standard error on {n:,} teams: +/-{se:.1f} points")
    print(f"  potential beats chance by "
          f"{(acc - 0.2) * 100 / se:.1f} standard errors")

    rho = spearman([p["raw"] for p in flat], [p["actual"] for p in flat])
    rho_r = spearman([p["c"].rating for p in flat], [p["actual"] for p in flat])
    rho_a = spearman([p["c"].acs for p in flat], [p["actual"] for p in flat])
    print(f"\n  Spearman correlation with actual performance:")
    print(f"    potential score        {rho:+.3f}")
    print(f"    existing rating alone  {rho_r:+.3f}")
    print(f"    career ACS alone       {rho_a:+.3f}")

    print("\n  Reminder: this measures ranking within a team, not whether any "
          "\n  individual number is right. Performance is noisy and strongly "
          "\n  mean-reverting; a modest lift over chance is the realistic "
          "\n  ceiling here, not a disappointment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
