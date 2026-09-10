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

    python tools/validate_potential.py [--teams 1500] [--json]
    python tools/validate_potential.py --flag [--json]

`--json` merges the measurement into reports/player_models.json, which
tools/model_metrics.py tabulates beside the win-prediction models.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

JSON_OUT = Path(__file__).resolve().parent.parent / "reports" / "player_models.json"


def write_json(key: str, payload: dict) -> None:
    """Merge one measurement into the shared file, keeping the other."""
    data = (json.loads(JSON_OUT.read_text(encoding="utf-8"))
            if JSON_OUT.exists() else {})
    data[key] = payload
    JSON_OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n  wrote {key} to {JSON_OUT}")


def ranking_scores(labels, scores) -> tuple[float, float]:
    """(AUC, average precision) of `scores` against binary `labels`."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    return (float(roc_auc_score(labels, scores)),
            float(average_precision_score(labels, scores)))

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


def flag_check(conn, b, index, norms, seed: int = 42, as_json: bool = False) -> int:
    """Do flagged players actually outperform their lobby?

    There is no smurf label in this data, so the flag cannot be validated
    against ground truth. This measures the nearest honest proxy: within a
    complete ten-player lobby, how often does a flagged player finish in the
    top third by actual performance? The base rate is 33.3% by construction,
    so anything at or below that means the flag is decoration.
    """
    import random
    from collections import defaultdict
    from valwr.rating.rating import rate_performance

    print(f"flag cut z>={index.flag_cut:.2f}   (test period, after {b.val_end})")
    rows = conn.execute("SELECT * FROM match_players WHERE started_at >= ? "
                        "AND rounds_played > 0", (b.val_end,)).fetchall()
    lobbies = defaultdict(list)
    for r in rows:
        lobbies[r["match_id"]].append(dict(r))
    full = [v for v in lobbies.values() if len(v) == 10]
    random.Random(seed).shuffle(full)
    print(f"{len(full):,} complete ten-player lobbies in the test period")

    flag_hits = flag_n = plain_hits = plain_n = scanned = 0
    zs: list[float] = []
    tops: list[int] = []
    for squad in full:
        as_of = squad[0]["started_at"]
        scored = []
        for r in squad:
            c = P.measure(conn, r["puuid"], as_of, r["map"], norms)
            actual = rate_performance(r, norms)
            if c is None or actual is None:
                continue
            verdict = P.above_rank(index, c)
            scored.append((verdict.flagged, actual.value,
                           index.z("rating", c.rating)))
        if len(scored) < 6:
            continue
        cut = sorted((v for _, v, _ in scored),
                     reverse=True)[: max(1, len(scored) // 3)][-1]
        for flagged, v, z in scored:
            top = v >= cut
            zs.append(z)
            tops.append(int(top))
            if flagged:
                flag_n += 1
                flag_hits += top
            else:
                plain_n += 1
                plain_hits += top
        scanned += 1
        if flag_n >= 400 or scanned >= 4000:
            break

    if flag_n < 50:
        print(f"only {flag_n} flagged players found; not enough to measure")
        return 1

    def pct(h, n):
        return 100.0 * h / n if n else float("nan")

    se = (0.33 * 0.67 / max(flag_n, 1)) ** 0.5 * 100
    delta = pct(flag_hits, flag_n) - pct(plain_hits, plain_n)
    print()
    print(f"  scanned {scanned:,} lobbies")
    print(f"  flagged players : {flag_n:>6,}   top-third rate "
          f"{pct(flag_hits, flag_n):5.1f}%")
    print(f"  everyone else   : {plain_n:>6,}   top-third rate "
          f"{pct(plain_hits, plain_n):5.1f}%")
    print(f"  base rate 33.3% by construction; SE on the flagged group "
          f"~{se:.1f} points")
    print(f"  difference      : {delta:+.1f} points  ({delta / se:+.1f} SE)")

    # The flag as a classifier of "finished in the top third". Precision is
    # the flagged group's top-third rate above; recall is how many of all
    # top-third finishers it caught, which is small by design -- it fires on
    # about one player in twenty, so it can never catch most strong games.
    top_all = flag_hits + plain_hits
    precision = flag_hits / flag_n
    recall = flag_hits / top_all if top_all else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    auc, ap = ranking_scores(tops, zs)
    print(f"  as a classifier : precision {precision:.3f}  recall {recall:.3f}"
          f"  F1 {f1:.3f}")
    print(f"  rating z alone  : AUC {auc:.3f}  PR-AUC {ap:.3f}  "
          f"(base rate {top_all / (flag_n + plain_n):.3f})")
    if as_json:
        write_json("flag", {
            "period": "test", "lobbies": scanned, "players": flag_n + plain_n,
            "flagged": flag_n, "flag_rate": flag_n / (flag_n + plain_n),
            "base_rate": top_all / (flag_n + plain_n),
            "precision": precision, "recall": recall, "f1": f1,
            "auc": auc, "pr_auc": ap})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="validate_potential")
    ap.add_argument("--teams", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--period", choices=("val", "test"), default="test",
                    help="tune on val; touch test once, at the end")
    ap.add_argument("--sweep", action="store_true",
                    help="compare candidate weightings (validation only)")
    ap.add_argument("--write-index", action="store_true",
                    help="store the measured top-1 rate in the index, so the "
                         "live view can state it without a hardcoded literal")
    ap.add_argument("--flag", action="store_true",
                    help="measure whether the above-rank flag predicts "
                         "outperformance")
    ap.add_argument("--json", action="store_true",
                    help=f"merge the result into {JSON_OUT.name}")
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

    if args.flag:
        return flag_check(conn, b, index, norms, as_json=args.json)

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

    if args.write_index:
        # The live view used to print "30.5%" as a literal, which went silently
        # wrong at the next retrain with nothing to catch it. The number now
        # travels with the index that produced it.
        import dataclasses
        updated = dataclasses.replace(index, top1_rate=round(acc, 4))
        P.INDEX_PATH.write_text(updated.to_json(), encoding="utf-8")
        print()
        print(f"  wrote top1_rate={acc:.4f} to {P.INDEX_PATH}")

    rho = spearman([p["raw"] for p in flat], [p["actual"] for p in flat])
    rho_r = spearman([p["c"].rating for p in flat], [p["actual"] for p in flat])
    rho_a = spearman([p["c"].acs for p in flat], [p["actual"] for p in flat])
    print("\n  Spearman correlation with actual performance:")
    print(f"    potential score        {rho:+.3f}")
    print(f"    existing rating alone  {rho_r:+.3f}")
    print(f"    career ACS alone       {rho_a:+.3f}")

    # As a classifier: every player is labelled by whether they had their
    # team's best game. Picking one player per team, a top pick is right
    # exactly when it is the best, so its precision and recall are the same
    # number -- the top-1 rate. AUC and average precision score the ranking of
    # every player rather than only the pick.
    labels = [int(abs(p["actual"] - max(q["actual"] for q in t)) < 1e-12)
              for t in teams for p in t]
    rng3 = random.Random(args.seed + 2)
    rankers = []
    for label, a, scores in (
            ("potential score", acc, [p["raw"] for p in flat]),
            ("existing rating alone", rat_acc, [p["c"].rating for p in flat]),
            ("career ACS alone", acs_acc, [p["c"].acs for p in flat]),
            ("shuffled (control)", shuf_acc, [rng3.random() for _ in flat])):
        auc, ap_ = ranking_scores(labels, scores)
        rankers.append({"name": label, "top1": a, "auc": auc, "pr_auc": ap_})
    base = sum(labels) / len(labels)
    print(f"\n  Against 'had their team's best game' (base rate {base:.3f}):")
    for r in rankers:
        print(f"    {r['name']:<24} top pick right {r['top1'] * 100:5.1f}%"
              f"  AUC {r['auc']:.3f}  PR-AUC {r['pr_auc']:.3f}")
    if args.json:
        write_json("potential", {"period": args.period, "teams": n,
                                 "positive_rate": base, "rankers": rankers})

    print("\n  Reminder: this measures ranking within a team, not whether any "
          "\n  individual number is right. Performance is noisy and strongly "
          "\n  mean-reverting; a modest lift over chance is the realistic "
          "\n  ceiling here, not a disappointment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
