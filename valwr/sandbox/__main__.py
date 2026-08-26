"""Sandbox CLI: python -m valwr.sandbox <command>

    list                              scenarios and categories
    run    --mode static|variance|both --scenario NAME|all
    sweep  --feature rating           single-factor curve
    grid   --pair rating,wr_map       pairwise interaction grid
    benchmark                         freeze the static catalog
    compare [path]                    diff a saved benchmark against now

Fully offline: synthetic data in an in-memory database, and a model bundle
loaded from disk. It never calls an API, reads the real database, or touches a
real player's history.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from valwr.sandbox import benchmark as bm
from valwr.sandbox import predictor as pred
from valwr.sandbox import report, runner, scenarios, sweeps
from valwr.sandbox.predictor import MissingModel

REPORTS = Path(__file__).resolve().parent.parent.parent / "reports" / "sandbox"


def _catalog(include_generated: bool) -> list:
    return scenarios.CATALOG + (sweeps.generated() if include_generated else [])


def _select(name: str, include_generated: bool) -> list:
    catalog = _catalog(include_generated)
    if name == "all":
        return catalog
    by_name = {s.name: s for s in catalog}
    if name in by_name:
        return [by_name[name]]
    matches = [s for s in catalog if s.category == name]
    if matches:
        return matches
    raise SystemExit(f"unknown scenario or category {name!r}; "
                     f"try `python -m valwr.sandbox list`")


def cmd_list(args) -> int:
    print(f"{len(scenarios.CATALOG)} curated scenarios\n")
    for category, n in sorted(scenarios.categories().items()):
        print(f"  {category:<16} {n:>4}")
        if args.verbose:
            for s in scenarios.CATALOG:
                if s.category == category:
                    print(f"      {s.name}")
    gen = sweeps.generated()
    print(f"\n  {'generated':<16} {len(gen):>4}   "
          f"(single-factor, pairwise, context -- use --generated to include)")
    print(f"\n  total available: {len(scenarios.CATALOG) + len(gen)}")
    missing = sweeps.missing_features()
    print(f"\n  production features covered: "
          f"{len(sweeps.covered_features())}/{len(sweeps.covered_features()) + len(missing)}")
    if missing:
        print(f"  NOT COVERED: {sorted(missing)}")
    return 0


def cmd_run(args) -> int:
    bundle = pred.load_bundle()
    predictors = pred.predictors(bundle, args.model)
    selected = _select(args.scenario, args.generated)
    single = len(selected) == 1

    exit_code = 0
    for p in predictors:
        if len(predictors) > 1:
            print(f"\n{'=' * 70}\nMODEL: {p.name}\n{'=' * 70}")
        results = runner.run_many(selected, p, bundle,
                                  with_mirror=not args.no_mirror)
        variances: dict = {}
        if args.mode in ("variance", "both"):
            for s in selected:
                variances[s.name] = runner.run_variance(
                    s, p, bundle, samples=args.samples, seed=args.seed)

        if single:
            v = variances.get(selected[0].name)
            print(report.scenario_detail(selected[0], results[0], v))
        else:
            print(report.batch_table(results, variances))
            print(report.highlights(results, variances))

        if args.strict and any(r.expectation_met is False for r in results):
            exit_code = 1
    return exit_code


def cmd_sweep(args) -> int:
    bundle = pred.load_bundle()
    predictors = pred.predictors(bundle, args.model)
    feature = args.feature
    if feature not in sweeps.KNOBS:
        raise SystemExit(f"no knob for {feature!r}; have {sorted(sweeps.KNOBS)}")

    selected = [s for s in sweeps.single_factor() if feature in s.tags]
    rows = []
    for p in predictors:
        print(f"\n{feature} -> P(A)   [{p.name}]")
        print(f"  {'level':<12}{'P(A)':>8}{'delta vs neutral':>20}")
        neutral = None
        for s in selected:
            r = runner.run(s, p, bundle, with_mirror=False)
            level = s.name.rsplit("__", 1)[-1]
            if level == "neutral":
                neutral = r.probability
            rows.append({"model": p.name, "feature": feature, "level": level,
                         "probability": round(r.probability, 6)})
            delta = "" if neutral is None else f"{(r.probability - neutral) * 100:+.2f} pts"
            print(f"  {level:<12}{r.probability * 100:>7.2f}%{delta:>20}")

    if args.csv:
        REPORTS.mkdir(parents=True, exist_ok=True)
        out = REPORTS / f"sweep_{feature}.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  wrote {out}")
    return 0


def cmd_grid(args) -> int:
    bundle = pred.load_bundle()
    p = pred.predictors(bundle, args.model)[0]
    left, right = (args.pair.split(",") + [""])[:2]
    selected = [s for s in sweeps.pairwise()
                if left in s.tags and right in s.tags]
    if not selected:
        raise SystemExit(f"no grid for {args.pair!r}; pairs are {sweeps.PAIRS}")

    print(f"\n{left} x {right}   [{p.name}]   values are P(A) %\n")
    header = f"  {left + ' \\ ' + right:<20}" + "".join(
        f"{lvl:>14}" for lvl in sweeps.GRID)
    print(header)
    for a_level in sweeps.GRID:
        cells = []
        for b_level in sweeps.GRID:
            match = next(s for s in selected
                         if s.name == f"grid__{left}_{a_level}__{right}_{b_level}")
            cells.append(runner.run(match, p, bundle,
                                    with_mirror=False).probability * 100)
        print(f"  {a_level:<20}" + "".join(f"{c:>14.2f}" for c in cells))
    return 0


def cmd_benchmark(args) -> int:
    bundle = pred.load_bundle()
    p = pred.shipped(bundle)
    results = runner.run_many(scenarios.CATALOG, p, bundle)
    path = bm.save(results, bundle, Path(args.out) if args.out else None)
    print(f"wrote {path}")
    print(f"  {len(results)} scenarios, model {p.name}")
    return 0


def cmd_compare(args) -> int:
    bundle = pred.load_bundle()
    p = pred.shipped(bundle)
    old = bm.load(Path(args.path) if args.path else None)
    results = runner.run_many(scenarios.CATALOG, p, bundle)
    print(bm.compare(old, results))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="valwr.sandbox", description=__doc__)
    ap.add_argument("--model", default="logistic",
                    help="logistic | gbm | margin | baselines | all "
                         "(comma-separated)")
    sub = ap.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="show scenarios and coverage")
    p_list.add_argument("-v", "--verbose", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="run scenarios")
    p_run.add_argument("--mode", choices=("static", "variance", "both"),
                       default="static")
    p_run.add_argument("--scenario", default="all",
                       help="a scenario name, a category, or 'all'")
    p_run.add_argument("--samples", type=int, default=1000)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--generated", action="store_true",
                       help="include the generated sweep and grid scenarios")
    p_run.add_argument("--no-mirror", action="store_true")
    p_run.add_argument("--strict", action="store_true",
                       help="exit non-zero if a directional expectation fails")
    p_run.set_defaults(func=cmd_run)

    p_sweep = sub.add_parser("sweep", help="single-factor curve")
    p_sweep.add_argument("--feature", required=True)
    p_sweep.add_argument("--csv", action="store_true")
    p_sweep.set_defaults(func=cmd_sweep)

    p_grid = sub.add_parser("grid", help="pairwise interaction grid")
    p_grid.add_argument("--pair", required=True, help="e.g. rating,wr_map")
    p_grid.set_defaults(func=cmd_grid)

    p_bench = sub.add_parser("benchmark", help="freeze the static catalog")
    p_bench.add_argument("--out")
    p_bench.set_defaults(func=cmd_benchmark)

    p_cmp = sub.add_parser("compare", help="diff a benchmark against now")
    p_cmp.add_argument("path", nargs="?")
    p_cmp.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except MissingModel as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
