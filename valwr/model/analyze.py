"""Post-training analysis: python -m valwr.model.analyze

The headline number is not the interesting part. These are:

- the equal-rank subset, where matchmaking did its job and any signal is the
  genuine residual rather than rank in disguise
- what the model actually relies on, including what it barely uses at all
- the reliability diagram, which says whether "58%" means anything

All three describe the model that **ships** -- `bundle["best"]`, served through
`serving` exactly as the live path serves it. This module has twice described a
different one: it loaded gbm.joblib by name after the bundle was renamed, and
later drew both README charts from the gradient booster while logistic
regression was what ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from valwr import config
from valwr.model import baselines, evaluate, serving, split
from valwr.store import schema

REPORTS = Path(__file__).resolve().parent.parent.parent / "reports"

# Key the charts' provenance is written under, read back by tools/audit.py.
STAMP_KEY = "Source"

# The team aggregates of one per-player stat.
AGGREGATES = ("_mean", "_max", "_min", "_std")


def load(conn):
    s = config.load(require_key=False)
    df = pd.read_parquet(s.database_path.parent / "features.parquet")
    b = split.compute(conn)
    # The same ordering train.py applies before de-duplicating, so both keep
    # the same copy of a match collected via several players.
    df = split.apply(df, b).sort_values("started_at").reset_index(drop=True)
    return df.drop_duplicates(subset="match_id", keep="first")


def equal_rank_subset(df, tol: float = 0.5) -> pd.DataFrame:
    """Matches where the teams' average ranks are within `tol` of a tier.

    This is the honest test. Where ranks differ, a model can score by
    rediscovering rank -- which is already a feature and already known. Where
    they are level, whatever remains is the part the feature engineering
    actually contributed.
    """
    return df[df["d_tier_mean"].abs() <= tol]


def label(feature: str) -> str:
    # removeprefix, not replace: replace("d_", "") also ate the "d_" inside
    # "games_played_mean" and "rating_trend_mean", and the chart shipped with
    # "games_playemean" on it.
    return feature.removeprefix("d_")


def family(feature: str) -> str:
    """acs_mean, acs_max, acs_min and acs_std are one stat asked four ways."""
    name = label(feature)
    for suffix in AGGREGATES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[:-len(suffix)]
    return name


def _log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def family_importance(bundle: dict, frame: pd.DataFrame, y,
                      repeats: int = 10, seed: int = 17) -> pd.DataFrame:
    """Rise in held-out log loss when each feature family is shuffled.

    Not coefficients. acs_mean and rating_mean correlate around 0.9, so the
    linear model splits their credit and hands rating_mean a *negative* weight
    -- which, charted, reads as "a better rating hurts". Shuffling a whole
    family at once breaks its link with the outcome without asking the model
    to tell near-duplicates apart. Families still overlap one another (ACS and
    ADR measure similar things), so a family can look smaller than it is
    because its neighbour covers for it; the ranking is the reliable part.
    """
    cols = bundle["columns"]
    X = frame[cols].reset_index(drop=True)
    base = _log_loss(y, serving.probabilities(bundle, X))
    groups: dict[str, list[str]] = {}
    for c in cols:
        groups.setdefault(family(c), []).append(c)

    rng = np.random.default_rng(seed)
    rows = []
    for name, members in groups.items():
        rises = []
        for _ in range(repeats):
            shuffled = X.copy()
            # One permutation for the whole family, so its members stay
            # consistent with each other and only their link to y is broken.
            shuffled[members] = X[members].to_numpy()[rng.permutation(len(X))]
            rises.append(_log_loss(y, serving.probabilities(bundle, shuffled))
                         - base)
        rows.append({"family": name, "features": len(members),
                     "rise": float(np.mean(rises)),
                     "se": float(np.std(rises, ddof=1) / np.sqrt(repeats))})
    return (pd.DataFrame(rows).sort_values("rise", ascending=False)
            .reset_index(drop=True))


def stamp(bundle: dict, n_test: int) -> str:
    return json.dumps({"model": bundle["best"],
                       "norms_as_of": bundle.get("norms_as_of"),
                       "n_test": n_test}, sort_keys=True)


def main(argv=None) -> int:
    import joblib

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    bundle_path = Path(s.database_path).parent.parent / "models" / "model.joblib"
    if not bundle_path.exists():
        print(f"no model at {bundle_path}; run python -m valwr.model.train first")
        return 1
    bundle = joblib.load(bundle_path)
    shipped = bundle["best"]

    df = load(conn)
    tr = df[df["slice"] == "train"]
    te = df[df["slice"] == "test"]

    # The split is recomputed from the database. If the crawl has grown since
    # training, the boundaries move, the test slice is no longer the one the
    # model was scored on, and some of it may be matches it trained on.
    results_path = REPORTS / "results.json"
    if results_path.exists():
        trained = json.loads(results_path.read_text(encoding="utf-8"))
        if trained.get("n_test") != len(te):
            print(f"  the test slice is now {len(te):,} matches but the model "
                  f"was scored on {trained.get('n_test'):,} -- the database "
                  f"has changed since training. Retrain before analysing.")
            return 1

    print(f"  shipped model: {shipped}   test set: {len(te):,} matches")
    y = te["target"].to_numpy(int)
    p = serving.probabilities(bundle, te)

    print("=" * 66)
    print("1. EQUAL-RANK SUBSET  (where matchmaking did its job)")
    print("=" * 66)
    sub = equal_rank_subset(te)
    print(f"  {len(sub):,} of {len(te):,} test matches have team ranks within "
          f"half a tier\n")
    if len(sub) >= 100:
        print(evaluate.header())
        rows = [evaluate.score(name, sub["target"], fn(sub))
                for name, fn in baselines.fitted(tr).items()]
        model_row = evaluate.score(shipped, sub["target"],
                                   serving.probabilities(bundle, sub))
        rows.append(model_row)
        for r in sorted(rows, key=lambda r: r.log_loss):
            print(r.row())
        ci = evaluate.confidence_interval(model_row.accuracy, model_row.n)
        print(f"\n  {shipped} here: {model_row.accuracy*100:.1f}% "
              f"+/- {ci*100:.1f}%")
        print("  If nothing beats a coin flip on this subset, the model is")
        print("  rediscovering rank rather than adding to it.")

    print("\n" + "=" * 66)
    print("2. WHAT THE MODEL RELIES ON")
    print("=" * 66)
    imp = family_importance(bundle, te, y)
    print("  rise in held-out log loss when a feature family is shuffled")
    print(f"  {'family':<22} {'features':>8} {'rise':>10} {'+/- se':>9}")
    for r in imp.itertuples():
        print(f"  {r.family:<22} {r.features:>8} {r.rise:>+10.5f} {r.se:>9.5f}")
    print("\n  Families near zero are a negative result worth reporting, not a")
    print("  bug -- they are the ones that turned out not to matter.")
    if "logistic" in serving.COMPONENTS.get(shipped, ()):
        clf = bundle["estimators"]["logistic"].named_steps["logisticregression"]
        coef = pd.Series(clf.coef_[0], index=[label(c) for c in bundle["columns"]])
        # Only a family with several members can split credit among them. A
        # negative weight on a family of one is the model's actual reading of
        # that stat -- more first deaths, or a recent win streak that has
        # pushed a player's matchmaking above their rank -- and saying
        # otherwise would explain away a real finding.
        shared = set(imp.family[(imp.features > 1)].head(5))
        negative = [k for k, v in coef.items() if v < 0 and family(k) in shared]
        if negative:
            print(f"\n  Raw coefficients are negative on {', '.join(negative)}.")
            print("  Those sit in families whose members correlate with each")
            print("  other and with other families, so the model splits credit")
            print("  among near-duplicates; it is not those stats hurting. That")
            print("  is why the chart shows shuffling, not coefficients.")

    print("\n" + "=" * 66)
    print("3. CALIBRATION")
    print("=" * 66)
    rel = evaluate.reliability_table(y, p)
    print(f"  {'predicted':>10} {'observed':>10} {'n':>7}")
    for pred, obs, n in rel:
        bar = "#" * int(abs(obs - pred) * 200)
        print(f"  {pred:>10.3f} {obs:>10.3f} {n:>7}  {bar}")
    print(f"\n  ECE {evaluate.expected_calibration_error(y, p):.4f}; "
          f"predictions span {p.min():.3f}-{p.max():.3f}, "
          f"middle 90% within {np.quantile(p, .05):.3f}-{np.quantile(p, .95):.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (charts skipped: pip install -e \".[research]\" for matplotlib)")
        return 0

    meta = {STAMP_KEY: stamp(bundle, len(te))}
    REPORTS.mkdir(exist_ok=True)
    ink, grid, accent, cool = "#1d2733", "#d9dee5", "#c8453b", "#3f6fb5"

    # --- reliability diagram ------------------------------------------
    fig, (ax, hist) = plt.subplots(
        2, 1, figsize=(6, 6.4), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08})
    xs = np.array([r[0] for r in rel])
    obs = np.array([r[1] for r in rel])
    ns = np.array([r[2] for r in rel])
    half = 1.96 * np.sqrt(np.clip(obs * (1 - obs), 1e-9, None) / ns)
    lo = min(xs.min(), (obs - half).min()) - 0.02
    hi = max(xs.max(), (obs + half).max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "--", color="#8a94a0", lw=1,
            label="perfectly calibrated")
    ax.errorbar(xs, obs, yerr=half, fmt="o-", color=accent, ecolor=accent,
                elinewidth=1, capsize=3, lw=1.6, ms=5,
                label="observed, with 95% interval")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("how often that team actually won", color=ink)
    ax.set_title(f"Calibration on {len(te):,} held-out matches\n"
                 f"{shipped}; each point is {ns.min():,}-{ns.max():,} matches",
                 fontsize=10, color=ink)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(color=grid, lw=0.6)
    # The histogram shares the diagram's x-axis so the two line up, which
    # crops the tails. Say how many predictions that hides rather than let
    # the crop pass as the whole distribution.
    hist.hist(p, bins=30, range=(lo, hi), color=cool, alpha=0.85)
    beyond = int(((p < lo) | (p > hi)).sum())
    if beyond:
        hist.text(0.99, 0.92, f"{beyond:,} predictions fall outside this range",
                  transform=hist.transAxes, ha="right", va="top", fontsize=7,
                  color=ink)
    hist.set_yticks([])
    hist.set_xlabel("predicted win probability", color=ink)
    hist.set_ylabel("matches", color=ink, fontsize=8)
    for a in (ax, hist):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.savefig(REPORTS / "reliability.png", bbox_inches="tight", metadata=meta)
    plt.close(fig)
    print(f"\n  wrote {REPORTS / 'reliability.png'}")

    # --- what the model relies on ----------------------------------------
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=150)
    ax.barh(range(len(top)), top["rise"], xerr=top["se"], color=cool,
            ecolor="#8a94a0", capsize=2)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(
        [f"{f}  ({n})" if n > 1 else f
         for f, n in zip(top["family"], top["features"])], fontsize=8, color=ink)
    ax.set_ylabel("feature family (number of features, if more than one)",
                  color=ink, fontsize=8)
    ax.axvline(0, color="#8a94a0", lw=0.8)
    ax.set_xlabel("rise in held-out log loss when shuffled", color=ink)
    ax.set_title(f"What the model relies on: top 15 of {len(imp)} feature "
                 f"families\n{shipped}, each shuffled on {len(te):,} held-out "
                 f"matches", fontsize=10, color=ink)
    ax.grid(axis="x", color=grid, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "importance.png", metadata=meta)
    plt.close(fig)
    print(f"  wrote {REPORTS / 'importance.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
