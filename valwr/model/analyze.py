"""Post-training analysis: python -m valwr.model.analyze

The headline number is not the interesting part. These are:

- the equal-rank subset, where matchmaking did its job and any signal is the
  genuine residual rather than rank in disguise
- which features actually carry weight, including the ones that carry none
- the reliability diagram, which is the most convincing single artefact the
  project produces and the one that says whether "58%" means anything
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from valwr import config
from valwr.features import build as fb  # noqa: F401  (kept for column names)
from valwr.model import baselines, evaluate, split
from valwr.store import schema

REPORTS = Path(__file__).resolve().parent.parent.parent / "reports"


def load(conn):
    s = config.load(require_key=False)
    df = pd.read_parquet(s.database_path.parent / "features.parquet")
    b = split.compute(conn)
    df = split.apply(df, b).drop_duplicates(subset="match_id", keep="first")
    return df


def equal_rank_subset(df, tol: float = 0.5) -> pd.DataFrame:
    """Matches where the teams' average ranks are within `tol` of a tier.

    This is the honest test. Where ranks differ, a model can score by
    rediscovering rank -- which is already a feature and already known. Where
    they are level, whatever remains is the part the feature engineering
    actually contributed.
    """
    return df[df["d_tier_mean"].abs() <= tol]


def main(argv=None) -> int:
    import joblib

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    df = load(conn)
    tr = df[df["slice"] == "train"]
    te = df[df["slice"] == "test"]

    bundle = joblib.load(Path(s.database_path).parent.parent / "models" / "gbm.joblib")
    gbm, cols = bundle["gbm"], bundle["columns"]

    print("=" * 66)
    print("1. EQUAL-RANK SUBSET  (where matchmaking did its job)")
    print("=" * 66)
    sub = equal_rank_subset(te)
    print(f"  {len(sub):,} of {len(te):,} test matches have team ranks within "
          f"half a tier\n")
    if len(sub) >= 100:
        print(evaluate.header())
        rows = []
        for name, fn in baselines.fitted(tr).items():
            rows.append(evaluate.score(name, sub["target"], fn(sub)))
        p = gbm.predict_proba(sub[cols].to_numpy(float))[:, 1]
        rows.append(evaluate.score("gradient boosting", sub["target"], p))
        for r in sorted(rows, key=lambda r: r.log_loss):
            print(r.row())
        best = min(rows, key=lambda r: r.log_loss)
        ci = evaluate.confidence_interval(best.accuracy, best.n)
        print(f"\n  best here: {best.name} at {best.accuracy*100:.1f}% "
              f"+/- {ci*100:.1f}%")
        print("  If nothing beats a coin flip on this subset, the model is")
        print("  rediscovering rank rather than adding to it.")

    print("\n" + "=" * 66)
    print("2. FEATURE IMPORTANCE")
    print("=" * 66)
    imp = pd.Series(gbm.feature_importances_, index=cols).sort_values(
        ascending=False)
    print("  top 12:")
    for k, v in imp.head(12).items():
        print(f"    {k:<28} {v:>7.0f}")
    dead = imp[imp == 0]
    print(f"\n  {len(dead)} of {len(cols)} features got zero splits:")
    for k in list(dead.index)[:8]:
        print(f"    {k}")
    print("\n  Zero-split features are a negative result worth reporting, not")
    print("  a bug -- they are the ones that turned out not to matter.")

    print("\n" + "=" * 66)
    print("3. CALIBRATION")
    print("=" * 66)
    res = json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))
    rel = res.get("reliability") or []
    print(f"  {'predicted':>10} {'observed':>10} {'n':>7}")
    for pred, obs, n in rel:
        bar = "#" * int(abs(obs - pred) * 200)
        print(f"  {pred:>10.3f} {obs:>10.3f} {n:>7}  {bar}")
    if rel:
        gap = np.mean([abs(o - p) for p, o, _ in rel])
        print(f"\n  mean |predicted - observed| = {gap:.4f}")

    # --- reliability diagram ------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=140)
        ax.plot([0.35, 0.65], [0.35, 0.65], "--", color="#888", lw=1,
                label="perfect calibration")
        if rel:
            ax.plot([p for p, _, _ in rel], [o for _, o, _ in rel],
                    "o-", color="#c94f4f", lw=1.8, ms=6, label="model")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed win frequency")
        ax.set_title("Reliability diagram (held-out test set)", fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        REPORTS.mkdir(exist_ok=True)
        fig.savefig(REPORTS / "reliability.png")
        print(f"\n  wrote {REPORTS / 'reliability.png'}")

        fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=140)
        top = imp.head(15)[::-1]
        ax2.barh(range(len(top)), top.values, color="#4f7fc9")
        ax2.set_yticks(range(len(top)))
        ax2.set_yticklabels([t.replace("d_", "") for t in top.index], fontsize=8)
        ax2.set_xlabel("LightGBM split count")
        ax2.set_title("Feature importance", fontsize=11)
        fig2.tight_layout()
        fig2.savefig(REPORTS / "importance.png")
        print(f"  wrote {REPORTS / 'importance.png'}")
    except Exception as e:
        print(f"  (plots skipped: {e})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
