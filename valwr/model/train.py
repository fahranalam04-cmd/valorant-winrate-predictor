"""The training pipeline: python -m valwr.model.train

Climbs the baseline ladder in order. Each rung must beat the one below on the
held-out test set before the next is worth anything.

Ordering matters and is the whole reason this is one script rather than
several: the split boundary is computed first, the feature matrix is built at
that cutoff, and only then is anything fitted. Building first and splitting
afterwards fits population statistics on held-out rows.

The test slice is scored once, at the end. Tuning against it repeatedly is a
slower form of overfitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from valwr import config
from valwr.features import build as fb
from valwr.model import baselines, evaluate, split
from valwr.store import schema

RANDOM_STATE = 17


def feature_columns(df) -> list[str]:
    """Difference features only, dropping any with no variance in training.

    A constant column teaches nothing and upsets some solvers. Twenty of them
    appeared once when the rating pipeline was silently dead, so this also
    doubles as a canary.
    """
    cols = [c for c in df.columns if c.startswith("d_")]
    train = df[df["slice"] == "train"]
    keep = [c for c in cols if train[c].nunique() > 1]
    dropped = sorted(set(cols) - set(keep))
    if dropped:
        print(f"  dropping {len(dropped)} zero-variance features: "
              f"{', '.join(dropped[:5])}{'...' if len(dropped) > 5 else ''}")
    return keep


def fit_logistic(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.1, random_state=RANDOM_STATE),
    )
    model.fit(X, y)
    return model


def fit_gbm(X, y, X_val, y_val):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=15,          # small: the signal is weak and easily overfit
        min_child_samples=60,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(X, y, eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def calibrate(p_val, y_val, p_test):
    """Calibrate on validation, applied to test. Never fitted on test.

    Tries isotonic and Platt and keeps whichever scores better on validation.
    Isotonic is the more flexible of the two and, on a signal this weak, it
    overfits: fitted blindly it made log loss WORSE than the uncalibrated
    model (0.6976 against 0.6922). Choosing between them on held-out data
    rather than by preference is the fix.
    """
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    iso.fit(p_val, y_val)

    platt = LogisticRegression(max_iter=1000)
    platt.fit(np.asarray(p_val).reshape(-1, 1), y_val)

    ll_iso = log_loss(y_val, np.clip(iso.predict(p_val), 1e-6, 1 - 1e-6),
                      labels=[0, 1])
    ll_platt = log_loss(
        y_val, platt.predict_proba(np.asarray(p_val).reshape(-1, 1))[:, 1],
        labels=[0, 1])

    if ll_iso <= ll_platt:
        return np.clip(iso.predict(p_test), 1e-6, 1 - 1e-6), ("isotonic", iso)
    return (platt.predict_proba(np.asarray(p_test).reshape(-1, 1))[:, 1],
            ("platt", platt))


def shuffled_target_check(X, y, X_test, y_test, draws: int = 25) -> dict:
    """Retrain on shuffled labels repeatedly; the AUCs must centre on 0.5.

    If a model can predict shuffled labels, a feature encodes the outcome.

    Deliberately many draws rather than one. A single shuffle has a standard
    deviation around 0.014 at this sample size, so one draw landing at 0.518
    looks alarming and means nothing -- which happened, and cost a round of
    investigation. The distribution is the test; a single sample is a rumour.
    """
    from sklearn.metrics import roc_auc_score
    aucs = []
    for seed in range(draws):
        rng = np.random.default_rng(seed)
        model = fit_logistic(X, rng.permutation(y))
        aucs.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    a = np.asarray(aucs)
    se = a.std() / max(len(a) ** 0.5, 1e-9)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "sigmas_from_chance": float(abs(a.mean() - 0.5) / se) if se else 0.0,
        "draws_over_threshold": int((a > 0.55).sum()),
        "draws": draws,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="valwr.model.train")
    ap.add_argument("--min-coverage", type=int, default=5)
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the feature matrix rather than reusing parquet")
    args = ap.parse_args(argv)

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)

    # --- 1. boundary first, matrix second ---------------------------
    b = split.compute(conn)
    print(f"split boundary: train ends {b.train_end}, val ends {b.val_end} "
          f"({b.n_matches:,} resolved matches)")

    cache = s.database_path.parent / "features.parquet"
    if args.rebuild or not cache.exists():
        rows = fb.build_all(conn, norms_as_of=b.train_end,
                            min_coverage=args.min_coverage)
        df = fb.to_frame(rows)
        df.to_parquet(cache, index=False)
    else:
        df = pd.read_parquet(cache)
        print(f"  reusing {cache.name} ({len(df):,} rows) -- --rebuild to refresh")

    df = split.apply(df, b).sort_values("started_at").reset_index(drop=True)
    # The crawler collects the same match via several players, so duplicates
    # across splits are a live risk rather than a theoretical one.
    before = len(df)
    df = df.drop_duplicates(subset="match_id", keep="first")
    if len(df) < before:
        print(f"  dropped {before - len(df):,} duplicate match_ids")
    print(split.describe(df))

    cols = feature_columns(df)
    tr, va, te = (df[df["slice"] == k] for k in ("train", "val", "test"))
    if min(len(tr), len(va), len(te)) < 50:
        print("\nnot enough data in one of the slices yet")
        return 1
    Xtr, ytr = tr[cols].to_numpy(float), tr["target"].to_numpy(int)
    Xva, yva = va[cols].to_numpy(float), va["target"].to_numpy(int)
    Xte, yte = te[cols].to_numpy(float), te["target"].to_numpy(int)
    print(f"\n  {len(cols)} features | train {len(tr):,} | val {len(va):,} "
          f"| test {len(te):,}")

    results = []

    # --- 2. baselines -----------------------------------------------
    for name, fn in baselines.fitted(tr).items():
        results.append(evaluate.score(name, yte, fn(te)))

    # --- 3. logistic ------------------------------------------------
    lr = fit_logistic(Xtr, ytr)
    p_lr_va = lr.predict_proba(Xva)[:, 1]
    p_lr_te = lr.predict_proba(Xte)[:, 1]
    results.append(evaluate.score("logistic regression", yte, p_lr_te))
    p_lr_cal, (lr_method, _) = calibrate(p_lr_va, yva, p_lr_te)
    results.append(evaluate.score(f"logistic + {lr_method}", yte, p_lr_cal))

    # --- 4. gradient boosting ---------------------------------------
    gbm = fit_gbm(Xtr, ytr, Xva, yva)
    p_gb_va = gbm.predict_proba(Xva)[:, 1]
    p_gb_te = gbm.predict_proba(Xte)[:, 1]
    results.append(evaluate.score("gradient boosting", yte, p_gb_te))
    p_gb_cal, (gb_method, iso) = calibrate(p_gb_va, yva, p_gb_te)
    results.append(evaluate.score(f"gbm + {gb_method}", yte, p_gb_cal))

    # --- 5. report ---------------------------------------------------
    print("\n" + "=" * 68)
    print("TEST SET RESULTS  (touched once)")
    print("=" * 68)
    print(evaluate.header())
    for r in sorted(results, key=lambda r: r.log_loss):
        print(r.row())

    rank = next(r for r in results if r.name == "avg rank (fitted)")
    best = min(results, key=lambda r: r.log_loss)
    ci = evaluate.confidence_interval(best.accuracy, best.n)
    print(f"\n  best by log loss: {best.name}")
    print(f"  accuracy {best.accuracy*100:.1f}% +/- {ci*100:.1f}%  "
          f"(95% CI, n={best.n:,})")
    print(f"  rank baseline    {rank.accuracy*100:.1f}%")
    beat = best.log_loss < rank.log_loss
    print(f"  beats rank baseline on log loss: {'YES' if beat else 'NO'}")
    if best.auc > 0.75:
        print("\n  !! AUC above 0.75 -- assume leakage and investigate. !!")

    # --- 5b. coverage strata -----------------------------------------
    # How many of the ten players we actually knew about matters more than
    # the headline number suggests, and reporting one blended figure hides it.
    print("\n  by how many of the 10 players had prior history:")
    print(f"    {'coverage':>10} {'n':>7} {'logloss':>9} {'auc':>7} {'acc':>8}")
    strata = []
    for lo, hi, label in [(5, 6, "5-6"), (7, 8, "7-8"), (9, 10, "9-10")]:
        m = (te["coverage"] >= lo) & (te["coverage"] <= hi)
        if m.sum() < 150:
            continue
        sc = evaluate.score(label, yte[m.to_numpy()], p_lr_te[m.to_numpy()])
        strata.append(sc)
        print(f"    {label:>10} {sc.n:>7,} {sc.log_loss:>9.4f} "
              f"{sc.auc:>7.3f} {sc.accuracy*100:>7.1f}%")

    # --- 6. leakage check --------------------------------------------
    shuf = shuffled_target_check(Xtr, ytr, Xte, yte)
    clean = shuf["sigmas_from_chance"] < 3.0 and shuf["draws_over_threshold"] == 0
    print(f"\n  shuffled-target ({shuf['draws']} draws): "
          f"mean AUC {shuf['mean']:.4f} +/- {shuf['std']:.4f}, "
          f"range {shuf['min']:.3f}-{shuf['max']:.3f}")
    print(f"    {shuf['sigmas_from_chance']:.1f} sigma from chance, "
          f"{shuf['draws_over_threshold']}/{shuf['draws']} over 0.55 -> "
          f"{'OK' if clean else 'LEAK -- INVESTIGATE'}")
    # --- 7. persist ---------------------------------------------------
    out = Path(s.database_path).parent.parent / "reports"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "n_features": len(cols),
        "shuffled": shuf,
        "results": [r.__dict__ for r in results],
        "coverage_strata": [r.__dict__ for r in strata],
        "reliability": evaluate.reliability_table(yte, p_gb_cal),
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {out / 'results.json'}")

    import joblib
    models = Path(s.database_path).parent.parent / "models"
    models.mkdir(exist_ok=True)
    joblib.dump({"gbm": gbm, "isotonic": iso, "columns": cols},
                models / "gbm.joblib")
    print(f"  wrote {models / 'gbm.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
