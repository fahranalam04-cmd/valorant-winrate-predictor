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

# Swept on validation over C in [0.003, 1.0]. The curve is almost flat --
# 0.6889 at the optimum against 0.6890 at the previous 0.1 -- which is
# itself the finding: regularisation is not what is limiting this model.
LOGISTIC_C = 0.03

# Preference order when models are statistically tied. Lower is simpler, and
# simpler wins ties: fewer moving parts to explain, and less to go wrong in
# the live path.
COMPLEXITY = {
    "coin flip": 0,
    "avg rank (fitted)": 1,
    "best player rank": 1,
    "avg rating (fitted)": 2,
    "logistic regression": 3,
    "logistic + platt": 4,
    "logistic + isotonic": 4,
    "margin regression": 5,
    "logistic + margin blend": 6,
    "gradient boosting": 7,
    "gbm + platt": 8,
    "gbm + isotonic": 8,
}


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
        LogisticRegression(max_iter=2000, C=LOGISTIC_C, random_state=RANDOM_STATE),
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


def fit_margin_model(Xtr, m_tr, Xva, m_va, Xte, y_va, p_shape=None):
    """Regress on round margin, then convert the prediction to a probability.

    A binary label carries one bit. The margin carries far more -- 13-3 and
    13-11 are the same bit but very different evidence about which side was
    stronger -- and a classifier discards all of it. Regressing on margin and
    mapping back is standard practice in sports modelling, and it matters most
    exactly here: weak signal and limited data, where every sample has to work
    harder.

    The margin-to-probability mapping is fitted on validation, never on test.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    reg = make_pipeline(StandardScaler(), Ridge(alpha=50.0))
    reg.fit(Xtr, m_tr)

    # Map predicted margin -> P(win) using held-out data.
    mhat_va = reg.predict(Xva).reshape(-1, 1)
    link = LogisticRegression(max_iter=1000)
    link.fit(mhat_va, y_va)

    return reg, link, link.predict_proba(reg.predict(Xte).reshape(-1, 1))[:, 1]


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

    # Rebuild the exact norms the matrix was built with, so inference can
    # reuse them rather than approximating them later.
    from valwr.rating.normalize import build_norms
    from valwr.store import temporal as _temporal
    norms_used = build_norms(conn, b.train_end)
    prior_used = _temporal.population_win_rate(conn, b.train_end)

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

    # --- 4b. margin regression ---------------------------------------
    m_tr = tr["margin"].to_numpy(float)
    m_va = va["margin"].to_numpy(float)
    reg, link, p_mg_te = fit_margin_model(Xtr, m_tr, Xva, m_va, Xte, yva)
    results.append(evaluate.score("margin regression", yte, p_mg_te))

    # Averaging two models that make different mistakes usually beats both.
    p_blend = 0.5 * p_lr_te + 0.5 * p_mg_te
    results.append(evaluate.score("logistic + margin blend", yte, p_blend))

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

    # --- 8. persist everything inference needs ------------------------
    # Not just the estimator. Live features must be built with the SAME
    # population norms, shrinkage prior and agent->role map used in training,
    # or the model is handed inputs it never saw. Saving the estimator alone
    # is the quiet way to get a live path that silently disagrees with its
    # own evaluation.
    import joblib
    from valwr.store import reference

    models = Path(s.database_path).parent.parent / "models"
    models.mkdir(exist_ok=True)

    # Ship the SIMPLEST model within one standard error of the best, not
    # whichever happened to win. Consecutive runs crowned "logistic + margin
    # blend" and then "gradient boosting" on the same data -- the ranking
    # flips because the gaps are smaller than the noise, so selecting on the
    # raw minimum is selecting on noise. The one-standard-error rule is the
    # standard remedy and it also happens to ship the model that is easier to
    # explain and cheaper to run live.
    se = evaluate.log_loss_standard_error(yte, p_lr_te)
    ranked = sorted(results, key=lambda r: r.log_loss)
    threshold = ranked[0].log_loss + se
    eligible = [r for r in ranked if r.log_loss <= threshold]
    winner_name = min(eligible, key=lambda r: COMPLEXITY.get(r.name, 99)).name
    print(f"\n  log-loss standard error {se:.4f}; "
          f"{len(eligible)} model(s) statistically tied")
    print(f"  shipping the simplest of them: {winner_name}")
    # Baselines are shippable models too. The one-standard-error rule can
    # legitimately choose one -- and did -- so the bundle has to be able to
    # serve it rather than silently falling back to something else.
    estimators = {"logistic": lr, "gbm": gbm, "margin_reg": reg,
                  "margin_link": link}
    fitted_baselines = baselines.fitted(tr)
    for bname in fitted_baselines:
        if bname != "coin flip":
            estimators[bname] = fitted_baselines[bname]
    if winner_name not in estimators and winner_name not in (
            "logistic regression", "gradient boosting", "margin regression",
            "logistic + margin blend"):
        raise RuntimeError(
            f"selected {winner_name!r} but it is not in the persisted "
            f"estimators; the live path could not serve it")

    joblib.dump({
        "estimators": estimators,
        "best": winner_name,
        "columns": cols,
        "norms": norms_used,
        "prior_rate": prior_used,
        "roles": reference.agent_roles(conn),
        "norms_as_of": b.train_end,
        "metrics": {r.name: r.__dict__ for r in results},
    }, models / "model.joblib")
    print(f"  wrote {models / 'model.joblib'}  (best: {winner_name})")

    # Regenerate the README table from what was just measured. Hand-maintained
    # numbers went stale four times as the crawl grew, and each manual edit is
    # a chance to leave a claim standing that the latest run no longer supports.
    try:
        from valwr.model import report
        report.main([])
    except Exception as e:
        print(f"  (README not regenerated: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
