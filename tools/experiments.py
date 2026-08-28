"""Accuracy experiments. Reports to validation; ships nothing.

Reads data/features.parquet and the database. Writes neither, and never loads
or overwrites models/model.joblib. The test slice is deliberately not scored
here -- it has been touched once, by train.py, and a candidate chosen by
repeatedly consulting it would just be overfitting more slowly.

The decision rule is fixed before any result is seen, because the differences
in this project live in the third decimal place and it is easy to talk yourself
into noise. A candidate must beat the incumbent by more than one log-loss
standard error on validation. Anything smaller is reported as null. A prior
refinement here measured 0.00017 against a standard error of 0.0020 and was
correctly abandoned; the same rule applies to everything below.

    python tools/experiments.py
    python tools/experiments.py --coverage    adds the min_coverage sweep,
                                              which needs tools/build_cov0.py
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from valwr import config
from valwr.model import evaluate, split, strength
from valwr.model.train import RANDOM_STATE, feature_columns, fit_gbm, fit_logistic
from valwr.store import schema


class Ledger:
    """Every candidate, judged against one standard error."""

    def __init__(self, incumbent_ll: float, se: float):
        self.incumbent_ll = incumbent_ll
        self.se = se
        self.rows: list[tuple] = []

    def add(self, name, ll, auc, acc, mirror=None, note=""):
        delta = ll - self.incumbent_ll          # negative is better
        verdict = "WIN" if delta < -self.se else "null"
        self.rows.append((name, ll, delta, auc, acc, mirror, verdict, note))
        return verdict

    def table(self) -> str:
        out = [
            "",
            "=" * 100,
            "VALIDATION RESULTS   (test slice deliberately not scored)",
            "=" * 100,
            f"  incumbent log loss {self.incumbent_ll:.4f}   "
            f"standard error {self.se:.4f}   "
            f"a win must beat it by more than {self.se:.4f}",
            "",
            f"  {'candidate':<36}{'logloss':>9}{'delta':>10}{'auc':>7}"
            f"{'acc':>8}{'mirror':>11}  verdict",
            "  " + "-" * 96,
        ]
        for name, ll, delta, auc, acc, mirror, verdict, note in self.rows:
            m = "-" if mirror is None else f"{mirror:.2e}"
            out.append(f"  {name:<36}{ll:>9.4f}{delta:>+10.4f}{auc:>7.3f}"
                       f"{acc:>7.1f}%{m:>11}  {verdict}"
                       + (f"   {note}" if note else ""))
        return "\n".join(out)


def mirror_error(model, X) -> float:
    """Mean |p(X) + p(-X) - 1|. Zero means swapping the teams mirrors exactly.

    Measured on real validation rows rather than on a synthetic scenario, so it
    reflects the feature distribution the model actually meets.
    """
    p = model.predict_proba(X)[:, 1]
    q = model.predict_proba(-X)[:, 1]
    return float(np.abs(p + q - 1).mean())


def fit_symmetric_logistic(X, y, C):
    """Logistic with no intercept and no mean-centering.

    The feature vector already negates exactly when the two teams swap. What
    breaks the mirror is the preprocessing: StandardScaler subtracts a non-zero
    training mean, and the intercept adds a constant that does not negate.
    Remove both and P(A) + P(B) == 1 holds by construction rather than to a
    documented tolerance.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, C=C, fit_intercept=False,
                           random_state=RANDOM_STATE),
    )
    model.fit(X, y)
    return model


def load_rosters(conn) -> dict:
    """match_id -> (blue puuids, red puuids) for every resolved match."""
    rows = conn.execute(
        "SELECT mp.match_id AS match_id, mp.puuid AS puuid, mp.team AS team "
        "FROM match_players mp JOIN matches m ON m.match_id = mp.match_id "
        "WHERE m.winner IN ('Blue','Red')").fetchall()
    out: dict = {}
    for r in rows:
        pair = out.setdefault(r["match_id"], ([], []))
        pair[0 if r["team"] == "Blue" else 1].append(r["puuid"])
    return out


def bt_columns(match_ids, rosters, st):
    """(d_bt_sum, d_bt_max) per match, in the order given."""
    sums = np.zeros(len(match_ids))
    maxs = np.zeros(len(match_ids))
    for i, mid in enumerate(match_ids):
        blue, red = rosters.get(mid, ([], []))
        bs, bm = st.team(blue)
        rs, rm = st.team(red)
        sums[i], maxs[i] = bs - rs, bm - rm
    return sums, maxs


def expanding_bt(conn, rosters, tr, quantiles=(0.40, 0.55, 0.70, 0.85), *,
                 min_appearances, C):
    """Out-of-sample Bradley-Terry features for the training rows.

    A training match's own outcome contributes to its players' strengths, so
    one fit across the whole training set makes those features partly
    self-fulfilling. The downstream model would over-trust them and then fail
    on validation, which reads as "Bradley-Terry does not work" when the real
    fault was the construction.

    So the training period is walked forward: rows in each window get strengths
    fitted only on matches from before that window opened. Rows before the
    first cut have no fit available and are dropped -- which is why the caller
    refits the incumbent on the same reduced set, or the comparison is rigged.
    """
    times = np.quantile(tr["started_at"].to_numpy(), quantiles)
    fits = [(cut, strength.fit(conn, int(cut),
                               min_appearances=min_appearances, C=C))
            for cut in times]

    sub = tr[tr["started_at"].to_numpy() >= times[0]]
    sums = np.zeros(len(sub))
    maxs = np.zeros(len(sub))
    for i, (mid, t) in enumerate(zip(sub["match_id"].tolist(),
                                     sub["started_at"].to_numpy())):
        st = next(s for cut, s in reversed(fits) if t >= cut)
        blue, red = rosters.get(mid, ([], []))
        bs, bm = st.team(blue)
        rs, rm = st.team(red)
        sums[i], maxs[i] = bs - rs, bm - rm
    return sub, sums, maxs


def matrix_boundaries(df) -> split.Boundaries:
    """A 70/15/15 time split over the matrix itself, not over the database.

    `split.compute` reads every resolved match in the database, and the crawler
    is still adding them. Re-deriving boundaries now puts almost the whole
    parquet before the newer train_end -- measured 18,652 train against 738
    validation, which inflates the standard error to 0.0042 and makes any
    candidate unfalsifiable.

    The matrix cannot be rebuilt to match the database without also rebuilding
    the norms it was made with, so for an internal comparison the honest split
    is the one taken over the rows actually present. Every candidate sees the
    same split, which is what the comparison needs; the absolute numbers are
    therefore close to, but not identical to, the README's.
    """
    t = np.sort(df["started_at"].to_numpy())
    return split.Boundaries(
        train_end=int(t[int(len(t) * split.TRAIN_FRAC)]),
        val_end=int(t[int(len(t) * (split.TRAIN_FRAC + split.VAL_FRAC))]),
        n_matches=len(t),
    )


def coverage_sweep(path, led):
    """Does admitting low-coverage matches to TRAINING help?

    The evaluation set is held fixed at coverage >= 5 -- the population the
    shipped model is actually scored on -- while the training threshold varies.
    Sweeping both at once would change the validation rows with every step and
    make the log losses incomparable, which is the easy way to read a different
    denominator as a better model.
    """
    df = pd.read_parquet(path).sort_values("started_at").reset_index(drop=True)
    df = df.drop_duplicates(subset="match_id", keep="first")
    b = matrix_boundaries(df)
    df = split.apply(df, b)
    cols = feature_columns(df[df["coverage"] >= 5])

    va = df[(df["slice"] == "val") & (df["coverage"] >= 5)]
    Xva, yva = va[cols].to_numpy(float), va["target"].to_numpy(int)

    def fit_at(k):
        tr = df[(df["slice"] == "train") & (df["coverage"] >= k)]
        m = fit_logistic(tr[cols].to_numpy(float), tr["target"].to_numpy(int))
        p = m.predict_proba(Xva)[:, 1]
        return len(tr), evaluate.score(f"cov{k}", yva, p), p

    # This sweep scores on its own validation rows, so it gets its own
    # baseline: the threshold currently shipped. Folding it into the main
    # ledger would compare log losses computed on different denominators.
    n5, base, p5 = fit_at(5)
    own = Ledger(base.log_loss, evaluate.log_loss_standard_error(yva, p5))
    print(f"  coverage sweep: fixed val of {len(va):,} rows at coverage >= 5, "
          f"baseline is the shipped threshold 5")
    own.add("train coverage >= 5 (shipped)", base.log_loss, base.auc,
            base.accuracy * 100, note=f"train {n5:,}")
    for k in (0, 3, 4, 6):
        n, sc, _ = fit_at(k)
        if n < 500:
            continue
        own.add(f"train coverage >= {k}", sc.log_loss, sc.auc,
                sc.accuracy * 100, note=f"train {n:,}")
    return own


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="experiments")
    ap.add_argument("--coverage", action="store_true",
                    help="include the min_coverage sweep")
    ap.add_argument("--bt-c", default="0.02,0.1,0.5",
                    help="L2 strengths to try for Bradley-Terry")
    ap.add_argument("--bt-min-appearances", type=int, default=5)
    args = ap.parse_args(argv)

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)

    df = pd.read_parquet(s.database_path.parent / "features.parquet")
    df = df.sort_values("started_at").reset_index(drop=True)
    df = df.drop_duplicates(subset="match_id", keep="first")
    b = matrix_boundaries(df)
    df = split.apply(df, b)
    cols = feature_columns(df)
    tr = df[df["slice"] == "train"]
    va = df[df["slice"] == "val"]
    Xtr, ytr = tr[cols].to_numpy(float), tr["target"].to_numpy(int)
    Xva, yva = va[cols].to_numpy(float), va["target"].to_numpy(int)
    print(f"  {len(cols)} features | train {len(tr):,} | val {len(va):,} "
          f"| test {(df['slice'] == 'test').sum():,} (held back)")

    # --- the incumbent, and the noise floor it is judged against -----
    inc = fit_logistic(Xtr, ytr)
    p_inc = inc.predict_proba(Xva)[:, 1]
    sc = evaluate.score("incumbent", yva, p_inc)
    se = evaluate.log_loss_standard_error(yva, p_inc)
    led = Ledger(sc.log_loss, se)
    led.add("logistic (incumbent)", sc.log_loss, sc.auc, sc.accuracy * 100,
            mirror_error(inc, Xva))

    # --- exact symmetry ----------------------------------------------
    sym = fit_symmetric_logistic(Xtr, ytr, C=0.03)
    sc = evaluate.score("sym", yva, sym.predict_proba(Xva)[:, 1])
    led.add("logistic, exact symmetry", sc.log_loss, sc.auc, sc.accuracy * 100,
            mirror_error(sym, Xva))

    # --- symmetric augmentation --------------------------------------
    Xaug = np.vstack([Xtr, -Xtr])
    yaug = np.concatenate([ytr, 1 - ytr])
    aug = fit_logistic(Xaug, yaug)
    sc = evaluate.score("aug", yva, aug.predict_proba(Xva)[:, 1])
    led.add("logistic, augmented", sc.log_loss, sc.auc, sc.accuracy * 100,
            mirror_error(aug, Xva))

    t = time.time()
    gbm = fit_gbm(Xtr, ytr, Xva, yva)
    sc = evaluate.score("gbm", yva, gbm.predict_proba(Xva)[:, 1])
    led.add("gradient booster", sc.log_loss, sc.auc, sc.accuracy * 100,
            mirror_error(gbm, Xva))

    gbm_aug = fit_gbm(Xaug, yaug, Xva, yva)
    sc = evaluate.score("gbm aug", yva, gbm_aug.predict_proba(Xva)[:, 1])
    led.add("gradient booster, augmented", sc.log_loss, sc.auc,
            sc.accuracy * 100, mirror_error(gbm_aug, Xva))
    print(f"  (booster fits took {time.time() - t:.0f}s)")

    # --- Bradley-Terry -------------------------------------------------
    rosters = load_rosters(conn)
    last = None
    diag: list[str] = []
    for C in [float(x) for x in args.bt_c.split(",")]:
        t = time.time()
        sub, s_tr, m_tr = expanding_bt(
            conn, rosters, tr, min_appearances=args.bt_min_appearances, C=C)
        last = strength.fit(conn, b.train_end,
                            min_appearances=args.bt_min_appearances, C=C)
        s_va, m_va = bt_columns(va["match_id"].tolist(), rosters, last)

        ytr2 = sub["target"].to_numpy(int)
        Xsub = sub[cols].to_numpy(float)

        # Fair comparison: the incumbent refitted on the SAME reduced rows.
        base = fit_logistic(Xsub, ytr2)
        sc_b = evaluate.score("base", yva, base.predict_proba(Xva)[:, 1])
        led.add(f"logistic, reduced train (C={C})", sc_b.log_loss, sc_b.auc,
                sc_b.accuracy * 100, mirror_error(base, Xva),
                note=f"n={len(sub):,}")

        Xtr2 = np.column_stack([Xsub, s_tr, m_tr])
        Xva2 = np.column_stack([Xva, s_va, m_va])
        bt = fit_logistic(Xtr2, ytr2)
        sc = evaluate.score("bt", yva, bt.predict_proba(Xva2)[:, 1])
        led.add(f"logistic + Bradley-Terry (C={C})", sc.log_loss, sc.auc,
                sc.accuracy * 100, mirror_error(bt, Xva2),
                note=f"{last.n_players:,} players, {time.time() - t:.0f}s")

        # A null result is only worth reporting if it says WHY. Strength on
        # its own separates "no signal at all" from "signal the other 52
        # features already carry", and those call for different conclusions.
        alone = evaluate.score("bt alone", yva, 1 / (1 + np.exp(-s_va)))
        coef = bt.named_steps["logisticregression"].coef_[0]
        diag.append(
            f"    C={C:<5} strength alone: auc {alone.auc:.3f}  "
            f"| weight the model gave it: sum {coef[-2]:+.4f} "
            f"max {coef[-1]:+.4f}  (mean |weight| on the other 52: "
            f"{np.abs(coef[:-2]).mean():.4f})")

    # --- coverage sweep ------------------------------------------------
    cov_led = None
    if args.coverage:
        path = s.database_path.parent / "features_cov0.parquet"
        if path.exists():
            cov_led = coverage_sweep(path, led)
        else:
            print(f"  (skipping coverage sweep: {path.name} not built)")

    print(led.table())
    if cov_led is not None:
        print(cov_led.table().replace(
            "VALIDATION RESULTS   (test slice deliberately not scored)",
            "COVERAGE SWEEP   (own baseline: its own validation rows)"))

    if diag:
        print("\n  Why Bradley-Terry landed where it did:")
        print("\n".join(diag))

    if last is not None:
        th = np.array(list(last.theta.values()))
        print("\n  Bradley-Terry, last fit:")
        print(f"    fitted on {last.n_matches:,} matches before {last.as_of}")
        print(f"    {last.n_players:,} players with a free parameter, "
              f"pooled strength {last.pooled:+.4f}")
        print(f"    theta std {th.std():.4f}, range "
              f"{th.min():+.3f} to {th.max():+.3f}")
    print(f"\n  A candidate counts only if delta beats -{se:.4f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
