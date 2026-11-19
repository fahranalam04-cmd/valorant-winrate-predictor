"""Do the team-composition features earn their place in the win model?

    python tools/ablate_role_coverage.py [--boot 2000]

Refits the shipped logistic on the cached feature matrix with different sets
of composition columns, so the only thing that changes between rows is which
columns go in. Same split, same fit, same matches.

Variants: the shipped model (every role count, has-role flag and
role_balance), no composition at all, and no composition plus single
"has a controller / sentinel / initiator" flags. Each one is scored on the
time-ordered test set, with a paired bootstrap against "no comp" so the gaps
come with their noise.

It also prints the raw win rate of the team that has a role, in matches
where exactly one team is missing it. That number isn't controlled for skill,
so treat it as a description, not an effect.

Agent pairings / synergy are deliberately not tested. They are left out of
the model by design: agents that pair well on paper don't bring chemistry
between the players every game, so the signal is too variable to trust.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, ".")

from valwr import config
from valwr.model import split
from valwr.model.train import feature_columns, fit_logistic
from valwr.store import schema

COMP = ["d_n_duelist", "d_has_duelist", "d_n_controller", "d_has_controller",
        "d_n_initiator", "d_has_initiator", "d_n_sentinel", "d_has_sentinel",
        "d_role_balance"]
BASELINE = "no comp at all"


def _ll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ablate_role_coverage")
    ap.add_argument("--boot", type=int, default=2000,
                    help="bootstrap resamples for the paired comparison")
    args = ap.parse_args(argv)

    s = config.load(require_key=False)
    cache = s.database_path.parent / "features.parquet"
    if not cache.exists():
        print(f"no {cache.name} -- run `python -m valwr.model.train` first")
        return 1
    b = split.compute(schema.connect(s.database_path))
    df = pd.read_parquet(cache)
    df = split.apply(df, b).sort_values("started_at").reset_index(drop=True)
    df = df.drop_duplicates(subset="match_id", keep="first")

    shipped = feature_columns(df)
    no_comp = [c for c in shipped if c not in COMP]
    variants = {
        "shipped (all comp features)": shipped,
        BASELINE: no_comp,
        "no comp + has_controller": no_comp + ["d_has_controller"],
        "no comp + has_controller + has_sentinel":
            no_comp + ["d_has_controller", "d_has_sentinel"],
        "no comp + has_controller + has_initiator":
            no_comp + ["d_has_controller", "d_has_initiator"],
    }

    tr, va, te = (df[df["slice"] == k] for k in ("train", "val", "test"))
    ytr, yva, yte = (d["target"].to_numpy(int) for d in (tr, va, te))
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}\n")

    probs, fitted = {}, {}
    for name, cols in variants.items():
        m = fit_logistic(tr[cols].to_numpy(float), ytr)
        fitted[name] = (m, cols)
        pv = m.predict_proba(va[cols].to_numpy(float))[:, 1]
        pt = m.predict_proba(te[cols].to_numpy(float))[:, 1]
        probs[name] = pt
        print(f"{name:<42} cols={len(cols):>2}  val LL {log_loss(yva, pv):.5f}  "
              f"test LL {log_loss(yte, pt):.5f}  "
              f"Brier {brier_score_loss(yte, pt):.5f}  "
              f"AUC {roc_auc_score(yte, pt):.4f}")

    print(f"\nPaired bootstrap on test log loss, minus '{BASELINE}' "
          "(negative is better; under ~2 SE is noise)")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(te), size=(args.boot, len(te)))
    base = _ll(yte, probs[BASELINE])
    for name in variants:
        if name == BASELINE:
            continue
        d = _ll(yte, probs[name]) - base
        boots = d[idx].mean(axis=1)
        print(f"  {name:<42} delta {d.mean():+.5f}  SE {boots.std():.5f}  "
              f"95% CI [{np.percentile(boots, 2.5):+.5f}, "
              f"{np.percentile(boots, 97.5):+.5f}]")

    print("\nRaw win rate of the team that HAS the role, when exactly one "
          "team is missing it (all slices, not skill-controlled)")
    for role in ("controller", "sentinel", "initiator", "duelist"):
        col = f"d_has_{role}"
        sub = df[df[col] != 0]
        # d = A - B, so +1 means team A has the role and team B does not.
        wins = np.where(sub[col] > 0, sub["target"], 1 - sub["target"])
        wr = wins.mean()
        half = 1.96 * np.sqrt(wr * (1 - wr) / len(sub))
        print(f"  {role:<11} {len(sub):>6,} matches ({len(sub) / len(df):5.1%})"
              f"  wins {wr:6.2%} +/- {half:.2%}")

    m, cols = fitted["no comp + has_controller + has_initiator"]
    coef = m[-1].coef_[0]
    order = list(np.argsort(-np.abs(coef)))
    print("\nStandardised coefficients in the controller + initiator variant")
    for c in ("d_has_controller", "d_has_initiator"):
        i = cols.index(c)
        print(f"  {c:<18} {coef[i]:+.4f}  rank {order.index(i) + 1} of {len(cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
