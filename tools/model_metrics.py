"""Every model this project built or tried, every metric, in one place.

    python tools/model_metrics.py

train.py scores each candidate once on the held-out test slice, and its report
records log loss, Brier, AUC, accuracy and calibration error -- never precision
or recall. This rebuilds every candidate's test predictions from the saved
bundle, on the same split, and **proves they are the same predictions** by
reproducing each log loss in reports/results.json before reporting anything
new. Nothing is selected on here: which model ships was decided by train.py.

Precision and recall need a positive class and a threshold: "Blue wins", the
training target, at 0.5 -- the same call accuracy makes. A model that always
says Blue has recall 1.0 and precision equal to Blue's win rate, so the share
of matches each model calls for Blue is reported beside them.

Writes reports/metrics.json and regenerates the tables between the markers in
docs/MODEL-CHOICE.md. Two further sources are folded in when present, each
produced by the tool that owns that measurement:

    python tools/experiments.py --coverage --json       validation-period experiments
    python tools/validate_potential.py --json           potential score (test period)
    python tools/validate_potential.py --flag --json    above-rank flag (test period)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from valwr import config
from valwr.model import evaluate, split
from valwr.model.baselines import FittedBaseline
from valwr.model.train import calibrate
from valwr.store import schema

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "reports" / "results.json"
OUT = ROOT / "reports" / "metrics.json"
EXPERIMENTS = ROOT / "reports" / "experiments.json"
PLAYERS = ROOT / "reports" / "player_models.json"
DOC = ROOT / "docs" / "MODEL-CHOICE.md"
START, END = "<!-- metrics:start -->", "<!-- metrics:end -->"

# Reproduced log losses must match the recorded ones to this. A looser match
# would mean different predictions, and every new number would be describing
# a model other than the one train.py scored.
TOLERANCE = 1e-6

# What each candidate is, in a sentence. `{n}` is the feature count.
WHAT = {
    "coin flip": "Always 50%. The floor.",
    "avg rank (fitted)": "One feature, the gap in average rank, through a logistic curve.",
    "best player rank": "One feature, the gap between each team's highest rank.",
    "avg rating (fitted)": "One feature, the gap in average player rating.",
    "acs alone": "One feature, the gap in average combat score. A diagnostic, not a training candidate.",
    "logistic regression": "A weighted sum of all {n} team-difference features, through a logistic curve.",
    "logistic + isotonic": "The logistic model, recalibrated by a stepwise fit on validation.",
    "logistic + platt": "The logistic model, recalibrated by a sigmoid fit on validation.",
    "gradient boosting": "LightGBM: many small decision trees over the same {n} features. Can learn interactions.",
    "gbm + isotonic": "The booster, recalibrated by a stepwise fit on validation.",
    "gbm + platt": "The booster, recalibrated by a sigmoid fit on validation.",
    "margin regression": "Ridge regression on the round margin (13-3 vs 13-11), mapped to a probability on validation.",
    "logistic + margin blend": "The average of the logistic and margin-regression probabilities.",
}


def load_slices(conn):
    """train, val, test -- ordered and de-duplicated exactly as train.py does."""
    s = config.load(require_key=False)
    df = pd.read_parquet(s.database_path.parent / "features.parquet")
    df = split.apply(df, split.compute(conn)).sort_values("started_at")
    df = df.reset_index(drop=True).drop_duplicates(subset="match_id", keep="first")
    return tuple(df[df["slice"] == k] for k in ("train", "val", "test"))


def predictions(bundle: dict, tr, va, te) -> dict[str, np.ndarray]:
    """Test-set probabilities for every candidate train.py scored.

    The calibrated variants are rebuilt by calling train.calibrate on the same
    validation probabilities, which is deterministic; the bundle does not keep
    the calibrators because nothing ships them.
    """
    est, cols = bundle["estimators"], bundle["columns"]
    Xva, Xte = va[cols].to_numpy(float), te[cols].to_numpy(float)
    yva = va["target"].to_numpy(int)

    p = {"coin flip": np.full(len(te), 0.5)}
    for name in ("avg rank (fitted)", "best player rank", "avg rating (fitted)"):
        if name in est:
            p[name] = np.asarray(est[name](te), dtype=float)

    for short, key in (("logistic", "logistic regression"), ("gbm", "gradient boosting")):
        p[key] = est[short].predict_proba(Xte)[:, 1]
        cal, (method, _) = calibrate(est[short].predict_proba(Xva)[:, 1], yva, p[key])
        p[f"{short} + {method}"] = np.asarray(cal, dtype=float)

    margin = est["margin_link"].predict_proba(
        est["margin_reg"].predict(Xte).reshape(-1, 1))[:, 1]
    p["margin regression"] = margin
    p["logistic + margin blend"] = 0.5 * p["logistic regression"] + 0.5 * margin
    return p


def verify(recorded: list[dict], probs: dict, y) -> list[str]:
    """Every recorded candidate, reproduced. Returns the ones that were not."""
    bad = []
    for r in recorded:
        if r["name"] not in probs:
            bad.append(f"{r['name']}: not reproduced")
            continue
        got = evaluate.score(r["name"], y, probs[r["name"]]).log_loss
        if abs(got - r["log_loss"]) > TOLERANCE:
            bad.append(f"{r['name']}: recorded {r['log_loss']:.6f}, "
                       f"reproduced {got:.6f}")
    return bad


def row(name: str, y, p, *, shipped: str, candidate: bool, n_features: int) -> dict:
    sc = evaluate.score(name, y, p)
    return {**sc.__dict__, "called_blue": float((np.asarray(p) >= 0.5).mean()),
            "candidate": candidate, "shipped": name == shipped,
            "what": WHAT.get(name, "").format(n=n_features)}


def _f(v, digits=3) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) \
        else f"{v:.{digits}f}"


def _pct(v) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) \
        else f"{v * 100:.1f}%"


def render(metrics: dict, experiments: dict | None = None,
           players: dict | None = None) -> str:
    rate = metrics["blue_win_rate"]
    out = [
        f"Held-out test set: **{metrics['n_test']:,} matches**, of which Blue "
        f"won {rate * 100:.1f}%. Precision and recall treat *Blue wins* as the "
        f"positive class at a 0.5 threshold; PR-AUC's chance level is Blue's "
        f"win rate ({rate:.3f}), not 0.5. Generated by "
        f"`tools/model_metrics.py`.",
        "",
        "| Model | What it does | Log loss | AUC | PR-AUC | Precision | Recall "
        "| F1 | Accuracy | Calls Blue |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in sorted(metrics["models"], key=lambda m: m["log_loss"]):
        b = "**" if m["shipped"] else ""
        label = m["name"] + (" (shipped)" if m["shipped"] else "")
        out.append(
            f"| {b}{label}{b} | {m['what']} | {b}{m['log_loss']:.4f}{b} | "
            f"{_f(m['auc'])} | {_f(m['pr_auc'])} | {_f(m['precision'])} | "
            f"{_f(m['recall'])} | {_f(m['f1'])} | {_pct(m['accuracy'])} | "
            f"{_pct(m['called_blue'])} |")

    if experiments:
        out += [
            "",
            f"**Tried on the validation period** ({experiments['n_val']:,} "
            f"matches; the test set is never used to choose). A candidate had "
            f"to beat the incumbent's log loss by more than one standard error "
            f"({experiments['se']:.4f}) to count. Generated by "
            f"`tools/experiments.py --json`.",
            "",
            "| Candidate | Log loss | vs incumbent | AUC | PR-AUC | Precision "
            "| Recall | F1 | Accuracy | Verdict |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in experiments["rows"]:
            note = f" ({r['note']})" if r.get("note") else ""
            out.append(
                f"| {r['name']}{note} | {r['log_loss']:.4f} | "
                f"{r['delta']:+.4f} | {_f(r['auc'])} | {_f(r['pr_auc'])} | "
                f"{_f(r['precision'])} | {_f(r['recall'])} | {_f(r['f1'])} | "
                f"{_pct(r['accuracy'])} | {r['verdict']} |")
        if experiments.get("coverage"):
            cov = experiments["coverage"]
            out += [
                "",
                f"Coverage sweep, on its own fixed validation rows "
                f"({cov['n_val']:,}, standard error {cov['se']:.4f}):",
                "",
                "| Training threshold | Log loss | vs shipped | AUC | PR-AUC | "
                "Precision | Recall | F1 | Accuracy | Verdict |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
            for r in cov["rows"]:
                out.append(
                    f"| {r['name']} ({r['note']}) | {r['log_loss']:.4f} | "
                    f"{r['delta']:+.4f} | {_f(r['auc'])} | {_f(r['pr_auc'])} | "
                    f"{_f(r['precision'])} | {_f(r['recall'])} | {_f(r['f1'])} | "
                    f"{_pct(r['accuracy'])} | {r['verdict']} |")

    pot = (players or {}).get("potential")
    if pot:
        out += [
            "",
            f"**The player score** ranks the five players on a team; the "
            f"question is who had the best game. Measured on "
            f"{pot['teams']:,} complete test-period teams. Picking one player "
            f"per team, precision and recall are the same number -- the top-1 "
            f"rate, chance 20%. AUC and PR-AUC score every player against "
            f"*had their team's best game* (base rate "
            f"{pot['positive_rate']:.3f}). Generated by "
            f"`tools/validate_potential.py --json`.",
            "",
            "| Ranked by | Top pick right (precision = recall) | AUC | PR-AUC |",
            "|---|---|---|---|",
        ]
        for r in pot["rankers"]:
            out.append(f"| {r['name']} | {_pct(r['top1'])} | {_f(r['auc'])} | "
                       f"{_f(r['pr_auc'])} |")

    flag = (players or {}).get("flag")
    if flag:
        out += [
            "",
            f"**The above-rank flag** is a yes/no call, checked against "
            f"*finished in the top third of their lobby* across "
            f"{flag['players']:,} players in {flag['lobbies']:,} test-period "
            f"lobbies (base rate {flag['base_rate']:.3f}). AUC scores the "
            f"rank-relative rating the flag thresholds. Generated by "
            f"`tools/validate_potential.py --flag --json`.",
            "",
            "| Flags | Precision | Recall | F1 | AUC (rating z) | PR-AUC |",
            "|---|---|---|---|---|---|",
            f"| {_pct(flag['flag_rate'])} of players | {_f(flag['precision'])} "
            f"| {_f(flag['recall'])} | {_f(flag['f1'])} | {_f(flag['auc'])} "
            f"| {_f(flag['pr_auc'])} |",
        ]
    return "\n".join(out)


def write_doc(text: str, doc: Path = DOC) -> bool:
    body = doc.read_text(encoding="utf-8")
    if START not in body or END not in body:
        return False
    head, rest = body.split(START, 1)
    _, tail = rest.split(END, 1)
    doc.write_text(f"{head}{START}\n{text}\n{END}{tail}", encoding="utf-8",
                   newline="\n")
    return True


def main(argv=None) -> int:
    import joblib

    ap = argparse.ArgumentParser(prog="model_metrics")
    ap.add_argument("--print-only", action="store_true",
                    help="print the tables; write neither JSON nor the doc")
    args = ap.parse_args(argv)

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    tr, va, te = load_slices(conn)
    y = te["target"].to_numpy(int)
    if len(te) != results["n_test"]:
        print(f"test slice is {len(te):,} matches but the model was scored on "
              f"{results['n_test']:,}; the database has changed. Retrain first.")
        return 1

    probs = predictions(bundle, tr, va, te)
    bad = verify(results["results"], probs, y)
    if bad:
        print("could not reproduce the recorded predictions:")
        print("\n".join(f"  {b}" for b in bad))
        return 1
    print(f"reproduced all {len(results['results'])} recorded candidates "
          f"to within {TOLERANCE:g} log loss")

    n = len(bundle["columns"])
    shipped = results["shipped"]
    models = [row(k, y, v, shipped=shipped, candidate=True, n_features=n)
              for k, v in probs.items()]
    acs = FittedBaseline("d_acs_mean").fit(tr).predict(te)
    models.append(row("acs alone", y, acs, shipped=shipped, candidate=False,
                      n_features=n))

    metrics = {"n_test": len(te), "blue_win_rate": float(y.mean()),
               "threshold": 0.5, "positive_class": "Blue wins",
               "shipped": shipped, "models": models}
    experiments = (json.loads(EXPERIMENTS.read_text(encoding="utf-8"))
                   if EXPERIMENTS.exists() else None)
    players = (json.loads(PLAYERS.read_text(encoding="utf-8"))
               if PLAYERS.exists() else None)
    text = render(metrics, experiments, players)
    print()
    print(text)
    if args.print_only:
        return 0

    OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    if write_doc(text):
        print(f"regenerated the tables in {DOC.relative_to(ROOT)}")
    else:
        print(f"{DOC.relative_to(ROOT)} has no {START} / {END} markers; "
              f"tables printed only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
