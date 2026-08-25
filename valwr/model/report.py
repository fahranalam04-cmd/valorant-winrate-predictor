"""Regenerate the README results table: python -m valwr.model.report

The numbers in a hand-written results table go stale the moment the crawler
adds another match, and this one had already drifted four times -- each edit a
chance to transcribe a figure wrong, or to leave a claim standing that the
latest run no longer supports.

So the table is generated from reports/results.json instead, between markers
in the README. Run it after every training run. Prose around the markers is
never touched; only the measured figures are.
"""

from __future__ import annotations

import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "reports" / "results.json"

START = "<!-- results:start -->"
END = "<!-- results:end -->"

# Presentation names, and the order the ladder should be read in. Anything not
# listed still appears, after these.
DISPLAY = {
    "logistic regression": "Logistic regression (52 features)",
    "logistic + margin blend": "Logistic + margin blend",
    "margin regression": "Margin regression",
    "gradient boosting": "Gradient boosting",
    "avg rating (fitted)": "Player rating alone (1 feature)",
    "best player rank": "Best player's rank (1 feature)",
    "avg rank (fitted)": "Average rank — *the baseline to beat*",
    "coin flip": "Coin flip",
}


def ci95(accuracy: float, n: int) -> float:
    return 1.96 * math.sqrt(max(accuracy * (1 - accuracy), 1e-9) / n) * 100


def render(res: dict) -> str:
    rows = sorted(res["results"], key=lambda r: r["log_loss"])
    best = rows[0]
    shuf = res.get("shuffled", {})

    out = [
        f"Measured on a held-out, time-ordered test set of "
        f"**{res['n_test']:,} matches** never touched during training or "
        f"tuning. {res['n_train']:,} training matches; "
        f"{res['n_features']} features.",
        "",
        "| Model | Log loss | AUC | Accuracy |",
        "|---|---|---|---|",
    ]
    for r in rows:
        # Calibrated variants are an implementation detail, not a rung.
        if "isotonic" in r["name"] or "platt" in r["name"]:
            continue
        label = DISPLAY.get(r["name"], r["name"])
        mark = "**" if r is best else ""
        acc = f"{r['accuracy'] * 100:.1f}%"
        if r is best:
            acc += f" ± {ci95(r['accuracy'], r['n']):.1f}%"
        out.append(f"| {mark}{label}{mark} | {mark}{r['log_loss']:.4f}{mark} | "
                   f"{r['auc']:.3f} | {mark}{acc}{mark} |")

    if shuf:
        out += [
            "",
            f"**Leakage check: {shuf['draws']} independent label shuffles, mean "
            f"AUC {shuf['mean']:.4f} ± {shuf['std']:.4f}, "
            f"{shuf['draws_over_threshold']} of {shuf['draws']} above the 0.55 "
            f"alarm threshold.** One draw is not a test — a single shuffle has "
            f"a standard deviation near {shuf['std']:.3f}, so any one of them "
            f"can land anywhere and mean nothing.",
        ]

    strata = res.get("coverage_strata") or []
    if strata:
        aucs = [c["auc"] for c in strata]
        monotonic = all(a <= b for a, b in zip(aucs, aucs[1:]))
        out += [
            "",
            "Split by how many of the ten players had prior history: "
            + ", ".join(f"**{c['name']}** {c['auc']:.3f}" for c in strata)
            + (". Rising with coverage, as expected."
               if monotonic else
               ". Not monotonic — see the retraction below."),
        ]
    return "\n".join(out)


def main(argv=None) -> int:
    if not RESULTS.exists():
        print(f"no results at {RESULTS}; run python -m valwr.model.train first")
        return 1
    res = json.loads(RESULTS.read_text(encoding="utf-8"))
    text = README.read_text(encoding="utf-8")

    if START not in text or END not in text:
        print(f"README is missing the {START} / {END} markers")
        return 1

    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    README.write_text(f"{head}{START}\n{render(res)}\n{END}{tail}",
                      encoding="utf-8", newline="\n")

    best = min(res["results"], key=lambda r: r["log_loss"])
    print(f"README results regenerated from {RESULTS.name}")
    print(f"  best: {best['name']} — log loss {best['log_loss']:.4f}, "
          f"AUC {best['auc']:.3f}, {best['accuracy'] * 100:.1f}% "
          f"(n={res['n_test']:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
