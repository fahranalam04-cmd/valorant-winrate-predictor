"""What your own matches say about the model, and whether a retrain helps.

    python tools/improve.py             # the report
    python tools/improve.py --retrain   # measure, retrain, measure again

The dashboard records every prediction it makes and, once the match is
published, the result. That record is the only measurement of this model that
is not taken on data collected alongside its own training set -- which makes it
the honest one, and the one worth acting on.

Two things it does.

**Reports where it is wrong.** Accuracy overall, then split by map, by how much
of the lobby was known, by stated confidence and by predicted band, with the
interval on every figure so a five-match streak is not read as a finding. It
also re-scores every recorded match with the *current* model, through the
replay path, so a model trained since then can be compared against the one that
was live at the time -- on the same matches, at the same `as_of`.

**Checks the measurement is still fair.** A match that has drifted into the
training window can no longer judge the model, and the report says so rather
than quietly counting it.

Nothing here retrains on the prediction log. The finished matches are stored
like any other and reach training the usual way, through the time-ordered
split; the log of what was predicted is never a feature or a label.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from valwr import config
from valwr.live import review
from valwr.model import split
from valwr.store import schema

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "live_review.json"


def bundle_path():
    s = config.load(require_key=False)
    return s.models_path / "model.joblib"


def load_bundle():
    import joblib
    path = bundle_path()
    if not path.exists():
        return None
    return joblib.load(path)


def rescore(conn, bundle, index, rows) -> dict:
    """Re-predict every recorded match with the model as it stands now.

    Through `dash/replay`, so each match is rebuilt at its own start time and
    the temporal firewall applies: the match cannot inform its own prediction.
    """
    from valwr.dash.replay import replay_state

    scored = {"n": 0, "correct": 0, "brier": 0.0, "failed": 0}
    for r in rows:
        try:
            state = replay_state(conn, r["match_id"], bundle, index,
                                 r["own_puuid"] or "")
        except Exception:                            # noqa: BLE001
            scored["failed"] += 1
            continue
        pred = (state or {}).get("prediction")
        if not pred or r["own_won"] is None:
            scored["failed"] += 1
            continue
        p_own = pred["own_probability"]
        scored["n"] += 1
        scored["correct"] += int((p_own >= 0.5) == bool(r["own_won"]))
        scored["brier"] += (p_own - r["own_won"]) ** 2
    if scored["n"]:
        scored["accuracy"] = scored["correct"] / scored["n"]
        scored["brier"] /= scored["n"]
    return scored


def fairness(conn, rows) -> dict:
    """Which recorded matches can still judge the model, and which cannot.

    Empty when the store holds too few matches to draw a split at all, which
    is a fresh install rather than a problem worth reporting.
    """
    try:
        b = split.compute(conn)
    except ValueError:
        return {}
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        started = conn.execute(
            "SELECT started_at FROM matches WHERE match_id = ?",
            (r["match_id"],)).fetchone()
        if started is None:
            continue
        out[b.slice_of(started["started_at"])].append(r["match_id"])
    return {k: len(v) for k, v in out.items()}


def table(rows, title, label="") -> None:
    if not rows:
        return
    print(f"\n  {title}")
    print(f"    {label or 'group':<22}{'n':>5}{'predicted':>11}{'won':>7}"
          f"{'called right':>14}")
    for r in sorted(rows, key=lambda r: -r["n"]):
        print(f"    {r['label'][:21]:<22}{r['n']:>5}{r['predicted'] * 100:>10.0f}%"
              f"{r['actual'] * 100:>6.0f}%{r['accuracy'] * 100:>13.0f}%")


def report(conn) -> dict:
    card = review.scorecard(conn)
    rows = list(conn.execute(
        "SELECT * FROM live_predictions WHERE settled_at IS NOT NULL "
        "AND correct IS NOT NULL AND standard_mode = 1"))

    print("=" * 70)
    print("WHAT YOUR OWN MATCHES SAY")
    print("=" * 70)
    print(f"  {card['recorded']} recorded, {card['settled']} scored, "
          f"{card['pending']} waiting for a result")
    for line in card["insights"]:
        print(f"  - {line}")

    table(card["by_map"], "By map", "map")
    table(card["by_coverage"], "By how much of the lobby was known", "coverage")
    table(card["by_confidence"], "By stated confidence", "confidence")
    table(card["calibration"], "By what it predicted", "predicted band")

    order = card.get("order") or {}
    if order.get("players"):
        print(f"\n  The 0-100 score ordered {order['players']} player rankings "
              f"at rho {order['order']:+.2f}; "
              f"{order['within_one_rate'] * 100:.0f}% within one place.")

    fair = fairness(conn, rows) if rows else {}
    if fair:
        print()
        print(f"  Where these matches sit in the current split: "
              f"{fair['train']} train, {fair['val']} validation, "
              f"{fair['test']} test.")
        if fair["train"]:
            print(f"  {fair['train']} have drifted into the training "
                  f"window and no longer judge the model fairly. They stay "
                  f"listed, and are what a retrain is measured *against*, "
                  f"never on.")
    card["fairness"] = fair
    return card


def suggestions(card: dict) -> list[str]:
    """What the record points at, ranked by how much evidence is behind it."""
    out = []
    comp = card["competitive"]
    if comp.get("n", 0) < 30:
        out.append("Play more with the dashboard open. Under about 30 matches "
                   "the interval is wider than any effect worth chasing.")
    cov = [b for b in card["by_coverage"] if b["n"] >= 5]
    if len(cov) >= 2:
        low = min(cov, key=lambda b: b["accuracy"])
        if low["label"].split("/")[0].isdigit() and int(low["label"].split("/")[0]) < 8:
            out.append("Keep the crawler running: the thin-coverage matches are "
                       "where it does worst, and coverage is the one input you "
                       "control.")
    order = card.get("order") or {}
    if order.get("players", 0) >= 50 and (order.get("order") or 0) < 0.2:
        out.append("The 0-100 score barely orders the scoreboard on your "
                   "lobbies. It is ACS-led, so it inherits role bias -- worth "
                   "trying a role-adjusted score before anything else.")
    gap = comp.get("actual", 0) - comp.get("predicted", 0)
    if comp.get("n", 0) >= 30 and abs(gap) > 2 * comp["se"]:
        out.append("Your results sit outside what it predicts on average. The "
                   "model has no term for you specifically; a per-account "
                   "offset, fitted on your own history, is the obvious test.")
    out.append("Retrain when the crawl has grown: python tools/improve.py "
               "--retrain measures the current model on these same matches, "
               "retrains, and measures again.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="improve")
    ap.add_argument("--retrain", action="store_true",
                    help="retrain, then re-measure on the same matches")
    args = ap.parse_args(argv)

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    card = report(conn)

    rows = list(conn.execute(
        "SELECT * FROM live_predictions WHERE settled_at IS NOT NULL "
        "AND correct IS NOT NULL AND standard_mode = 1"))
    if not rows:
        print("\n  Nothing scored yet, so there is nothing to act on.")
        OUT.write_text(json.dumps(card, indent=2), encoding="utf-8")
        return 0

    bundle = load_bundle()
    index = None
    try:
        from valwr.rating import potential as pot
        index = pot.PerfIndex.load()
    except Exception:                                # noqa: BLE001
        index = None

    before = None
    if bundle is not None:
        before = rescore(conn, bundle, index, rows)
        live = sum(r["correct"] for r in rows) / len(rows)
        print(f"\n  On these {len(rows)} matches: the model that was live at "
              f"the time called {live * 100:.0f}% right; the model as it "
              f"stands now calls {before.get('accuracy', 0) * 100:.0f}% "
              f"({before['n']} replayed, {before['failed']} could not be).")

    print("\n" + "=" * 70)
    print("WHAT TO DO WITH IT")
    print("=" * 70)
    for line in suggestions(card):
        print(f"  - {line}")

    if args.retrain:
        print("\n" + "=" * 70)
        print("RETRAINING")
        print("=" * 70)
        started = time.time()
        # Everything, through the one command. This used to retrain and then
        # refit only the old score's index -- not the per-role one the page
        # scores with -- so every 0-100 on the page sat against a reference
        # older than its model.
        sys.path.insert(0, str(ROOT / "tools"))
        import rebuild
        if rebuild.run(s.models_path, s.database_path) != 0:
            print("  the rebuild failed; the previous models are back in place")
            return 1
        print(f"\n  retrained in {time.time() - started:.0f}s")

        conn = schema.connect(s.database_path)       # the bundle changed
        after = rescore(conn, load_bundle(), index, rows)
        if before and before.get("n") and after.get("n"):
            print(f"\n  On the same {after['n']} matches: "
                  f"{before['accuracy'] * 100:.0f}% -> "
                  f"{after['accuracy'] * 100:.0f}% called right, "
                  f"Brier {before['brier']:.3f} -> {after['brier']:.3f}")
            fair = fairness(conn, rows)
            if fair["train"]:
                print(f"  {fair['train']} of them are now inside the training "
                      f"window, so that comparison flatters the new model. "
                      f"Read the {fair['val'] + fair['test']} outside it.")
        print("\n  Then, to bring the rest of the project with it:")
        print("    python -m valwr.model.analyze")
        print("    python tools/validate_potential.py --write-index")
        print("    python tools/model_metrics.py")
        print("    python -m valwr.sandbox benchmark")

    card["suggestions"] = suggestions(card)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
