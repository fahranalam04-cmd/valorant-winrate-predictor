"""Replay historical matches through the LIVE path, end to end.

Every other check in this project exercises the *training* path: build the
feature matrix, fit, score. That leaves the code you actually run in a match
verified by whatever games you happened to play -- five, in this case.

This replays real matches exactly as the live client would deliver them: a
roster of ten puuids, teams and agents, and nothing else. No stored features,
no outcome, no stats from the match itself. It then asks two questions.

**Does the live path agree with the training path?** For the same match and
the same model, `predict.predict()` and `features.build.build_match()` must
produce the same probability. They share `build_match`, so any divergence is
in how the live path assembles its inputs -- which is precisely where a
train/serve skew hides. Divergence is reported per feature, not just as a
total, so a mismatch names the column responsible.

**Is it right as often as the training path?** Accuracy on the same matches,
against the same labels.

    python tools/backtest_live.py [--matches 500] [--period test]

Reads only. Writes nothing, and never loads or modifies the model bundle
beyond reading it.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")

import joblib

from valwr import config
from valwr.features import build as fb
from valwr.live import predict as LP
from valwr.live.roster import LiveMatch, LivePlayer
from valwr.live.resolve import Resolution
from valwr.model import evaluate, split
from valwr.store import schema

# The two paths share build_match, so they should agree to floating-point
# noise. Anything above this is a real difference in how inputs are assembled.
AGREEMENT_TOLERANCE = 1e-9


def as_live(rows: list[dict], map_name: str) -> LiveMatch:
    """A stored match, in the shape the client would hand over mid-game.

    Deliberately lossy: puuid, team and agent only. Everything the client does
    not expose -- score, tier, party, the outcome -- is dropped, so the replay
    cannot accidentally use something the live path would never have.
    """
    return LiveMatch(
        match_id=rows[0]["match_id"], phase="coregame", map_name=map_name,
        mode="BombGameMode", flow="Matchmaking",
        players=[LivePlayer(puuid=r["puuid"], team=r["team"], agent_id="x",
                            agent=r["agent"] or "?") for r in rows])


def training_path(conn, rows, match_row, bundle) -> tuple[float, dict] | None:
    """What the training pipeline would compute for this match."""
    mf = fb.build_match(conn, dict(match_row), rows, bundle["norms"],
                        bundle["prior_rate"], bundle["roles"],
                        require_outcome=False)
    if mf is None:
        return None
    cols = bundle["columns"]
    est = (bundle["estimators"].get(bundle["best"])
           or bundle["estimators"]["logistic"])
    x = [[mf.values.get(c, 0.0) for c in cols]]
    if not hasattr(est, "predict_proba"):
        return None                     # a fitted baseline; not comparable here
    return float(est.predict_proba(x)[0][1]), mf.values


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="backtest_live")
    ap.add_argument("--matches", type=int, default=500)
    ap.add_argument("--period", choices=("val", "test"), default="test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    b = split.compute(conn)
    bundle = joblib.load(s.database_path.parent.parent / "models" /
                         "model.joblib")
    print(f"model: {bundle['best']}, {len(bundle['columns'])} features")

    lo, hi = ((b.train_end, b.val_end) if args.period == "val"
              else (b.val_end, 1 << 62))
    matches = conn.execute(
        "SELECT * FROM matches WHERE started_at >= ? AND started_at < ? "
        "AND winner IN ('Blue','Red')", (lo, hi)).fetchall()
    print(f"{len(matches):,} resolved matches in the {args.period} period")

    rng = random.Random(args.seed)
    rng.shuffle(matches)

    agree_max = 0.0
    diverged: list[tuple[str, float]] = []
    per_feature: dict[str, float] = defaultdict(float)
    live_p, train_p, labels = [], [], []
    skipped = 0
    started = time.time()

    for m in matches:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM match_players WHERE match_id = ?",
            (m["match_id"],)).fetchall()]
        if len(rows) != 10:
            skipped += 1
            continue
        as_of = m["started_at"]

        tp = training_path(conn, rows, m, bundle)
        if tp is None:
            skipped += 1
            continue
        p_train, train_values = tp

        live = as_live(rows, m["map"])
        res = Resolution(known={r["puuid"] for r in rows})
        pred = LP.predict(conn, live, bundle, res, rows[0]["puuid"],
                          as_of=as_of)
        if pred is None:
            skipped += 1
            continue

        gap = abs(pred.win_probability - p_train)
        agree_max = max(agree_max, gap)
        if gap > AGREEMENT_TOLERANCE:
            diverged.append((m["match_id"], gap))
            live_values = fb.build_match(
                conn, {**dict(m), "winner": None, "rounds_blue": None,
                       "rounds_red": None, "started_at": as_of},
                LP._roster_rows(live, conn, as_of), bundle["norms"],
                bundle["prior_rate"], bundle["roles"],
                require_outcome=False)
            if live_values is not None:
                for k, v in live_values.values.items():
                    per_feature[k] = max(per_feature[k],
                                         abs(v - train_values.get(k, 0.0)))

        live_p.append(pred.win_probability)
        train_p.append(p_train)
        labels.append(1 if m["winner"] == "Blue" else 0)
        if len(live_p) >= args.matches:
            break

    if not live_p:
        print("no matches could be replayed")
        return 1

    n = len(live_p)
    print(f"replayed {n:,} matches in {time.time() - started:.0f}s "
          f"({skipped:,} skipped)\n")

    print("=" * 66)
    print("DOES THE LIVE PATH AGREE WITH THE TRAINING PATH?")
    print("=" * 66)
    print(f"  largest probability gap : {agree_max:.2e}")
    print(f"  matches over {AGREEMENT_TOLERANCE:.0e}      : "
          f"{len(diverged):,} of {n:,}")
    if per_feature:
        print("\n  features responsible, worst first:")
        for k, v in sorted(per_feature.items(), key=lambda kv: -kv[1])[:8]:
            if v > 1e-12:
                print(f"    {k:<28} max difference {v:.4g}")
    if not diverged:
        print("\n  identical. The live path is the training path.")

    sl = evaluate.score("live", labels, live_p)
    st = evaluate.score("train", labels, train_p)
    print("\n" + "=" * 66)
    print(f"ACCURACY ON THE SAME {n:,} MATCHES")
    print("=" * 66)
    print(f"  {'path':<10}{'logloss':>10}{'auc':>8}{'acc':>9}")
    for name, sc in (("live", sl), ("training", st)):
        print(f"  {name:<10}{sc.log_loss:>10.4f}{sc.auc:>8.3f}"
              f"{sc.accuracy * 100:>8.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
