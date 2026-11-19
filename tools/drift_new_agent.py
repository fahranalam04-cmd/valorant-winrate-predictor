"""What happened to the model when a new agent shipped?

    python tools/drift_new_agent.py --old models/backup-rebuild-<stamp>
        [--manifest-rev <commit>] [--agent Hush] [--release 2026-10-27]

Takes a model bundle saved *before* the agent existed and replays every match
it never saw -- everything after the newest match in its training database --
exactly as it would have run live: its own norms, its own prior, and its own
agent->role table, which does not know the new agent. Then it splits those
matches at the release date and asks whether the model got worse.

Three comparisons, each with a bootstrap standard error:

  1. before vs after the release, on the old model as it would have run
  2. after the release, with the new agent in the lobby vs without
  3. the same post-release matches with the agent's role filled in, to see
     how much of any gap was the missing role rather than the agent itself

Everything here is unseen by the old model, so it is a fair test of drift.
The gaps are expected to be small and the post-release sample is a few
thousand matches; read the standard errors before reading the gaps.

Reads only. Writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")

from valwr import config
from valwr.features import build as fb
from valwr.model import serving
from valwr.store import reference, schema, temporal

MIN_COVERAGE = 5


def _epoch(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp())


def _frame(conn, matches, bundle, roles) -> pd.DataFrame:
    recs = []
    for m in matches:
        rows = temporal.match_roster(conn, m["match_id"])
        mf = fb.build_match(conn, dict(m), rows, bundle["norms"],
                            bundle["prior_rate"], roles)
        if mf is None or mf.target is None or mf.coverage < MIN_COVERAGE:
            continue
        d = {"match_id": mf.match_id, "started_at": mf.started_at,
             "target": mf.target}
        d.update(mf.values)
        recs.append(d)
    return pd.DataFrame.from_records(recs)


def _score(y: np.ndarray, p: np.ndarray, boot: int, rng) -> dict:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    idx = rng.integers(0, len(y), size=(boot, len(y)))
    return {"n": len(y), "log_loss": float(ll.mean()),
            "ll_se": float(ll[idx].mean(axis=1).std()),
            "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
            "acc": float(((p > 0.5) == y).mean()),
            "ll": ll}


def _row(label: str, s: dict) -> None:
    print(f"  {label:<38} n={s['n']:>5,}  log loss {s['log_loss']:.4f} "
          f"+/- {s['ll_se']:.4f}  AUC {s['auc']:.3f}  acc {s['acc']:.1%}")


def _gap(label: str, a: dict, b: dict) -> None:
    gap = b["log_loss"] - a["log_loss"]
    se = (a["ll_se"] ** 2 + b["ll_se"] ** 2) ** 0.5
    word = "worse" if gap > 0 else "better"
    print(f"  {label}: {gap:+.4f} ({word}), {abs(gap) / se:.1f} SE "
          f"-- {'noise' if abs(gap) < 2 * se else 'a real difference'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="drift_new_agent")
    ap.add_argument("--old", required=True, type=Path,
                    help="directory holding a model.joblib saved before the agent")
    ap.add_argument("--agent", default="Hush")
    ap.add_argument("--release", default="2026-10-27", help="YYYY-MM-DD, UTC")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--manifest-rev", default="HEAD",
                    help="git revision whose reports/model_manifest.json "
                         "describes the old model")
    args = ap.parse_args(argv)

    bundle = joblib.load(args.old / "model.joblib")
    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    release = _epoch(args.release)

    # Unseen by the old model means newer than anything in the database it
    # was built from. The committed manifest records that until the next
    # rebuild rewrites it, so read it from git, as of --manifest-rev.
    import json
    import subprocess
    try:
        text = subprocess.run(
            ["git", "show", f"{args.manifest_rev}:reports/model_manifest.json"],
            capture_output=True, text=True, check=True).stdout
        cutoff = json.loads(text)["built"]["database"]["newest_match"]
    except (subprocess.CalledProcessError, KeyError, ValueError):
        print(f"no model manifest at {args.manifest_rev}; pass --manifest-rev "
              "for the commit the old model was built at")
        return 1
    matches = conn.execute(
        "SELECT * FROM matches WHERE winner IN (?,?) AND started_at > ? "
        "ORDER BY started_at", (fb.TEAM_A, fb.TEAM_B, cutoff)).fetchall()
    has_agent = {r[0] for r in conn.execute(
        "SELECT DISTINCT match_id FROM match_players WHERE agent = ?",
        (args.agent,))}

    old_roles = bundle["roles"]
    fixed_roles = reference.agent_roles(conn)
    print(f"old model: {bundle['best']}, {len(bundle['columns'])} features; "
          f"{args.agent} role in its table: {old_roles.get(args.agent)!r}")
    print(f"matches it never saw: {len(matches):,} "
          f"(after {datetime.fromtimestamp(cutoff, timezone.utc):%Y-%m-%d})\n")

    df = _frame(conn, matches, bundle, old_roles)
    df["post"] = df["started_at"] >= release
    df["with_agent"] = df["match_id"].isin(has_agent)
    p = serving.probabilities(bundle, df)
    y = df["target"].to_numpy(int)
    rng = np.random.default_rng(0)

    pre = _score(y[~df.post], p[~df.post], args.boot, rng)
    post = _score(y[df.post], p[df.post], args.boot, rng)
    print(f"1. Old model as it ran, before vs after {args.agent} "
          f"({args.release})")
    _row("before release", pre)
    _row("after release", post)
    _gap("after minus before", pre, post)

    w = df.post & df.with_agent
    wo = df.post & ~df.with_agent
    with_s = _score(y[w], p[w], args.boot, rng)
    without_s = _score(y[wo], p[wo], args.boot, rng)
    print(f"\n2. After release: {args.agent} in the lobby or not")
    _row(f"without {args.agent}", without_s)
    _row(f"with {args.agent}", with_s)
    _gap(f"with minus without", without_s, with_s)

    post_matches = [m for m in matches if m["started_at"] >= release
                    and m["match_id"] in has_agent]
    fixed = _frame(conn, post_matches, bundle, fixed_roles)
    if len(fixed):
        pf_ = serving.probabilities(bundle, fixed)
        fixed_s = _score(fixed["target"].to_numpy(int), pf_, args.boot, rng)
        print(f"\n3. Lobbies with {args.agent}, role missing vs filled in "
              f"({fixed_roles.get(args.agent)})")
        _row("as it ran (no role)", with_s)
        _row("role filled in", fixed_s)
        # The same matches scored twice, so the gap is paired: bootstrap the
        # per-match difference rather than adding two independent errors.
        as_ran = df.loc[w, ["match_id"]].assign(ll=with_s["ll"])
        both = as_ran.merge(fixed[["match_id"]].assign(ll_fixed=fixed_s["ll"]),
                            on="match_id")
        d = (both["ll_fixed"] - both["ll"]).to_numpy()
        idx = rng.integers(0, len(d), size=(args.boot, len(d)))
        se = d[idx].mean(axis=1).std()
        print(f"  filled minus as-ran (paired, {len(d):,} matches): "
              f"{d.mean():+.4f}, {abs(d.mean()) / se:.1f} SE -- "
              f"{'noise' if abs(d.mean()) < 2 * se else 'a real difference'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
