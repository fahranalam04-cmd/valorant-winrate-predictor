"""Fit the population reference for the 0-100 potential score.

Writes models/perf_index.json: component means and standard deviations, plus
the sorted composite values the percentile mapping reads.

Fitted on the **training period only**. Fitting on everything would let the
score's own scale be informed by matches it is later evaluated on -- the same
leak `train.py` avoids by computing its split boundary before building the
feature matrix.

    python tools/build_perf_index.py [--sample 15000]
"""

from __future__ import annotations

import argparse
import random
import sys
import time

sys.path.insert(0, ".")

from valwr import config
from valwr.model import split
from valwr.rating import potential as P
from valwr.rating.normalize import build_norms
from valwr.store import schema


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_perf_index")
    ap.add_argument("--sample", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    b = split.compute(conn)
    print(f"training period ends {b.train_end}")

    norms = build_norms(conn, b.train_end)

    rows = conn.execute(
        "SELECT puuid, started_at, map FROM match_players "
        "WHERE started_at < ? AND rounds_played > 0", (b.train_end,)).fetchall()
    print(f"{len(rows):,} candidate player-matches in the training period")

    rng = random.Random(args.seed)
    picked = rng.sample(list(rows), min(args.sample, len(rows)))

    started = time.time()
    collected: list[P.Components] = []
    for i, r in enumerate(picked, 1):
        c = P.measure(conn, r["puuid"], r["started_at"], r["map"], norms)
        if c is not None:
            collected.append(c)
        if i % 2000 == 0:
            rate = i / (time.time() - started)
            print(f"  {i:,}/{len(picked):,}  ({rate:.0f}/s, "
                  f"{len(collected):,} with history)")

    if len(collected) < 500:
        print(f"only {len(collected)} usable samples; too few to fit an index")
        return 1

    means, stds = {}, {}
    for name in P.WEIGHTS:
        vals = [getattr(c, name) for c in collected]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)
        means[name], stds[name] = mu, var ** 0.5

    index = P.PerfIndex(means=means, stds=stds, quantiles=[],
                        as_of=b.train_end, n=len(collected))
    quantiles = sorted(index.composite(c) for c in collected)

    # Calibrate the above-rank cut so the flag fires on FLAG_TARGET of players
    # rather than on whatever a hand-picked z-score happens to catch. Only
    # players who dominate their lobbies can flag, so the cut is chosen over
    # that subset -- but the target is a fraction of ALL players, which is the
    # number actually worth controlling.
    eligible = [index.z("rating", c.rating) for c in collected
                if c.n_dominance >= P.MIN_DOMINANCE_GAMES
                and c.dominance >= P.DOMINANT]
    want = int(round(P.FLAG_TARGET * len(collected)))
    if eligible and 0 < want <= len(eligible):
        flag_cut = sorted(eligible, reverse=True)[want - 1]
    else:
        flag_cut = P.DEFAULT_FLAG_CUT
        print(f"  (could not calibrate: only {len(eligible)} of "
              f"{len(collected)} dominate their lobbies; keeping {flag_cut})")

    index = P.PerfIndex(means=means, stds=stds, quantiles=quantiles,
                        as_of=b.train_end, n=len(collected),
                        flag_cut=flag_cut)
    fires = sum(1 for c in collected if P.above_rank(index, c).flagged)
    print()
    print(f"  above-rank flag: cut z>={flag_cut:.2f}, fires on "
          f"{fires:,}/{len(collected):,} ({fires / len(collected) * 100:.1f}%)"
          f"  -- {len(eligible):,} players dominated enough to qualify")

    P.INDEX_PATH.parent.mkdir(exist_ok=True)
    P.INDEX_PATH.write_text(index.to_json(), encoding="utf-8")
    print(f"\nwrote {P.INDEX_PATH}  ({len(collected):,} samples, "
          f"{time.time() - started:.0f}s)")
    for name in P.WEIGHTS:
        print(f"  {name:<10} mean {means[name]:+8.3f}   sd {stds[name]:8.3f}")
    print(f"  composite  p10 {quantiles[len(quantiles)//10]:+.3f}   "
          f"p50 {quantiles[len(quantiles)//2]:+.3f}   "
          f"p90 {quantiles[9*len(quantiles)//10]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
