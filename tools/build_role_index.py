"""Fit the per-role reference population.

    python tools/build_role_index.py [--sample 40000]

Samples player-matches from the TRAINING period, measures each player as they
were before that match, and fits `models/role_index.json`: band-relative
moments for every component, per-agent ability distributions, the role averages
thin players are shrunk toward, and the per-role percentile tables.

Training period only. Fitting on anything later would put the test set inside
the reference the test set is later scored against, which is the leak this
project spends most of its effort avoiding.

Re-run after a crawl grows the database, alongside the other rebuild steps in
docs/SCORE-SPEC.md.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

from valwr import config
from valwr.model import split
from valwr.rating import role_score as R
from valwr.rating import roleindex
from valwr.store import schema


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="build_role_index")
    ap.add_argument("--sample", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(roleindex.index_path()))
    args = ap.parse_args(argv)

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    b = split.compute(conn)
    roles_by_agent = R.roles_by_agent(conn)
    if not roles_by_agent:
        print("ref_agents is empty; run tools/fetch_agent_art.py first")
        return 1
    print(f"training period ends {b.train_end}")

    rows = conn.execute(
        "SELECT puuid, started_at, map, agent FROM match_players "
        "WHERE started_at < ? AND rounds_played > 0", (b.train_end,)).fetchall()
    print(f"{len(rows):,} candidate player-matches in the training period")

    rng = random.Random(args.seed)
    picked = rng.sample(list(rows), min(args.sample, len(rows)))

    started = time.time()
    samples, by_role, no_history = [], Counter(), 0
    for i, r in enumerate(picked, 1):
        role = roles_by_agent.get(r["agent"])
        # Measured unshrunk: the role average is fitted from these, and it
        # cannot be built out of values already pulled toward it.
        c = R.measure(conn, r["puuid"], r["started_at"], r["map"], role,
                      roles_by_agent, role_means=None, agent=r["agent"])
        if c is None:
            no_history += 1
        else:
            samples.append(c)
            by_role[role or "?"] += 1
        if i % 5000 == 0:
            print(f"  measured {i:,}/{len(picked):,} "
                  f"({time.time() - started:.0f}s)")

    print(f"\n{len(samples):,} usable samples "
          f"({no_history:,} had no prior history), "
          f"{time.time() - started:.0f}s")
    if len(samples) < 2000:
        print("too few samples to fit a reference population")
        return 1

    for role, n in sorted(by_role.items()):
        print(f"  {role:<12}{n:>8,}")

    with_casts = sum(1 for c in samples if c.values.get("abilities") is not None)
    print(f"\n  ability casts available for {with_casts:,} of {len(samples):,} "
          f"samples ({with_casts / len(samples) * 100:.1f}%)")
    if with_casts / len(samples) < 0.5:
        print("  WARNING: abilities carry 11-19% of every role's weight and are"
              "\n  missing for most of this sample. Finish the re-parse"
              "\n  (python -m valwr.store.normalize) before trusting the index.")

    index = roleindex.fit(samples, as_of=b.train_end)

    if index.fell_back:
        print(f"\n  {len(index.fell_back)} agent(s) below "
              f"{roleindex.MIN_AGENT_SAMPLE} samples, scored against their role"
              f" instead: {', '.join(index.fell_back)}")

    print("\n  percentile tables: " + ", ".join(
        f"{role} {len(q):,}" for role, q in sorted(index.quantiles.items())
        if role != "__all__"))

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(index.to_json(), encoding="utf-8")
    print(f"\nwrote {path}  ({len(samples):,} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
