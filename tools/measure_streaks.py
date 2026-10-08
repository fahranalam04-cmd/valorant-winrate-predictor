"""Does a run of wins or losses say anything about the next game?

The dashboard badges a teammate on a run -- three or more results the same this
session (valwr/live/streak.py). This is the measurement behind what
docs/DASHBOARD.md says about that badge: the next game's win rate, and the
player's damage per round against their own average, by the run going in.

Only back-to-back stored games are chained: each begun within 55 minutes of
the one before, so that no game can fit between them. The crawl does not hold
every game a player played, and a gap in what is stored would otherwise make
two separate wins look like a streak.

    python tools/measure_streaks.py

Read-only. Prints aggregates; no player is named.
"""

from __future__ import annotations

import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from valwr import config  # noqa: E402

BACK_TO_BACK_SECONDS = 55 * 60      # live/streak.py COMPLETE_WITHIN_SECONDS

BANDS = [(-99, -4, "4+ losses in a row"), (-3, -3, "3 losses in a row"),
         (-2, -2, "2 losses in a row"), (-1, -1, "a loss"),
         (0, 0, "first game of a sitting"), (1, 1, "a win"),
         (2, 2, "2 wins in a row"), (3, 3, "3 wins in a row"),
         (4, 99, "4+ wins in a row")]


def games(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT mp.puuid, mp.started_at AS t, mp.won, "
        "mp.damage_dealt * 1.0 / mp.rounds_played AS adr "
        "FROM match_players mp JOIN matches m USING (match_id) "
        "WHERE m.mode = 'competitive' AND mp.rounds_played > 0 "
        "AND mp.won IS NOT NULL", conn)
    df = df.sort_values(["puuid", "t"]).reset_index(drop=True)
    df["adr_vs_own"] = df.adr - df.groupby("puuid").adr.transform("mean")
    chained = ((df.puuid == df.puuid.shift())
               & (df.t - df.t.shift() <= BACK_TO_BACK_SECONDS))
    df["sitting"] = (~chained).cumsum()
    df["game_no"] = df.groupby("sitting").cumcount() + 1
    return df.assign(run=runs_going_in(df.sitting.to_numpy(),
                                       df.won.to_numpy()))


def runs_going_in(sitting: np.ndarray, won: np.ndarray) -> np.ndarray:
    """+n after n wins in a row, -n after n losses, 0 for a sitting's first."""
    run = np.zeros(len(won), dtype=int)
    for i in range(1, len(won)):
        if sitting[i] != sitting[i - 1]:
            continue
        step = 1 if won[i - 1] else -1
        prev = run[i - 1]
        run[i] = prev + step if prev and (prev > 0) == (step > 0) else step
    return run


def line(label: str, g: pd.DataFrame) -> str:
    p = g.won.mean()
    margin = 1.96 * math.sqrt(p * (1 - p) / len(g)) if len(g) else float("nan")
    return (f"  {label:<26} {len(g):>9,}   {p:6.1%} ± {margin:.1%}"
            f"   {g.adr_vs_own.mean():+6.1f}")


def main() -> int:
    db = config.load(require_key=False).database_path
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    df = games(conn)
    print(f"  {len(df):,} competitive games, {df.puuid.nunique():,} players; "
          f"next-game win rate overall {df.won.mean():.1%}\n")
    head = "games   win rate (95%)   ADR vs own"
    print(f"  {'Going into the game after':<26} {head}")
    for lo, hi, label in BANDS:
        print(line(label, df[(df.run >= lo) & (df.run <= hi)]))
    print(f"\n  {'Game of the sitting':<26} {head}")
    for n in (1, 2, 3, 4):
        print(line(f"game {n}", df[df.game_no == n]))
    print(line("game 5+", df[df.game_no >= 5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
