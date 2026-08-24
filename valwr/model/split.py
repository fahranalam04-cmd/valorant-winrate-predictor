"""Time-ordered train / validation / test splits.

Match data is a time series. Shuffling it trains on the future and tests on
the past, and the inflation that produces is large -- so there is no k-fold
here and there should never be one.

The boundary has to be computed *before* the feature matrix is built, because
population norms and the shrinkage prior are fitted at that cutoff. Building
first and splitting afterwards would fit preprocessing on held-out rows.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test is the remainder


@dataclass(frozen=True)
class Boundaries:
    """Unix timestamps separating the three slices."""
    train_end: int          # also the norms cutoff
    val_end: int
    n_matches: int

    def slice_of(self, started_at: int) -> str:
        if started_at < self.train_end:
            return "train"
        if started_at < self.val_end:
            return "val"
        return "test"


def compute(conn: sqlite3.Connection) -> Boundaries:
    """Boundaries over every resolved match, by start time.

    Computed from `matches` rather than from the built matrix, because the
    matrix cannot exist yet -- it needs the cutoff this returns. Coverage
    filtering later removes rows unevenly, so the realised proportions drift
    a little from 70/15/15; `describe()` reports what they actually came out
    as rather than assuming.
    """
    times = [r["started_at"] for r in conn.execute(
        "SELECT started_at FROM matches WHERE winner IN ('Blue','Red') "
        "ORDER BY started_at")]
    if len(times) < 10:
        raise ValueError(f"only {len(times)} resolved matches; too few to split")
    return Boundaries(
        train_end=times[int(len(times) * TRAIN_FRAC)],
        val_end=times[int(len(times) * (TRAIN_FRAC + VAL_FRAC))],
        n_matches=len(times),
    )


def apply(df, boundaries: Boundaries):
    """Add a `slice` column to a built feature matrix."""
    df = df.copy()
    df["slice"] = df["started_at"].map(boundaries.slice_of)
    return df


def describe(df) -> str:
    lines = []
    for name in ("train", "val", "test"):
        part = df[df["slice"] == name]
        if part.empty:
            lines.append(f"  {name:<6} EMPTY")
            continue
        lines.append(
            f"  {name:<6} {len(part):>6,} rows  "
            f"({len(part)/len(df)*100:>4.1f}%)  "
            f"target {part['target'].mean():.3f}")
    return "\n".join(lines)
