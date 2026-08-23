"""Context normalisation for performance components.

Raw ACS is not comparable across contexts: 200 ACS in Iron is not 200 ACS in
Immortal, and some maps produce systematically more damage than others. Each
component is therefore turned into a z-score *within rank band and map*.

Thin cells are the difficulty. With 9 bands x 15 maps there are 135 cells, and
the sparse ones give unstable means and standard deviations. So norms are
hierarchical and shrunk:

    global  ->  per band  ->  per (band, map)

each level shrunk toward its parent in proportion to how little data it has.
This is the same empirical-Bayes idea applied to rate features elsewhere in the
project (CLAUDE.md rule 4), and for the same reason: a cell with four
observations should not be trusted to define its own centre.

Norms are always built with an `as_of` cutoff. They are derived from data, so
computing them over the full history would leak the future into a feature.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field

from valwr.collect.frontier import band_of
from valwr.rating.components import per_round_rates

# Components that make up the rating. Ordering is irrelevant; membership is not.
COMPONENTS = ("acs", "adr", "kast", "kpr", "dpr", "fb_rate", "fd_rate",
              "trade_rate", "multikill_rate", "hs_pct")

# How much evidence a cell needs before it is trusted over its parent. At
# PRIOR_WEIGHT observations, a cell is weighted 50/50 against its parent.
PRIOR_WEIGHT = 30.0


@dataclass
class Moments:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        self.total += x
        self.total_sq += x * x

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        var = self.total_sq / self.n - self.mean ** 2
        return math.sqrt(max(var, 0.0))

    def shrunk_toward(self, parent: "Moments") -> tuple[float, float]:
        """Blend this cell's moments with its parent's by sample size."""
        w = self.n / (self.n + PRIOR_WEIGHT)
        mean = w * self.mean + (1 - w) * parent.mean
        std = w * self.std + (1 - w) * parent.std
        return mean, (std if std > 1e-9 else parent.std)


@dataclass
class Norms:
    """Population moments per component, at three levels of specificity."""
    glob: dict[str, Moments] = field(default_factory=dict)
    by_band: dict[tuple[str, int], Moments] = field(default_factory=dict)
    by_band_map: dict[tuple[str, int, str], Moments] = field(default_factory=dict)
    rows_used: int = 0

    def _cells(self, comp: str, band: int, map_name: str):
        g = self.glob.get(comp, Moments())
        b = self.by_band.get((comp, band), Moments())
        bm = self.by_band_map.get((comp, band, map_name), Moments())
        return g, b, bm

    def z(self, comp: str, value: float | None, band: int, map_name: str) -> float | None:
        """z-score `value` against its (band, map) cell, shrunk toward parents."""
        if value is None:
            return None
        g, b, bm = self._cells(comp, band, map_name)
        if g.n == 0:
            return None
        b_mean, b_std = b.shrunk_toward(g)
        parent = Moments(n=b.n, total=b_mean * max(b.n, 1),
                         total_sq=(b_std ** 2 + b_mean ** 2) * max(b.n, 1))
        mean, std = bm.shrunk_toward(parent)
        if std <= 1e-9:
            return 0.0
        return (value - mean) / std


def build_norms(conn: sqlite3.Connection, as_of: int) -> Norms:
    """Build population norms from matches strictly before `as_of`."""
    norms = Norms()
    cur = conn.execute(
        "SELECT tier, map, rounds_played, score, damage_dealt, kills, deaths, "
        "assists, headshots, bodyshots, legshots, kast_rounds, first_bloods, "
        "first_deaths, trade_kills, multikills "
        "FROM match_players WHERE started_at < ? AND rounds_played > 0",
        (as_of,),
    )
    for row in cur:
        band = band_of(row["tier"])
        map_name = row["map"] or "?"
        rates = per_round_rates(dict(row))
        norms.rows_used += 1
        for comp in COMPONENTS:
            v = rates.get(comp)
            if v is None:
                continue
            norms.glob.setdefault(comp, Moments()).add(v)
            norms.by_band.setdefault((comp, band), Moments()).add(v)
            norms.by_band_map.setdefault((comp, band, map_name), Moments()).add(v)
    return norms
