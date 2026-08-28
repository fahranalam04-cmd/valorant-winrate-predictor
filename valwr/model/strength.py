"""Bradley-Terry player strength, learned from who actually won.

The composite in `valwr/rating/rating.py` is built from *performance*
components -- damage, KAST, ACS. Those are a proxy for skill: a player can farm
damage and still lose. This learns a single strength per player directly from
match outcomes, which is the quantity the model is ultimately asking about.

The model is the standard one for team-vs-team outcomes, the same family as Elo:

    P(Blue wins) = sigmoid( sum(strength of Blue's five)
                          - sum(strength of Red's five) )

Fitting it is ordinary logistic regression on an unusual design matrix -- one
column per player, +1 for Blue's five, -1 for Red's five, zero elsewhere. The
fitted coefficients *are* the strengths. No new dependency: scipy's sparse
matrices plus scikit-learn.

Two things keep it honest.

**The `as_of` boundary.** Strengths are fitted only on matches that started
strictly before `as_of`. A strength that absorbed information from a test match
would hand the model the answer, so this is the single most important line in
the file. It mirrors the discipline in `valwr/store/temporal.py`.

**Pooling the rare players.** 56.8% of the 162,002 players in this dataset
appear in exactly one match, and a player seen once has no learnable strength
-- only 45,000 matches constrain the whole system. Players below
`min_appearances` therefore share one pooled column instead of each getting a
free parameter. Two pooled players on opposite teams cancel, which is the
correct behaviour: two unknowns are even odds. The L2 penalty then pulls
thinly-evidenced players toward zero, which is the population average.

There is no intercept. The design matrix negates exactly when the two teams
swap, so without an intercept the model is antisymmetric by construction:
P(Blue) + P(Red) == 1 exactly.
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

POOLED = "<pooled>"


@dataclass(frozen=True)
class Strengths:
    """Fitted strengths plus the provenance needed to trust them."""
    theta: dict[str, float]
    pooled: float
    as_of: int
    n_players: int          # players given a free parameter
    n_matches: int          # matches the fit saw
    min_appearances: int
    C: float

    def of(self, puuid: str) -> float:
        """Strength of one player; pooled value for anyone not fitted."""
        return self.theta.get(puuid, self.pooled)

    def team(self, puuids) -> tuple[float, float]:
        """(sum, max) strength for a roster -- the two features worth having."""
        vals = [self.of(p) for p in puuids]
        if not vals:
            return 0.0, 0.0
        return float(sum(vals)), float(max(vals))


def _rosters(conn: sqlite3.Connection, as_of: int):
    """Blue/Red rosters and outcome for every resolved match before `as_of`."""
    rows = conn.execute(
        "SELECT mp.match_id AS match_id, mp.puuid AS puuid, mp.team AS team, "
        "       m.winner AS winner "
        "FROM match_players mp "
        "JOIN matches m ON m.match_id = mp.match_id "
        "WHERE m.started_at < ? AND m.winner IN ('Blue','Red')",
        (as_of,)).fetchall()

    by_match: dict[str, dict] = defaultdict(
        lambda: {"Blue": [], "Red": [], "winner": None})
    for r in rows:
        m = by_match[r["match_id"]]
        m["winner"] = r["winner"]
        if r["team"] in ("Blue", "Red"):
            m[r["team"]].append(r["puuid"])
    # A partial roster would silently bias the fit toward whichever side was
    # scraped more completely, so drop anything that is not a full 5v5.
    return {k: v for k, v in by_match.items()
            if len(v["Blue"]) == 5 and len(v["Red"]) == 5}


def fit(conn: sqlite3.Connection, as_of: int, *,
        min_appearances: int = 5, C: float = 1.0) -> Strengths:
    """Fit strengths on matches strictly before `as_of`."""
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression

    matches = _rosters(conn, as_of)
    if len(matches) < 100:
        raise ValueError(f"only {len(matches)} usable matches before {as_of}")

    appearances = Counter()
    for m in matches.values():
        appearances.update(m["Blue"])
        appearances.update(m["Red"])

    free = sorted(p for p, n in appearances.items() if n >= min_appearances)
    index = {p: i for i, p in enumerate(free)}
    pooled_col = len(free)
    n_cols = pooled_col + 1

    rows_i, cols_i, vals = [], [], []
    y = np.empty(len(matches), dtype=np.int8)
    for row, m in enumerate(matches.values()):
        y[row] = 1 if m["winner"] == "Blue" else 0
        for sign, side in ((1.0, "Blue"), (-1.0, "Red")):
            for puuid in m[side]:
                rows_i.append(row)
                cols_i.append(index.get(puuid, pooled_col))
                vals.append(sign)

    # Duplicate (row, col) entries sum, which is exactly what should happen
    # when several pooled players land on the same column.
    X = sparse.coo_matrix((vals, (rows_i, cols_i)),
                          shape=(len(matches), n_cols)).tocsr()

    model = LogisticRegression(fit_intercept=False, C=C, max_iter=1000,
                               solver="lbfgs")
    model.fit(X, y)
    coef = model.coef_[0]

    return Strengths(
        theta={p: float(coef[i]) for p, i in index.items()},
        pooled=float(coef[pooled_col]),
        as_of=as_of,
        n_players=len(free),
        n_matches=len(matches),
        min_appearances=min_appearances,
        C=C,
    )
