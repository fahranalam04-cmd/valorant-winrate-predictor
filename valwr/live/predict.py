"""Live roster -> win probability.

The important rule here is that this does **not** build features. It assembles
a match dict and hands it to `features.build.build_match` with
`require_outcome=False`, so training and inference run the identical code.

Writing a second feature builder for the live path is the quiet way to feed a
model inputs it never saw: every column would still be present, every number
would still look plausible, and nothing would raise. The shared path is why
`build_match` grew that flag rather than a sibling function.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from valwr.features import build as fb
from valwr.live.roster import LiveMatch
from valwr.live.resolve import Resolution

TEAM_A = fb.TEAM_A          # "Blue" -- the perspective the target is defined from


@dataclass(frozen=True)
class Prediction:
    win_probability: float          # for TEAM_A
    own_team: str
    own_probability: float
    coverage: int
    confidence: str
    model: str
    factors: list[tuple[str, float]]

    def summary(self) -> str:
        return (f"{self.own_team} {self.own_probability * 100:.1f}%  "
                f"({self.confidence} confidence, {self.coverage}/10 known)")


def _roster_rows(match: LiveMatch, conn: sqlite3.Connection) -> list[dict]:
    """Roster in the shape build_match expects.

    Tier and account level come from whatever the store last saw for each
    player. Live, the client does not hand them over, and fetching them would
    cost requests the deadline cannot spare -- so a player we have never seen
    contributes neutral values, which is what shrinkage would produce anyway.
    """
    rows = []
    for p in match.players:
        seen = conn.execute(
            "SELECT tier, account_level FROM match_players WHERE puuid = ? "
            "ORDER BY started_at DESC LIMIT 1", (p.puuid,)).fetchone()
        rows.append({
            "match_id": match.match_id,
            "puuid": p.puuid,
            "team": p.team,
            "agent": p.agent,
            "party_id": None,       # not exposed pre-match
            "tier": seen["tier"] if seen else None,
            "account_level": seen["account_level"] if seen else None,
        })
    return rows


def predict(conn: sqlite3.Connection, match: LiveMatch, bundle: dict,
            resolution: Resolution, own_puuid: str,
            as_of: int | None = None) -> Prediction | None:
    """Predict a live match. None when the roster is too thin to say anything."""
    if len(match.players) < 2:
        return None

    as_of = as_of or int(time.time())
    live_match = {
        "match_id": match.match_id,
        "started_at": as_of,
        "map": match.map_name or "?",
        "season": None,
        "region": "na",
        "winner": None,             # it has not happened yet
        "rounds_blue": None,
        "rounds_red": None,
    }

    mf = fb.build_match(conn, live_match, _roster_rows(match, conn),
                        bundle["norms"], bundle["prior_rate"], bundle["roles"],
                        require_outcome=False)
    if mf is None:
        return None

    cols = bundle["columns"]
    x = [[mf.values.get(c, 0.0) for c in cols]]

    name = bundle["best"]
    estimator = bundle["estimators"].get(name) or bundle["estimators"]["logistic"]
    if callable(estimator) and not hasattr(estimator, "predict_proba"):
        # A fitted single-feature baseline is a plain callable over a frame.
        import pandas as pd
        p_a = float(estimator(pd.DataFrame(
            {c: [mf.values.get(c, 0.0)] for c in cols}))[0])
    else:
        p_a = float(estimator.predict_proba(x)[0][1])

    own_team = match.team_of(own_puuid) or TEAM_A
    own_p = p_a if own_team == TEAM_A else 1.0 - p_a

    return Prediction(
        win_probability=p_a,
        own_team=own_team,
        own_probability=own_p,
        coverage=resolution.coverage,
        confidence=resolution.confidence,
        model=name,
        factors=top_factors(mf, bundle),
    )


def top_factors(mf, bundle, n: int = 5) -> list[tuple[str, float]]:
    """The features pushing the prediction hardest, signed toward TEAM_A.

    Contribution is weight x value for the linear model, which is exact rather
    than an approximation -- a real advantage of shipping the simple model.
    """
    cols = bundle["columns"]
    est = bundle["estimators"].get("logistic")
    if est is None or not hasattr(est, "named_steps"):
        return []
    try:
        scaler = est.named_steps["standardscaler"]
        clf = est.named_steps["logisticregression"]
    except (AttributeError, KeyError):
        return []

    contributions = []
    for i, c in enumerate(cols):
        raw = mf.values.get(c, 0.0)
        scaled = (raw - scaler.mean_[i]) / (scaler.scale_[i] or 1.0)
        contributions.append((c, float(clf.coef_[0][i] * scaled)))
    contributions.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return contributions[:n]
