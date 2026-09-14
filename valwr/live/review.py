"""Reading back what was predicted against what happened.

`outcomes.py` writes; this reads. Two questions:

- **One match.** What did the page say at the loading screen, and how did it
  turn out -- the winner, the score, and whether the player the score put first
  actually played best.
- **All of them.** Accuracy, calibration and the per-player hit rate across
  every match the dashboard has recorded, with the honest caveat attached:
  these are tens of matches, not thousands, so the interval around each figure
  is wide and is reported beside it.

Only standard bomb-defusal matches count toward the headline figures. The model
never saw swiftplay or deathmatch, so folding them in would measure the wrong
thing -- but they are recorded, listed, and counted separately, because a
session of them is still worth seeing.
"""

from __future__ import annotations

import json
import math
import sqlite3

# Buckets for the calibration table. Predictions cluster hard around the
# middle, so the edges are wide and the centre is not.
BANDS = ((0.0, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 1.0))

# The per-player score picks one of five, so chance is 20%.
TOP_PICK_CHANCE = 0.2


def _se(p: float, n: int) -> float:
    """Standard error of a proportion. The number that keeps this honest."""
    return math.sqrt(max(p * (1 - p), 1e-9) / n) if n else float("nan")


def verdict(p: float | None) -> str:
    """The same wording the page uses, so the review reads like the page."""
    if p is None:
        return "no prediction"
    edge = abs(p - 0.5)
    side = "your side" if p > 0.5 else "the enemy"
    if edge < 0.015:
        return "too close to call"
    if edge < 0.05:
        return f"{side} marginally ahead"
    if edge < 0.12:
        return f"{side} favoured"
    if edge < 0.25:
        return f"{side} clearly favoured"
    return f"{side} heavily favoured"


def compare(conn: sqlite3.Connection, match_id: str) -> dict | None:
    """One match: the prediction, the result, and every player either way."""
    row = conn.execute("SELECT * FROM live_predictions WHERE match_id = ?",
                       (match_id,)).fetchone()
    if row is None:
        return None
    state = json.loads(row["state_json"])
    own_team = row["own_team"]

    played = {
        r["puuid"]: r for r in conn.execute(
            "SELECT puuid, team, agent, kills, deaths, assists, score, "
            "rounds_played, headshots, bodyshots, legshots "
            "FROM match_players WHERE match_id = ?", (match_id,))
    }

    predicted_rank = {
        p["puuid"]: i + 1
        for i, p in enumerate(sorted(
            (p for p in state.get("players", []) if p.get("score") is not None),
            key=lambda p: p["score"], reverse=True))
    }
    actual_acs = {}
    for puuid, r in played.items():
        rounds = r["rounds_played"] or 0
        if rounds:
            actual_acs[puuid] = (r["score"] or 0) / rounds
    actual_rank = {
        puuid: i + 1
        for i, puuid in enumerate(sorted(actual_acs, key=actual_acs.get,
                                         reverse=True))
    }

    players = []
    for p in state.get("players", []):
        r = played.get(p["puuid"])
        shots = None
        if r:
            hits = (r["headshots"] or 0) + (r["bodyshots"] or 0) + (r["legshots"] or 0)
            shots = (r["headshots"] or 0) / hits if hits else None
        players.append({
            "puuid": p["puuid"], "name": p.get("name"), "team": p.get("team"),
            "agent": p.get("agent"), "is_you": p.get("is_you", False),
            "predicted_score": p.get("score"),
            "predicted_rank": predicted_rank.get(p["puuid"]),
            "reason": p.get("reason"),
            "played": r is not None,
            "acs": round(actual_acs[p["puuid"]], 1) if p["puuid"] in actual_acs else None,
            "kills": r["kills"] if r else None,
            "deaths": r["deaths"] if r else None,
            "assists": r["assists"] if r else None,
            "kd": (round((r["kills"] or 0) / r["deaths"], 2)
                   if r and r["deaths"] else None),
            "headshot_rate": round(shots, 4) if shots is not None else None,
            "actual_rank": actual_rank.get(p["puuid"]),
        })

    out = {
        "match_id": match_id, "map": row["map"], "mode": row["mode"],
        "standard_mode": bool(row["standard_mode"]),
        "is_custom": bool(row["is_custom"]),
        "made_at": row["made_at"], "settled_at": row["settled_at"],
        "model": row["model"], "coverage": row["coverage"],
        "confidence": row["confidence"], "own_team": own_team,
        "predicted": {
            "own_probability": row["own_probability"],
            "win_probability": row["win_probability"],
            "verdict": verdict(row["own_probability"]),
        },
        "settled": row["settled_at"] is not None,
        "players": players,
        "state": state,
    }
    if row["settled_at"] is None:
        out["actual"] = None
        out["waiting"] = (row["last_error"] or
                          "the result has not been published yet")
        return out

    own_rounds = row["rounds_blue"] if own_team == "Blue" else row["rounds_red"]
    their_rounds = row["rounds_red"] if own_team == "Blue" else row["rounds_blue"]
    picked = next((p for p in players
                   if p["puuid"] == _top_pick_id(state, own_team)), None)
    best = next((p for p in players if p["actual_rank"] == 1
                 and p["team"] == own_team), None)
    out["actual"] = {
        "winner": row["winner"], "own_won": row["own_won"],
        "rounds_blue": row["rounds_blue"], "rounds_red": row["rounds_red"],
        "score": (f"{own_rounds}-{their_rounds}"
                  if own_rounds is not None and their_rounds is not None else None),
    }
    out["correct"] = row["correct"]
    out["brier"] = row["brier"]
    out["top_pick"] = {
        "hit": row["top_pick_hit"],
        "picked": picked["name"] if picked else None,
        "actually_best": best["name"] if best else None,
    }
    return out


def _top_pick_id(state: dict, team: str | None) -> str | None:
    from valwr.live.outcomes import top_pick
    return top_pick(state, team)


def _bucket(rows: list, label_of) -> list[dict]:
    """Group settled rows and score each group the same way."""
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(label_of(r), []).append(r)
    out = []
    for label, group in groups.items():
        n = len(group)
        hits = sum(r["correct"] or 0 for r in group)
        predicted = sum(r["own_probability"] for r in group) / n
        won = sum(r["own_won"] or 0 for r in group) / n
        out.append({"label": label, "n": n, "accuracy": hits / n,
                    "predicted": predicted, "actual": won,
                    "se": _se(hits / n, n)})
    return sorted(out, key=lambda d: d["label"])


def scorecard(conn: sqlite3.Connection, limit: int = 50) -> dict:
    """Every recorded match, scored, with the caveats attached."""
    rows = list(conn.execute(
        "SELECT * FROM live_predictions ORDER BY made_at DESC"))
    settled = [r for r in rows
               if r["settled_at"] is not None and r["correct"] is not None]
    standard = [r for r in settled if r["standard_mode"]]

    def summarise(group: list) -> dict:
        n = len(group)
        if not n:
            return {"n": 0}
        hits = sum(r["correct"] or 0 for r in group)
        accuracy = hits / n
        return {
            "n": n, "accuracy": accuracy, "se": _se(accuracy, n),
            "brier": sum(r["brier"] for r in group) / n,
            "predicted": sum(r["own_probability"] for r in group) / n,
            "actual": sum(r["own_won"] or 0 for r in group) / n,
            "won": sum(r["own_won"] or 0 for r in group),
        }

    picks = [r for r in standard if r["top_pick_hit"] is not None]
    hits = sum(r["top_pick_hit"] for r in picks)
    out = {
        "recorded": len(rows),
        "settled": len(settled),
        "pending": sum(1 for r in rows if r["settled_at"] is None),
        "competitive": summarise(standard),
        "everything": summarise(settled),
        "calibration": _bucket(standard, lambda r: _band(r["own_probability"])),
        "by_confidence": _bucket(standard, lambda r: r["confidence"] or "?"),
        "by_coverage": _bucket(standard, lambda r: f"{r['coverage']}/10 known"),
        "top_pick": {
            "n": len(picks), "hits": hits,
            "rate": hits / len(picks) if picks else None,
            "se": _se(hits / len(picks), len(picks)) if picks else None,
            "chance": TOP_PICK_CHANCE,
        },
        "matches": [{
            "match_id": r["match_id"], "map": r["map"], "mode": r["mode"],
            "made_at": r["made_at"], "standard_mode": bool(r["standard_mode"]),
            "own_probability": r["own_probability"],
            "verdict": verdict(r["own_probability"]),
            "settled": r["settled_at"] is not None,
            "own_won": r["own_won"], "correct": r["correct"],
            "score": _score_line(r),
        } for r in rows[:limit]],
    }
    out["insights"] = insights(out)
    return out


def _band(p: float) -> str:
    for low, high in BANDS:
        if low <= p < high:
            return f"{low * 100:.0f}-{high * 100:.0f}%"
    return "100%"


def _score_line(r) -> str | None:
    if r["rounds_blue"] is None or r["rounds_red"] is None:
        return None
    own = r["rounds_blue"] if r["own_team"] == "Blue" else r["rounds_red"]
    them = r["rounds_red"] if r["own_team"] == "Blue" else r["rounds_blue"]
    return f"{own}-{them}"


def insights(card: dict) -> list[str]:
    """What the numbers support saying, and nothing more.

    Every claim here is checked against its own standard error first, because
    a handful of matches will happily show a 30-point swing that means nothing.
    """
    out: list[str] = []
    comp = card["competitive"]
    n = comp.get("n", 0)
    if n == 0:
        out.append("No competitive match has been scored yet. Play one with "
                   "the dashboard open and it will appear here.")
        return out

    span = 1.96 * comp["se"] * 100
    out.append(f"{n} competitive match{'es' if n != 1 else ''} scored: "
               f"{comp['accuracy'] * 100:.0f}% correct, give or take "
               f"{span:.0f} points at 95% confidence.")
    if n < 30:
        out.append("That interval is wider than the edge being measured. "
                   "Around 30 matches is where this starts to mean anything, "
                   "and a few hundred is where it settles.")

    gap = comp["actual"] - comp["predicted"]
    gap_se = comp["se"]
    if abs(gap) > 2 * gap_se:
        direction = "more" if gap > 0 else "fewer"
        out.append(f"You win {direction} than predicted: {comp['actual'] * 100:.0f}% "
                   f"against {comp['predicted'] * 100:.0f}% expected. On this "
                   f"many matches that gap is larger than chance explains, so "
                   f"it is worth watching.")
    else:
        out.append(f"Predicted {comp['predicted'] * 100:.0f}% on average, won "
                   f"{comp['actual'] * 100:.0f}%. The two agree within noise.")

    pick = card["top_pick"]
    if pick["n"] >= 10 and pick["rate"] is not None:
        edge = (pick["rate"] - pick["chance"]) / pick["se"] if pick["se"] else 0
        if edge > 2:
            out.append(f"The player score picked your best teammate "
                       f"{pick['rate'] * 100:.0f}% of the time against 20% by "
                       f"chance, across {pick['n']} matches.")
        else:
            out.append(f"The player score picked your best teammate "
                       f"{pick['rate'] * 100:.0f}% of the time; against 20% by "
                       f"chance that is not yet outside noise on "
                       f"{pick['n']} matches.")

    worst = [b for b in card["by_coverage"] if b["n"] >= 5]
    if len(worst) >= 2:
        low = min(worst, key=lambda b: b["accuracy"])
        high = max(worst, key=lambda b: b["accuracy"])
        if high["accuracy"] - low["accuracy"] > 2 * (low["se"] + high["se"]):
            out.append(f"It does better with more of the lobby known: "
                       f"{high['accuracy'] * 100:.0f}% at {high['label']} "
                       f"against {low['accuracy'] * 100:.0f}% at {low['label']}. "
                       f"Leaving the crawler running would help.")

    others = card["everything"]["n"] - n
    if others:
        out.append(f"{others} non-competitive match"
                   f"{'es are' if others != 1 else ' is'} recorded but left out "
                   f"of these figures: the model only ever saw bomb defusal.")
    return out
