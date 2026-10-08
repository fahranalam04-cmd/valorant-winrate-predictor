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

from valwr.rating import role_score

# Buckets for the calibration table. Predictions cluster hard around the
# middle, so the edges are wide and the centre is not.
BANDS = ((0.0, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 1.0))

# The per-player score picks one of five, so chance is 20%.
TOP_PICK_CHANCE = 0.2

# How far from a player's own career average counts as a real departure. ACS
# swings hugely match to match, so the bands are wide on purpose: the question
# is "did they play like themselves", not "were they exactly average".
WELL_ABOVE, ABOVE, BELOW, WELL_BELOW = 1.15, 1.05, 0.95, 0.85


def against_usual(actual: float | None, career: float | None) -> str | None:
    """How a match compares with the player's own career average.

    Measured on damage per round. It was combat score until patch 13.06 removed
    that from the game; damage is what the card shows in its place, and the two
    correlate 0.98, so the verdict this produces barely moves.

    The 0-100 score ranks players against each other. This asks a different and
    more answerable question -- did this player do what they usually do -- which
    is what makes a per-player comparison worth reading rather than a restatement
    of who topped the scoreboard.
    """
    if actual is None or not career:
        return None
    ratio = actual / career
    if ratio >= WELL_ABOVE:
        return "well above their usual"
    if ratio >= ABOVE:
        return "above their usual"
    if ratio <= WELL_BELOW:
        return "well below their usual"
    if ratio <= BELOW:
        return "below their usual"
    return "about their usual"


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation, for ordering players rather than scoring them.

    The 0-100 score is a ranking claim -- who will play best -- so the honest
    check is whether its order matched the scoreboard's, not whether the number
    was close to anything.
    """
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def _se(p: float, n: int) -> float:
    """Standard error of a proportion. The number that keeps this honest."""
    return math.sqrt(max(p * (1 - p), 1e-9) / n) if n else float("nan")


def verdict(p: float | None) -> str:
    """The same wording the page uses (index.html `verdict`), so the review
    reads like the page: a lean, never "favoured", which claimed more than a
    model right 54% of the time can."""
    if p is None:
        return "no prediction"
    edge = abs(p - 0.5)
    if edge < 0.02:
        return "coin flip"
    way = "your way" if p > 0.5 else "their way"
    return f"slight lean {way}" if edge < 0.05 else f"leans {way}"


def recent(conn: sqlite3.Connection, limit: int = 6) -> list[dict]:
    """The last few matches the dashboard recorded, newest first.

    The live page follows whatever match is happening now, so when one ends it
    drops back to "waiting for a match" and the game just played disappears
    from view. Getting back to it relied entirely on catching the tab the
    server popped open at the time. This is the list that makes it reachable
    from the page itself.
    """
    out = []
    for r in conn.execute(
            "SELECT match_id, map, mode, made_at, settled_at, own_won, correct, "
            "rounds_blue, rounds_red, own_team FROM live_predictions "
            "ORDER BY made_at DESC LIMIT ?", (limit,)):
        own = r["rounds_blue"] if r["own_team"] == "Blue" else r["rounds_red"]
        theirs = r["rounds_red"] if r["own_team"] == "Blue" else r["rounds_blue"]
        out.append({
            "match_id": r["match_id"], "map": r["map"], "mode": r["mode"],
            "made_at": r["made_at"], "settled": r["settled_at"] is not None,
            "own_won": r["own_won"], "correct": r["correct"],
            "score": (f"{own}-{theirs}" if own is not None and theirs is not None
                      else None),
        })
    return out


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
            # Every column: rating a match needs the round-derived ones
            # (KAST, trades, entries) as well as the scoreboard line.
            "SELECT * FROM match_players WHERE match_id = ?", (match_id,))
    }

    # Ranked the way the scoreboard was ordered, so "put #2 of the players the
    # page could score" describes the list the player was actually looking at.
    predicted_rank = {
        p["puuid"]: i + 1
        for i, p in enumerate(sorted(
            (p for p in state.get("players", [])
             if role_score.standing(p) is not None),
            key=role_score.standing, reverse=True))
    }
    # Where each player actually finished, by the one definition of "played
    # best" the verdict and the scorecard also use: the game's own Performance
    # Score where the client gave it, match impact otherwise.
    from valwr.live import client_scores
    from valwr.live.outcomes import played_best_values
    performance = client_scores.scores_for(conn, match_id)
    values, best_by = played_best_values(conn, match_id, performance)
    actual_rank = {
        puuid: i + 1
        for i, puuid in enumerate(sorted(values, key=values.get, reverse=True))
    }
    actual_adr, actual_acs = {}, {}
    for puuid, r in played.items():
        rounds = r["rounds_played"] or 0
        if not rounds:
            continue
        actual_adr[puuid] = (r["damage_dealt"] or 0) / rounds
        # Still carried for matches recorded before 13.06, whose stored state
        # has a career ACS to compare against and no career ADR.
        actual_acs[puuid] = (r["score"] or 0) / rounds

    players = []
    for p in state.get("players", []):
        r = played.get(p["puuid"])
        shots = None
        if r:
            hits = (r["headshots"] or 0) + (r["bodyshots"] or 0) + (r["legshots"] or 0)
            shots = (r["headshots"] or 0) / hits if hits else None
        # What the page knew about them beforehand: their own career and form,
        # which is the baseline the match is read against.
        career = p.get("career") or {}
        recent = p.get("recent") or {}
        acs_now = actual_acs.get(p["puuid"])
        adr_now = actual_adr.get(p["puuid"])
        kd_now = ((r["kills"] or 0) / r["deaths"]) if r and r["deaths"] else None
        # Matches recorded before 13.06 stored a career ACS and no career ADR.
        # The verdict is a ratio against the player's own average, so it holds
        # on either -- but the two must never be mixed in one comparison.
        usual = (against_usual(adr_now, career.get("adr"))
                 if career.get("adr") else against_usual(acs_now, career.get("acs")))
        players.append({
            "career_adr": career.get("adr"),
            "adr_delta": (round(adr_now - career["adr"], 1)
                          if adr_now is not None and career.get("adr") else None),
            "adr": round(adr_now, 1) if adr_now is not None else None,
            "career_acs": career.get("acs"),
            "career_kd": career.get("kd"),
            "career_hs": career.get("headshot_rate"),
            "career_games": career.get("games"),
            "recent_kd": recent.get("kd"),
            "acs_delta": (round(acs_now - career["acs"], 1)
                          if acs_now is not None and career.get("acs") else None),
            "kd_delta": (round(kd_now - career["kd"], 2)
                         if kd_now is not None and career.get("kd") else None),
            "hs_delta": (round(shots - career["headshot_rate"], 4)
                         if shots is not None and career.get("headshot_rate")
                         else None),
            "versus_usual": usual,
            "puuid": p["puuid"], "name": p.get("name"), "team": p.get("team"),
            "agent": p.get("agent"), "is_you": p.get("is_you", False),
            "rank": p.get("rank"),
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
            "performance_score": client_scores.shown(performance.get(p["puuid"])),
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
        "best_by": best_by,
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
    for p in players:
        if p["predicted_rank"] and p["actual_rank"]:
            p["place_delta"] = p["predicted_rank"] - p["actual_rank"]
        else:
            p["place_delta"] = None
    out["summary"] = summarise_match(out)
    return out


def summarise_match(got: dict) -> dict:
    """How much of this match the prediction got right, in four numbers."""
    rated = [p for p in got["players"]
             if p["predicted_rank"] and p["actual_rank"]]
    within = sum(1 for p in rated
                 if abs(p["predicted_rank"] - p["actual_rank"]) <= 1)
    return {
        "winner_called": got.get("correct"),
        "top_pick_hit": got.get("top_pick", {}).get("hit"),
        "rated": len(rated),
        "within_one": within,
        "order": spearman([p["predicted_rank"] for p in rated],
                          [p["actual_rank"] for p in rated]),
    }


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
    # Each verdict is judged on the measure its match had: the game's own
    # Performance Score where the client gave it, match impact otherwise. One
    # figure, with the split said, rather than two each resting on less.
    from valwr.live.outcomes import played_best_values
    judged_by: dict[str, int] = {}
    for r in picks:
        basis = played_best_values(conn, r["match_id"])[1]
        judged_by[basis] = judged_by.get(basis, 0) + 1
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
            "judged_by": judged_by,
        },
        "by_map": _bucket(standard, lambda r: r["map"] or "?"),
        "by_mode": _bucket(settled, lambda r: r["mode"] or "?"),
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
    out["order"] = _order_quality(conn, standard)
    out["insights"] = insights(out)
    return out


def _order_quality(conn: sqlite3.Connection, rows: list) -> dict:
    """How well the 0-100 score ordered each lobby, across every match.

    One match says nothing -- five players and a lot of luck. Pooled over a
    season it is the sharpest thing this record measures, because every match
    contributes ten rankings rather than one binary outcome.
    """
    pairs: list[tuple[float, float]] = []
    within = rated = 0
    for r in rows:
        got = compare(conn, r["match_id"])
        if not got or not got.get("summary"):
            continue
        for p in got["players"]:
            if p["predicted_rank"] and p["actual_rank"]:
                pairs.append((p["predicted_rank"], p["actual_rank"]))
                rated += 1
                within += abs(p["predicted_rank"] - p["actual_rank"]) <= 1
    if not pairs:
        return {"players": 0}
    return {
        "players": rated,
        "within_one": within,
        "within_one_rate": within / rated,
        "order": spearman([a for a, _ in pairs], [b for _, b in pairs]),
    }


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
        by = pick.get("judged_by") or {}
        if len(by) > 1:
            out.append(f"\"Best teammate\" is judged on the game's own "
                       f"Performance Score in {by.get('performance score', 0)} "
                       f"of those and on match impact in "
                       f"{by.get('match impact', 0)} -- whichever each match "
                       f"had.")

    worst = [b for b in card["by_coverage"] if b["n"] >= 5]
    if len(worst) >= 2:
        low = min(worst, key=lambda b: b["accuracy"])
        high = max(worst, key=lambda b: b["accuracy"])
        if high["accuracy"] - low["accuracy"] > 2 * (low["se"] + high["se"]):
            out.append(f"It does better with more of the lobby known: "
                       f"{high['accuracy'] * 100:.0f}% at {high['label']} "
                       f"against {low['accuracy'] * 100:.0f}% at {low['label']}. "
                       f"Leaving the crawler running would help.")

    order = card.get("order") or {}
    if order.get("players", 0) >= 50 and order.get("order") is not None:
        rho = order["order"]
        strength = ("no better than shuffling them" if abs(rho) < 0.1
                    else "a weak but real ordering" if rho < 0.3
                    else "a clear ordering")
        out.append(f"Across {order['players']} player rankings, the 0-100 score "
                   f"ordered the scoreboard at rho {rho:+.2f} -- {strength}. "
                   f"{order['within_one_rate'] * 100:.0f}% landed within one "
                   f"place of where it put them.")

    maps = [b for b in card.get("by_map", []) if b["n"] >= 5]
    if len(maps) >= 3:
        low = min(maps, key=lambda b: b["accuracy"])
        high = max(maps, key=lambda b: b["accuracy"])
        if high["accuracy"] - low["accuracy"] > 2 * (low["se"] + high["se"]):
            out.append(f"By map, {high['label']} is called right "
                       f"{high['accuracy'] * 100:.0f}% of the time against "
                       f"{low['accuracy'] * 100:.0f}% on {low['label']}. Worth "
                       f"a look: the model has no map-specific term beyond a "
                       f"small per-player one.")

    others = card["everything"]["n"] - n
    if others:
        out.append(f"{others} non-competitive match"
                   f"{'es are' if others != 1 else ' is'} recorded but left out "
                   f"of these figures: the model only ever saw bomb defusal.")
    return out
