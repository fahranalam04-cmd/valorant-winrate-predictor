"""Does the rating measure anything real?

Three checks, run against live data. A rating that fails these is a weighted
sum of noise, and it is far better to know that before building features on it.
"""

from __future__ import annotations

import sqlite3
from statistics import mean

from valwr.rating.normalize import build_norms
from valwr.rating.rating import rate_performance


def _corr(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else None


def _rated_rows(conn, as_of, norms, min_matches):
    """{puuid: [(started_at, rating, acs), ...]} for players with enough history."""
    rows = conn.execute(
        "SELECT * FROM match_players WHERE started_at < ? AND rounds_played > 0 "
        "ORDER BY puuid, started_at", (as_of,)).fetchall()
    by_player: dict[str, list] = {}
    for r in rows:
        d = dict(r)
        rating = rate_performance(d, norms)
        if rating is None:
            continue
        acs = (d["score"] or 0) / d["rounds_played"]
        by_player.setdefault(d["puuid"], []).append((d["started_at"], rating.value, acs))
    return {p: v for p, v in by_player.items() if len(v) >= min_matches}


def check_rank_correlation(conn, as_of, norms) -> dict:
    """Sanity. Expect WEAK positive correlation, not strong.

    The rating normalises within rank band on purpose, so most of the rank
    signal is removed by construction. A strong correlation would mean the
    normalisation failed and the rating is just a rank proxy -- which would be
    useless, since rank is already a feature.
    """
    xs, ys = [], []
    for r in conn.execute(
        "SELECT * FROM match_players WHERE started_at < ? AND rounds_played > 0 "
        "AND tier > 0 LIMIT 20000", (as_of,)):
        d = dict(r)
        rating = rate_performance(d, norms)
        if rating:
            xs.append(float(d["tier"]))
            ys.append(rating.value)
    return {"n": len(xs), "corr": _corr(xs, ys)}


def check_split_half(conn, as_of, norms, min_matches=6) -> dict:
    """Reliability: does the rating measure a stable property, or noise?

    Split each player's matches into odd and even, rate each half, correlate
    the halves across players. Low correlation means the composite is
    dominated by match-to-match variance.
    """
    by_player = _rated_rows(conn, as_of, norms, min_matches)
    odd, even = [], []
    for _, matches in by_player.items():
        a = [m[1] for m in matches[0::2]]
        b = [m[1] for m in matches[1::2]]
        if a and b:
            odd.append(mean(a))
            even.append(mean(b))
    r = _corr(odd, even)
    # Spearman-Brown: split-half underestimates full-test reliability.
    adjusted = (2 * r / (1 + r)) if r is not None and r > -1 else None
    return {"players": len(odd), "split_half_r": r, "spearman_brown": adjusted}


def check_predicts_next_match(conn, as_of, norms, min_matches=4) -> dict:
    """Usefulness: does the rating beat raw ACS at predicting the next match?

    For each player, use matches 1..n-1 to predict match n. Compares the mean
    prior RATING against the mean prior ACS as predictors of the same target
    (next-match rating). If ACS wins, the extra machinery earned nothing.
    """
    by_player = _rated_rows(conn, as_of, norms, min_matches)
    tgt, from_rating, from_acs = [], [], []
    for _, matches in by_player.items():
        prior, last = matches[:-1], matches[-1]
        tgt.append(last[1])
        from_rating.append(mean(m[1] for m in prior))
        from_acs.append(mean(m[2] for m in prior))
    return {
        "players": len(tgt),
        "rating_r": _corr(from_rating, tgt),
        "acs_r": _corr(from_acs, tgt),
    }


def run(conn: sqlite3.Connection) -> dict:
    as_of = conn.execute("SELECT MAX(started_at) + 1 t FROM matches").fetchone()["t"]
    norms = build_norms(conn, as_of)
    return {
        "norms_rows": norms.rows_used,
        "rank": check_rank_correlation(conn, as_of, norms),
        "reliability": check_split_half(conn, as_of, norms),
        "usefulness": check_predicts_next_match(conn, as_of, norms),
    }


def main(argv=None) -> int:
    from valwr import config
    from valwr.store import schema

    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    res = run(conn)

    def fmt(v):
        return "n/a" if v is None else f"{v:+.3f}"

    print(f"norms built from {res['norms_rows']:,} player-match rows\n")
    r = res["rank"]
    print("1. rank correlation (sanity)")
    print(f"   n={r['n']:,}  r={fmt(r['corr'])}")
    print("   expect WEAK positive -- the rating normalises within band on")
    print("   purpose, so a strong r would mean it is just a rank proxy.\n")

    r = res["reliability"]
    print("2. split-half reliability")
    print(f"   players={r['players']:,}  r={fmt(r['split_half_r'])}"
          f"  spearman-brown={fmt(r['spearman_brown'])}")
    print("   is it measuring a stable property, or match-to-match noise?\n")

    r = res["usefulness"]
    print("3. predicts next match better than raw ACS?")
    print(f"   players={r['players']:,}  rating r={fmt(r['rating_r'])}"
          f"  acs r={fmt(r['acs_r'])}")
    if r["rating_r"] is not None and r["acs_r"] is not None:
        verdict = "rating wins" if r["rating_r"] > r["acs_r"] else "ACS wins -- rating earned nothing"
        print(f"   -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
