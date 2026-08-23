"""Per-player, per-match performance components derived from round and kill data.

These are the inputs to the rating metric. They come from the `rounds` and
`kills` arrays in the raw match body, which the aggregate `stats` block does
not expose -- see docs/API-NOTES.md.

Everything here is computed from what the API actually returns. Where a metric
would need guessing, it is marked as inferred (clutches) or left out entirely.
A noisy component makes the composite worse, not more complete.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

# A kill counts as a trade if it lands within this long after the victim
# killed one of the avenger's teammates. Five seconds is the community
# convention; it is a judgement call, not a fact, so it lives here as a
# named constant rather than buried in the loop.
TRADE_WINDOW_MS = 5_000

MULTIKILL_THRESHOLD = 3


def _puuid(obj: Any) -> str | None:
    if isinstance(obj, dict):
        p = obj.get("puuid")
        if p:
            return p
        inner = obj.get("player")
        if isinstance(inner, dict):
            return inner.get("puuid")
    return None


def blank() -> dict[str, int]:
    return {
        "rounds_played": 0,
        "first_bloods": 0,
        "first_deaths": 0,
        "multikills": 0,
        "trade_kills": 0,
        "traded_deaths": 0,
        "kast_rounds": 0,
        "clutches": 0,
    }


def match_components(m: dict) -> dict[str, dict[str, int]]:
    """Derive per-player components for one match. Returns {puuid: stats}."""
    rounds = m.get("rounds") or []
    kills = m.get("kills") or []
    players = m.get("players") or []

    teams: dict[str, str] = {}
    for p in players:
        if p.get("puuid"):
            teams[p["puuid"]] = p.get("team_id")

    out = {puuid: blank() for puuid in teams}
    n_rounds = len(rounds)
    for stats in out.values():
        stats["rounds_played"] = n_rounds

    by_round: dict[int, list[dict]] = defaultdict(list)
    for k in kills:
        by_round[k.get("round", -1)].append(k)

    for idx, rnd in enumerate(rounds):
        rk = sorted(by_round.get(idx, []),
                    key=lambda k: k.get("time_in_round_in_ms") or 0)

        killers_this_round: dict[str, int] = defaultdict(int)
        assisted: set[str] = set()
        victims: set[str] = set()
        traded: set[str] = set()

        if rk:
            first = rk[0]
            fb, fd = _puuid(first.get("killer")), _puuid(first.get("victim"))
            if fb in out:
                out[fb]["first_bloods"] += 1
            if fd in out:
                out[fd]["first_deaths"] += 1

        for k in rk:
            killer, victim = _puuid(k.get("killer")), _puuid(k.get("victim"))
            t = k.get("time_in_round_in_ms") or 0
            if killer:
                killers_this_round[killer] += 1
            if victim:
                victims.add(victim)
            for a in k.get("assistants") or []:
                pa = _puuid(a)
                if pa:
                    assisted.add(pa)

            # A trade: `killer` avenges a teammate that `victim` killed
            # moments earlier.
            if killer and victim and killer in teams:
                for prior in rk:
                    if (prior.get("time_in_round_in_ms") or 0) >= t:
                        break
                    pk, pv = _puuid(prior.get("killer")), _puuid(prior.get("victim"))
                    if pk != victim or pv not in teams:
                        continue
                    if teams.get(pv) != teams.get(killer):
                        continue
                    if t - (prior.get("time_in_round_in_ms") or 0) <= TRADE_WINDOW_MS:
                        out[killer]["trade_kills"] += 1
                        if pv in out:
                            out[pv]["traded_deaths"] += 1
                            traded.add(pv)
                        break

        for puuid, n in killers_this_round.items():
            if puuid in out and n >= MULTIKILL_THRESHOLD:
                out[puuid]["multikills"] += 1

        # KAST: the player got a Kill, an Assist, Survived, or was Traded.
        for puuid in out:
            if (puuid in killers_this_round or puuid in assisted
                    or puuid not in victims or puuid in traded):
                out[puuid]["kast_rounds"] += 1

        # Clutches are INFERRED. `ceremony` marks that a clutch happened but
        # not who won it, so we take the sole survivor on the winning team.
        # Ambiguous cases are skipped rather than guessed.
        if rnd.get("ceremony") == "CeremonyClutch":
            winner = rnd.get("winning_team")
            survivors = [p for p, t in teams.items()
                         if t == winner and p not in victims]
            if len(survivors) == 1:
                out[survivors[0]]["clutches"] += 1

    return out


def per_round_rates(row: dict) -> dict[str, float | None]:
    """Turn stored counts into rate features. None when there is no denominator."""
    r = row.get("rounds_played") or 0
    if not r:
        return {k: None for k in
                ("acs", "adr", "kpr", "dpr", "apr", "kast",
                 "fb_rate", "fd_rate", "multikill_rate", "trade_rate")}

    def rate(v):
        return (v or 0) / r

    shots = sum(row.get(k) or 0 for k in ("headshots", "bodyshots", "legshots"))
    return {
        "acs": rate(row.get("score")),
        "adr": rate(row.get("damage_dealt")),
        "kpr": rate(row.get("kills")),
        "dpr": rate(row.get("deaths")),
        "apr": rate(row.get("assists")),
        "kast": rate(row.get("kast_rounds")),
        "fb_rate": rate(row.get("first_bloods")),
        "fd_rate": rate(row.get("first_deaths")),
        "multikill_rate": rate(row.get("multikills")),
        "trade_rate": rate(row.get("trade_kills")),
        "hs_pct": (row.get("headshots") or 0) / shots if shots else None,
    }
