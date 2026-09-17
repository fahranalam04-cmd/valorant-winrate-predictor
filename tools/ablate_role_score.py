"""What each role's formula is worth, component by component and variant by variant.

    python tools/ablate_role_score.py [--teams 3000] [--sample 30000]

One measurement pass, many scores. Players are measured once -- the expensive
part -- and then scored under several weightings, so the variants are compared
on identical players rather than on separate samples that differ by noise.

Every variant gets its own percentile tables, refitted from the training
samples. That matters: a team is ranked by the 0-100, each role has its own
mapping, and a variant that changed the weights but kept someone else's
mapping would be measured on a score nobody would ship.

Reports, per role: how well the score orders that role's players (rho), how
often it names them when they genuinely had their team's best game (recall),
and which components actually move the number.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, ".")

from valwr import config
from valwr.model import split
from valwr.rating import potential as P
from valwr.rating import role_score as R
from valwr.rating import roleindex, roles
from valwr.rating.normalize import build_norms
from valwr.rating.rating import rate_performance
from valwr.store import schema

ROOT = Path(__file__).resolve().parent.parent
TEAM_SIZE = 5


def _harness():
    spec = importlib.util.spec_from_file_location(
        "validate_potential", ROOT / "tools" / "validate_potential.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rescale(table: dict[str, float], drop: tuple[str, ...] = (),
            set_to: dict[str, float] | None = None) -> dict[str, float]:
    """A role's table with components dropped or re-weighted, back to 100."""
    out = {k: v for k, v in table.items() if k not in drop}
    for k, v in (set_to or {}).items():
        if k in out or v:
            out[k] = v
    fixed = set(set_to or {})
    room = 100.0 - sum(abs(v) for k, v in out.items() if k in fixed)
    rest = sum(abs(v) for k, v in out.items() if k not in fixed)
    if rest <= 0:
        return out
    return {k: (v if k in fixed else v * room / rest) for k, v in out.items()}


def variants() -> dict[str, dict[str, dict[str, float]]]:
    """Each variant is a full set of role tables, in percentage points."""
    base = roles.ROLE_WEIGHTS
    out: dict[str, dict[str, dict[str, float]]] = {}
    out["shipped"] = {r: dict(w) for r, w in base.items()}
    out["no abilities"] = {r: rescale(w, drop=("abilities",))
                           for r, w in base.items()}
    out["abilities at 5"] = {r: rescale(w, set_to={"abilities": 5.0})
                             for r, w in base.items()}
    # Damage back where it was before the support roles were re-cut.
    restored = {r: dict(w) for r, w in base.items()}
    restored["Controller"] = rescale(base["Controller"],
                                     set_to={"acs": 22.0, "adr": 9.0})
    restored["Initiator"] = rescale(base["Initiator"],
                                    set_to={"acs": 24.0, "adr": 13.0})
    out["damage restored"] = restored
    # Dying first is negative for every role, which is what the data shows.
    fd_all = {}
    for r, w in base.items():
        fd_all[r] = rescale(w, set_to={"fd": -6.0})
    out["first deaths everywhere"] = fd_all
    # Both of the above together, with abilities cut.
    combo = {}
    for r, w in restored.items():
        combo[r] = rescale(w, set_to={"abilities": 5.0, "fd": -6.0})
    out["cut abilities + restore damage + fd"] = combo

    # Weighted by what each input was measured to predict rather than by what
    # the role is for. Damage leads everywhere, K/D behind it, KAST small,
    # abilities near zero, opening kills only where they showed signal
    # (Initiators; for Duelists the correlation is negative).
    evidence = {
        "Duelist":    {"acs": 30, "adr": 27, "kd": 22, "kast": 10, "hs": 6,
                       "abilities": 2, "fd": -3},
        "Initiator":  {"acs": 27, "adr": 24, "kda": 20, "kast": 10, "hs": 8,
                       "fb": 6, "abilities": 2, "fd": -3},
        "Controller": {"acs": 28, "adr": 28, "kd": 18, "kast": 11, "hs": 8,
                       "abilities": 2, "assists": 2, "fd": -3},
        "Sentinel":   {"acs": 29, "adr": 29, "kda": 17, "kast": 10, "hs": 8,
                       "abilities": 2, "fd": -5},
    }
    out["evidence-led"] = {r: rescale(w) for r, w in evidence.items()}

    # Only the Duelist table changes; the support roles keep what ships. The
    # ablation says the aggregate loss is concentrated in Duelists, so this
    # asks whether the support gains can be kept while giving that back.
    duelist_fix = {r: dict(w) for r, w in base.items()}
    duelist_fix["Duelist"] = rescale(evidence["Duelist"])
    out["shipped, Duelist by evidence"] = duelist_fix
    return out


def as_fractions(table: dict[str, float]) -> dict[str, float]:
    """Percentage points -> fractions of one, with room for the map term."""
    room = (100.0 - roles.MAP_WEIGHT) / 100.0
    out = {k: v / 100.0 * room for k, v in table.items() if v}
    out["map_edge"] = roles.MAP_WEIGHT / 100.0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ablate_role_score")
    ap.add_argument("--teams", type=int, default=3000)
    ap.add_argument("--sample", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    h = _harness()
    s = config.load(require_key=False)
    conn = schema.connect(s.database_path)
    b = split.compute(conn)
    norms = build_norms(conn, b.train_end)
    old = P.PerfIndex.load()
    shipped = roleindex.RoleIndex.load()
    rba = R.roles_by_agent(conn)

    # --- training samples, for each variant's percentile tables ---------
    started = time.time()
    rows = conn.execute(
        "SELECT puuid, started_at, map, agent FROM match_players "
        "WHERE started_at < ? AND rounds_played > 0", (b.train_end,)).fetchall()
    rng = random.Random(args.seed)
    train = []
    for r in rng.sample(list(rows), min(args.sample, len(rows))):
        role = rba.get(r["agent"])
        # Shrunk, exactly as a live player is. The shipped index fits its
        # tables on UNSHRUNK values, which is the inconsistency this run is
        # partly here to measure.
        c = R.measure(conn, r["puuid"], r["started_at"], r["map"], role, rba,
                      role_means=shipped.role_means.get(role or "?"),
                      agent=r["agent"])
        if c is not None:
            train.append(c)
    print(f"{len(train):,} training samples ({time.time() - started:.0f}s)")

    # --- test teams, measured once --------------------------------------
    started = time.time()
    trows = conn.execute(
        "SELECT * FROM match_players WHERE started_at >= ? AND rounds_played > 0",
        (b.val_end,)).fetchall()
    by_team = defaultdict(list)
    for r in trows:
        by_team[(r["match_id"], r["team"])].append(dict(r))
    full = [v for v in by_team.values() if len(v) == TEAM_SIZE]
    rng.shuffle(full)

    teams = []
    for squad in full:
        as_of = squad[0]["started_at"]
        scored = []
        for row in squad:
            role = rba.get(row["agent"])
            comp = R.measure(conn, row["puuid"], as_of, row["map"], role, rba,
                             role_means=shipped.role_means.get(role or "?"),
                             agent=row["agent"])
            comp_old = P.measure(conn, row["puuid"], as_of, row["map"], norms)
            actual = rate_performance(row, norms)
            if comp is None or comp_old is None or actual is None:
                break
            scored.append({"c": comp, "role": role or "?",
                           "raw": old.composite(comp_old),
                           "acs": comp_old.acs, "actual": actual.value})
        if len(scored) == TEAM_SIZE:
            teams.append(scored)
        if len(teams) >= args.teams:
            break
    flat = [p for t in teams for p in t]
    n = len(teams)
    se = (0.2 * 0.8 / n) ** 0.5 * 100
    print(f"{n:,} test teams / {len(flat):,} players "
          f"({time.time() - started:.0f}s), standard error +/-{se:.1f}\n")

    def score_all(tables: dict[str, dict[str, float]], key: str) -> None:
        """Fit this variant's percentile tables, then score every test player."""
        quant: dict[str, list[float]] = defaultdict(list)
        for c in train:
            w = as_fractions(tables.get(c.role or "", tables.get("Duelist", {})))
            raw = shipped.composite(c, w)
            if raw is not None:
                quant[c.role or "?"].append(raw)
        for v in quant.values():
            v.sort()
        import bisect
        for p in flat:
            w = as_fractions(tables.get(p["role"], tables.get("Duelist", {})))
            raw = shipped.composite(p["c"], w)
            qs = quant.get(p["role"]) or []
            if raw is None or not qs:
                p[key] = None
                continue
            p[key] = 100.0 * bisect.bisect_left(qs, raw) / len(qs)
            # Kept so the same weights can also be measured ranked by the
            # raw composite. The percentile is taken within a role, which
            # erases the fact that some roles top their team far more
            # often than others -- fine for reading a number, possibly
            # costly for picking one player out of five.
            p[key + " |raw"] = raw

    names = list(variants())
    for label, tables in variants().items():
        score_all(tables, label)
    # Everyone must be scored under every variant, or the comparison is
    # between different sets of players.
    keep = [t for t in teams if all(p[k] is not None for p in t for k in names)]
    print(f"{len(keep):,} teams scored under every variant\n")

    print("=" * 74)
    print("PICKING THE BEST PLAYER OUT OF FIVE   (chance = 20.0%)")
    print("=" * 74)
    rng2 = random.Random(args.seed + 1)
    rank = [("the old score", lambda p: p["raw"]),
            ("career ACS alone", lambda p: p["acs"])]
    rank += [(label, (lambda k: (lambda p: p[k]))(label)) for label in names]
    rank += [("shuffled (control)", lambda p: rng2.random())]
    for label, key in rank:
        acc, _ = h.top1(keep, key)
        print(f"  {label:<34}{acc * 100:>7.1f}%{(acc - 0.2) * 100:>+9.1f}")

    print("THE SAME WEIGHTS, RANKED BY RAW COMPOSITE NOT THE PERCENTILE")
    print("=" * 74)
    print("  A within-role percentile throws away the base rate: a Duelist"
          "\n  tops their team far more often than an Initiator does."
          "\n  Ranking by the raw composite keeps that, and the page"
          "\n  could still display the percentile.")
    for label in names:
        acc, _ = h.top1(keep, (lambda p, k=label: p[k + " |raw"]))
        print(f"  {label:<34}{acc * 100:>7.1f}%{(acc - 0.2) * 100:>+9.1f}")
    print("\n" + "=" * 74)
    print("\n" + "=" * 74)
    print("BY ROLE: how well each variant orders that role's players (rho)")
    print("=" * 74)
    kept_flat = [p for t in keep for p in t]
    roles_seen = ("Duelist", "Initiator", "Controller", "Sentinel")
    print(f"  {'variant':<34}" + "".join(f"{r[:9]:>11}" for r in roles_seen))
    print("  " + "-" * 70)
    for label in ["the old score"] + names:
        key = (lambda p: p["raw"]) if label == "the old score" \
            else (lambda p, k=label: p[k])
        cells = []
        for role in roles_seen:
            ps = [p for p in kept_flat if p["role"] == role]
            cells.append(h.spearman([key(p) for p in ps],
                                    [p["actual"] for p in ps]) if ps else 0.0)
        print(f"  {label:<34}" + "".join(f"{c:>+11.3f}" for c in cells))

    print("\n" + "=" * 74)
    print("BY ROLE: named when they genuinely had the team's best game")
    print("=" * 74)
    print(f"  {'variant':<34}" + "".join(f"{r[:9]:>11}" for r in roles_seen))
    print("  " + "-" * 70)
    counts = Counter(p["role"] for t in keep for p in t
                     if abs(p["actual"] - max(q["actual"] for q in t)) < 1e-12)
    for label in ["the old score"] + names:
        key = (lambda p: p["raw"]) if label == "the old score" \
            else (lambda p, k=label: p[k])
        cells = []
        for role in roles_seen:
            rows_ = h.role_breakdown(keep, kept_flat, key=key)
            got = next((r for r in rows_ if r["role"] == role), None)
            cells.append((got or {}).get("recall", 0.0) * 100)
        print(f"  {label:<34}" + "".join(f"{c:>10.1f}%" for c in cells))
    print("  " + "-" * 70)
    print(f"  {'teams where this role was best':<34}"
          + "".join(f"{counts.get(r, 0):>11,}" for r in roles_seen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
