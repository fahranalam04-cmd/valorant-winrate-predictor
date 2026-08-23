"""Team-level aggregation of the five players' features.

The aggregations that matter are not only the means. A team with one smurf and
four weak players behaves very differently from five average players, and the
mean erases that distinction entirely -- hence max, min and standard deviation
on the skill features.
"""

from __future__ import annotations

from statistics import pstdev

from valwr.features.player import PlayerFeatures

ROLES = ["Duelist", "Controller", "Initiator", "Sentinel"]

# Features summarised with the full spread rather than just a mean, because
# their distribution within a team carries information.
SPREAD_FEATURES = ["rating", "wr", "tier", "acs", "adr", "kast"]

# Features where the team mean is the whole story.
MEAN_FEATURES = ["games_played", "account_level", "wr_map", "wr_agent",
                 "wr_map_agent", "wr_recent", "games_map", "games_agent",
                 "games_map_agent", "fb_rate", "fd_rate", "rating_trend",
                 "days_since_last", "rating_n"]


def _stat(values: list[float], how: str) -> float:
    if not values:
        return 0.0
    if how == "mean":
        return sum(values) / len(values)
    if how == "max":
        return max(values)
    if how == "min":
        return min(values)
    if how == "std":
        return pstdev(values) if len(values) > 1 else 0.0
    raise ValueError(how)


def build(players: list[PlayerFeatures], agents: list[str],
          party_ids: list[str | None],
          roles: dict[str, str | None]) -> dict[str, float]:
    """Aggregate one team. `players`, `agents` and `party_ids` are parallel."""
    f: dict[str, float] = {}

    for name in SPREAD_FEATURES:
        vals = [p.values.get(name, 0.0) for p in players]
        for how in ("mean", "max", "min", "std"):
            f[f"{name}_{how}"] = _stat(vals, how)

    for name in MEAN_FEATURES:
        f[f"{name}_mean"] = _stat([p.values.get(name, 0.0) for p in players], "mean")

    # --- composition -------------------------------------------------
    team_roles = [roles.get(a) for a in agents]
    for role in ROLES:
        f[f"n_{role.lower()}"] = float(sum(1 for r in team_roles if r == role))
        f[f"has_{role.lower()}"] = float(any(r == role for r in team_roles))

    # Balance: how far the composition sits from one of each role. Low is
    # balanced, high means stacked.
    counts = [sum(1 for r in team_roles if r == role) for role in ROLES]
    f["role_balance"] = float(sum(abs(c - 1.25) for c in counts))
    f["n_off_role"] = float(sum(p.values.get("off_role", 0.0) for p in players))

    # --- party structure ---------------------------------------------
    # party_id is reported directly, so stacks are observed rather than
    # inferred. A five-stack coordinates in ways five solo queuers do not.
    sizes: dict[str, int] = {}
    for pid in party_ids:
        if pid:
            sizes[pid] = sizes.get(pid, 0) + 1
    grouped = [n for n in sizes.values() if n > 1]
    f["max_party"] = float(max(sizes.values()) if sizes else 1)
    f["n_parties"] = float(len(grouped))
    f["n_grouped"] = float(sum(grouped))

    # --- coverage ----------------------------------------------------
    # How much of this team we actually know about. The model can learn to
    # discount teams built mostly from strangers, and the live path uses it
    # to widen its confidence band.
    f["n_with_history"] = float(sum(1 for p in players if p.has_history))

    return f


def feature_names() -> list[str]:
    names = [f"{n}_{how}" for n in SPREAD_FEATURES
             for how in ("mean", "max", "min", "std")]
    names += [f"{n}_mean" for n in MEAN_FEATURES]
    names += [f"n_{r.lower()}" for r in ROLES] + [f"has_{r.lower()}" for r in ROLES]
    names += ["role_balance", "n_off_role", "max_party", "n_parties",
              "n_grouped", "n_with_history"]
    return names
