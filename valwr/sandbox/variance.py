"""Realistic stochastic realisations around a benchmark scenario.

Static scenarios are exact. Variance asks the different question: does the
prediction survive plausible noise, or was the benchmark balanced on a knife
edge?

**Correlated, not independent.** Perturbing every statistic on its own produces
players who do not exist -- elite damage with terrible KAST, a rising trend
with a collapsing win rate. Instead each player draws a few latent shocks and
the observable statistics are derived from them:

    skill  -> ACS, ADR, KAST, first-blood rate, win rate  (one ability)
    form   -> recent win rate, trend
    map    -> map win rate
    agent  -> agent win rate, map x agent win rate
    noise  -> small independent per-statistic measurement error

That is a deliberately coarse mechanism. The goal is plausible variance, not a
generative model of VALORANT, and pretending otherwise would be worse than
being crude on purpose.

**Identity is preserved.** A realisation of `single_smurf` where the smurf is
the worst player in the lobby is not a noisy version of that scenario, it is a
different one. Scenarios may declare a predicate and realisations are
rejection-sampled against it.

Everything is seeded. `--seed 42` reproduces exactly; a different seed does not.
"""

from __future__ import annotations

import numpy as np

from valwr.sandbox.schema import (MatchScenario, PlayerProfile, TeamProfile,
                                  VarianceSpec)

# How strongly each observable follows the latent skill shock. Judgement, not
# fitted -- documented in docs/SANDBOX.md.
SKILL_LOADING = {"acs": 1.0, "adr": 0.95, "kast": 0.45, "fb_rate": 0.7}
# Better players die first *less*, so this loading is negative on purpose.
FD_LOADING = -0.5
WIN_RATE_LOADING = 0.35


def _clip(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def _jitter_count(rng, n: int, proportion: float) -> int:
    """Integer counts stay integers -- there is no such thing as 3.4 games."""
    if n <= 0:
        return 0
    delta = rng.normal(0.0, proportion * n)
    return max(0, int(round(n + delta)))


def perturb_player(p: PlayerProfile, rng: np.random.Generator,
                   spec: VarianceSpec) -> PlayerProfile:
    skill = rng.normal(0.0, spec.skill_sigma)
    form = rng.normal(0.0, spec.form_sigma)
    map_shock = rng.normal(0.0, spec.map_sigma)
    agent_shock = rng.normal(0.0, spec.agent_sigma)

    def noisy(base: float, loading: float = 0.0) -> float:
        return base * (1.0 + loading * skill + rng.normal(0.0, spec.noise_sigma))

    acs = _clip(noisy(p.acs, SKILL_LOADING["acs"]), 0.0, 600.0)
    adr = _clip(noisy(p.adr, SKILL_LOADING["adr"]), 0.0, 400.0)
    kast = _clip(noisy(p.kast, SKILL_LOADING["kast"]), 0.0, 1.0)
    fb = _clip(noisy(p.fb_rate, SKILL_LOADING["fb_rate"]), 0.0, 1.0)
    fd = _clip(noisy(p.fd_rate, FD_LOADING), 0.0, 1.0)

    # Counts move first, because the rates below must stay consistent with them.
    games = _jitter_count(rng, p.games, spec.games_jitter)
    map_games = min(games, _jitter_count(rng, p.map_games, spec.games_jitter))
    agent_games = min(games, _jitter_count(rng, p.agent_games, spec.games_jitter))
    ma_games = min(map_games, agent_games,
                   _jitter_count(rng, p.map_agent_games, spec.games_jitter))

    wr = _clip(p.win_rate + WIN_RATE_LOADING * skill
               + rng.normal(0.0, spec.noise_sigma), 0.0, 1.0)
    map_wr = _clip(p.map_win_rate + map_shock + 0.5 * WIN_RATE_LOADING * skill,
                   0.0, 1.0)
    agent_wr = _clip(p.agent_win_rate + agent_shock
                     + 0.5 * WIN_RATE_LOADING * skill, 0.0, 1.0)
    ma_wr = _clip(p.map_agent_win_rate + 0.7 * (map_shock + agent_shock),
                  0.0, 1.0)
    recent = None
    if p.recent_win_rate is not None:
        recent = _clip(p.recent_win_rate + form + WIN_RATE_LOADING * skill,
                       0.0, 1.0)

    tier = int(_clip(p.tier + rng.integers(-spec.tier_jitter,
                                           spec.tier_jitter + 1), 0, 27))
    days = _clip(p.days_since_last * (1.0 + rng.normal(0.0, 0.25)), 0.0, 1500.0)
    trend = _clip(p.trend + form, -1.0, 1.0)

    return p.with_(
        games=games, map_games=map_games, agent_games=agent_games,
        map_agent_games=ma_games, win_rate=wr, map_win_rate=map_wr,
        agent_win_rate=agent_wr, map_agent_win_rate=ma_wr,
        recent_win_rate=recent, acs=acs, adr=adr, kast=kast, fb_rate=fb,
        fd_rate=fd, tier=tier, days_since_last=days, trend=trend,
    )


def perturb_team(t: TeamProfile, rng, spec: VarianceSpec) -> TeamProfile:
    return TeamProfile(
        players=tuple(perturb_player(p, rng, spec) for p in t.players),
        agents=t.agents)


def realise(scenario: MatchScenario, rng: np.random.Generator) -> MatchScenario:
    """One plausible realisation, respecting the scenario's identity guard."""
    spec = scenario.variance
    for _ in range(max(1, spec.max_attempts)):
        candidate = MatchScenario(
            name=scenario.name, category=scenario.category,
            description=scenario.description,
            team_a=perturb_team(scenario.team_a, rng, spec),
            team_b=perturb_team(scenario.team_b, rng, spec),
            map_name=scenario.map_name, variance=spec, expect=scenario.expect,
            tags=scenario.tags)
        try:
            candidate.validate()
        except Exception:
            continue                       # an impossible draw; try again
        if spec.preserve is None or spec.preserve(candidate):
            return candidate
    # Falling back to the exact scenario is the honest failure: it keeps the
    # sample count truthful rather than silently returning fewer draws, and
    # shows up as an unusually low variance rather than as missing data.
    return scenario


def sample(scenario: MatchScenario, samples: int, seed: int):
    """Yield `samples` realisations, reproducibly."""
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        yield realise(scenario, rng)
