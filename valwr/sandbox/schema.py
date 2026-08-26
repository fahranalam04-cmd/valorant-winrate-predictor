"""Sandbox data model.

The central decision: a `PlayerProfile` declares **raw match history**, not
target feature values.

Declaring features directly would be easier and would prove nothing -- it would
bypass shrinkage, recency weighting, the rating composite and the opponent
adjustment, which are exactly the parts most worth testing. A profile saying
"3 games, 100% win rate" exists so the pipeline can be observed shrinking it.

Everything here is a frozen dataclass. Scenario construction must be
deterministic, and mutable defaults are how that quietly stops being true.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable

# Domain bounds. Values outside these are a bug in a scenario, not an
# interesting edge case -- `validate()` rejects them rather than letting a
# NaN reach the model several layers later.
RATE_BOUNDS = (0.0, 1.0)
ACS_BOUNDS = (0.0, 600.0)
ADR_BOUNDS = (0.0, 400.0)
TIER_BOUNDS = (0, 27)
LEVEL_BOUNDS = (1, 3000)
DAYS_BOUNDS = (0.0, 1500.0)
MAX_GAMES = 5000


class InvalidProfile(ValueError):
    """A profile whose values cannot describe a real player."""


@dataclass(frozen=True)
class PlayerProfile:
    """One synthetic player, described by the history they would have.

    `games` is the total prior match count. The map/agent counts are subsets of
    it, and `map_agent_games` is a subset of both -- `validate()` enforces that
    rather than letting an impossible partition through.
    """

    name: str = "average"

    # --- volume -------------------------------------------------------
    games: int = 60
    map_games: int = 12
    agent_games: int = 20
    map_agent_games: int = 5

    # --- outcomes (raw rates, pre-shrinkage) --------------------------
    win_rate: float = 0.50
    map_win_rate: float = 0.50
    agent_win_rate: float = 0.50
    map_agent_win_rate: float = 0.50
    recent_win_rate: float | None = None      # None = same as win_rate

    # --- per-match performance ----------------------------------------
    acs: float = 210.0
    adr: float = 140.0
    kast: float = 0.71
    fb_rate: float = 0.10
    fd_rate: float = 0.10

    # --- identity / context -------------------------------------------
    tier: int = 15                             # Platinum 1
    account_level: int = 120
    days_since_last: float = 1.0
    trend: float = 0.0                         # >0 improving, <0 declining
    role: str = "Duelist"                      # the role they usually play
    party: str | None = None                   # shared id = same party

    def validate(self) -> None:
        def bounded(label, value, lo, hi):
            if value is None:
                return
            if not math.isfinite(value):
                raise InvalidProfile(f"{self.name}.{label} is not finite")
            if not lo <= value <= hi:
                raise InvalidProfile(
                    f"{self.name}.{label}={value} outside [{lo}, {hi}]")

        if self.games < 0 or self.games > MAX_GAMES:
            raise InvalidProfile(f"{self.name}.games={self.games} implausible")
        for label in ("map_games", "agent_games", "map_agent_games"):
            n = getattr(self, label)
            if n < 0:
                raise InvalidProfile(f"{self.name}.{label} is negative")
            if n > self.games:
                raise InvalidProfile(
                    f"{self.name}.{label}={n} exceeds games={self.games}")
        if self.map_agent_games > min(self.map_games, self.agent_games):
            raise InvalidProfile(
                f"{self.name}.map_agent_games={self.map_agent_games} exceeds "
                f"map_games={self.map_games} or agent_games={self.agent_games}")

        for label in ("win_rate", "map_win_rate", "agent_win_rate",
                      "map_agent_win_rate", "recent_win_rate", "kast",
                      "fb_rate", "fd_rate"):
            bounded(label, getattr(self, label), *RATE_BOUNDS)
        bounded("acs", self.acs, *ACS_BOUNDS)
        bounded("adr", self.adr, *ADR_BOUNDS)
        bounded("days_since_last", self.days_since_last, *DAYS_BOUNDS)
        bounded("trend", self.trend, -1.0, 1.0)
        if not TIER_BOUNDS[0] <= self.tier <= TIER_BOUNDS[1]:
            raise InvalidProfile(f"{self.name}.tier={self.tier} outside range")
        if not LEVEL_BOUNDS[0] <= self.account_level <= LEVEL_BOUNDS[1]:
            raise InvalidProfile(f"{self.name}.account_level implausible")

    @property
    def has_history(self) -> bool:
        return self.games > 0

    def with_(self, **changes) -> "PlayerProfile":
        """A copy with fields replaced -- the only way scenarios vary a profile."""
        return replace(self, **changes)


@dataclass(frozen=True)
class TeamProfile:
    players: tuple[PlayerProfile, ...]
    agents: tuple[str, ...] = ()      # agent each player picks this match

    def validate(self) -> None:
        if len(self.players) != 5:
            raise InvalidProfile(f"team has {len(self.players)} players, need 5")
        if self.agents and len(self.agents) != 5:
            raise InvalidProfile("agents must be empty or name all five picks")
        for p in self.players:
            p.validate()

    def agent_for(self, index: int) -> str:
        if self.agents:
            return self.agents[index]
        return "Jett"


@dataclass(frozen=True)
class VarianceSpec:
    """How much a scenario's values may move, and what must stay true.

    `preserve` guards scenario identity: `single_smurf` is meaningless if a
    realisation makes the smurf the weakest player in the lobby, so variance is
    rejection-sampled against it.
    """

    skill_sigma: float = 0.06         # latent ability shock, multiplicative
    form_sigma: float = 0.05
    map_sigma: float = 0.05
    agent_sigma: float = 0.05
    noise_sigma: float = 0.02         # independent per-stat measurement error
    tier_jitter: int = 1
    games_jitter: float = 0.15        # proportion of the count
    preserve: Callable[["MatchScenario"], bool] | None = None
    max_attempts: int = 50


@dataclass(frozen=True)
class MatchScenario:
    name: str
    category: str
    description: str
    team_a: TeamProfile
    team_b: TeamProfile
    map_name: str = "Ascent"
    variance: VarianceSpec = field(default_factory=VarianceSpec)
    # Directional expectation, e.g. "a_favoured". Reported, never enforced --
    # a trained model disagreeing with a reasonable assumption is a finding.
    expect: str | None = None
    tags: tuple[str, ...] = ()

    def validate(self) -> None:
        self.team_a.validate()
        self.team_b.validate()

    def mirrored(self) -> "MatchScenario":
        """The same match with the sides swapped.

        Generated rather than hand-written, so no scenario can have a
        mirror that silently drifts out of step with it.
        """
        flip = {"a_favoured": "b_favoured", "b_favoured": "a_favoured"}
        return MatchScenario(
            name=f"{self.name}__mirror",
            category=self.category,
            description=f"{self.description} (sides swapped)",
            team_a=self.team_b,
            team_b=self.team_a,
            map_name=self.map_name,
            variance=self.variance,
            expect=flip.get(self.expect or "", self.expect),
            tags=self.tags + ("mirror",),
        )


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    category: str
    model: str
    probability: float                 # P(team A wins)
    features: dict[str, float]
    mirror_probability: float | None = None
    expect: str | None = None
    factors: tuple[tuple[str, float], ...] = ()

    @property
    def mirror_error(self) -> float | None:
        if self.mirror_probability is None:
            return None
        return abs(self.probability + self.mirror_probability - 1.0)

    @property
    def expectation_met(self) -> bool | None:
        """None when the scenario makes no directional claim."""
        if self.expect is None:
            return None
        if self.expect == "a_favoured":
            return self.probability > 0.5
        if self.expect == "b_favoured":
            return self.probability < 0.5
        if self.expect == "even":
            return abs(self.probability - 0.5) < 0.02
        return None


@dataclass(frozen=True)
class VarianceResult:
    scenario: str
    model: str
    samples: int
    seed: int
    static_probability: float
    probabilities: tuple[float, ...]

    def _q(self, p: float) -> float:
        xs = sorted(self.probabilities)
        if not xs:
            return float("nan")
        k = (len(xs) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    @property
    def mean(self) -> float:
        return sum(self.probabilities) / len(self.probabilities)

    @property
    def std(self) -> float:
        m = self.mean
        var = sum((p - m) ** 2 for p in self.probabilities) / len(self.probabilities)
        return math.sqrt(var)

    @property
    def median(self) -> float:
        return self._q(0.5)

    def percentile(self, p: float) -> float:
        return self._q(p)

    @property
    def flip_rate(self) -> float:
        """How often the favourite changes side between static and sample."""
        favoured_a = self.static_probability > 0.5
        flips = sum(1 for p in self.probabilities if (p > 0.5) != favoured_a)
        return flips / len(self.probabilities)

    @property
    def drift(self) -> float:
        return self.mean - self.static_probability

    @property
    def robustness(self) -> str:
        """A developer heuristic, deliberately not a scientific claim.

        Thresholds are judgement calls documented in docs/SANDBOX.md; they
        exist to sort a long report, not to certify anything.
        """
        if self.std < 0.02 and self.flip_rate < 0.10:
            return "stable"
        if self.std < 0.05 and self.flip_rate < 0.30:
            return "moderately sensitive"
        return "highly sensitive"


@dataclass(frozen=True)
class SandboxRun:
    results: tuple[ScenarioResult, ...]
    variance: tuple[VarianceResult, ...] = ()
    model: str = "logistic"
    seed: int | None = None
