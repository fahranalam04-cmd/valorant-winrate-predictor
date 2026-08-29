"""The 0-100 potential score, run against players whose truth is known.

Live data can tell you *how often* the score picks the best player (30.5%
against 20% chance -- docs/MODEL-CHOICE.md). It cannot tell you *whether the
score is behaving sensibly*, because a real player has no declared skill to
check the answer against.

Synthetic players do. `profiles._ladder` builds every archetype from a single
ability parameter, so `elite` is known to be better than `strong`, which is
known to be better than `average`. That makes ordering an assertion rather
than an impression, and it is the one thing the held-out measurement cannot
provide.

It also makes known defects testable. The score is ACS-led, and ACS depends
heavily on the role a player mains -- measured on real data, duelists average
225 ACS against initiators' 194. `profiles.ROLE_MAINS` reproduces that spread,
so the bias is pinned down by a scenario that fails if it silently changes,
rather than living only in a paragraph of documentation.
"""

from __future__ import annotations

from dataclasses import dataclass

from valwr.rating import potential as pot
from valwr.sandbox import world
from valwr.sandbox.runner import AS_OF
from valwr.sandbox.schema import MatchScenario


@dataclass(frozen=True)
class ScoredPlayer:
    """One synthetic player's score, next to the archetype that produced it."""
    slot: int
    team: str
    profile: str                # the archetype name -- the ground truth label
    agent: str
    role: str | None
    score: int | None           # None when the profile has no history
    reason: str
    components: pot.Components | None

    @property
    def known(self) -> bool:
        return self.score is not None


def score_scenario(scenario: MatchScenario, bundle: dict,
                   index: pot.PerfIndex, as_of: int = AS_OF,
                   team: str | None = None) -> list[ScoredPlayer]:
    """Score every player in a scenario, or just one side.

    Uses the production scorer unchanged -- `rating/potential.evaluate` -- so
    this measures what ships, not a sandbox reimplementation of it.
    """
    conn, roster = world.build_world(scenario, as_of)
    roles = bundle.get("roles") or {}
    by_team = {"Blue": scenario.team_a, "Red": scenario.team_b}

    out: list[ScoredPlayer] = []
    for row in roster:
        if team and row["team"] != team:
            continue
        slot = int(row["puuid"][-1])
        profile = by_team[row["team"]].players[slot]
        p = pot.evaluate(conn, row["puuid"], as_of, scenario.map_name,
                         bundle["norms"], index)
        out.append(ScoredPlayer(
            slot=slot, team=row["team"], profile=profile.name,
            agent=row["agent"], role=roles.get(row["agent"]),
            score=p.score if p else None,
            reason=p.reason if p else "no history",
            components=p.components if p else None))
    conn.close()
    return out


def ranked(scored: list[ScoredPlayer]) -> list[ScoredPlayer]:
    """Best first; players with no history sort last rather than vanishing."""
    return sorted(scored, key=lambda p: (p.known, p.score or 0), reverse=True)


def ordering_matches(scored: list[ScoredPlayer], truth: list[str]) -> bool:
    """Does the score rank these archetypes in the expected order?

    `truth` is best-first. Only players present and scored are compared, so a
    profile with no history does not silently pass the check.
    """
    got = [p.profile for p in ranked(scored) if p.known]
    want = [name for name in truth if name in got]
    return got[:len(want)] == want


def spread(scored: list[ScoredPlayer]) -> int:
    """Highest score minus lowest, over the players that have one."""
    got = [p.score for p in scored if p.known]
    return max(got) - min(got) if got else 0
