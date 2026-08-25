"""Who is in the match right now.

Two phases matter. In **pre-game** (agent select) the roster exists but agents
are still being locked, so it arrives incomplete and fills in. In **core-game**
the match is underway and everything is settled.

`404 RESOURCE_NOT_FOUND` from either endpoint is the normal "not in a match"
answer, not an error -- verified against a live client sitting in the lobby.

Read-only. This module never selects, locks, hovers or dodges an agent; that is
where bans actually happen. See docs/ETHICS-AND-TOS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from valwr.live.session import Session

TIMEOUT = 20.0

# Agents arrive as UUIDs; ref_agents maps them to names.
UNKNOWN_AGENT = "?"


@dataclass(frozen=True)
class LivePlayer:
    puuid: str
    team: str                       # "Blue" | "Red"
    agent_id: str | None            # None while still picking
    agent: str = UNKNOWN_AGENT      # resolved name, filled by resolve_agents


@dataclass(frozen=True)
class LiveMatch:
    match_id: str
    phase: str                      # "pregame" | "coregame"
    map_name: str | None
    mode: str | None
    players: list[LivePlayer] = field(default_factory=list)

    @property
    def locked_in(self) -> int:
        return sum(1 for p in self.players if p.agent_id)

    def team_of(self, puuid: str) -> str | None:
        for p in self.players:
            if p.puuid == puuid:
                return p.team
        return None


def _get(session: Session, url: str) -> dict | None:
    """GET a glz endpoint. None means 'not in a match', which is not an error."""
    r = httpx.get(url, headers=session.headers, timeout=TIMEOUT)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def current_match_id(session: Session) -> tuple[str, str] | None:
    """(match_id, phase) for the match in progress, or None.

    Checks core-game first: if a match is underway that is the authoritative
    answer, and a stale pre-game id can linger briefly after the transition.
    """
    core = _get(session, f"{session.glz}/core-game/v1/players/{session.puuid}")
    if core and core.get("MatchID"):
        return core["MatchID"], "coregame"
    pre = _get(session, f"{session.glz}/pregame/v1/players/{session.puuid}")
    if pre and pre.get("MatchID"):
        return pre["MatchID"], "pregame"
    return None


def _map_name(url: str | None) -> str | None:
    """Map arrives as an asset path like /Game/Maps/Ascent/Ascent."""
    if not url:
        return None
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail or None


def fetch(session: Session, match_id: str, phase: str) -> LiveMatch | None:
    """The full roster for a match id."""
    if phase == "coregame":
        data = _get(session, f"{session.glz}/core-game/v1/matches/{match_id}")
        if not data:
            return None
        players = [
            LivePlayer(puuid=p["Subject"],
                       team="Blue" if p.get("TeamID") == "Blue" else "Red",
                       agent_id=p.get("CharacterID") or None)
            for p in data.get("Players", []) if p.get("Subject")
        ]
        return LiveMatch(match_id=match_id, phase=phase,
                         map_name=_map_name(data.get("MapID")),
                         mode=_map_name(data.get("ModeID")), players=players)

    data = _get(session, f"{session.glz}/pregame/v1/matches/{match_id}")
    if not data:
        return None

    # Pre-game splits the two sides: AllyTeam is fully described, the enemy
    # side is only counted. So a pre-game roster is genuinely partial -- five
    # known players, not ten -- and the live path has to say so rather than
    # pretend otherwise.
    players: list[LivePlayer] = []
    ally = data.get("AllyTeam") or {}
    ally_side = "Blue" if ally.get("TeamID") == "Blue" else "Red"
    for p in ally.get("Players", []):
        if p.get("Subject"):
            players.append(LivePlayer(puuid=p["Subject"], team=ally_side,
                                      agent_id=p.get("CharacterID") or None))
    return LiveMatch(match_id=match_id, phase=phase,
                     map_name=_map_name(data.get("MapID")),
                     mode=_map_name(data.get("ModeID")), players=players)


def resolve_agents(match: LiveMatch, agents_by_id: dict[str, str]) -> LiveMatch:
    """Turn agent UUIDs into names using the ref_agents table."""
    return LiveMatch(
        match_id=match.match_id, phase=match.phase, map_name=match.map_name,
        mode=match.mode,
        players=[
            LivePlayer(puuid=p.puuid, team=p.team, agent_id=p.agent_id,
                       agent=agents_by_id.get((p.agent_id or "").lower(),
                                              UNKNOWN_AGENT))
            for p in match.players
        ],
    )


def current(session: Session, agents_by_id: dict[str, str] | None = None
            ) -> LiveMatch | None:
    """One call: find the current match and return its roster, or None."""
    found = current_match_id(session)
    if not found:
        return None
    match = fetch(session, *found)
    if match and agents_by_id:
        match = resolve_agents(match, agents_by_id)
    return match
