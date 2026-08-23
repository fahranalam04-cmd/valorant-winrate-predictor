"""Match-level context. Symmetric -- it belongs to neither team."""

from __future__ import annotations

# Attacking first is a genuine asymmetry on some maps, so which side a team
# starts on is context rather than a team property.
FIRST_SIDE_TEAM = "Red"   # Red attacks first in standard competitive


def build(match: dict) -> dict[str, float | str]:
    return {
        "map": match.get("map") or "?",
        "season": match.get("season") or "?",
        "region": match.get("region") or "?",
    }


CONTEXT_CATEGORICAL = ["map", "season", "region"]
