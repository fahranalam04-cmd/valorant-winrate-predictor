"""Static game metadata from valorant-api.com.

No API key required, and no rate limit worth worrying about. Pull once; refresh
only when a new agent or map ships.

The agent->role mapping is the important one -- it drives every composition
feature later. Roles are read from the API rather than hardcoded because agents
get reworked and reassigned.
"""

from __future__ import annotations

import sqlite3

import httpx

BASE = "https://valorant-api.com/v1"
TIMEOUT = 30.0


def _get(client: httpx.Client, path: str, **params) -> list[dict]:
    r = client.get(f"{BASE}/{path}", params=params or None)
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data")
    if data is None:
        raise RuntimeError(f"valorant-api.com/{path} returned no data field")
    return data


def load_agents(conn: sqlite3.Connection, client: httpx.Client) -> int:
    rows = []
    for a in _get(client, "agents", isPlayableCharacter="true"):
        role = a.get("role") or {}
        rows.append((a["uuid"], a["displayName"], role.get("displayName")))
    conn.executemany(
        "INSERT INTO ref_agents (uuid, name, role) VALUES (?,?,?) "
        "ON CONFLICT(uuid) DO UPDATE SET name=excluded.name, role=excluded.role",
        rows,
    )
    return len(rows)


def load_maps(conn: sqlite3.Connection, client: httpx.Client) -> int:
    rows = [
        (m["uuid"], m["displayName"])
        for m in _get(client, "maps")
        # Non-playable entries (the range, tutorial) have no displayName or no
        # coordinates. Keep only real maps -- they are what matches happen on.
        if m.get("displayName") and m.get("coordinates")
    ]
    conn.executemany(
        "INSERT INTO ref_maps (uuid, name) VALUES (?,?) "
        "ON CONFLICT(uuid) DO UPDATE SET name=excluded.name",
        rows,
    )
    return len(rows)


def load_tiers(conn: sqlite3.Connection, client: httpx.Client) -> int:
    """Competitive tiers, from the most recent tier table.

    The endpoint returns one entry per historical tier layout; the last is
    current. `tier` is the numeric ordering that makes ranks comparable.
    """
    tables = _get(client, "competitivetiers")
    if not tables:
        raise RuntimeError("no competitive tier tables returned")
    tiers = tables[-1]["tiers"]
    rows = [
        (t["tier"], t["tierName"], t.get("divisionName"))
        for t in tiers
        # Tier 0 is "Unranked" / unused placeholder rows share blank names.
        if t.get("tierName")
    ]
    conn.executemany(
        "INSERT INTO ref_tiers (tier, name, division) VALUES (?,?,?) "
        "ON CONFLICT(tier) DO UPDATE SET name=excluded.name, division=excluded.division",
        rows,
    )
    return len(rows)


def load_seasons(conn: sqlite3.Connection, client: httpx.Client) -> int:
    rows = [
        (
            s["uuid"],
            s.get("displayName") or "",
            s.get("type"),
            s.get("parentUuid"),
            s.get("startTime"),
            s.get("endTime"),
        )
        for s in _get(client, "seasons")
    ]
    conn.executemany(
        "INSERT INTO ref_seasons (uuid, name, type, parent_uuid, start_time, end_time) "
        "VALUES (?,?,?,?,?,?) ON CONFLICT(uuid) DO UPDATE SET "
        "name=excluded.name, type=excluded.type, parent_uuid=excluded.parent_uuid, "
        "start_time=excluded.start_time, end_time=excluded.end_time",
        rows,
    )
    return len(rows)


def load_all(conn: sqlite3.Connection) -> dict[str, int]:
    """Refresh every reference table. Safe to re-run."""
    with httpx.Client(timeout=TIMEOUT) as client:
        counts = {
            "agents": load_agents(conn, client),
            "maps": load_maps(conn, client),
            "tiers": load_tiers(conn, client),
            "seasons": load_seasons(conn, client),
        }
    conn.commit()
    return counts


def agent_roles(conn: sqlite3.Connection) -> dict[str, str | None]:
    """Agent name -> role. The lookup every composition feature needs."""
    return {r["name"]: r["role"] for r in conn.execute("SELECT name, role FROM ref_agents")}
