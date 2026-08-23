"""Phase 0 tests: schema creation and config validation."""

from __future__ import annotations

import sqlite3

import pytest

from valwr import config
from valwr.store import schema


def test_create_all_is_idempotent(tmp_path):
    """Re-running the schema must never error or duplicate.

    It gets run on every startup, so this is a real requirement rather than a
    theoretical one.
    """
    db = tmp_path / "t.db"
    conn = schema.connect(db)

    schema.create_all(conn)
    first = schema.table_counts(conn)

    schema.create_all(conn)
    second = schema.table_counts(conn)

    assert first == second
    assert "matches" in first
    assert "match_players" in first
    assert "raw_response" in first


def test_connect_creates_parent_directory(tmp_path):
    db = tmp_path / "nested" / "deeper" / "t.db"
    schema.connect(db)
    assert db.parent.is_dir()


def test_foreign_keys_are_enforced(tmp_path):
    """match_players referencing a missing match must fail loudly.

    Phase 2's normaliser relies on this to catch parse ordering bugs.
    """
    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO match_players (match_id, puuid, team, agent) VALUES (?,?,?,?)",
            ("no-such-match", "p1", "Red", "Jett"),
        )
        conn.commit()


@pytest.mark.parametrize("tier,expected", [("basic", 30), ("enhanced", 90)])
def test_tier_limits(tier, expected):
    assert config.TIER_LIMITS[tier] == expected


def test_bad_region_is_rejected(monkeypatch):
    """Config errors explain the fix rather than raising a bare KeyError."""
    monkeypatch.setenv("REGION", "atlantis")
    monkeypatch.setenv("HENRIK_API_KEY", "HDEV-test")
    with pytest.raises(config.ConfigError, match="not a valid region"):
        config.load()
