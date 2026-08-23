"""Tests for the compressed raw-response layer."""

from __future__ import annotations

import json

from valwr.store import raw, schema


def test_roundtrip_preserves_bytes_exactly(tmp_path):
    """Verbatim is the point -- a re-parse must see the original text."""
    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)

    payload = {"data": [{"metadata": {"match_id": "abc"}, "players": [{"puuid": "p"}]}]}
    text = json.dumps(payload)
    raw.record(conn, "/v4/matches", {"size": 5}, 200, text)

    assert raw.load(conn, 1) == payload
    stored = conn.execute("SELECT body FROM raw_response WHERE id=1").fetchone()["body"]
    assert isinstance(stored, bytes)
    assert raw.decompress(stored) == text


def test_decompress_tolerates_uncompressed_rows(tmp_path):
    """A database written before compression existed must still read."""
    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)
    conn.execute(
        "INSERT INTO raw_response (endpoint, params, fetched_at, status, body) "
        "VALUES (?,?,?,?,?)",
        ("/v4/matches", "{}", 0, 200, '{"data": []}'),
    )
    conn.commit()
    assert raw.load(conn, 1) == {"data": []}


def test_iter_responses_filters_by_endpoint_and_status(tmp_path):
    conn = schema.connect(tmp_path / "t.db")
    schema.create_all(conn)
    raw.record(conn, "/v4/matches", {}, 200, '{"n": 1}')
    raw.record(conn, "/v3/leaderboard", {}, 200, '{"n": 2}')
    raw.record(conn, "/v4/matches", {}, 429, '{"n": 3}')

    got = [d for _, d in raw.iter_responses(conn, "%matches%")]
    assert got == [{"n": 1}]


def test_compression_is_worth_doing(tmp_path):
    """Guards the storage assumption the schema comment documents."""
    # Repetitive JSON, like a real match blob's player/round arrays.
    text = json.dumps([{"puuid": "x" * 36, "kills": 10, "deaths": 8}] * 500)
    ratio = len(raw.compress(text)) / len(text.encode())
    assert ratio < 0.20, f"expected strong compression, got {ratio:.1%}"
