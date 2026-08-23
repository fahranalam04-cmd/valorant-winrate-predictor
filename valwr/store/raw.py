"""Read and write the raw response layer.

Bodies are zlib-compressed JSON. A v4 matchlist response measured ~450 KB per
match uncompressed -- roughly 23 GB across a 50k-match crawl -- and compresses
to about 7.4% of that (~1.7 GB). Compressing rather than trimming fields keeps
the response verbatim, which is the entire reason layer 1 exists: a parsing
mistake should cost a re-parse, never a re-crawl.
"""

from __future__ import annotations

import json
import sqlite3
import time
import zlib
from typing import Any, Iterator

# Level 6 is the useful knee. Level 9 buys ~0.2 percentage points, and lzma is
# meaningfully smaller but far slower -- the crawler writes continuously, so
# compression speed matters more here than the last few percent of size.
LEVEL = 6


def compress(text: str) -> bytes:
    return zlib.compress(text.encode("utf-8"), LEVEL)


def decompress(blob: bytes | str) -> str:
    """Decode a stored body.

    Tolerates uncompressed rows so a database written before compression was
    introduced keeps working.
    """
    if isinstance(blob, str):
        return blob
    try:
        return zlib.decompress(blob).decode("utf-8")
    except zlib.error:
        return blob.decode("utf-8")


def record(
    conn: sqlite3.Connection, endpoint: str, params: dict, status: int, text: str
) -> None:
    conn.execute(
        "INSERT INTO raw_response (endpoint, params, fetched_at, status, body) "
        "VALUES (?,?,?,?,?)",
        (
            endpoint,
            json.dumps(params, sort_keys=True),
            int(time.time()),
            status,
            compress(text),
        ),
    )
    conn.commit()


def load(conn: sqlite3.Connection, row_id: int) -> Any:
    row = conn.execute("SELECT body FROM raw_response WHERE id=?", (row_id,)).fetchone()
    if row is None:
        raise KeyError(f"no raw_response with id={row_id}")
    return json.loads(decompress(row["body"]))


def iter_responses(
    conn: sqlite3.Connection, endpoint_like: str = "%", status: int = 200
) -> Iterator[tuple[int, Any]]:
    """Stream decoded responses. Phase 2's normaliser reads through this.

    Yields one at a time rather than materialising the set -- decompressed
    match blobs are large enough that loading them all at once is a real
    memory problem.
    """
    cur = conn.execute(
        "SELECT id, body FROM raw_response WHERE endpoint LIKE ? AND status = ? ORDER BY id",
        (endpoint_like, status),
    )
    for row in cur:
        yield row["id"], json.loads(decompress(row["body"]))


def storage_summary(conn: sqlite3.Connection) -> dict[str, int]:
    row = conn.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(LENGTH(body)), 0) bytes FROM raw_response"
    ).fetchone()
    return {"responses": row["n"], "compressed_bytes": row["bytes"]}
