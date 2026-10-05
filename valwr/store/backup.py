"""Back the database up, and check every copy before trusting it.

    python -m valwr.store.backup            # or backup.bat

Two copies, for two different ways of losing the data:

- **Full**, to the internal drive (``~/valwr-backups``). The database lives on a
  portable SSD, which can be unplugged, dropped or lost; this copy is on a
  different physical disk. Everything, raw API responses included.
- **Slim**, to OneDrive when it is present (``~/OneDrive/valwr-backups``). Every
  match, player and prediction, without the raw responses -- 9.8 of the 10.4
  GB, and only needed to re-parse old matches. Small enough for a free cloud
  tier, so the data survives the PC itself.

A copy is written under a ``.partial`` name, checked with ``quick_check`` and
its row counts, and only then given its real name. A half-written file never
looks like a backup, and older copies are pruned only after a new one passes.

The full copy uses SQLite's online backup, which reads one consistent snapshot
while the crawler and the dashboard carry on writing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

# The scheduled run waits for VALORANT to close rather than copy 10 GB onto
# the drive the game is running from, which can cost frames mid-match.
GAME_WAIT_POLL_SECONDS = 60
GAME_WAIT_LIMIT_SECONDS = 12 * 3600

FULL_PREFIX = "valwr-full-"
SLIM_PREFIX = "valwr-slim-"
# The raw API bodies: the bulk of the file, and only needed for re-parsing.
LEFT_OUT_OF_SLIM = ("raw_response",)


class BackupError(RuntimeError):
    """Raised with what went wrong and what to do about it."""


def _uri(path: Path, readonly: bool = True) -> str:
    """A SQLite URI for `path`, escaped, read-only unless asked otherwise."""
    return f"file:{quote(path.as_posix(), safe='/:')}" + ("?mode=ro" if readonly else "")


def default_full_dir() -> Path:
    return Path.home() / "valwr-backups"


def default_slim_dir() -> Path | None:
    onedrive = Path.home() / "OneDrive"
    return onedrive / "valwr-backups" if onedrive.is_dir() else None


def _counts(conn: sqlite3.Connection, schema: str = "main") -> dict[str, int]:
    names = [r[0] for r in conn.execute(
        f"SELECT name FROM {schema}.sqlite_master WHERE type='table' "
        f"AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {n: conn.execute(f'SELECT COUNT(*) FROM {schema}."{n}"').fetchone()[0]
            for n in names}


def _verify(path: Path) -> dict[str, int]:
    """quick_check, then the row counts. Raises if the copy is not sound."""
    conn = sqlite3.connect(_uri(path), uri=True)
    try:
        got = conn.execute("PRAGMA quick_check").fetchone()[0]
        if got != "ok":
            raise BackupError(f"{path.name} failed its integrity check: {got}")
        return _counts(conn)
    finally:
        conn.close()


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H%M%S")


def _prune(directory: Path, prefix: str, keep: int) -> list[Path]:
    """Delete all but the newest `keep` finished copies. Names sort by time."""
    done = sorted(directory.glob(f"{prefix}*.db"))
    gone = done[:-keep] if keep > 0 else []
    for old in gone:
        old.unlink()
        old.with_suffix(".json").unlink(missing_ok=True)
    return gone


def _need_space(directory: Path, needed: int) -> None:
    free = shutil.disk_usage(directory).free
    # Room for the copy and a margin, so a backup never fills the disk the way
    # a runaway write-ahead log once did.
    if free < needed * 1.1 + 2 * 1024 ** 3:
        raise BackupError(
            f"not enough space in {directory}: {free / 1e9:.1f} GB free, "
            f"{needed / 1e9:.1f} GB needed plus a 2 GB margin")


def _discard(partial: Path) -> None:
    """A failed copy, and any journal files opening it left beside it."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        partial.with_name(partial.name + suffix).unlink(missing_ok=True)


def _finish(partial: Path, final: Path, counts: dict, source: Path) -> None:
    partial.replace(final)
    final.with_suffix(".json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": str(source), "bytes": final.stat().st_size,
        "quick_check": "ok", "rows": counts}, indent=2), encoding="utf-8")


def full_backup(source: Path, directory: Path, keep: int = 2) -> Path:
    """A consistent copy of the whole database, checked, then the old pruned."""
    if not source.exists():
        raise BackupError(
            f"no database at {source}. If it lives on the portable drive, "
            f"is the drive connected?")
    directory.mkdir(parents=True, exist_ok=True)
    _need_space(directory, source.stat().st_size)
    final = directory / f"{FULL_PREFIX}{_stamp()}.db"
    partial = final.with_name(final.name + ".partial")
    src = sqlite3.connect(_uri(source), uri=True)
    dst = sqlite3.connect(partial)
    try:
        # One step: the whole copy is read inside a single snapshot, so it is
        # consistent even with the crawler and dashboard writing meanwhile.
        src.backup(dst)
        # The copy inherits the live database's write-ahead log, so merely
        # opening it to check it left -wal and -shm files beside it. A backup
        # should be one self-contained file.
        dst.execute("PRAGMA journal_mode=DELETE")
    finally:
        dst.close()
        src.close()
    try:
        counts = _verify(partial)
    except Exception:
        _discard(partial)
        raise
    _finish(partial, final, counts, source)
    _prune(directory, FULL_PREFIX, keep)
    return final


def slim_backup(full: Path, directory: Path, keep: int = 4) -> Path:
    """Everything in a checked full copy except the raw API responses.

    Built from the full copy rather than the live database, so both describe
    the same moment, and from each table's own CREATE statement so columns
    added over time keep their order.
    """
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{SLIM_PREFIX}{_stamp()}.db"
    partial = final.with_name(final.name + ".partial")
    _discard(partial)
    # URI mode on the main connection too: without it SQLite reads the
    # ATTACH argument as a plain filename and creates an empty database
    # called "file:...", attaching that instead of the backup.
    conn = sqlite3.connect(_uri(partial, readonly=False), uri=True)
    try:
        conn.execute("ATTACH DATABASE ? AS src", (_uri(full),))
        objects = conn.execute(
            "SELECT type, name, tbl_name, sql FROM src.sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "ORDER BY type = 'index'").fetchall()
        for kind, name, table, sql in objects:
            if table in LEFT_OUT_OF_SLIM:
                continue
            conn.execute(sql)
            if kind == "table":
                conn.execute(f'INSERT INTO main."{name}" SELECT * FROM src."{name}"')
        conn.commit()
        expected = {t: n for t, n in _counts(conn, "src").items()
                    if t not in LEFT_OUT_OF_SLIM}
        conn.execute("DETACH DATABASE src")
    finally:
        conn.close()
    try:
        counts = _verify(partial)
        if counts != expected:
            raise BackupError(f"slim copy rows differ from the full copy: "
                              f"{counts} != {expected}")
    except Exception:
        _discard(partial)
        raise
    _finish(partial, final, counts, full)
    _prune(directory, SLIM_PREFIX, keep)
    return final


def _wait_for_the_game_to_close(say) -> bool:
    """True once VALORANT is not running; False if it never closed in time."""
    from valwr.live import lockfile
    waited = 0
    if lockfile.game_is_running():
        say("VALORANT is running -- waiting for it to close before copying")
    while lockfile.game_is_running():
        if waited >= GAME_WAIT_LIMIT_SECONDS:
            say(f"FAILED: VALORANT still running after "
                f"{GAME_WAIT_LIMIT_SECONDS // 3600} hours; the next run will try")
            return False
        time.sleep(GAME_WAIT_POLL_SECONDS)
        waited += GAME_WAIT_POLL_SECONDS
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="valwr.store.backup")
    ap.add_argument("--full-dir", type=Path, default=default_full_dir())
    ap.add_argument("--slim-dir", type=Path, default=default_slim_dir(),
                    help="default: OneDrive/valwr-backups when OneDrive exists")
    ap.add_argument("--keep-full", type=int, default=2)
    ap.add_argument("--keep-slim", type=int, default=4)
    ap.add_argument("--after-game", action="store_true",
                    help="wait for VALORANT to close first; the scheduled run "
                         "uses this")
    args = ap.parse_args(argv)

    from valwr import config
    source = config.load(require_key=False).database_path
    log = args.full_dir / "backup.log"

    def say(msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        try:
            args.full_dir.mkdir(parents=True, exist_ok=True)
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    if args.after_game and not _wait_for_the_game_to_close(say):
        return 1

    started = time.monotonic()
    try:
        full = full_backup(source, args.full_dir, args.keep_full)
        say(f"full: {full}  ({full.stat().st_size / 1e9:.2f} GB, checked)")
        if args.slim_dir is not None:
            slim = slim_backup(full, args.slim_dir, args.keep_slim)
            say(f"slim: {slim}  ({slim.stat().st_size / 1e9:.2f} GB, checked)")
    except (BackupError, sqlite3.Error, OSError) as e:
        say(f"FAILED: {e}")
        return 1
    say(f"done in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
