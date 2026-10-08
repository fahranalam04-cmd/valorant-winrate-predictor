"""Backups: complete, checked before they are trusted, and pruned only after."""

from __future__ import annotations

import itertools
import json
import sqlite3

import pytest

from valwr.store import backup as B
from valwr.store import raw, schema


@pytest.fixture
def source(tmp_path):
    """A small database with every table, a raw response, and a column added
    the way the real schema has grown -- by ALTER TABLE, at the end."""
    path = tmp_path / "live" / "valwr.db"
    path.parent.mkdir()
    conn = schema.connect(path)
    schema.create_all(conn)
    conn.execute(
        "INSERT INTO matches (match_id, started_at, map, mode, queue, region, "
        "season, rounds_red, rounds_blue, winner, data_quality, ingested_at) "
        "VALUES ('m1', 1000, 'Ascent', 'competitive', 'Standard', 'na', 's', "
        "9, 13, 'Blue', NULL, 0)")
    conn.execute("INSERT INTO players (puuid, name, tag) VALUES ('p1', 'n', 't')")
    raw.record(conn, "/valorant/v4/match/na/m1", {}, 200, '{"data": {}}')
    conn.execute("ALTER TABLE players ADD COLUMN added_later TEXT")
    conn.execute("UPDATE players SET added_later = 'kept' WHERE puuid = 'p1'")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def stamps(monkeypatch):
    """Distinct, increasing timestamps: copies made within one second would
    otherwise share a name."""
    counter = itertools.count()
    monkeypatch.setattr(B, "_stamp", lambda: f"2026-10-04-{next(counter):06d}")


def _only_finished_copies(directory):
    """Nothing in a backup directory but copies, their notes, and the log."""
    stray = [p.name for p in directory.iterdir()
             if p.suffix not in (".db", ".json") and p.name != "backup.log"]
    assert not stray, f"left behind: {stray}"


def _rows(path):
    conn = sqlite3.connect(path)
    try:
        return B._counts(conn)
    finally:
        conn.close()


def test_a_full_backup_is_a_checked_copy_of_everything(source, tmp_path, stamps):
    out = B.full_backup(source, tmp_path / "full")
    assert _rows(out) == _rows(source)
    _only_finished_copies(tmp_path / "full")
    note = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert note["quick_check"] == "ok" and note["rows"] == _rows(source)


def test_the_slim_copy_has_everything_but_the_raw_responses(source, tmp_path, stamps):
    full = B.full_backup(source, tmp_path / "full")
    slim = B.slim_backup(full, tmp_path / "slim")
    expected = {t: n for t, n in _rows(full).items() if t != "raw_response"}
    assert _rows(slim) == expected
    assert slim.stat().st_size < full.stat().st_size
    _only_finished_copies(tmp_path / "slim")

    def indexes(path):
        conn = sqlite3.connect(path)
        try:
            return {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND sql IS NOT NULL AND tbl_name != 'raw_response'")}
        finally:
            conn.close()
    assert indexes(slim) == indexes(full), "indexes were not carried over"


def test_columns_added_over_time_keep_their_values(source, tmp_path, stamps):
    """Rebuilt from a fresh schema, a column added by ALTER TABLE could land
    in a different position and take another column's values."""
    full = B.full_backup(source, tmp_path / "full")
    slim = B.slim_backup(full, tmp_path / "slim")
    conn = sqlite3.connect(slim)
    try:
        row = conn.execute("SELECT name, tag, added_later FROM players").fetchone()
    finally:
        conn.close()
    assert row == ("n", "t", "kept")


def test_only_the_newest_copies_are_kept(source, tmp_path, stamps):
    for _ in range(4):
        newest = B.full_backup(source, tmp_path / "full", keep=2)
    kept = sorted((tmp_path / "full").glob(f"{B.FULL_PREFIX}*.db"))
    assert len(kept) == 2 and kept[-1] == newest
    assert len(list((tmp_path / "full").glob("*.json"))) == 2


def test_a_copy_that_fails_its_check_is_not_kept_and_costs_no_older_one(
        source, tmp_path, stamps, monkeypatch):
    good = B.full_backup(source, tmp_path / "full", keep=1)

    def broken(path):
        raise B.BackupError("failed its integrity check")
    monkeypatch.setattr(B, "_verify", broken)
    with pytest.raises(B.BackupError):
        B.full_backup(source, tmp_path / "full", keep=1)
    assert sorted((tmp_path / "full").glob(f"{B.FULL_PREFIX}*")) == [
        good, good.with_suffix(".json")], "the good copy must survive"
    _only_finished_copies(tmp_path / "full")


@pytest.mark.parametrize("kind", ["full", "slim"])
def test_whatever_fails_the_partial_copy_is_removed(source, tmp_path, stamps,
                                                    monkeypatch, kind):
    """Each run has a new name and pruning sees only finished copies, so a
    copy that failed after the check -- OneDrive holding the file during the
    rename, say -- was a file left for good."""
    full = B.full_backup(source, tmp_path / "full") if kind == "slim" else None

    def locked(*a, **k):
        raise PermissionError("the file is in use")
    monkeypatch.setattr(B, "_finish", locked)
    with pytest.raises(PermissionError):
        if kind == "full":
            B.full_backup(source, tmp_path / "full")
        else:
            B.slim_backup(full, tmp_path / "slim")
    _only_finished_copies(tmp_path / kind)


def test_the_slim_copy_checks_for_space_too(source, tmp_path, stamps,
                                             monkeypatch):
    full = B.full_backup(source, tmp_path / "full")
    monkeypatch.setattr(B.shutil, "disk_usage",
                        lambda p: type("U", (), {"free": 1024})())
    with pytest.raises(B.BackupError, match="not enough space"):
        B.slim_backup(full, tmp_path / "slim")
    assert not list((tmp_path / "slim").glob("*"))


def test_a_path_sqlite_would_misread_still_opens_the_right_file(tmp_path):
    # Both legal in a Windows file name, and both read as URI syntax.
    odd = tmp_path / "100% #1.db"
    conn = sqlite3.connect(odd)
    conn.execute("CREATE TABLE t (x)")
    conn.execute("INSERT INTO t VALUES (7)")
    conn.commit()
    conn.close()
    ro = sqlite3.connect(schema.uri(odd), uri=True)
    try:
        assert ro.execute("SELECT x FROM t").fetchone() == (7,)
    finally:
        ro.close()


def test_a_missing_database_asks_whether_the_drive_is_connected(tmp_path):
    with pytest.raises(B.BackupError, match="drive connected"):
        B.full_backup(tmp_path / "unplugged" / "valwr.db", tmp_path / "full")


def test_a_backup_never_fills_the_disk(source, tmp_path, monkeypatch):
    monkeypatch.setattr(B.shutil, "disk_usage",
                        lambda p: type("U", (), {"free": 1024})())
    with pytest.raises(B.BackupError, match="not enough space"):
        B.full_backup(source, tmp_path / "full")
    assert not list((tmp_path / "full").glob("*"))


def test_the_command_logs_a_failure_rather_than_crashing(tmp_path):
    """Run by a scheduled task with no window, the log is all anyone sees."""
    code = B.main(["--full-dir", str(tmp_path / "full"), "--slim-dir",
                   str(tmp_path / "slim")])
    assert code == 1
    log = (tmp_path / "full" / "backup.log").read_text(encoding="utf-8")
    assert "FAILED" in log and "drive connected" in log


def test_the_scheduled_run_waits_for_valorant_to_close(source, tmp_path, monkeypatch):
    """Copying 10 GB onto the game's own drive mid-match can cost frames."""
    from valwr.live import lockfile
    running = iter([True, True, True, False])
    monkeypatch.setattr(lockfile, "game_is_running", lambda: next(running, False))
    monkeypatch.setattr(B.time, "sleep", lambda s: None)
    monkeypatch.setenv("DATABASE_PATH", str(source))
    code = B.main(["--after-game", "--full-dir", str(tmp_path / "full"),
                   "--slim-dir", str(tmp_path / "slim")])
    assert code == 0
    log = (tmp_path / "full" / "backup.log").read_text(encoding="utf-8")
    assert log.index("waiting for it to close") < log.index("full:")


def test_a_game_that_never_closes_is_given_up_on(tmp_path, monkeypatch):
    from valwr.live import lockfile
    monkeypatch.setattr(lockfile, "game_is_running", lambda: True)
    monkeypatch.setattr(B.time, "sleep", lambda s: None)
    code = B.main(["--after-game", "--full-dir", str(tmp_path / "full")])
    assert code == 1
    assert "still running" in (tmp_path / "full" / "backup.log").read_text(
        encoding="utf-8")
