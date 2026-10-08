"""Rebuild every model the dashboard uses, in order, or none of them.

    python tools/rebuild.py

Retraining is ten commands, and they depend on one another: the player scores
are percentiles against the training period, their hit rates are measured
against those indexes, and the docs tabulate the measurements. Run by hand,
one was eventually skipped -- `improve.py --retrain` refitted the old score's
index and not the per-role one the page actually scores with, and nothing
noticed. This runs them all, stopping at the first that fails.

The models are backed up first. If any step fails, the backup is put back, so
the dashboard never runs on a mixed set -- a new model scored against an old
index. Reports and docs regenerated before the failure are left for `git diff`
to show (`git checkout -- reports docs README.md` returns them).

On success it writes reports/model_manifest.json, recording what was built,
from which commit and database, and with which library versions, then runs the
audit. The audit is reported, not enforced: after a retrain the prose in the
docs is expected to need its numbers updating, and that is a reading job.

Takes as long as training does. Close the dashboard first, or restart it after:
it keeps the models it loaded.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from valwr.model import manifest  # noqa: E402

PY = sys.executable
STEPS = (
    ("train the win model", [PY, "-m", "valwr.model.train", "--rebuild"]),
    ("draw the README charts", [PY, "-m", "valwr.model.analyze"]),
    ("refit the card's reference index", [PY, "tools/build_perf_index.py"]),
    ("measure it", [PY, "tools/validate_potential.py", "--write-index", "--json"]),
    ("measure the above-rank flag",
     [PY, "tools/validate_potential.py", "--flag", "--json"]),
    ("refit the per-role index", [PY, "tools/build_role_index.py"]),
    # 3,000 teams: the sample docs/SCORE-SPEC.md quotes the figure on.
    ("measure the rate the dashboard quotes",
     [PY, "tools/compare_role_score.py", "--teams", "3000", "--write-index"]),
    ("re-export the sandbox dashboard", [PY, "tools/export_dashboard.py"]),
    ("re-freeze the sandbox benchmark", [PY, "-m", "valwr.sandbox", "benchmark"]),
    ("tabulate every model into the docs", [PY, "tools/model_metrics.py"]),
)
AUDIT = [PY, "tools/audit.py"]
KEEP_BACKUPS = 3
LIBRARIES = ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib")


def backup(models: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = models / f"backup-rebuild-{stamp}"
    dest.mkdir(parents=True)
    for name in manifest.ARTIFACTS:
        if (models / name).exists():
            shutil.copy2(models / name, dest / name)
    # Only this tool's own backups are pruned; the others were made by hand.
    for old in sorted(models.glob("backup-rebuild-*"))[:-KEEP_BACKUPS]:
        shutil.rmtree(old, ignore_errors=True)
    return dest


def restore(models: Path, saved: Path) -> None:
    for name in manifest.ARTIFACTS:
        if (saved / name).exists():
            shutil.copy2(saved / name, models / name)
        else:
            (models / name).unlink(missing_ok=True)    # it did not exist before


def provenance(database: Path, seconds: dict) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    commit = git("rev-parse", "HEAD") or None
    if commit and git("status", "--porcelain", "--untracked-files=no"):
        commit += "+uncommitted"
    versions = {"python": platform.python_version()}
    for lib in LIBRARIES:
        try:
            versions[lib] = version(lib)
        except PackageNotFoundError:
            versions[lib] = None
    import sqlite3
    conn = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        matches, newest = conn.execute(
            "SELECT COUNT(*), MAX(started_at) FROM matches").fetchone()
    finally:
        conn.close()
    return {"by": "tools/rebuild.py",
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commit": commit,
            "database": {"matches": matches, "newest_match": newest},
            "versions": versions,
            # A list, in the order they ran: the manifest sorts its keys.
            "steps": [{"step": k, "seconds": round(v)}
                      for k, v in seconds.items()]}


def run(models: Path, database: Path, runner=subprocess.run,
        manifest_path: Path | None = None) -> int:
    manifest_path = manifest_path or manifest.PATH
    saved = backup(models)
    print(f"  models backed up to {saved.relative_to(models.parent)}")
    seconds: dict[str, float] = {}
    for i, (label, cmd) in enumerate(STEPS, 1):
        print(f"\n  [{i}/{len(STEPS)}] {label}", flush=True)
        started = time.monotonic()
        try:
            code = runner(cmd, cwd=ROOT).returncode
        except (OSError, subprocess.SubprocessError) as e:
            print(f"  could not run it: {e}")
            code = -1
        seconds[label] = time.monotonic() - started
        if code != 0:
            restore(models, saved)
            print(f"\n  '{label}' failed (exit {code}). The previous models are "
                  f"back in place; the dashboard is unaffected.")
            print("  Reports and docs regenerated before it are left for "
                  "`git diff`;\n  `git checkout -- reports docs README.md` "
                  "returns them.")
            return 1
    manifest.write(models, provenance(database, seconds), manifest_path)
    print(f"\n  wrote {manifest_path.name}. "
          f"Rebuilt in {sum(seconds.values()) / 60:.0f} min.")
    print("\n  Audit -- after a retrain, prose figures are expected to need "
          "updating:")
    runner(AUDIT, cwd=ROOT)
    print("\n  Restart the dashboard to load the new models.")
    return 0


def main(argv=None) -> int:
    from valwr import config
    s = config.load(require_key=False)
    if not s.database_path.exists():
        print(f"  no database at {s.database_path}")
        return 1
    return run(s.models_path, s.database_path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
