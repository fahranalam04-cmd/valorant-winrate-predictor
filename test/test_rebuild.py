"""One command rebuilds every model, or leaves the old ones in place."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sqlite3
import subprocess

import numpy as np
import pytest

from valwr.model import manifest as M

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def models(tmp_path):
    """A models/ directory holding a small but real-shaped set."""
    import joblib
    d = tmp_path / "models"
    d.mkdir()
    joblib.dump({"best": "logistic regression", "columns": ["a", "b", "c"],
                 "norms_as_of": 1_000,
                 "metrics": {"logistic regression": {
                     "n": np.int64(10_304), "log_loss": np.float64(0.6874271),
                     "auc": 0.56, "accuracy": 0.5426}}}, d / "model.joblib")
    (d / "role_index.json").write_text(
        json.dumps({"as_of": 1_500, "n": 26_427, "top1_rate": 0.3}), "utf-8")
    (d / "perf_index.json").write_text(
        json.dumps({"as_of": 1_000, "n": 39_869, "top1_rate": 0.3047}), "utf-8")
    return d


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "v.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE matches (match_id TEXT, started_at INTEGER)")
    conn.executemany("INSERT INTO matches VALUES (?, ?)", [("a", 5), ("b", 9)])
    conn.commit()
    conn.close()
    return path


# --- the manifest ------------------------------------------------------

def test_the_manifest_carries_each_files_fingerprint_and_facts(models):
    got = M.describe(models)["artifacts"]
    assert list(got) == list(M.ARTIFACTS)
    assert got["model.joblib"]["model"] == "logistic regression"
    assert got["model.joblib"]["features"] == 3
    test = got["model.joblib"]["test"]
    assert test["n"] == 10_304 and isinstance(test["n"], int), \
        "a count must not turn into a float on the way through"
    assert test["log_loss"] == 0.687427
    assert got["role_index.json"] == {**M.fingerprint(models / "role_index.json"),
                                      "as_of": 1_500, "n": 26_427,
                                      "top1_rate": 0.3}


def test_a_described_set_matches_until_a_file_changes(models, tmp_path):
    m = M.write(models, path=tmp_path / "manifest.json")
    assert M.mismatches(models, m) == []
    (models / "role_index.json").write_text("{}", "utf-8")
    (models / "perf_index.json").unlink()
    assert M.mismatches(models, m) == ["role_index.json", "perf_index.json"]


def test_a_manifest_written_after_the_fact_says_so(models, tmp_path):
    path = tmp_path / "manifest.json"
    M.write(models, path=path)
    text = path.read_bytes()
    assert text.endswith(b"}\n") and b"\r\n" not in text, \
        "LF and a final newline, or every rebuild is a whole-file diff"
    assert M.load(path)["built"]["by"] is None
    assert "not recorded" in M.load(path)["built"]["note"]


def test_no_manifest_is_none_not_an_error(tmp_path):
    assert M.load(tmp_path / "nope.json") is None


# --- the rebuild -------------------------------------------------------

class Runner:
    """Stands in for subprocess.run: records each command, fails on one."""

    def __init__(self, models, fail_at=None, writes=None):
        self.models, self.fail_at, self.writes = models, fail_at, writes or {}
        self.ran = []

    def __call__(self, cmd, cwd=None):
        self.ran.append(cmd)
        step = len(self.ran)
        for name, body in self.writes.get(step, {}).items():
            (self.models / name).write_text(body, "utf-8")
        return subprocess.CompletedProcess(cmd, 1 if step == self.fail_at else 0)


def test_the_steps_run_in_dependency_order():
    """Each index after the model whose norms it is fitted against, each
    measurement after its index, and the tables after the measurements."""
    rebuild = _tool("rebuild")
    names = [" ".join(cmd[1:]) for _, cmd in rebuild.STEPS]

    def at(fragment):
        hits = [i for i, n in enumerate(names) if fragment in n]
        assert len(hits) == 1, f"{fragment}: {names}"
        return hits[0]
    train = at("valwr.model.train --rebuild")
    assert train == 0
    assert train < at("build_perf_index") < at("validate_potential.py --write-index")
    assert train < at("build_role_index") < at("compare_role_score")
    assert at("model_metrics") > max(at("--write-index --json"), at("--flag"),
                                     at("compare_role_score"))


def test_the_rate_the_dashboard_quotes_is_measured_as_the_docs_state_it():
    """docs/SCORE-SPEC.md quotes it on 3,000 teams; measured on the default
    1,500 the audit would compare two different samples."""
    rebuild = _tool("rebuild")
    cmd = next(c for _, c in rebuild.STEPS if "compare_role_score" in " ".join(c))
    assert cmd[cmd.index("--teams") + 1] == "3000" and "--write-index" in cmd


def test_a_successful_rebuild_records_how_it_was_built(models, database, tmp_path):
    rebuild = _tool("rebuild")
    runner = Runner(models)
    path = tmp_path / "manifest.json"
    assert rebuild.run(models, database, runner=runner, manifest_path=path) == 0
    assert runner.ran == [cmd for _, cmd in rebuild.STEPS] + [rebuild.AUDIT]
    built = M.load(path)["built"]
    assert built["by"] == "tools/rebuild.py"
    assert built["database"] == {"matches": 2, "newest_match": 9}
    assert [s["step"] for s in built["steps"]] == [
        label for label, _ in rebuild.STEPS], "in the order they ran"
    assert set(built["versions"]) >= {"python", "scikit-learn", "lightgbm"}
    assert M.mismatches(models, M.load(path)) == []


def test_a_failed_step_puts_every_previous_model_back(models, database, tmp_path):
    """Training succeeded and the role index did not: left alone, the page
    would score with a new model against an old reference."""
    rebuild = _tool("rebuild")
    before = {n: (models / n).read_bytes() for n in M.ARTIFACTS}
    role_step = next(i for i, (_, c) in enumerate(rebuild.STEPS, 1)
                     if "build_role_index" in " ".join(c))
    runner = Runner(models, fail_at=role_step,
                    writes={1: {"model.joblib": "new model"},
                            3: {"perf_index.json": "{}"}})
    path = tmp_path / "manifest.json"
    assert rebuild.run(models, database, runner=runner, manifest_path=path) == 1
    assert {n: (models / n).read_bytes() for n in M.ARTIFACTS} == before
    assert len(runner.ran) == role_step, "it went on past the failure"
    assert rebuild.AUDIT not in runner.ran
    assert M.load(path) is None, "a failed build must not be described"


def test_stopping_it_mid_step_puts_the_old_models_back(models, database,
                                                      tmp_path):
    """Ctrl-C in a half-hour training run is the likeliest way to stop it."""
    rebuild = _tool("rebuild")
    before = (models / "model.joblib").read_bytes()

    def interrupted(cmd, cwd=None):
        (models / "model.joblib").write_text("half-written", "utf-8")
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        rebuild.run(models, database, runner=interrupted,
                    manifest_path=tmp_path / "manifest.json")
    assert (models / "model.joblib").read_bytes() == before


def test_the_commit_is_read_before_the_steps_change_the_tree(
        models, database, tmp_path, monkeypatch):
    """The steps rewrite tracked reports and docs; read afterwards, the tree
    was always modified and every build was recorded as uncommitted."""
    rebuild = _tool("rebuild")
    order = []
    monkeypatch.setattr(rebuild, "commit", lambda: order.append("commit") or "abc")

    def runner(cmd, cwd=None):
        order.append("step")
        return subprocess.CompletedProcess(cmd, 0)
    path = tmp_path / "manifest.json"
    rebuild.run(models, database, runner=runner, manifest_path=path)
    assert order[0] == "commit" and order.count("commit") == 1
    assert M.load(path)["built"]["commit"] == "abc"


def test_describing_unchanged_models_keeps_how_they_were_built(
        models, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "PATH", tmp_path / "manifest.json")
    monkeypatch.setenv("MODELS_PATH", str(models))
    built = {"by": "tools/rebuild.py", "commit": "abc"}
    M.write(models, built)
    assert M.main([]) == 0
    assert M.load()["built"] == built, "the same files, the same build"
    (models / "perf_index.json").write_text("{}", "utf-8")
    assert M.main([]) == 0
    assert M.load()["built"]["by"] is None, "different files: not that build"


def test_a_file_that_did_not_exist_before_is_removed_again(models, database,
                                                           tmp_path):
    rebuild = _tool("rebuild")
    (models / "role_index.json").unlink()
    runner = Runner(models, fail_at=2, writes={1: {"role_index.json": "{}"}})
    rebuild.run(models, database, runner=runner,
                manifest_path=tmp_path / "manifest.json")
    assert not (models / "role_index.json").exists()


def test_a_command_that_cannot_start_counts_as_a_failure(models, database,
                                                         tmp_path):
    rebuild = _tool("rebuild")

    def missing(cmd, cwd=None):
        raise FileNotFoundError(cmd[0])
    assert rebuild.run(models, database, runner=missing,
                       manifest_path=tmp_path / "manifest.json") == 1


def test_only_its_own_backups_are_pruned(models, monkeypatch):
    rebuild = _tool("rebuild")
    (models / "backup-2026-09-26").mkdir()
    for stamp in ("20260101-000000", "20260102-000000", "20260103-000000",
                  "20260104-000000"):
        (models / f"backup-rebuild-{stamp}").mkdir()
    newest = rebuild.backup(models)
    kept = sorted(p.name for p in models.glob("backup-rebuild-*"))
    assert len(kept) == rebuild.KEEP_BACKUPS and newest.name in kept
    assert "backup-rebuild-20260101-000000" not in kept
    assert (models / "backup-2026-09-26").exists(), "a hand-made backup"


# --- the audit and the preflight --------------------------------------

def _audit_against(tmp_path, monkeypatch, models, spec_rows, shipped=None,
                   role_as_of=1_500):
    audit = _tool("audit")
    (models / "role_index.json").write_text(
        json.dumps({"as_of": role_as_of, "n": 26_427, "top1_rate": 0.3}), "utf-8")
    root = tmp_path / "repo"
    (root / "reports").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "reports" / "results.json").write_text(json.dumps(
        {"shipped": shipped or "logistic regression", "n_test": 10_304}), "utf-8")
    (root / "docs" / "SCORE-SPEC.md").write_text(
        "| | top-1 |\n|---|---|\n" + "".join(
            f"| **the per-role score** | **{r}%** |\n" for r in spec_rows), "utf-8")
    monkeypatch.setattr(audit, "ROOT", root)
    monkeypatch.setattr(M, "PATH", root / "reports" / "model_manifest.json")
    monkeypatch.setenv("MODELS_PATH", str(models))
    M.write(models)
    rep = audit.Report()
    audit.check_manifest(rep)
    return audit, rep


def test_the_audit_passes_a_described_set(tmp_path, monkeypatch, models):
    _, rep = _audit_against(tmp_path, monkeypatch, models, ["27.7", "30.0"])
    assert rep.problems == [], "the earlier row is history, not the claim"


def test_the_audit_catches_a_stale_quote(tmp_path, monkeypatch, models):
    _, rep = _audit_against(tmp_path, monkeypatch, models, ["30.0", "28.8"])
    assert any("28.8%" in p for p in rep.problems)


def test_the_audit_catches_an_index_older_than_its_model(tmp_path, monkeypatch,
                                                         models):
    _, rep = _audit_against(tmp_path, monkeypatch, models, ["30.0"],
                            role_as_of=900)
    assert any("role_index.json" in p and "before the model" in p
               for p in rep.problems)


def test_the_audit_catches_a_manifest_for_another_model(tmp_path, monkeypatch,
                                                        models):
    _, rep = _audit_against(tmp_path, monkeypatch, models, ["30.0"],
                            shipped="lightgbm")
    assert any("results.json ships 'lightgbm'" in p for p in rep.problems)


def test_the_audit_catches_files_that_are_not_the_described_ones(
        tmp_path, monkeypatch, models):
    audit, rep = _audit_against(tmp_path, monkeypatch, models, ["30.0"])
    assert rep.problems == []
    (models / "perf_index.json").write_text("{}", "utf-8")
    rep = audit.Report()
    audit.check_manifest(rep)
    assert any("perf_index.json differ" in p for p in rep.problems)


def test_without_models_the_audit_still_checks_the_rest(tmp_path, monkeypatch,
                                                        models):
    """CI has no models; the committed manifest is still checked against the
    committed reports and docs."""
    audit, _ = _audit_against(tmp_path, monkeypatch, models, ["28.8"])
    monkeypatch.setenv("MODELS_PATH", str(tmp_path / "none"))
    rep = audit.Report()
    audit.check_manifest(rep)
    assert any("28.8%" in p for p in rep.problems)
    assert any("no models on disk" in n for n in rep.notes)


def test_the_preflight_warns_but_does_not_stop(tmp_path, monkeypatch, models):
    preflight = _tool("preflight")
    monkeypatch.setattr(M, "PATH", tmp_path / "manifest.json")
    monkeypatch.setenv("MODELS_PATH", str(models))
    # Its own, so this never depends on what the shared sandbox holds.
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "absent.db"))
    monkeypatch.setattr("valwr.live.lockfile.game_is_running", lambda: False)
    # The indexes here are stand-ins; loading them is not what is under test.
    monkeypatch.setattr("valwr.rating.roleindex.RoleIndex.load",
                        classmethod(lambda cls: (_ for _ in ()).throw(
                            FileNotFoundError("stand-in"))))
    monkeypatch.setattr("valwr.rating.potential.PerfIndex.load",
                        classmethod(lambda cls: (_ for _ in ()).throw(
                            FileNotFoundError("stand-in"))))
    M.write(models)
    lines, problems, _ = preflight.check_all()
    assert any("model manifest" in ln and "matches" in ln for ln in lines)
    (models / "model.joblib").write_bytes((models / "model.joblib").read_bytes()
                                          + b"\0")
    lines, problems, _ = preflight.check_all()
    warned = [p for p in problems if "model_manifest.json" in p]
    assert warned and all(p.startswith("Optional") for p in warned), \
        "a warning, never a reason to refuse to start"
    assert "tools\\rebuild.py" in warned[0]
